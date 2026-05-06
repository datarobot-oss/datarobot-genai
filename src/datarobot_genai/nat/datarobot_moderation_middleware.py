# Copyright 2026 DataRobot, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""NAT (NeMo Agent Toolkit) middleware: DataRobot LLM guardrails for DRAgent workflows."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import queue
import threading
import uuid
from collections.abc import AsyncIterator
from collections.abc import Iterator
from datetime import UTC
from datetime import datetime
from typing import Any
from typing import Literal
from typing import cast

import numpy as np
import pandas as pd
from ag_ui.core import AssistantMessage
from ag_ui.core import Event
from ag_ui.core import EventType
from ag_ui.core import RunAgentInput
from ag_ui.core import SystemMessage
from ag_ui.core import TextMessageChunkEvent
from ag_ui.core import TextMessageContentEvent
from ag_ui.core import TextMessageEndEvent
from ag_ui.core import TextMessageStartEvent
from ag_ui.core import ToolCallArgsEvent
from ag_ui.core import ToolCallChunkEvent
from ag_ui.core import ToolCallEndEvent
from ag_ui.core import ToolCallStartEvent
from ag_ui.core import ToolMessage
from ag_ui.core import UserMessage
from datarobot_dome.api import ModerationPipeline
from datarobot_dome.chat_helper import run_postscore_guards
from datarobot_dome.constants import CHAT_COMPLETION_OBJECT
from datarobot_dome.constants import DATAROBOT_MODERATIONS_ATTR
from datarobot_dome.constants import DISABLE_MODERATION_RUNTIME_PARAM_NAME
from datarobot_dome.constants import MODERATION_CONFIG_FILE_NAME
from datarobot_dome.constants import NONE_CUSTOM_PY_RESPONSE
from datarobot_dome.constants import GuardStage
from datarobot_dome.otel_helpers import report_otel_evaluation_set_metric
from datarobot_dome.runtime import get_runtime_parameter_value_bool
from datarobot_dome.streaming import ModerationIterator
from datarobot_dome.streaming import StreamingContextBuilder
from datarobot_moderation_interface.drum_integration import (
    _handle_result_df_error_cases as handle_result_df_error_cases,
)
from datarobot_moderation_interface.drum_integration import (
    _set_moderation_attribute_to_completion as set_moderation_attribute_to_completion,
)
from datarobot_moderation_interface.drum_integration import build_non_streaming_chat_completion
from datarobot_moderation_interface.drum_integration import build_predictions_df_from_completion
from datarobot_moderation_interface.drum_integration import format_result_df
from datarobot_moderation_interface.drum_integration import get_chat_prompt
from datarobot_moderation_interface.drum_integration import run_prescore_guards
from nat.builder.builder import Builder
from nat.cli.register_workflow import register_middleware
from nat.data_models.api_server import ChatRequest
from nat.data_models.api_server import ChatRequestOrMessage
from nat.data_models.api_server import ChatResponse
from nat.data_models.api_server import ChatResponseChunk
from nat.data_models.api_server import ChatResponseChunkChoice
from nat.data_models.api_server import ChoiceDelta
from nat.data_models.api_server import ChoiceDeltaToolCall
from nat.data_models.api_server import ChoiceDeltaToolCallFunction
from nat.data_models.api_server import Usage as NATUsage
from nat.data_models.api_server import UserMessageContentRoleType
from nat.data_models.middleware import FunctionMiddlewareBaseConfig
from nat.middleware.function_middleware import FunctionMiddleware
from nat.middleware.middleware import CallNext
from nat.middleware.middleware import CallNextStream
from nat.middleware.middleware import FunctionMiddlewareContext
from nat.middleware.middleware import InvocationContext
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion import Choice as OpenAIChoice
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice as OpenAIChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta as OpenAIChoiceDelta
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall as OpenAIChoiceDeltaToolCall
from openai.types.chat.chat_completion_chunk import (
    ChoiceDeltaToolCallFunction as OpenAIChoiceDeltaToolCallFunction,
)
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message_tool_call import Function as OpenAIToolFunction
from openai.types.completion_usage import CompletionUsage
from pydantic import Field

from datarobot_genai.core.agents import default_usage_metrics
from datarobot_genai.dragent.frontends.converters import (
    convert_dragent_event_response_to_chat_response_chunk,
)
from datarobot_genai.dragent.frontends.response import DRAgentEventResponse
from datarobot_genai.dragent.frontends.tool_call_registry import register_tool_call

_logger = logging.getLogger(__name__)


def load_llm_moderation_pipeline(model_dir: str | None) -> ModerationPipeline | None:
    """Load YAML LLM moderation for NAT via ``ModerationPipeline.from_yaml`` (not DRUM ``init``)."""
    if get_runtime_parameter_value_bool(DISABLE_MODERATION_RUNTIME_PARAM_NAME, default_value=False):
        _logger.warning("Moderation is disabled via runtime parameter on the model")
        return None

    os.environ["RAGAS_DO_NOT_TRACK"] = "true"
    os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"

    base = model_dir if model_dir is not None else os.getcwd()
    guard_config_file = os.path.join(base, MODERATION_CONFIG_FILE_NAME)
    if not os.path.exists(guard_config_file):
        _logger.warning(
            "Guard config file: %s not found; moderations will not be enforced",
            guard_config_file,
        )
        return None

    pipeline = ModerationPipeline.from_yaml(guard_config_file)
    os.environ["PROMPT_COLUMN_NAME"] = pipeline._pipeline.get_input_column(GuardStage.PROMPT)
    os.environ["RESPONSE_COLUMN_NAME"] = pipeline._pipeline.get_input_column(GuardStage.RESPONSE)
    return pipeline


def _text_for_moderation_eval(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value)


def _optional_prompt_for_moderation_eval(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    s = str(value)
    return s or None


_FINISH_REASON = Literal["stop", "length", "tool_calls", "content_filter", "function_call"]


def _nat_chat_response_chunk_to_openai_chat_completion(
    chunk: ChatResponseChunk,
) -> ChatCompletion:
    if not chunk.choices:
        raise ValueError("ChatResponseChunk has no choices")
    c0 = chunk.choices[0]
    delta = c0.delta
    openai_tool_calls: list[ChatCompletionMessageToolCall] | None = None
    if delta.tool_calls:
        openai_tool_calls = []
        for tc in delta.tool_calls:
            fn = tc.function
            openai_tool_calls.append(
                ChatCompletionMessageToolCall(
                    id=tc.id or str(uuid.uuid4()),
                    type="function",
                    function=OpenAIToolFunction(
                        name=(fn.name if fn else None) or "",
                        arguments=(fn.arguments if fn else None) or "",
                    ),
                )
            )
    finish: _FINISH_REASON
    if c0.finish_reason is not None:
        finish = cast(_FINISH_REASON, c0.finish_reason)
    elif openai_tool_calls:
        finish = "tool_calls"
    else:
        finish = "stop"
    msg = ChatCompletionMessage(
        role="assistant",
        content=delta.content,
        tool_calls=openai_tool_calls,
    )
    created = chunk.created
    if created.tzinfo is None:
        created = created.replace(tzinfo=UTC)
    created_ts = int(created.timestamp())
    return ChatCompletion(
        id=chunk.id,
        choices=[OpenAIChoice(finish_reason=finish, index=0, message=msg)],
        created=created_ts,
        model=chunk.model,
        object=CHAT_COMPLETION_OBJECT,
        usage=None,
    )


def _tool_calls_from_ag_ui_events(
    events: list[Any],
) -> list[ChatCompletionMessageToolCall] | None:
    """Rebuild OpenAI tool_calls from AG-UI streaming events (original_chunk is often absent)."""
    by_id: dict[str, dict[str, Any]] = {}
    for ev in events:
        if isinstance(ev, ToolCallStartEvent):
            by_id[ev.tool_call_id] = {"name": ev.tool_call_name, "args": []}
        elif isinstance(ev, ToolCallArgsEvent):
            st = by_id.get(ev.tool_call_id)
            if st is not None and ev.delta:
                st["args"].append(ev.delta)
        elif isinstance(ev, ToolCallChunkEvent):
            tid = ev.tool_call_id
            if tid and tid in by_id and ev.delta:
                by_id[tid]["args"].append(ev.delta)
        elif isinstance(ev, ToolCallEndEvent):
            by_id.setdefault(ev.tool_call_id, {"name": "", "args": []})
    calls: list[ChatCompletionMessageToolCall] = []
    for tid, st in by_id.items():
        name = st.get("name") or ""
        arguments = "".join(st.get("args") or [])
        if not name and not arguments:
            continue
        calls.append(
            ChatCompletionMessageToolCall(
                id=tid,
                type="function",
                function=OpenAIToolFunction(name=name, arguments=arguments),
            )
        )
    return calls or None


def _nat_choice_delta_tool_calls_to_openai(
    nat_tool_calls: list[ChoiceDeltaToolCall] | None,
) -> list[OpenAIChoiceDeltaToolCall] | None:
    if not nat_tool_calls:
        return None
    out: list[OpenAIChoiceDeltaToolCall] = []
    for tc in nat_tool_calls:
        fn = tc.function
        out.append(
            OpenAIChoiceDeltaToolCall(
                index=tc.index,
                id=tc.id,
                type=tc.type or "function",
                function=(
                    OpenAIChoiceDeltaToolCallFunction(
                        name=(fn.name if fn else None) or "",
                        arguments=(fn.arguments if fn else None) or "",
                    )
                    if fn is not None
                    else None
                ),
            )
        )
    return out or None


def _nat_chat_response_chunk_to_openai_chat_completion_chunk(
    chunk: ChatResponseChunk,
) -> ChatCompletionChunk:
    """Map NAT streaming chunk to OpenAI ``chat.completion.chunk`` (delta, not aggregated message)."""
    if not chunk.choices:
        raise ValueError("ChatResponseChunk has no choices")
    c0 = chunk.choices[0]
    delta = c0.delta
    openai_delta = OpenAIChoiceDelta(
        content=delta.content,
        role=delta.role.value if delta.role is not None else None,
        tool_calls=_nat_choice_delta_tool_calls_to_openai(delta.tool_calls),
    )
    finish = cast(_FINISH_REASON | None, c0.finish_reason)
    choice = OpenAIChunkChoice(
        index=0,
        delta=openai_delta,
        finish_reason=finish,
    )
    created = chunk.created
    if created.tzinfo is None:
        created = created.replace(tzinfo=UTC)
    created_ts = int(created.timestamp())
    usage_openai: CompletionUsage | None = None
    if chunk.usage is not None:
        u = chunk.usage
        usage_openai = CompletionUsage(
            prompt_tokens=u.prompt_tokens,
            completion_tokens=u.completion_tokens,
            total_tokens=u.total_tokens,
        )
    return ChatCompletionChunk(
        id=chunk.id,
        choices=[choice],
        created=created_ts,
        model=chunk.model,
        object="chat.completion.chunk",
        usage=usage_openai,
    )


def _message_tool_calls_to_openai_delta_tool_calls(
    tool_calls: list[ChatCompletionMessageToolCall],
) -> list[OpenAIChoiceDeltaToolCall]:
    out: list[OpenAIChoiceDeltaToolCall] = []
    for idx, tc in enumerate(tool_calls):
        if tc.type != "function":
            continue
        out.append(
            OpenAIChoiceDeltaToolCall(
                index=idx,
                id=tc.id,
                type=tc.type,
                function=OpenAIChoiceDeltaToolCallFunction(
                    name=tc.function.name,
                    arguments=tc.function.arguments,
                ),
            )
        )
    return out


def _openai_choice_delta_tool_calls_to_nat(
    tool_calls: list[OpenAIChoiceDeltaToolCall] | None,
) -> list[ChoiceDeltaToolCall] | None:
    if not tool_calls:
        return None
    built: list[ChoiceDeltaToolCall] = []
    for tc in tool_calls:
        if tc.type and tc.type != "function":
            continue
        fn = tc.function
        built.append(
            ChoiceDeltaToolCall(
                index=tc.index,
                id=tc.id,
                type=tc.type or "function",
                function=(
                    ChoiceDeltaToolCallFunction(
                        name=fn.name if fn else None,
                        arguments=fn.arguments if fn else None,
                    )
                    if fn is not None
                    else None
                ),
            )
        )
    return built or None


def _openai_stream_role_to_nat(
    role: str | None,
) -> UserMessageContentRoleType | None:
    if role is None:
        return None
    if role == "assistant":
        return UserMessageContentRoleType.ASSISTANT
    if role == "user":
        return UserMessageContentRoleType.USER
    if role == "system":
        return UserMessageContentRoleType.SYSTEM
    return None


def _openai_chat_completion_chunk_to_nat_chat_response_chunk(
    completion: ChatCompletionChunk,
) -> ChatResponseChunk:
    if not completion.choices:
        raise ValueError("ChatCompletionChunk has no choices")
    c0 = completion.choices[0]
    d = c0.delta
    nat_delta = ChoiceDelta(
        content=d.content,
        role=_openai_stream_role_to_nat(d.role),
        tool_calls=_openai_choice_delta_tool_calls_to_nat(d.tool_calls),
    )
    nat_usage: NATUsage | None = None
    if completion.usage is not None:
        u = completion.usage
        nat_usage = NATUsage(
            prompt_tokens=u.prompt_tokens,
            completion_tokens=u.completion_tokens,
            total_tokens=u.total_tokens,
        )
    created_dt = datetime.fromtimestamp(int(completion.created), tz=UTC)
    return ChatResponseChunk(
        id=completion.id,
        choices=[
            ChatResponseChunkChoice(index=0, delta=nat_delta, finish_reason=c0.finish_reason),
        ],
        created=created_dt,
        model=completion.model,
        object="chat.completion.chunk",
        usage=nat_usage,
    )


def _streaming_text_events_from_openai_chunk(
    completion: ChatCompletionChunk,
    source_ag_ui_events: list[Any] | None,
) -> list[Any]:
    """Rebuild AG-UI text events from a streaming OpenAI chunk (single delta, no synthetic start/end)."""
    delta_content = completion.choices[0].delta.content
    text = "" if delta_content is None else delta_content
    if source_ag_ui_events:
        ev0 = source_ag_ui_events[0]
        if isinstance(ev0, TextMessageContentEvent):
            if text:
                return [TextMessageContentEvent(message_id=ev0.message_id, delta=text)]
            return [TextMessageChunkEvent(message_id=ev0.message_id, role="assistant", delta="")]
        if isinstance(ev0, TextMessageChunkEvent):
            return [
                TextMessageChunkEvent(
                    message_id=ev0.message_id,
                    role=ev0.role,
                    delta=text or "",
                )
            ]
    mid = str(uuid.uuid4())
    if text:
        return [TextMessageContentEvent(message_id=mid, delta=text)]
    return [TextMessageChunkEvent(message_id=mid, role="assistant", delta="")]


def dragent_event_response_to_chat_completion(
    response: DRAgentEventResponse,
    *,
    as_streaming_chunk: bool = False,
) -> ChatCompletion | ChatCompletionChunk:
    """Convert a DRAgent response to OpenAI format for moderation.

    Non-streaming callers use the default aggregated ``ChatCompletion`` (``message``).
    Streaming callers should pass ``as_streaming_chunk=True`` to preserve
    ``chat.completion.chunk`` shape with incremental ``delta`` content.
    """
    chunk = convert_dragent_event_response_to_chat_response_chunk(response)
    event_tool_calls = _tool_calls_from_ag_ui_events(response.events)

    if as_streaming_chunk:
        completion_chunk = _nat_chat_response_chunk_to_openai_chat_completion_chunk(chunk)
        if not event_tool_calls:
            return completion_chunk
        stream_choice = completion_chunk.choices[0]
        new_delta = stream_choice.delta.model_copy(
            update={"tool_calls": _message_tool_calls_to_openai_delta_tool_calls(event_tool_calls)}
        )
        return ChatCompletionChunk(
            id=completion_chunk.id,
            choices=[
                OpenAIChunkChoice(
                    index=0,
                    delta=new_delta,
                    finish_reason="tool_calls",
                )
            ],
            created=completion_chunk.created,
            model=completion_chunk.model,
            object="chat.completion.chunk",
            usage=completion_chunk.usage,
        )

    non_stream = _nat_chat_response_chunk_to_openai_chat_completion(chunk)
    if not event_tool_calls:
        return non_stream
    ns_choice = non_stream.choices[0]
    msg = ns_choice.message
    return ChatCompletion(
        id=non_stream.id,
        choices=[
            OpenAIChoice(
                finish_reason="tool_calls",
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content=msg.content,
                    tool_calls=event_tool_calls,
                ),
            )
        ],
        created=non_stream.created,
        model=non_stream.model,
        object=non_stream.object,
        usage=non_stream.usage,
    )


def _openai_usage_to_usage_metrics(usage: Any) -> dict[str, int] | None:
    if usage is None:
        return None
    pt = getattr(usage, "prompt_tokens", None)
    ct = getattr(usage, "completion_tokens", None)
    tt = getattr(usage, "total_tokens", None)
    if pt is None and ct is None and tt is None:
        return None
    return {
        "prompt_tokens": int(pt or 0),
        "completion_tokens": int(ct or 0),
        "total_tokens": int(tt or 0),
    }


def _openai_chat_completion_to_nat_chat_response_chunk(
    completion: ChatCompletion,
) -> ChatResponseChunk:
    if not completion.choices:
        raise ValueError("ChatCompletion has no choices")
    c0 = completion.choices[0]
    msg = c0.message
    delta_tool_calls: list[ChoiceDeltaToolCall] | None = None
    if msg.tool_calls:
        built: list[ChoiceDeltaToolCall] = []
        idx = 0
        for tc in msg.tool_calls:
            if tc.type != "function":
                continue
            built.append(
                ChoiceDeltaToolCall(
                    index=idx,
                    id=tc.id,
                    type=tc.type,
                    function=ChoiceDeltaToolCallFunction(
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                    ),
                )
            )
            idx += 1
        if built:
            delta_tool_calls = built
    finish = c0.finish_reason
    delta = ChoiceDelta(
        content=msg.content,
        role=UserMessageContentRoleType.ASSISTANT,
        tool_calls=delta_tool_calls,
    )
    nat_usage: NATUsage | None = None
    if completion.usage is not None:
        u = completion.usage
        nat_usage = NATUsage(
            prompt_tokens=u.prompt_tokens,
            completion_tokens=u.completion_tokens,
            total_tokens=u.total_tokens,
        )
    created_dt = datetime.fromtimestamp(int(completion.created), tz=UTC)
    return ChatResponseChunk(
        id=completion.id,
        choices=[
            ChatResponseChunkChoice(index=0, delta=delta, finish_reason=finish),
        ],
        created=created_dt,
        model=completion.model,
        object="chat.completion.chunk",
        usage=nat_usage,
    )


def _json_safe_moderation_metadata(obj: Any) -> Any:  # noqa: PLR0911
    """Recursively convert dome moderation metadata to JSON-safe values.

    ``build_moderations_attribute_for_completion`` may attach numpy scalars or pandas
    timestamps. Those values break ``DRAgentEventResponse.model_dump_json()`` during NAT
    SSE serialization; the FastAPI layer then emits a bare workflow error JSON line (no
    ``data:`` prefix), which e2e parses as an empty stream.
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {str(k): _json_safe_moderation_metadata(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe_moderation_metadata(v) for v in obj]
    if isinstance(obj, str):
        return obj
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    if isinstance(obj, np.generic):
        return _json_safe_moderation_metadata(obj.item())
    if isinstance(obj, np.ndarray):
        return _json_safe_moderation_metadata(obj.tolist())
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return _json_safe_moderation_metadata(obj.item())
        except Exception:
            pass
    return str(obj)


def _assistant_text_events_from_message(content: str | None) -> list[Any]:
    message_id = str(uuid.uuid4())
    events: list[Any] = [
        TextMessageStartEvent(message_id=message_id, role="assistant"),
    ]
    text = content or ""
    if text:
        events.append(TextMessageContentEvent(message_id=message_id, delta=text))
    else:
        events.append(TextMessageChunkEvent(message_id=message_id, role="assistant", delta=""))
    events.append(TextMessageEndEvent(message_id=message_id))
    return events


def _datarobot_moderations_from_completion(
    completion: ChatCompletion | ChatCompletionChunk,
) -> dict[str, Any] | None:
    raw = getattr(completion, DATAROBOT_MODERATIONS_ATTR, None)
    if not isinstance(raw, dict):
        return None
    return cast(dict[str, Any], _json_safe_moderation_metadata(raw))


def _infer_parent_message_id_for_tool_calls(
    source_ag_ui_events: list[Any] | None,
    built_text_events: list[Any],
) -> str:
    """Best-effort parent_message_id for ToolCallStart (matches stream_converter semantics)."""
    for ev in reversed(built_text_events):
        if isinstance(ev, (TextMessageContentEvent, TextMessageChunkEvent)):
            return ev.message_id
    if source_ag_ui_events:
        for ev in reversed(source_ag_ui_events):
            if isinstance(ev, (TextMessageContentEvent, TextMessageChunkEvent)):
                return ev.message_id
            if isinstance(ev, ToolCallStartEvent):
                return ev.parent_message_id
    return ""


def _agui_tool_events_from_openai_delta_tool_calls(
    tool_calls: list[OpenAIChoiceDeltaToolCall] | None,
    *,
    parent_message_id: str,
    tool_index_map: dict[int, str],
) -> list[Any]:
    """Rebuild ToolCall* AG-UI events from one OpenAI streaming delta (per stream_converter)."""
    events: list[Any] = []
    if not tool_calls:
        return events
    for tc in tool_calls:
        tc_id = tc.id or tool_index_map.get(tc.index)
        if tc_id is None:
            continue
        is_new = tc.id is not None and tc.index not in tool_index_map
        fn = tc.function
        if is_new:
            tool_index_map[tc.index] = tc_id
            tool_name = (fn.name if fn else None) or ""
            events.append(
                ToolCallStartEvent(
                    tool_call_id=tc_id,
                    tool_call_name=tool_name,
                    parent_message_id=parent_message_id,
                )
            )
            if tool_name:
                register_tool_call(tool_name, tc_id)
        arguments = fn.arguments if fn else None
        if arguments:
            events.append(ToolCallArgsEvent(tool_call_id=tc_id, delta=arguments))
    return events


def chat_completion_to_dragent_event_response(
    completion: ChatCompletion | ChatCompletionChunk,
    *,
    source_ag_ui_events: list[Any] | None = None,
    stream_tool_index_map: dict[int, str] | None = None,
) -> DRAgentEventResponse:
    """Convert OpenAI completion or streaming chunk back to DRAgentEventResponse."""
    if isinstance(completion, ChatCompletionChunk):
        chunk = _openai_chat_completion_chunk_to_nat_chat_response_chunk(completion)
        d = completion.choices[0].delta
        idx_map = stream_tool_index_map if stream_tool_index_map is not None else {}
        events: list[Any] = []
        if d.content:
            idx_map.clear()
            events.extend(
                _streaming_text_events_from_openai_chunk(
                    completion, source_ag_ui_events=source_ag_ui_events
                )
            )
        elif not d.tool_calls:
            events.extend(
                _streaming_text_events_from_openai_chunk(
                    completion, source_ag_ui_events=source_ag_ui_events
                )
            )
        if d.tool_calls:
            if d.content and events:
                last = events[-1]
                if isinstance(last, (TextMessageContentEvent, TextMessageChunkEvent)):
                    events.append(TextMessageEndEvent(message_id=last.message_id))
            parent_id = _infer_parent_message_id_for_tool_calls(source_ag_ui_events, events)
            events.extend(
                _agui_tool_events_from_openai_delta_tool_calls(
                    d.tool_calls,
                    parent_message_id=parent_id,
                    tool_index_map=idx_map,
                )
            )
    else:
        chunk = _openai_chat_completion_to_nat_chat_response_chunk(completion)
        msg = completion.choices[0].message
        events = _assistant_text_events_from_message(msg.content)
    usage_metrics = _openai_usage_to_usage_metrics(completion.usage) or default_usage_metrics()
    return DRAgentEventResponse(
        events=events,
        usage_metrics=usage_metrics,
        original_chunk=chunk,
        model=completion.model,
        datarobot_moderations=_datarobot_moderations_from_completion(completion),
    )


def _ag_ui_message_to_openai(msg: object) -> dict[str, Any]:
    if isinstance(msg, UserMessage):
        return {"role": "user", "content": _user_content_to_openai(msg.content)}
    if isinstance(msg, SystemMessage):
        return {"role": "system", "content": msg.content}
    if isinstance(msg, AssistantMessage):
        out: dict[str, Any] = {
            "role": "assistant",
            "content": msg.content,
        }
        if msg.tool_calls:
            out["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]
        return out
    if isinstance(msg, ToolMessage):
        return {
            "role": "tool",
            "content": msg.content,
            "tool_call_id": msg.tool_call_id,
        }
    return {}


def _user_content_to_openai(content: object) -> object:
    if isinstance(content, str):
        return content
    if isinstance(content, list) and content:
        parts: list[dict[str, Any]] = []
        for c in content:
            if not hasattr(c, "type"):
                continue
            if getattr(c, "type", None) == "text" and hasattr(c, "text"):
                parts.append({"type": "text", "text": c.text})
        if parts:
            return parts
    return str(content)


def run_agent_input_to_completion_dict(rai: RunAgentInput) -> dict[str, Any]:
    ccp: dict[str, Any] = {
        "messages": [],
        "stream": True,
    }
    for m in rai.messages:
        ccp["messages"].append(_ag_ui_message_to_openai(m))
    ccp["tools"] = [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in (rai.tools or [])
    ]
    props = rai.forwarded_props
    if isinstance(props, dict):
        merged = {**props, **ccp, "messages": ccp["messages"]}
        return merged
    return ccp


def _normalize_nat_chat_request_completion_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Normalize ChatRequest/ChatRequestOrMessage JSON dumps for moderation helpers.

    ``get_chat_prompt`` expects ``tools`` to be absent or iterable; NAT dumps often
    include ``tools: null``.
    """
    out = dict(d)
    if out.get("tools") is None:
        out["tools"] = []
    return out


def nat_chat_request_like_to_completion_dict(
    request: ChatRequest | ChatRequestOrMessage,
) -> dict[str, Any]:
    """Build completion-style params from NAT chat request models (LLM Gateway path)."""
    return _normalize_nat_chat_request_completion_dict(request.model_dump(mode="json"))


def workflow_input_to_completion_dict(
    workflow_input: RunAgentInput | ChatRequest | ChatRequestOrMessage,
) -> dict[str, Any]:
    """Build OpenAI-style completion params for prescore from AG-UI or NAT chat inputs."""
    if isinstance(workflow_input, RunAgentInput):
        return run_agent_input_to_completion_dict(workflow_input)
    if isinstance(workflow_input, (ChatRequest, ChatRequestOrMessage)):
        return nat_chat_request_like_to_completion_dict(workflow_input)
    raise TypeError(
        f"Unsupported workflow input type for moderation: {type(workflow_input).__name__}"
    )


def _apply_moderated_prompt_text_to_run_agent_input(
    run_agent_input: RunAgentInput, moderated_text: str
) -> bool:
    """Write prescore-moderated prompt text onto the AG-UI message list (last ``UserMessage``).

    Prescore guards set ``replaced_<prompt_col>``; the moderated string lives in
    ``filtered_df`` under the prompt column. ``RunAgentInput.messages`` are Pydantic
    models, not OpenAI-style dicts, so we must use ``model_copy`` rather than
    ``msg[\"content\"] = ...``.
    """
    messages = run_agent_input.messages
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if isinstance(msg, UserMessage):
            messages[idx] = msg.model_copy(update={"content": moderated_text})
            return True
    _logger.warning(
        "Prescore replaced the prompt but no UserMessage was found to apply it; "
        "leaving messages unchanged."
    )
    return False


def _apply_moderated_prompt_text_to_nat_chat_messages(
    messages: list[Any], moderated_text: str
) -> bool:
    """Apply prescore replacement to the last NAT ``Message`` with user role."""
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if getattr(msg, "role", None) == UserMessageContentRoleType.USER:
            messages[idx] = msg.model_copy(update={"content": moderated_text})
            return True
    _logger.warning(
        "Prescore replaced the prompt but no user-role NAT message was found; "
        "leaving messages unchanged."
    )
    return False


def _apply_moderated_prompt_text_to_workflow_input(
    workflow_input: RunAgentInput | ChatRequest | ChatRequestOrMessage,
    moderated_text: str,
) -> bool:
    if isinstance(workflow_input, RunAgentInput):
        return _apply_moderated_prompt_text_to_run_agent_input(workflow_input, moderated_text)
    messages = getattr(workflow_input, "messages", None)
    if messages:
        return _apply_moderated_prompt_text_to_nat_chat_messages(messages, moderated_text)
    return False


def skip_event_type(event: Event) -> bool:
    return event.type not in {
        EventType.TEXT_MESSAGE_CONTENT,
        EventType.TEXT_MESSAGE_CHUNK,
    }


def _response_has_assistant_text_deltas(response: DRAgentEventResponse) -> bool:
    """Check if the payload includes assistant text AG-UI deltas
    (possibly after lifecycle events).
    """
    return any(
        isinstance(ev, (TextMessageContentEvent, TextMessageChunkEvent)) for ev in response.events
    )


def _openai_chat_completion_has_assistant_message_content(completion: ChatCompletion) -> bool:
    if not completion.choices:
        return False
    content = completion.choices[0].message.content
    return bool(content)


def _final_openai_completion_to_nat_chat_response(
    final_completion: ChatCompletion,
    original_nat_response: ChatResponse,
) -> ChatResponse:
    """Map a moderated OpenAI completion back to NAT's model (usage is required on ``ChatResponse``)."""
    data = final_completion.model_dump(mode="json")
    if data.get("usage") is None:
        if original_nat_response.usage is not None:
            data["usage"] = original_nat_response.usage.model_dump(mode="json")
        else:
            data["usage"] = NATUsage(
                prompt_tokens=0, completion_tokens=0, total_tokens=0
            ).model_dump(mode="json")
    return ChatResponse.model_validate(data)


def _assistant_text_joined_from_ag_ui(response: DRAgentEventResponse) -> str:
    return "".join(
        ev.delta
        for ev in response.events
        if isinstance(ev, (TextMessageContentEvent, TextMessageChunkEvent))
    )


def _merge_moderations_into_multi_event_response(
    incoming: DRAgentEventResponse,
    moderated_completion_response: DRAgentEventResponse,
) -> DRAgentEventResponse:
    """Attach serialized moderations (and optional usage/model) without dropping envelope events."""
    return incoming.model_copy(
        update={
            "datarobot_moderations": moderated_completion_response.datarobot_moderations,
            "usage_metrics": moderated_completion_response.usage_metrics or incoming.usage_metrics,
            "model": moderated_completion_response.model or incoming.model,
        }
    )


def _defer_until_after_moderated_chunk(event: Event) -> bool:
    """Defer START/END until after the moderated chunk: late moderated deltas still use the prior message_id.

    If TEXT_MESSAGE_START for the next segment is yielded before moderated content for the
    previous segment, storage switches active_message and can drop or mis-attribute the
    last deltas (truncation in DB). TEXT_MESSAGE_END must stay after moderated text for AG-UI.
    """
    return event.type in (
        EventType.TEXT_MESSAGE_END,
        EventType.TEXT_MESSAGE_START,
    )


def _pending_deferred_in_emit_order(
    pending: list[DRAgentEventResponse],
) -> list[DRAgentEventResponse]:
    """TEXT_MESSAGE_END before other deferred items so END always precedes a following START."""
    ends_first = [
        item
        for item in pending
        if item.events and item.events[0].type == EventType.TEXT_MESSAGE_END
    ]
    return ends_first + [item for item in pending if item not in ends_first]


_STREAM_INPUT_END = object()
_WORKER_QUEUE_END = object()


class DataRobotModerationConfig(
    FunctionMiddlewareBaseConfig,  # type: ignore[misc]
    name="datarobot_moderation",  # type: ignore[call-arg]
):
    """NAT middleware: DataRobot prescore / postscore guard pipeline (moderation_config.yaml)."""

    model_dir: str | None = Field(
        default=None,
        description="Directory that contains moderation_config.yaml (defaults to process CWD).",
    )


class DataRobotModerationMiddleware(
    FunctionMiddleware,  # type: ignore[misc]
):
    def __init__(self, config: DataRobotModerationConfig, builder: Builder) -> None:  # noqa: ARG002
        super().__init__()
        self._moderation = load_llm_moderation_pipeline(config.model_dir)
        self.data = None
        self.prescore_df = None
        self.input_df: pd.DataFrame | None = None
        self.latency_so_far: float = 0.0
        self.association_id: str | None = None
        self._stream_tool_index_map: dict[int, str] = {}

    @property
    def enabled(self) -> bool:
        return self._moderation is not None

    async def function_middleware_invoke(
        self,
        *args: Any,
        call_next: CallNext,
        context: FunctionMiddlewareContext,
        **kwargs: Any,
    ) -> Any:
        """Run prescore guards; skip the agent when prescore blocks the prompt."""
        ctx = InvocationContext(
            function_context=context,
            original_args=args,
            original_kwargs=dict(kwargs),
            modified_args=args,
            modified_kwargs=dict(kwargs),
            output=None,
        )
        result = await self.pre_invoke(ctx)
        if result is not None:
            ctx = result
        if ctx.output is not None:
            return ctx.output
        ctx.output = await call_next(*ctx.modified_args, **ctx.modified_kwargs)
        result = await self.post_invoke(ctx)
        if result is not None:
            ctx = result
        return ctx.output

    async def pre_invoke(self, context: InvocationContext) -> InvocationContext | None:
        """Pre-invocation hook called before the function is invoked.

        Args:
            context: Invocation context containing function metadata and args

        Returns:
            InvocationContext if modified (including when the prompt is blocked and
            ``context.output`` holds the guard message), or None to pass through unchanged.
        """
        workflow_input: RunAgentInput | ChatRequest | ChatRequestOrMessage | None = (
            context.original_args[0] if context.original_args else None
        )
        if workflow_input is None:
            return None
        if self._moderation is None:
            return None

        pipeline = self._moderation._pipeline
        completion_create_params = workflow_input_to_completion_dict(workflow_input)

        # the chat request is not a dataframe, but we'll build a DF internally for moderation.
        prompt_column_name = pipeline.get_input_column(GuardStage.PROMPT)
        prompt = get_chat_prompt(completion_create_params)

        data = pd.DataFrame({prompt_column_name: [prompt]})

        # ==================================================================
        # Step 1: Prescore Guards processing
        #
        # ``run_prescore_guards`` keeps the full prescore DataFrame (for postscore / streaming);
        # ``evaluate_prompt`` returns the public ``EvaluationResult`` for routing.
        prescore_df, _, _ = run_prescore_guards(pipeline, data)
        prompt_eval, prescore_latency = self._moderation.evaluate_prompt(prompt)
        self.data = data
        self.input_df = data
        self.prescore_df = prescore_df
        self.latency_so_far = prescore_latency

        if prompt_eval.blocked:
            pipeline.report_custom_metrics(prescore_df)
            # If all prompts in the input are blocked, means history as well as the prompt
            # are not worthy to be sent to LLM
            chat_completion = build_non_streaming_chat_completion(
                prompt_eval.blocked_message,
                "content_filter",
            )
            result_df = handle_result_df_error_cases(
                prompt_column_name, prescore_df, prescore_latency
            )
            completion = set_moderation_attribute_to_completion(
                pipeline, chat_completion, result_df
            )
            report_otel_evaluation_set_metric(pipeline, result_df)
            context.output = chat_completion_to_dragent_event_response(completion)
            return context

        if prompt_eval.replaced and prompt_eval.replacement is not None:
            # PII-style guards may redact the prompt; apply the replacement text to the
            # request and align downstream postscore input.
            moderated_prompt = prompt_eval.replacement
            if _apply_moderated_prompt_text_to_workflow_input(workflow_input, moderated_prompt):
                data.at[0, prompt_column_name] = moderated_prompt
            return context
        else:
            return None

    async def post_invoke(self, context: InvocationContext) -> InvocationContext | None:
        """Post-invocation hook called after the function returns.

        Args:
            context: Invocation context containing function metadata, args, and output

        Returns:
            InvocationContext if modified, or None to pass through unchanged.
            Default implementation does nothing.
        """
        original_output = context.output
        if self._moderation is None:
            return None

        incoming_agui: DRAgentEventResponse | None = None
        if isinstance(original_output, DRAgentEventResponse):
            incoming_agui = original_output
            if not _response_has_assistant_text_deltas(incoming_agui):
                return None
            chat_completion = dragent_event_response_to_chat_completion(incoming_agui)
        elif isinstance(original_output, ChatResponse):
            chat_completion = ChatCompletion.model_validate(original_output.model_dump(mode="json"))
            if not _openai_chat_completion_has_assistant_message_content(chat_completion):
                return None
        elif isinstance(original_output, str):
            if not original_output.strip():
                return None
            chat_completion = build_non_streaming_chat_completion(original_output, "stop")
        else:
            return None

        pipeline = self._moderation._pipeline

        # ==================================================================
        # Step 3: Postscore Guards processing
        #
        # Prompt column name is already part of data and gets included for
        # faithfulness calculation processing
        response_column_name = pipeline.get_input_column(GuardStage.RESPONSE)
        prompt_column_name = pipeline.get_input_column(GuardStage.PROMPT)
        predictions_df, extra_attributes = build_predictions_df_from_completion(
            self.data, pipeline, chat_completion
        )
        response_text = predictions_df.loc[0, response_column_name]
        prompt_for_eval = predictions_df.loc[0, prompt_column_name]

        if response_text is not None:
            none_predictions_df = None
            postscore_df, _ = run_postscore_guards(pipeline, predictions_df)
        else:
            postscore_df, _ = pd.DataFrame(), 0
            none_predictions_df = predictions_df

        # ``run_postscore_guards`` feeds ``format_result_df`` / OTEL; ``evaluate_response``
        # returns the public ``EvaluationResult`` for the completion (mirrors ``evaluate_prompt``).
        response_eval, _ = self._moderation.evaluate_response(
            _text_for_moderation_eval(response_text),
            prompt=_optional_prompt_for_moderation_eval(prompt_for_eval),
        )

        # ==================================================================
        # Step 4: Assemble the result - we need to merge prescore, postscore
        #         Dataframes.
        #
        result_df = format_result_df(
            pipeline,
            self.prescore_df,
            postscore_df,
            self.data,
            none_predictions_df=none_predictions_df,
        )

        # ==================================================================
        # Step 5: Additional metadata calculations
        #
        # TBD: handle latency
        # result_df["datarobot_latency"] = (
        #                                          score_latency + prescore_latency + postscore_latency
        #                                  ) / result_df.shape[0]

        blocked_message_completion_column_name = f"blocked_message_{response_column_name}"
        if response_eval.blocked:
            response_message = response_eval.blocked_message
            if response_message is None and not postscore_df.empty:
                response_message = postscore_df.loc[0, blocked_message_completion_column_name]
            finish_reason = "content_filter"
        elif response_eval.replaced and response_eval.replacement is not None:
            response_message = response_eval.replacement
            finish_reason = "content_filter"
        elif postscore_df.empty:
            response_message = NONE_CUSTOM_PY_RESPONSE
            finish_reason = "stop"
        else:
            response_message = postscore_df.loc[0, response_column_name]
            finish_reason = "stop"
        report_otel_evaluation_set_metric(pipeline, result_df)

        final_completion = build_non_streaming_chat_completion(
            response_message, finish_reason, extra_attributes
        )
        final_completion = set_moderation_attribute_to_completion(
            pipeline, final_completion, result_df, association_id=self.association_id
        )
        moderated_dr = chat_completion_to_dragent_event_response(final_completion)

        if incoming_agui is not None:
            preserve_ag_ui_envelope = (
                len(incoming_agui.events) > 1
                and finish_reason == "stop"
                and not response_eval.blocked
                and not (response_eval.replaced and response_eval.replacement is not None)
                and response_message == _assistant_text_joined_from_ag_ui(incoming_agui)
            )
            if preserve_ag_ui_envelope:
                context.output = _merge_moderations_into_multi_event_response(
                    incoming_agui, moderated_dr
                )
            else:
                context.output = moderated_dr
        elif isinstance(original_output, ChatResponse):
            context.output = _final_openai_completion_to_nat_chat_response(
                final_completion, original_output
            )
        else:
            context.output = response_message

        return context

    async def function_middleware_stream(
        self,
        *args: Any,
        call_next: CallNextStream,
        context: FunctionMiddlewareContext,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Execute middleware hooks around streaming function call.

        Pre-invoke runs once before streaming starts.
        Post-invoke runs per-chunk as they stream through.

        Override for custom streaming behavior (e.g., buffering,
        aggregation, chunk filtering).

        Note: Framework checks ``enabled`` before calling this method.
        You do NOT need to check ``enabled`` yourself.

        Args:
            args: Positional arguments for the function (first arg is typically the input value).
            call_next: Callable to invoke next middleware or target stream.
            context: Static function metadata.
            kwargs: Keyword arguments for the function.

        Yields:
            Stream chunks (potentially transformed by post_invoke).
        """
        ctx = InvocationContext(
            function_context=context,
            original_args=args,
            original_kwargs=dict(kwargs),
            modified_args=args,
            modified_kwargs=dict(kwargs),
            output=None,
        )

        result = await self.pre_invoke(ctx)
        if result is not None:
            ctx = result
        if ctx.output is not None:
            yield ctx.output
            return

        # Prescore populates ``input_df`` / ``latency_so_far``. If ``pre_invoke`` returned
        # early (e.g. no workflow input / prescore skipped), pass the stream through unchanged.
        if self.input_df is None:
            async for chunk in call_next(*ctx.modified_args, **ctx.modified_kwargs):
                yield chunk
            return

        moderation = self._moderation
        assert moderation is not None

        self._stream_tool_index_map.clear()

        streaming_context = (
            StreamingContextBuilder()
            .set_pipeline(moderation._pipeline)
            .set_prescore_df(self.prescore_df)
            .set_prescore_latency(self.latency_so_far)
            .set_input_df(self.input_df)
            .build()
        )

        in_q: queue.Queue[Any] = queue.Queue()
        out_q: queue.Queue[Any] = queue.Queue()
        mod_errors: list[BaseException] = []
        worker: threading.Thread | None = None

        def input_chunk_iter() -> Iterator[Any]:
            while True:
                item = in_q.get()
                if item is _STREAM_INPUT_END:
                    break
                yield item

        def moderation_worker() -> None:
            try:
                mit = ModerationIterator(streaming_context, input_chunk_iter())
                for moderated_chunk in mit:
                    out_q.put(moderated_chunk)
            except BaseException as exc:
                mod_errors.append(exc)
            finally:
                out_q.put(_WORKER_QUEUE_END)

        def start_moderation_worker() -> threading.Thread:
            t = threading.Thread(
                target=moderation_worker,
                name="datarobot-moderation-stream",
                daemon=True,
            )
            t.start()
            return t

        upstream = call_next(*ctx.modified_args, **ctx.modified_kwargs).__aiter__()
        loop = asyncio.get_running_loop()
        worker_sentinel_received = False

        async def read_upstream() -> DRAgentEventResponse | None:
            try:
                return cast(DRAgentEventResponse, await upstream.__anext__())
            except StopAsyncIteration:
                return None

        try:
            # Leading pass-through (non–text-message) events
            while True:
                response = await read_upstream()
                if response is None:
                    return
                if not response.events or skip_event_type(response.events[0]):
                    yield response
                    continue
                break

            # ``ModerationIterator`` peeks one chunk ahead: each ``__next__`` pulls the next
            # item from the iterable before returning the moderated current chunk. Feed that
            # peek *before* awaiting each moderated output or the worker deadlocks.
            worker = start_moderation_worker()
            current_response = response
            in_q.put(
                cast(
                    ChatCompletionChunk,
                    dragent_event_response_to_chat_completion(
                        current_response, as_streaming_chunk=True
                    ),
                )
            )

            # Defer TEXT_MESSAGE_START/END until after the moderated chunk for this step.
            # (See _defer_until_after_moderated_chunk.) Other pass-through events keep
            # upstream order relative to non-text events.
            pending_deferred_pass_through: list[DRAgentEventResponse] = []

            while True:
                # Pass-through events read while advancing to the next moderation input chunk
                # must follow the moderated output for ``current_response`` (ModerationIterator
                # peeks one chunk ahead). Yielding them early breaks ordering (e.g. RUN_FINISHED
                # before TEXT_MESSAGE_CONTENT / TEXT_MESSAGE_END).
                pending_pass_through: list[DRAgentEventResponse] = []
                peek = await read_upstream()
                while peek is not None and (not peek.events or skip_event_type(peek.events[0])):
                    if peek.events and _defer_until_after_moderated_chunk(peek.events[0]):
                        pending_deferred_pass_through.append(peek)
                    else:
                        pending_pass_through.append(peek)
                    peek = await read_upstream()

                if peek is None:
                    in_q.put(_STREAM_INPUT_END)
                else:
                    in_q.put(
                        cast(
                            ChatCompletionChunk,
                            dragent_event_response_to_chat_completion(
                                peek, as_streaming_chunk=True
                            ),
                        )
                    )

                moderated = await loop.run_in_executor(None, out_q.get)
                if moderated is _WORKER_QUEUE_END:
                    for item in _pending_deferred_in_emit_order(pending_deferred_pass_through):
                        yield item
                    pending_deferred_pass_through.clear()
                    for item in pending_pass_through:
                        yield item
                    worker_sentinel_received = True
                    if mod_errors:
                        raise mod_errors[0]
                    break

                ctx.output = chat_completion_to_dragent_event_response(
                    moderated,
                    source_ag_ui_events=current_response.events,
                    stream_tool_index_map=self._stream_tool_index_map,
                )
                yield ctx.output
                for item in _pending_deferred_in_emit_order(pending_deferred_pass_through):
                    yield item
                pending_deferred_pass_through.clear()
                for item in pending_pass_through:
                    yield item
                finish = moderated.choices[0].finish_reason if moderated.choices else None
                if finish == "content_filter":
                    break

                if peek is None:
                    break

                current_response = peek
        finally:
            if worker is not None:
                in_q.put(_STREAM_INPUT_END)
                if not worker_sentinel_received:
                    sentinel = await loop.run_in_executor(None, out_q.get)
                    if sentinel is not _WORKER_QUEUE_END:
                        _logger.warning(
                            "Unexpected item from moderation worker queue: %r",
                            sentinel,
                        )
                worker.join(timeout=120.0)


@register_middleware(  # type: ignore[untyped-decorator]
    config_type=DataRobotModerationConfig
)
async def datarobot_moderation_middleware(
    config: DataRobotModerationConfig,
    builder: Builder,  # noqa: ARG001
) -> AsyncIterator[DataRobotModerationMiddleware]:
    """Register DataRobot LLM guard middleware for NAT/DRAgent workflows."""
    yield DataRobotModerationMiddleware(config, builder)
