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
import contextvars
import logging
import math
import os
import queue
import threading
import uuid
from collections.abc import AsyncIterator
from collections.abc import Iterator
from dataclasses import dataclass
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
from datarobot_dome.api import EvaluationResult
from datarobot_dome.api import ModerationPipeline
from datarobot_dome.api import _from_dataframe
from datarobot_dome.constants import AGENTIC_PIPELINE_INTERACTIONS_ATTR
from datarobot_dome.constants import CHAT_COMPLETION_OBJECT
from datarobot_dome.constants import DATAROBOT_MODERATIONS_ATTR
from datarobot_dome.constants import DISABLE_MODERATION_RUNTIME_PARAM_NAME
from datarobot_dome.constants import MODERATION_CONFIG_FILE_NAME
from datarobot_dome.constants import GuardStage
from datarobot_dome.runtime import get_runtime_parameter_value_bool
from datarobot_dome.streaming import ModerationIterator
from datarobot_dome.streaming import StreamingContextBuilder
from datarobot_moderation_interface.drum_integration import MODERATION_MODEL_NAME
from datarobot_moderation_interface.drum_integration import build_non_streaming_chat_completion
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


def _predictions_response_missing_for_postscore(value: Any) -> bool:
    """Check if the response cell is null-like (``None`` or pandas missing), not merely falsy."""
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except TypeError:
        pass
    return False


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


def _postscore_evaluation_context_columns(
    postscore_df: pd.DataFrame,
    prompt_column_name: str,
    prompt_for_eval: Any,
) -> frozenset[str]:
    """Columns omitted from ``EvaluationResult.metrics`` (aligned with ``evaluate_response``)."""
    cols: set[str] = set()
    if _optional_prompt_for_moderation_eval(prompt_for_eval) is not None:
        cols.add(prompt_column_name)
    if AGENTIC_PIPELINE_INTERACTIONS_ATTR in postscore_df.columns:
        cols.add(AGENTIC_PIPELINE_INTERACTIONS_ATTR)
    return frozenset(cols)


def _predictions_df_with_response_evaluation(
    predictions_df: pd.DataFrame,
    response_column_name: str,
    response_eval: EvaluationResult,
) -> pd.DataFrame:
    """Merge ``ModerationPipeline.evaluate_response`` output into the predictions frame.

    ``format_result_df`` / completion metadata need the postscore-shaped DataFrame produced
    by the guard executor; ``evaluate_response`` returns only ``EvaluationResult``, so we
    copy predictions and apply the evaluation fields and metric columns on row 0.
    """
    postscore_df = predictions_df.copy(deep=True)
    postscore_df.index = predictions_df.index
    postscore_df.loc[0, f"blocked_{response_column_name}"] = response_eval.blocked
    postscore_df.loc[0, f"blocked_message_{response_column_name}"] = response_eval.blocked_message
    postscore_df.loc[0, f"replaced_{response_column_name}"] = response_eval.replaced
    postscore_df.loc[0, f"replaced_message_{response_column_name}"] = response_eval.replacement
    for metric_key, metric_val in response_eval.metrics.items():
        postscore_df.loc[0, metric_key] = metric_val
    return postscore_df


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


def _datarobot_moderations_from_evaluation_result(
    eval_result: EvaluationResult,
) -> dict[str, Any] | None:
    if not eval_result.metrics:
        return None
    return cast(dict[str, Any], _json_safe_moderation_metadata(eval_result.metrics))


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


def _dragent_event_response_from_blocked_prompt_eval(
    prompt_eval: EvaluationResult,
) -> DRAgentEventResponse:
    """Build AG-UI output for prescore prompt blocking without an intermediate ChatCompletion.

    Serializes non-internal prescore columns from ``prompt_eval.metrics`` into
    ``datarobot_moderations`` (JSON-safe), matching how streamed completions expose guard output.
    """
    content = prompt_eval.blocked_message or ""
    events = _assistant_text_events_from_message(content)
    completion_id = str(uuid.uuid4())
    created_ts = int(datetime.now(tz=UTC).timestamp())
    created_dt = datetime.fromtimestamp(created_ts, tz=UTC)
    chunk = ChatResponseChunk(
        id=completion_id,
        choices=[
            ChatResponseChunkChoice(
                index=0,
                delta=ChoiceDelta(
                    content=content,
                    role=UserMessageContentRoleType.ASSISTANT,
                    tool_calls=None,
                ),
                finish_reason="content_filter",
            )
        ],
        created=created_dt,
        model=MODERATION_MODEL_NAME,
        object="chat.completion.chunk",
        usage=None,
    )
    return DRAgentEventResponse(
        events=events,
        usage_metrics=default_usage_metrics(),
        original_chunk=chunk,
        model=MODERATION_MODEL_NAME,
        datarobot_moderations=_datarobot_moderations_from_evaluation_result(prompt_eval),
    )


def chat_completion_to_dragent_event_response(
    completion: ChatCompletion | ChatCompletionChunk,
    *,
    response_eval: EvaluationResult | None = None,
    source_ag_ui_events: list[Any] | None = None,
    stream_tool_index_map: dict[int, str] | None = None,
) -> DRAgentEventResponse:
    """Convert OpenAI completion or streaming chunk back to DRAgentEventResponse.

    When ``response_eval`` is set (non-streaming postscore path), ``datarobot_moderations`` is
    taken from ``response_eval.metrics``; otherwise it is read from the completion's moderation
    sidecar attribute (streaming chunks from ``ModerationIterator``).
    """
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
    datarobot_moderations = (
        _datarobot_moderations_from_evaluation_result(response_eval)
        if response_eval is not None
        else _datarobot_moderations_from_completion(completion)
    )
    return DRAgentEventResponse(
        events=events,
        usage_metrics=usage_metrics,
        original_chunk=chunk,
        model=completion.model,
        datarobot_moderations=datarobot_moderations,
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


def _moderation_prompt_from_openai_style_content(prompt_content: Any) -> str:
    """Stringify user message content like ``get_chat_prompt`` (multimodal OpenAI-style parts)."""
    if isinstance(prompt_content, str):
        return prompt_content
    if isinstance(prompt_content, list):
        concatenated_prompt: list[str] = []
        for content in prompt_content:
            if not isinstance(content, dict):
                continue
            ctype = content.get("type")
            if ctype == "text":
                concatenated_prompt.append(content["text"])
            elif ctype == "image_url":
                concatenated_prompt.append(f"Image URL: {content['image_url']['url']}")
            elif ctype == "input_audio":
                concatenated_prompt.append(
                    f"Audio Input, Format: {content['input_audio']['format']}"
                )
            else:
                concatenated_prompt.append(f"Unhandled content type: {ctype}")
        return "\n".join(concatenated_prompt)
    raise ValueError(f"Unhandled prompt type: {type(prompt_content)}")


def _nat_user_message_content_to_prompt_string(content: str | list[Any]) -> str:
    if isinstance(content, str):
        return content
    dumped = [
        part.model_dump(mode="json") if hasattr(part, "model_dump") else part for part in content
    ]
    return _moderation_prompt_from_openai_style_content(dumped)


def _tool_names_from_nat_tools(tools: list[dict[str, Any]] | None) -> list[str]:
    names: list[str] = []
    for tool in tools or []:
        fn = tool.get("function") if isinstance(tool, dict) else None
        if isinstance(fn, dict):
            name = fn.get("name")
            if name:
                names.append(name)
    return names


def _tool_call_lines_from_openai_style_messages(messages: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for message in messages:
        if message.get("role") == "tool":
            lines.append(f"{message.get('name', '')}_{message['content']}")
    return lines


def moderation_prompt_from_workflow_input(
    workflow_input: RunAgentInput | ChatRequest | ChatRequestOrMessage,
) -> str:
    """Extract the prescore prompt string from AG-UI or NAT chat input.

    Matches ``get_chat_prompt(workflow_input_to_completion_dict(...))`` without building a full
    completion-params dict.
    """
    if isinstance(workflow_input, RunAgentInput):
        msgs = workflow_input.messages
        if not msgs:
            raise ValueError(
                f"Chat input for moderation does not contain a message: {workflow_input}"
            )
        openai_msgs = [_ag_ui_message_to_openai(m) for m in msgs]
        last = openai_msgs[-1]
        if not isinstance(last, dict) or "content" not in last or last["content"] is None:
            raise ValueError(
                f"Chat input for moderation does not contain a message: {workflow_input}"
            )
        last_user_ag: UserMessage | None = None
        for m in reversed(workflow_input.messages):
            if isinstance(m, UserMessage):
                last_user_ag = m
                break
        if last_user_ag is None:
            raise ValueError("No message with 'user' role found in input")
        inner = _user_content_to_openai(last_user_ag.content)
        chat_prompt = _moderation_prompt_from_openai_style_content(inner)
        tool_lines = _tool_call_lines_from_openai_style_messages(openai_msgs)
        tool_names = [t.name for t in (workflow_input.tools or [])]
        if tool_lines:
            return "\n".join([chat_prompt, "Tool Calls:", "\n".join(tool_lines)])
        if tool_names:
            return "\n".join([chat_prompt, "Tool Names:", "\n".join(tool_names)])
        return chat_prompt

    if (
        isinstance(workflow_input, ChatRequestOrMessage)
        and workflow_input.input_message is not None
    ):
        return workflow_input.input_message

    if isinstance(workflow_input, (ChatRequest, ChatRequestOrMessage)):
        messages = workflow_input.messages
        if messages is None or len(messages) == 0:
            raise ValueError(
                f"Chat input for moderation does not contain a message: {workflow_input}"
            )
        last_dump = messages[-1].model_dump(mode="json")
        if not isinstance(last_dump, dict) or "content" not in last_dump:
            raise ValueError(
                f"Chat input for moderation does not contain a message: {workflow_input}"
            )
        last_user_msg = None
        for m in messages:
            if m.role == UserMessageContentRoleType.USER:
                last_user_msg = m
        if last_user_msg is None:
            raise ValueError("No message with 'user' role found in input")
        chat_prompt = _nat_user_message_content_to_prompt_string(last_user_msg.content)
        tool_names = _tool_names_from_nat_tools(getattr(workflow_input, "tools", None))
        if tool_names:
            return "\n".join([chat_prompt, "Tool Names:", "\n".join(tool_names)])
        return chat_prompt

    raise TypeError(
        f"Unsupported workflow input type for moderation: {type(workflow_input).__name__}"
    )


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
    if (
        isinstance(workflow_input, ChatRequestOrMessage)
        and workflow_input.input_message is not None
    ):
        # String-only gateway input; ``messages`` is unset so the list-based path below is skipped.
        workflow_input.input_message = moderated_text
        return True
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


def _openai_assistant_content_as_str(completion: ChatCompletion) -> str:
    """Assistant message text from a completion (for moderation eval, not dataframe columns)."""
    if not completion.choices:
        return ""
    content = completion.choices[0].message.content
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    joined: list[str] = []
    for part in content:
        text: str | None = None
        if isinstance(part, dict):
            if part.get("type") == "text":
                text = part.get("text")  # type: ignore[assignment]
        elif getattr(part, "type", None) == "text":
            text = getattr(part, "text", None)
        if text:
            joined.append(str(text))
    return "".join(joined)


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


@dataclass
class _ModerationInvokeState:
    """Per-async-task prescore payload for post_invoke / streaming (middleware may be shared)."""

    data: pd.DataFrame
    prescore_df: pd.DataFrame
    input_df: pd.DataFrame
    latency_so_far: float
    association_id: str | None
    ctx_token: contextvars.Token[_ModerationInvokeState | None] | None = None


_moderation_invoke_state_ctx: contextvars.ContextVar[_ModerationInvokeState | None] = (
    contextvars.ContextVar("datarobot_genai_moderation_invoke_state", default=None)
)


def _set_moderation_invoke_state(
    *,
    data: pd.DataFrame,
    prescore_df: pd.DataFrame,
    latency_so_far: float,
    association_id: str | None = None,
) -> None:
    state = _ModerationInvokeState(
        data=data,
        prescore_df=prescore_df,
        input_df=data,
        latency_so_far=latency_so_far,
        association_id=association_id,
    )
    token = _moderation_invoke_state_ctx.set(state)
    state.ctx_token = token


def _clear_moderation_invoke_state_if_set() -> None:
    state = _moderation_invoke_state_ctx.get()
    if state is None or state.ctx_token is None:
        return
    _moderation_invoke_state_ctx.reset(state.ctx_token)


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
        """Run prescore guards; skip the agent when prescore blocks the prompt.

        The default NAT ``FunctionMiddleware.function_middleware_invoke`` always awaits
        ``call_next`` after ``pre_invoke``. When prescore blocks, ``pre_invoke`` sets
        ``ctx.output`` to the guard response; we return it immediately so ``call_next``
        (and thus the LLM) is never invoked.
        """
        ctx = InvocationContext(
            function_context=context,
            original_args=args,
            original_kwargs=dict(kwargs),
            modified_args=args,
            modified_kwargs=dict(kwargs),
            output=None,
        )
        try:
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
        finally:
            _clear_moderation_invoke_state_if_set()

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

        # the chat request is not a dataframe, but we'll build a DF internally for moderation.
        prompt_column_name = pipeline.get_input_column(GuardStage.PROMPT)
        prompt = moderation_prompt_from_workflow_input(workflow_input)

        data = pd.DataFrame({prompt_column_name: [prompt]})

        # ==================================================================
        # Step 1: Prescore guards (single run, same path as
        # ``ModerationPipeline.evaluate_prompt`` / ``_run_stage``)
        #
        # We need the full prescore DataFrame for postscore, ``format_result_df``, and
        # streaming; we derive ``EvaluationResult`` with the same ``_from_dataframe`` helper
        # used by the public API (avoids ``run_prescore_guards`` + ``evaluate_prompt``,
        # which would execute prescore twice).
        guards = pipeline.get_prescore_guards()
        if not guards:
            prescore_df = data.copy(deep=True)
            prescore_df[f"blocked_{prompt_column_name}"] = False
            prescore_latency = 0.0
        else:
            prescore_df, prescore_latency = self._moderation._executor.run_guards(
                data.copy(deep=True),
                guards,
                GuardStage.PROMPT,
            )
        prompt_eval = _from_dataframe(prescore_df, prompt_column_name)
        _set_moderation_invoke_state(
            data=data,
            prescore_df=prescore_df,
            latency_so_far=prescore_latency,
            association_id=None,
        )

        if prompt_eval.blocked:
            # If all prompts in the input are blocked, means history as well as the prompt
            # are not worthy to be sent to LLM
            context.output = _dragent_event_response_from_blocked_prompt_eval(prompt_eval)
            return context

        if prompt_eval.replaced and prompt_eval.replacement is not None:
            # PII-style guards may redact the prompt; apply the replacement on the workflow
            # object so ``call_next`` (LLM) receives moderated text; align postscore ``data``.
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
        """
        original_output = context.output
        if self._moderation is None:
            return None

        if isinstance(original_output, DRAgentEventResponse):
            if not _response_has_assistant_text_deltas(original_output):
                return None
            response_text = _assistant_text_joined_from_ag_ui(original_output)
        elif isinstance(original_output, ChatResponse):
            chat_completion = ChatCompletion.model_validate(original_output.model_dump(mode="json"))
            if not _openai_chat_completion_has_assistant_message_content(chat_completion):
                return None
            response_text = _openai_assistant_content_as_str(chat_completion)
        elif isinstance(original_output, str):
            if not original_output.strip():
                return None
            response_text = original_output
        else:
            return None

        pipeline = self._moderation._pipeline
        state = _moderation_invoke_state_ctx.get()
        if state is None:
            return None

        # ==================================================================
        # Step 3: Postscore via ``ModerationPipeline.evaluate_response`` (same path as
        # ``_run_stage`` in dome) when response text is present.
        prompt_column_name = pipeline.get_input_column(GuardStage.PROMPT)
        prompt_for_eval = state.data.loc[0, prompt_column_name]

        response_eval, _ = self._moderation.evaluate_response(
            _text_for_moderation_eval(response_text),
            prompt=_optional_prompt_for_moderation_eval(prompt_for_eval),
        )

        if response_eval.blocked:
            response_message = response_eval.blocked_message or ""
            finish_reason = "content_filter"
        elif response_eval.replaced:
            response_message = response_eval.replacement or ""
            finish_reason = "content_filter"
        else:
            response_message = response_text
            finish_reason = "stop"

        final_completion = build_non_streaming_chat_completion(response_message, finish_reason)
        moderated_dr = chat_completion_to_dragent_event_response(
            final_completion,
            response_eval=response_eval,
        )

        if isinstance(original_output, DRAgentEventResponse):
            preserve_ag_ui_envelope = (
                len(original_output.events) > 1
                and finish_reason == "stop"
                and not response_eval.blocked
                and not (response_eval.replaced and response_eval.replacement is not None)
                and response_message == _assistant_text_joined_from_ag_ui(original_output)
            )
            if preserve_ag_ui_envelope:
                context.output = _merge_moderations_into_multi_event_response(
                    original_output, moderated_dr
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

        When prescore blocks the prompt, ``pre_invoke`` sets ``ctx.output``; we yield that
        response once and return without calling ``call_next``, so the LLM stream is never
        started (NAT's default stream middleware always iterates ``call_next`` after
        ``pre_invoke``).
        """
        ctx = InvocationContext(
            function_context=context,
            original_args=args,
            original_kwargs=dict(kwargs),
            modified_args=args,
            modified_kwargs=dict(kwargs),
            output=None,
        )

        try:
            result = await self.pre_invoke(ctx)
            if result is not None:
                ctx = result
            if ctx.output is not None:
                yield ctx.output
                return

            # Prescore populates invoke state. If ``pre_invoke`` returned early (e.g. no
            # workflow input / prescore skipped), pass the stream through unchanged.
            stream_state = _moderation_invoke_state_ctx.get()
            if stream_state is None:
                async for chunk in call_next(*ctx.modified_args, **ctx.modified_kwargs):
                    yield chunk
                return

            moderation = self._moderation
            assert moderation is not None

            stream_tool_index_map: dict[int, str] = {}

            streaming_context = (
                StreamingContextBuilder()
                .set_pipeline(moderation._pipeline)
                .set_prescore_df(stream_state.prescore_df)
                .set_prescore_latency(stream_state.latency_so_far)
                .set_input_df(stream_state.input_df)
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
                        stream_tool_index_map=stream_tool_index_map,
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
        finally:
            _clear_moderation_invoke_state_if_set()


@register_middleware(  # type: ignore[untyped-decorator]
    config_type=DataRobotModerationConfig
)
async def datarobot_moderation_middleware(
    config: DataRobotModerationConfig,
    builder: Builder,  # noqa: ARG001
) -> AsyncIterator[DataRobotModerationMiddleware]:
    """Register DataRobot LLM guard middleware for NAT/DRAgent workflows."""
    yield DataRobotModerationMiddleware(config, builder)
