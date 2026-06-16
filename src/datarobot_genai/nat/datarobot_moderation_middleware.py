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
"""NAT (NeMo Agent Toolkit) middleware: DataRobot LLM guardrails for DRAgent workflows.

Registered under the ``nat.plugins`` distribution entry ``datarobot_moderation_middleware`` so NAT
loads ``@register_middleware`` without a custom recipe mapping (``_type: datarobot_moderation``).

Expected workflow contracts:

* **DRAgent** (``dragent_fastapi`` + LangGraph / LlamaIndex / CrewAI per-user functions): input
  ``RunAgentInput`` (or ``DRAgentRunAgentInput``), streaming output ``DRAgentEventResponse``.
* **Native NAT chat** (LLM Gateway agents): input ``ChatRequest`` / ``ChatRequestOrMessage``,
  non-streaming output ``ChatResponse``.

Guard configuration (``_type: datarobot_moderation``):

* **Inline (preferred)** — nest guards under ``middleware.<name>.moderation`` in ``workflow.yaml``.
  When present, this block is used even if ``moderation_config.yaml`` also exists.
* **DRUM-style file (fallback)** — when ``moderation`` is omitted, load ``moderation_config.yaml``
  from ``model_dir`` (defaults to the directory containing ``workflow.yaml``, resolved from
  ``DRAGENT_CONFIG_FILE`` when set, otherwise the process working directory).
* If neither source is present or both are empty, the middleware is a no-op.

``ModerationPipeline.stream_response_async`` only accepts OpenAI ``ChatCompletionChunk``; DRAgent
streaming uses ``convert_dragent_event_response_to_openai_chat_completion_chunk`` at that
boundary, then reverses to AG-UI on the way out.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import math
import os
import uuid
from collections.abc import AsyncGenerator
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Literal
from typing import TypeAlias
from typing import cast

import numpy as np
import pandas as pd
import yaml
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
from ag_ui.core import ToolCallResultEvent
from ag_ui.core import ToolCallStartEvent
from ag_ui.core import ToolMessage
from ag_ui.core import UserMessage
from datarobot_dome.api import EvaluationResult
from datarobot_dome.api import ModerationPipeline
from datarobot_dome.api import _from_dataframe
from datarobot_dome.chat_helper import build_moderations_attribute_for_completion
from datarobot_dome.constants import CHAT_COMPLETION_OBJECT
from datarobot_dome.constants import DATAROBOT_MODERATIONS_ATTR
from datarobot_dome.constants import DISABLE_MODERATION_RUNTIME_PARAM_NAME
from datarobot_dome.constants import MODERATION_CONFIG_FILE_NAME
from datarobot_dome.constants import MODERATION_MODEL_NAME
from datarobot_dome.constants import GuardStage
from datarobot_dome.runtime import get_runtime_parameter_value_bool
from datarobot_dome.schema.moderation_config import ModerationConfig
from datarobot_moderation_interface.drum_integration import get_chat_prompt
from nat.builder.builder import Builder
from nat.cli.register_workflow import register_middleware
from nat.data_models.api_server import ChatRequest
from nat.data_models.api_server import ChatRequestOrMessage
from nat.data_models.api_server import ChatResponse
from nat.data_models.api_server import ChatResponseChoice
from nat.data_models.api_server import ChatResponseChunk
from nat.data_models.api_server import ChatResponseChunkChoice
from nat.data_models.api_server import ChoiceDelta
from nat.data_models.api_server import ChoiceDeltaToolCall
from nat.data_models.api_server import ChoiceDeltaToolCallFunction
from nat.data_models.api_server import ChoiceMessage
from nat.data_models.api_server import Usage as NATUsage
from nat.data_models.api_server import UserMessageContentRoleType
from nat.data_models.middleware import FunctionMiddlewareBaseConfig
from nat.middleware.function_middleware import FunctionMiddleware
from nat.middleware.middleware import CallNext
from nat.middleware.middleware import CallNextStream
from nat.middleware.middleware import FunctionMiddlewareContext
from nat.middleware.middleware import InvocationContext
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice as OpenAIChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall as OpenAIChoiceDeltaToolCall
from openai.types.chat.chat_completion_chunk import (
    ChoiceDeltaToolCallFunction as OpenAIChoiceDeltaToolCallFunction,
)
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message_tool_call import Function as OpenAIToolFunction
from pydantic import Field

from datarobot_genai.core.agents import default_usage_metrics
from datarobot_genai.dragent.constants import DRAGENT_CONFIG_FILE_ENV
from datarobot_genai.dragent.frontends.converters import (
    convert_dragent_event_response_to_openai_chat_completion_chunk,
)
from datarobot_genai.dragent.frontends.converters import convert_dragent_event_response_to_str
from datarobot_genai.dragent.frontends.response import DRAgentEventResponse
from datarobot_genai.dragent.frontends.tool_call_registry import mark_args_done
from datarobot_genai.dragent.frontends.tool_call_registry import register_tool_call
from datarobot_genai.dragent.workflow_paths import discover_workflow_yaml
from datarobot_genai.dragent.workflow_paths import publish_dragent_config_file_env

_logger = logging.getLogger(__name__)

WorkflowInput: TypeAlias = RunAgentInput | ChatRequest | ChatRequestOrMessage


def _workflow_input_from_args(args: tuple[Any, ...]) -> WorkflowInput | None:
    if not args:
        return None
    candidate = args[0]
    if isinstance(candidate, (RunAgentInput, ChatRequest, ChatRequestOrMessage)):
        return candidate
    raise TypeError(f"Unsupported workflow input type for moderation: {type(candidate).__name__}")


class DataRobotModerationConfig(
    FunctionMiddlewareBaseConfig,  # type: ignore[misc]
    name="datarobot_moderation",  # type: ignore[call-arg]
):
    """NAT middleware: DataRobot prescore / postscore guards.

    The middleware is a no-op (``enabled`` is ``False``) when no guards are configured in the
    inline ``moderation`` block or in ``moderation_config.yaml``.
    """

    model_dir: str | None = Field(
        default=None,
        description=(
            "Directory containing ``moderation_config.yaml`` and guard assets (DRUM custom model "
            "layout). Used for the inline ``moderation`` block and as the fallback file location. "
            "Defaults to the directory containing ``workflow.yaml`` (``DRAGENT_CONFIG_FILE``), "
            "or the process working directory when that env var is unset."
        ),
    )
    moderation: ModerationConfig | None = Field(
        default=None,
        description=(
            "Inline guard configuration (``ModerationConfig`` from datarobot_dome). When set, "
            "takes priority over ``moderation_config.yaml``."
        ),
    )


def moderation_config_has_guards(moderation: ModerationConfig) -> bool:
    """Return whether ``moderation`` defines at least one guard across all targets."""
    return any(target.guards for target in moderation.targets)


def _default_moderation_model_dir() -> str:
    """Return the directory that holds ``workflow.yaml`` when known, else CWD."""
    publish_dragent_config_file_env()
    config_file = os.environ.get(DRAGENT_CONFIG_FILE_ENV)
    if config_file:
        found = discover_workflow_yaml()
        if found is not None:
            return str(found.parent)
        try:
            return str(Path(config_file).expanduser().resolve().parent)
        except OSError:
            pass
    discovered = discover_workflow_yaml()
    if discovered is not None:
        return str(discovered.parent)
    return os.path.abspath(os.getcwd())


def resolve_moderation_model_dir(model_dir: str | None) -> str:
    """Resolve the base directory for guard assets and ``moderation_config.yaml``."""
    if model_dir is not None:
        return os.path.abspath(model_dir)
    return _default_moderation_model_dir()


def moderation_config_file_path(model_dir: str | None) -> Path:
    """Return the DRUM-style ``moderation_config.yaml`` path under ``model_dir`` (or workflow dir)."""
    return Path(resolve_moderation_model_dir(model_dir)) / MODERATION_CONFIG_FILE_NAME


def load_moderation_config_from_file(model_dir: str | None) -> ModerationConfig | None:
    """Load and validate ``moderation_config.yaml`` from a model directory (DRUM layout)."""
    config_path = moderation_config_file_path(model_dir)
    if not config_path.is_file():
        return None
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return None
    return ModerationConfig.model_validate(raw)


def _load_llm_moderation_pipeline_from_inline(
    config: DataRobotModerationConfig,
) -> ModerationPipeline | None:
    """Load guards from the inline ``moderation`` block."""
    assert config.moderation is not None
    if not moderation_config_has_guards(config.moderation):
        _logger.debug("Inline ``moderation`` has no guards; moderation middleware is a no-op")
        return None
    resolved_model_dir = resolve_moderation_model_dir(config.model_dir)
    return ModerationPipeline.from_config(config.moderation, model_dir=resolved_model_dir)


def _load_llm_moderation_pipeline_from_config_file(
    config: DataRobotModerationConfig,
) -> ModerationPipeline | None:
    """Load guards from ``moderation_config.yaml`` when no inline ``moderation`` block is set."""
    config_path = moderation_config_file_path(config.model_dir)
    if not config_path.is_file():
        _logger.debug(
            "No inline ``moderation`` block and no %s at %s; moderation middleware is a no-op",
            MODERATION_CONFIG_FILE_NAME,
            config_path,
        )
        return None

    moderation = load_moderation_config_from_file(config.model_dir)
    if moderation is None:
        _logger.debug(
            "No inline ``moderation`` block and %s could not be parsed; "
            "moderation middleware is a no-op",
            config_path,
        )
        return None
    if not moderation_config_has_guards(moderation):
        _logger.debug(
            "No inline ``moderation`` block and %s has no guards; moderation middleware is a no-op",
            config_path,
        )
        return None

    return ModerationPipeline.from_yaml(str(config_path))


def load_llm_moderation_pipeline(config: DataRobotModerationConfig) -> ModerationPipeline | None:
    """Build an LLM moderation pipeline from inline ``moderation`` or ``moderation_config.yaml``.

    Returns ``None`` when moderation is disabled, no configuration source is available, or the
    resolved source has no guards, so the middleware is a no-op and can be listed unconditionally
    in ``workflow.yaml``. The inline ``moderation`` block takes priority over the YAML file.
    """
    if get_runtime_parameter_value_bool(DISABLE_MODERATION_RUNTIME_PARAM_NAME, default_value=False):
        _logger.warning("Moderation is disabled via runtime parameter on the model")
        return None

    if config.moderation is not None:
        return _load_llm_moderation_pipeline_from_inline(config)
    return _load_llm_moderation_pipeline_from_config_file(config)


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


def _first_text_delta_in_events(
    events: list[Any],
) -> TextMessageContentEvent | TextMessageChunkEvent | None:
    for event in events:
        if isinstance(event, (TextMessageContentEvent, TextMessageChunkEvent)):
            return event
    return None


def _streaming_text_events_from_openai_chunk(
    completion: ChatCompletionChunk,
    source_ag_ui_events: list[Any] | None,
) -> list[Any]:
    """Rebuild AG-UI text events from a streaming OpenAI chunk (single delta, no synthetic start/end)."""
    delta_content = completion.choices[0].delta.content
    text = "" if delta_content is None else delta_content
    if source_ag_ui_events:
        delta_event = _first_text_delta_in_events(source_ag_ui_events)
        if isinstance(delta_event, TextMessageContentEvent):
            if text:
                return [TextMessageContentEvent(message_id=delta_event.message_id, delta=text)]
            return [
                TextMessageChunkEvent(message_id=delta_event.message_id, role="assistant", delta="")
            ]
        if isinstance(delta_event, TextMessageChunkEvent):
            return [
                TextMessageChunkEvent(
                    message_id=delta_event.message_id,
                    role=delta_event.role,
                    delta=text or "",
                )
            ]
    mid = str(uuid.uuid4())
    if text:
        return [TextMessageContentEvent(message_id=mid, delta=text)]
    return [TextMessageChunkEvent(message_id=mid, role="assistant", delta="")]


def dragent_event_response_to_dome_chunk(
    response: DRAgentEventResponse,
) -> ChatCompletionChunk:
    """Convert one DRAgent stream chunk to an OpenAI chunk for ``ModerationPipeline`` streaming."""
    completion_chunk = convert_dragent_event_response_to_openai_chat_completion_chunk(response)
    event_tool_calls = _tool_calls_from_ag_ui_events(response.events)
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
    completion: ChatCompletionChunk,
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


def _datarobot_moderations_merged_prompt_and_response_eval(
    response_eval: EvaluationResult,
    *,
    prompt_eval: EvaluationResult | None,
) -> dict[str, Any] | None:
    """Merge prescore (prompt-stage) and postscore metrics for a single client metadata dict."""
    merged: dict[str, Any] = {}
    if prompt_eval is not None and prompt_eval.metrics:
        merged.update(_json_safe_moderation_metadata(prompt_eval.metrics))
    if response_eval.metrics:
        merged.update(_json_safe_moderation_metadata(response_eval.metrics))
    return cast(dict[str, Any], merged) if merged else None


def _prescore_datarobot_moderations_from_df(
    pipeline: Any,
    prescore_df: pd.DataFrame,
) -> dict[str, Any] | None:
    """Serialize prescore guard metrics the same way dome attaches them to the first stream chunk."""
    raw = build_moderations_attribute_for_completion(pipeline, prescore_df)
    if not raw:
        return None
    return cast(dict[str, Any], _json_safe_moderation_metadata(raw))


def _postscore_only_datarobot_moderations(
    moderations: dict[str, Any] | None,
    prescore_moderations: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Drop prescore keys so response-stage metrics stay on moderated text chunks only."""
    if not moderations:
        return None
    if not prescore_moderations:
        return moderations
    stripped = {k: v for k, v in moderations.items() if k not in prescore_moderations}
    return cast(dict[str, Any], stripped) if stripped else None


def _is_text_message_start_response(response: DRAgentEventResponse) -> bool:
    return any(isinstance(ev, TextMessageStartEvent) for ev in response.events)


@dataclass
class _StreamingPrescoreModerationState:
    """Track prescore attachment across a moderated DRAgent stream (dome first-chunk semantics)."""

    prescore_moderations: dict[str, Any] | None
    prescore_attached: bool = False
    saw_moderated_content: bool = False

    def emit(self, response: DRAgentEventResponse) -> DRAgentEventResponse:
        if (
            not self.prescore_attached
            and self.prescore_moderations
            and _is_text_message_start_response(response)
        ):
            self.prescore_attached = True
            return response.model_copy(update={"datarobot_moderations": self.prescore_moderations})
        return response

    def emit_moderated(self, response: DRAgentEventResponse) -> DRAgentEventResponse:
        mods = response.datarobot_moderations
        if self.prescore_moderations and mods:
            if self.prescore_attached or self.saw_moderated_content:
                mods = _postscore_only_datarobot_moderations(mods, self.prescore_moderations)
                response = response.model_copy(update={"datarobot_moderations": mods})
        self.saw_moderated_content = True
        return response


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


def _upstream_model_from_dragent_response(response: DRAgentEventResponse) -> str | None:
    if response.model:
        return response.model
    if response.original_chunk is not None:
        return response.original_chunk.model or None
    return None


def _should_use_moderation_model_name(
    response_eval: EvaluationResult,
    *,
    prompt_eval: EvaluationResult | None = None,
) -> bool:
    """Use ``MODERATION_MODEL_NAME`` when guards blocked or replaced prompt or response text."""
    if response_eval.blocked or response_eval.replaced:
        return True
    if prompt_eval is not None and prompt_eval.replaced:
        return True
    return False


def _dragent_event_response_from_postscore_assistant_text(
    response_message: str,
    finish_reason: str,
    response_eval: EvaluationResult,
    *,
    upstream_model: str | None = None,
    prompt_eval: EvaluationResult | None = None,
) -> DRAgentEventResponse:
    """Build AG-UI output after postscore from message, finish reason, and eval (no ChatCompletion).

    Matches ``dome_chunk_to_dragent_event_response`` for a text-only chunk without going through dome.
    """
    use_moderation_model = _should_use_moderation_model_name(response_eval, prompt_eval=prompt_eval)
    chunk_model = (
        MODERATION_MODEL_NAME
        if use_moderation_model
        else (upstream_model if upstream_model is not None else "unknown-model")
    )
    response_model = MODERATION_MODEL_NAME if use_moderation_model else upstream_model
    events = _assistant_text_events_from_message(response_message)
    completion_id = str(uuid.uuid4())
    created_ts = int(datetime.now(tz=UTC).timestamp())
    created_dt = datetime.fromtimestamp(created_ts, tz=UTC)
    delta = ChoiceDelta(
        content=response_message,
        role=UserMessageContentRoleType.ASSISTANT,
        tool_calls=None,
    )
    chunk = ChatResponseChunk(
        id=completion_id,
        choices=[
            ChatResponseChunkChoice(index=0, delta=delta, finish_reason=finish_reason),
        ],
        created=created_dt,
        model=chunk_model,
        object="chat.completion.chunk",
        usage=None,
    )
    return DRAgentEventResponse(
        events=events,
        usage_metrics=default_usage_metrics(),
        original_chunk=chunk,
        model=response_model,
        datarobot_moderations=_datarobot_moderations_merged_prompt_and_response_eval(
            response_eval,
            prompt_eval=prompt_eval,
        ),
    )


def _nat_chat_response_from_postscore_assistant_text(
    response_message: str,
    finish_reason: str,
    original_nat_response: ChatResponse,
    response_eval: EvaluationResult,
    *,
    prompt_eval: EvaluationResult | None = None,
) -> ChatResponse:
    """Build NAT ``ChatResponse`` after postscore without an intermediate OpenAI ``ChatCompletion``.

    Preserves ``usage`` from ``original_nat_response`` when present (same as the former
    ``build_non_streaming_chat_completion`` + ``model_validate`` path). Attaches
    ``datarobot_moderations`` as an extra field (NAT ``ChatResponse`` allows extras), aligned
    with ``_dragent_event_response_from_postscore_assistant_text``.
    """
    usage = original_nat_response.usage
    if usage is None:
        usage = NATUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    out_model = (
        MODERATION_MODEL_NAME
        if _should_use_moderation_model_name(response_eval, prompt_eval=prompt_eval)
        else original_nat_response.model
    )
    return ChatResponse(
        id=str(uuid.uuid4()),
        object=CHAT_COMPLETION_OBJECT,
        model=out_model,
        created=datetime.now(tz=UTC),
        choices=[
            ChatResponseChoice(
                index=0,
                finish_reason=cast(
                    Literal["stop", "length", "tool_calls", "content_filter", "function_call"]
                    | None,
                    finish_reason,
                ),
                message=ChoiceMessage(
                    content=response_message,
                    role=UserMessageContentRoleType.ASSISTANT,
                ),
            )
        ],
        usage=usage,
        datarobot_moderations=_datarobot_moderations_merged_prompt_and_response_eval(
            response_eval,
            prompt_eval=prompt_eval,
        ),
    )


def dome_chunk_to_dragent_event_response(
    completion: ChatCompletionChunk,
    *,
    response_eval: EvaluationResult | None = None,
    source_ag_ui_events: list[Any] | None = None,
    stream_tool_index_map: dict[int, str] | None = None,
) -> DRAgentEventResponse:
    """Convert a moderated OpenAI streaming chunk back to ``DRAgentEventResponse``.

    When ``response_eval`` is set (non-streaming postscore path), ``datarobot_moderations`` is
    taken from ``response_eval.metrics``; otherwise it is read from the completion's moderation
    sidecar attribute (streaming chunks from ``ModerationIterator``).
    """
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


def moderation_prompt_from_workflow_input(workflow_input: WorkflowInput) -> str:
    """Extract the prescore prompt string from AG-UI or NAT chat input.

    Delegates to ``get_chat_prompt`` on completion-params built from ``workflow_input``.
    ``ChatRequestOrMessage`` with only ``input_message`` (no ``messages``) is handled directly
    because ``get_chat_prompt`` requires a non-empty ``messages`` list.
    """
    if (
        isinstance(workflow_input, ChatRequestOrMessage)
        and workflow_input.input_message is not None
    ):
        return workflow_input.input_message
    return get_chat_prompt(workflow_input_to_completion_dict(workflow_input))


def run_agent_input_to_completion_dict(rai: RunAgentInput) -> dict[str, Any]:
    ccp: dict[str, Any] = {
        "messages": [],
        "stream": True,
    }
    for m in rai.messages:
        converted = _ag_ui_message_to_openai(m)
        # Skip AG-UI message types with no OpenAI equivalent (e.g. ReasoningMessage),
        # which map to an empty dict. get_chat_prompt indexes message["role"] and would
        # raise KeyError on a role-less entry (multi-turn history can replay reasoning).
        if converted:
            ccp["messages"].append(converted)
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


def workflow_input_to_completion_dict(workflow_input: WorkflowInput) -> dict[str, Any]:
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
    workflow_input: WorkflowInput,
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
    _logger.warning(
        "Prescore replaced the prompt but workflow input has no applicable message field "
        "(empty or missing ``messages``, no ``input_message``); leaving input unchanged."
    )
    return False


def _clear_prompt_replacement_flags_in_prescore_df(df: pd.DataFrame, prompt_column: str) -> None:
    """Clear replacement columns when moderated text could not be written to the workflow object.

    Keeps ``state.prescore_df`` / streaming ``ModerationIterator`` metadata aligned with the
    prompt text actually sent to the LLM (``state.prompt``).
    """
    replaced_col = f"replaced_{prompt_column}"
    if replaced_col in df.columns:
        df.loc[df.index[0], replaced_col] = False
    replaced_msg_col = f"replaced_message_{prompt_column}"
    if replaced_msg_col in df.columns:
        df.loc[df.index[0], replaced_msg_col] = np.nan


def _prompt_sent_after_prescore_replacement(
    workflow_input: WorkflowInput,
    *,
    original_prompt: str,
    prompt_eval: EvaluationResult,
    prescore_df: pd.DataFrame,
    prompt_column: str,
) -> tuple[str, bool]:
    """Apply prescore replacement to workflow input; return prompt for postscore and whether it applied."""
    replacement = prompt_eval.replacement
    if not (prompt_eval.replaced and replacement is not None):
        return original_prompt, False
    if _apply_moderated_prompt_text_to_workflow_input(workflow_input, replacement):
        return replacement, True
    _clear_prompt_replacement_flags_in_prescore_df(prescore_df, prompt_column)
    return original_prompt, False


def skip_event_type(event: Event) -> bool:
    return event.type not in {
        EventType.TEXT_MESSAGE_CONTENT,
        EventType.TEXT_MESSAGE_CHUNK,
    }


def _track_open_text_message(open_message_ids: set[str], event: Event) -> None:
    """Track assistant text segments that received content but did not end."""
    if isinstance(event, TextMessageEndEvent):
        open_message_ids.discard(event.message_id)
    elif isinstance(event, (TextMessageContentEvent, TextMessageChunkEvent)):
        if event.message_id:
            open_message_ids.add(event.message_id)


def _track_open_tool_call(open_tool_call_ids: set[str], event: Event) -> None:
    """Track tool calls that started but did not end."""
    if isinstance(event, ToolCallEndEvent):
        if event.tool_call_id:
            open_tool_call_ids.discard(event.tool_call_id)
    elif isinstance(event, ToolCallStartEvent):
        if event.tool_call_id:
            open_tool_call_ids.add(event.tool_call_id)


def _synthetic_text_message_end_events(
    open_message_ids: set[str],
) -> list[TextMessageEndEvent]:
    """Close dangling text segments when upstream ends the stream without TEXT_MESSAGE_END."""
    end_events = [TextMessageEndEvent(message_id=message_id) for message_id in open_message_ids]
    open_message_ids.clear()
    return end_events


def _synthetic_text_message_end_responses(
    open_message_ids: set[str],
) -> list[DRAgentEventResponse]:
    """Wrap synthetic ``TEXT_MESSAGE_END`` events as ``DRAgentEventResponse`` batches."""
    zero = default_usage_metrics()
    return [
        DRAgentEventResponse(events=[end_event], usage_metrics=zero)
        for end_event in _synthetic_text_message_end_events(open_message_ids)
    ]


def _synthetic_tool_call_end_events(open_tool_call_ids: set[str]) -> list[ToolCallEndEvent]:
    """Close dangling tool calls when upstream ends the stream without ``TOOL_CALL_END``."""
    end_events = [
        ToolCallEndEvent(tool_call_id=tool_call_id) for tool_call_id in open_tool_call_ids
    ]
    open_tool_call_ids.clear()
    return end_events


def _synthetic_tool_call_end_responses(
    open_tool_call_ids: set[str],
) -> list[DRAgentEventResponse]:
    """Wrap synthetic ``TOOL_CALL_END`` events as ``DRAgentEventResponse`` batches."""
    zero = default_usage_metrics()
    return [
        DRAgentEventResponse(events=[end_event], usage_metrics=zero)
        for end_event in _synthetic_tool_call_end_events(open_tool_call_ids)
    ]


def _track_dragent_response_events(
    open_message_ids: set[str],
    open_tool_call_ids: set[str],
    response: DRAgentEventResponse,
) -> None:
    for event in response.events:
        _track_open_text_message(open_message_ids, event)
        _track_open_tool_call(open_tool_call_ids, event)


def _response_has_assistant_text_deltas(response: DRAgentEventResponse) -> bool:
    """Check if the payload includes assistant text AG-UI deltas
    (possibly after lifecycle events).
    """
    return any(
        isinstance(ev, (TextMessageContentEvent, TextMessageChunkEvent)) for ev in response.events
    )


def _tool_lifecycle_passthrough_responses(
    response: DRAgentEventResponse,
) -> list[DRAgentEventResponse]:
    """Re-emit tool end/result events from a moderated source batch.

    ``dome_chunk_to_dragent_event_response`` rebuilds text from the moderated OpenAI
    chunk and drops co-located ``TOOL_CALL_END`` / ``TOOL_CALL_RESULT`` events that
    ``mark_args_done`` or the step adaptor attached to the same batch.
    """
    zero = default_usage_metrics()
    return [
        DRAgentEventResponse(events=[event], usage_metrics=zero)
        for event in response.events
        if isinstance(event, (ToolCallEndEvent, ToolCallResultEvent))
    ]


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

    ``RUN_FINISHED`` is buffered separately and only emitted after dangling text segments close.
    """
    return event.type in (
        EventType.TEXT_MESSAGE_END,
        EventType.TEXT_MESSAGE_START,
    )


def _pending_after_moderated_chunk(
    pending_deferred: list[DRAgentEventResponse],
    pending_pass_through: list[DRAgentEventResponse],
) -> list[DRAgentEventResponse]:
    """Order buffered events after a moderated text chunk.

    Deferred ``TEXT_MESSAGE_END`` closes the prior segment first. Pass-through events (for example
    ``STEP_FINISHED`` / ``STEP_STARTED``) keep upstream order. Deferred ``TEXT_MESSAGE_START`` for
    the next segment follows so step boundaries stay valid for AG-UI verification.
    """
    deferred_ends = [
        item
        for item in pending_deferred
        if item.events and item.events[0].type == EventType.TEXT_MESSAGE_END
    ]
    deferred_starts = [
        item
        for item in pending_deferred
        if item.events and item.events[0].type == EventType.TEXT_MESSAGE_START
    ]
    deferred_other = [
        item
        for item in pending_deferred
        if item not in deferred_ends and item not in deferred_starts
    ]
    return deferred_ends + list(pending_pass_through) + deferred_starts + deferred_other


def _drain_pending_after_moderated_chunk(
    pending_deferred: list[DRAgentEventResponse],
    pending_pass_through: list[DRAgentEventResponse],
) -> list[DRAgentEventResponse]:
    ordered = _pending_after_moderated_chunk(pending_deferred, pending_pass_through)
    pending_deferred.clear()
    pending_pass_through.clear()
    return ordered


def _pending_end_message_ids(pending_deferred: list[DRAgentEventResponse]) -> set[str]:
    return {
        item.events[0].message_id
        for item in pending_deferred
        if item.events
        and item.events[0].type == EventType.TEXT_MESSAGE_END
        and item.events[0].message_id
    }


def _pending_tool_end_ids(pending_deferred: list[DRAgentEventResponse]) -> set[str]:
    return {
        item.events[0].tool_call_id
        for item in pending_deferred
        if item.events
        and item.events[0].type == EventType.TOOL_CALL_END
        and item.events[0].tool_call_id
    }


def _pending_tool_end_ids_in_responses(responses: list[DRAgentEventResponse]) -> set[str]:
    """Collect ``tool_call_id`` values from ``TOOL_CALL_END`` events in *responses*."""
    end_ids: set[str] = set()
    for item in responses:
        for event in item.events:
            if isinstance(event, ToolCallEndEvent) and event.tool_call_id:
                end_ids.add(event.tool_call_id)
    return end_ids


def _flushed_deferred_tool_call_responses(
    open_tool_call_ids: set[str],
) -> list[DRAgentEventResponse]:
    """Return deferred ``TOOL_CALL_END``/``RESULT`` pairs from the stream-converter registry."""
    zero = default_usage_metrics()
    flushed: list[DRAgentEventResponse] = []
    for tool_call_id in list(open_tool_call_ids):
        deferred_events = mark_args_done(tool_call_id)
        if deferred_events:
            flushed.append(DRAgentEventResponse(events=deferred_events, usage_metrics=zero))
    return flushed


def _prepend_flushed_deferred_tool_call_responses(
    pending_deferred: list[DRAgentEventResponse],
    open_tool_call_ids: set[str],
) -> None:
    """Flush step-adaptor deferred ``TOOL_CALL_END``/``RESULT`` pairs before synthetic ends."""
    flushed = _flushed_deferred_tool_call_responses(open_tool_call_ids)
    if flushed:
        pending_deferred[:0] = list(reversed(flushed))


def _prepend_synthetic_text_message_ends_for_dangling_segments(
    pending_deferred: list[DRAgentEventResponse],
    open_text_message_ids: set[str],
) -> None:
    """Close dangling text segments before ``RUN_FINISHED`` reaches the client.

    ``TEXT_MESSAGE_CHUNK`` responses (and other deltas without a matching upstream END) leave
    message ids in ``open_text_message_ids``. Synthetic ends are prepended to the deferred
    queue so they drain before pass-through lifecycle events such as ``RUN_FINISHED``.
    """
    still_open = open_text_message_ids - _pending_end_message_ids(pending_deferred)
    if not still_open:
        return
    ids_to_close = set(still_open)
    pending_deferred[:0] = _synthetic_text_message_end_responses(ids_to_close)
    open_text_message_ids.difference_update(ids_to_close)


def _prepend_synthetic_tool_call_ends_for_dangling_calls(
    pending_deferred: list[DRAgentEventResponse],
    pending_pass_through: list[DRAgentEventResponse],
    open_tool_call_ids: set[str],
) -> None:
    """Close dangling tool calls before ``RUN_FINISHED`` reaches the client."""
    pending_ends = _pending_tool_end_ids(pending_deferred) | _pending_tool_end_ids_in_responses(
        pending_pass_through
    )
    still_open = open_tool_call_ids - pending_ends
    if not still_open:
        return
    ids_to_close = set(still_open)
    pending_deferred[:0] = _synthetic_tool_call_end_responses(ids_to_close)
    open_tool_call_ids.difference_update(ids_to_close)


def _drain_pending_with_dangling_lifecycle_closed(
    pending_deferred: list[DRAgentEventResponse],
    pending_pass_through: list[DRAgentEventResponse],
    open_text_message_ids: set[str],
    open_tool_call_ids: set[str],
) -> list[DRAgentEventResponse]:
    """Drain pending events after closing dangling text segments and tool calls."""
    for item in pending_pass_through:
        _track_dragent_response_events(open_text_message_ids, open_tool_call_ids, item)
    for item in pending_deferred:
        _track_dragent_response_events(open_text_message_ids, open_tool_call_ids, item)
    _prepend_flushed_deferred_tool_call_responses(pending_deferred, open_tool_call_ids)
    for item in pending_deferred:
        _track_dragent_response_events(open_text_message_ids, open_tool_call_ids, item)
    _prepend_synthetic_tool_call_ends_for_dangling_calls(
        pending_deferred, pending_pass_through, open_tool_call_ids
    )
    _prepend_synthetic_text_message_ends_for_dangling_segments(
        pending_deferred, open_text_message_ids
    )
    return _drain_pending_after_moderated_chunk(pending_deferred, pending_pass_through)


def _first_text_delta_event(
    response: DRAgentEventResponse,
) -> TextMessageContentEvent | TextMessageChunkEvent | None:
    return _first_text_delta_in_events(response.events)


def _leading_text_message_starts(response: DRAgentEventResponse) -> list[TextMessageStartEvent]:
    """Return ``TEXT_MESSAGE_START`` events that precede the first text delta in *response*."""
    starts: list[TextMessageStartEvent] = []
    for event in response.events:
        if isinstance(event, (TextMessageContentEvent, TextMessageChunkEvent)):
            break
        if isinstance(event, TextMessageStartEvent):
            starts.append(event)
    return starts


def _text_message_id_from_response(response: DRAgentEventResponse) -> str | None:
    delta_event = _first_text_delta_event(response)
    return delta_event.message_id if delta_event is not None else None


def _record_emitted_text_message_starts(
    emitted_message_ids: set[str], response: DRAgentEventResponse
) -> None:
    for event in response.events:
        if isinstance(event, TextMessageStartEvent) and event.message_id:
            emitted_message_ids.add(event.message_id)


def _take_pending_text_message_start_for_message(
    pending_deferred: list[DRAgentEventResponse],
    message_id: str,
) -> DRAgentEventResponse | None:
    """Remove and return a buffered ``TEXT_MESSAGE_START`` for *message_id* only."""
    for index, item in enumerate(pending_deferred):
        if any(
            isinstance(event, TextMessageStartEvent) and event.message_id == message_id
            for event in item.events
        ):
            return pending_deferred.pop(index)
    return None


async def _aclose_async_iterator(iterator: AsyncGenerator[Any]) -> None:
    """Close an async generator, ignoring errors from double-close or partial consumption."""
    try:
        await iterator.aclose()
    except Exception:
        _logger.debug("Error closing async iterator during stream teardown", exc_info=True)


async def _moderated_dragent_stream(
    upstream: AsyncGenerator[DRAgentEventResponse],
    *,
    moderation: ModerationPipeline,
    stream_state: _ModerationInvokeState,
) -> AsyncGenerator[DRAgentEventResponse]:
    """Yield DRAgent stream chunks with AG-UI-safe ordering around moderated text deltas.

    Non-text upstream events pass through immediately until the first text delta. Text deltas are
    moderated via ``stream_response_async``; ``TEXT_MESSAGE_START`` / ``END`` and other events read
    during peek-ahead are buffered and emitted after each moderated chunk.
    """
    stream_tool_index_map: dict[int, str] = {}
    open_text_message_ids: set[str] = set()
    open_tool_call_ids: set[str] = set()
    emitted_text_message_starts: set[str] = set()
    pending_deferred: list[DRAgentEventResponse] = []
    pending_pass_through: list[DRAgentEventResponse] = []
    pending_run_finished: list[DRAgentEventResponse] = []
    moderation_source_responses: list[DRAgentEventResponse] = []
    stopped_for_content_filter = False
    prescore_state = _StreamingPrescoreModerationState(
        prescore_moderations=_prescore_datarobot_moderations_from_df(
            moderation._pipeline,
            stream_state.prescore_df,
        ),
    )

    def buffer_passthrough(response: DRAgentEventResponse) -> None:
        if response.events and response.events[0].type == EventType.RUN_FINISHED:
            pending_run_finished.append(response)
        elif response.events and _defer_until_after_moderated_chunk(response.events[0]):
            pending_deferred.append(response)
        else:
            pending_pass_through.append(response)

    async def next_text_response() -> DRAgentEventResponse | None:
        async for response in upstream:
            if not response.events or not _response_has_assistant_text_deltas(response):
                buffer_passthrough(response)
                continue
            return response
        return None

    async def completion_chunks(
        first_text: DRAgentEventResponse,
    ) -> AsyncIterator[ChatCompletionChunk]:
        current: DRAgentEventResponse | None = first_text
        while current is not None:
            moderation_source_responses.append(current)
            yield dragent_event_response_to_dome_chunk(current)
            current = await next_text_response()

    first_text: DRAgentEventResponse | None = None
    try:
        async for response in upstream:
            if response.events and _response_has_assistant_text_deltas(response):
                first_text = response
                break
            if response.events and response.events[0].type == EventType.RUN_FINISHED:
                pending_run_finished.append(response)
            else:
                _track_dragent_response_events(open_text_message_ids, open_tool_call_ids, response)
                _record_emitted_text_message_starts(emitted_text_message_starts, response)
                yield prescore_state.emit(response)
        if first_text is None:
            for item in _drain_pending_with_dangling_lifecycle_closed(
                pending_deferred,
                pending_pass_through,
                open_text_message_ids,
                open_tool_call_ids,
            ):
                _track_dragent_response_events(open_text_message_ids, open_tool_call_ids, item)
                _record_emitted_text_message_starts(emitted_text_message_starts, item)
                yield prescore_state.emit(item)
            for item in pending_run_finished:
                _track_dragent_response_events(open_text_message_ids, open_tool_call_ids, item)
                _record_emitted_text_message_starts(emitted_text_message_starts, item)
                yield prescore_state.emit(item)
            return

        async with contextlib.aclosing(
            moderation.stream_response_async(
                completion_chunks(first_text),
                prompt=stream_state.prompt,
                prescore_df=stream_state.prescore_df,
                prescore_latency=stream_state.latency_so_far,
            )
        ) as moderation_stream:
            async for moderated in moderation_stream:
                source_response = moderation_source_responses.pop(0)
                moderated_response = prescore_state.emit_moderated(
                    dome_chunk_to_dragent_event_response(
                        moderated,
                        source_ag_ui_events=source_response.events,
                        stream_tool_index_map=stream_tool_index_map,
                    )
                )
                message_id = _text_message_id_from_response(source_response)
                first_delta = _first_text_delta_event(source_response)
                if (
                    message_id
                    and message_id not in emitted_text_message_starts
                    and first_delta is not None
                ):
                    extra_start_events: list[Any] = list(
                        _leading_text_message_starts(source_response)
                    )
                    if not extra_start_events:
                        pending_start = _take_pending_text_message_start_for_message(
                            pending_deferred, message_id
                        )
                        if pending_start is not None:
                            extra_start_events.extend(
                                ev
                                for ev in pending_start.events
                                if isinstance(ev, TextMessageStartEvent)
                                and ev.message_id == message_id
                            )
                        else:
                            extra_start_events.append(
                                TextMessageStartEvent(message_id=message_id, role="assistant")
                            )
                    emitted_text_message_starts.add(message_id)
                    moderated_response = moderated_response.model_copy(
                        update={"events": extra_start_events + moderated_response.events}
                    )
                    if not prescore_state.prescore_attached and prescore_state.prescore_moderations:
                        prescore_state.prescore_attached = True
                        existing_mods = moderated_response.datarobot_moderations or {}
                        moderated_response = moderated_response.model_copy(
                            update={
                                "datarobot_moderations": {
                                    **prescore_state.prescore_moderations,
                                    **existing_mods,
                                }
                            }
                        )
                _track_dragent_response_events(
                    open_text_message_ids, open_tool_call_ids, moderated_response
                )
                yield moderated_response
                pending_pass_through.extend(_tool_lifecycle_passthrough_responses(source_response))
                for item in _drain_pending_after_moderated_chunk(
                    pending_deferred, pending_pass_through
                ):
                    _track_dragent_response_events(open_text_message_ids, open_tool_call_ids, item)
                    _record_emitted_text_message_starts(emitted_text_message_starts, item)
                    yield prescore_state.emit(item)
                finish = moderated.choices[0].finish_reason if moderated.choices else None
                if finish == "content_filter":
                    for flushed in _flushed_deferred_tool_call_responses(open_tool_call_ids):
                        _track_dragent_response_events(
                            open_text_message_ids, open_tool_call_ids, flushed
                        )
                        yield flushed
                    for end_response in _synthetic_tool_call_end_responses(open_tool_call_ids):
                        yield end_response
                    for end_response in _synthetic_text_message_end_responses(
                        open_text_message_ids
                    ):
                        yield end_response
                    stopped_for_content_filter = True
                    break

        if not stopped_for_content_filter:
            for item in _drain_pending_with_dangling_lifecycle_closed(
                pending_deferred,
                pending_pass_through,
                open_text_message_ids,
                open_tool_call_ids,
            ):
                _track_dragent_response_events(open_text_message_ids, open_tool_call_ids, item)
                _record_emitted_text_message_starts(emitted_text_message_starts, item)
                yield prescore_state.emit(item)
            for item in pending_run_finished:
                _track_dragent_response_events(open_text_message_ids, open_tool_call_ids, item)
                _record_emitted_text_message_starts(emitted_text_message_starts, item)
                yield prescore_state.emit(item)
    finally:
        await _aclose_async_iterator(upstream)


@dataclass
class _ModerationInvokeState:
    """Per-async-task prescore payload for post_invoke / streaming (middleware may be shared)."""

    prompt: str
    prescore_df: pd.DataFrame
    latency_so_far: float
    ctx_token: contextvars.Token[_ModerationInvokeState | None] | None = None


_moderation_invoke_state_ctx: contextvars.ContextVar[_ModerationInvokeState | None] = (
    contextvars.ContextVar("datarobot_genai_moderation_invoke_state", default=None)
)


def _set_moderation_invoke_state(
    *,
    prompt: str,
    prescore_df: pd.DataFrame,
    latency_so_far: float,
) -> None:
    state = _ModerationInvokeState(
        prompt=prompt,
        prescore_df=prescore_df,
        latency_so_far=latency_so_far,
    )
    token = _moderation_invoke_state_ctx.set(state)
    state.ctx_token = token


def _clear_moderation_invoke_state_if_set() -> None:
    state = _moderation_invoke_state_ctx.get()
    if state is None or state.ctx_token is None:
        return
    _moderation_invoke_state_ctx.reset(state.ctx_token)


class DataRobotModerationMiddleware(
    FunctionMiddleware,  # type: ignore[misc]
):
    """Guardrails middleware for DRAgent NAT workflows and native NAT chat agents.

    * **DRAgent** (``RunAgentInput`` in, ``DRAgentEventResponse`` stream out): prescore/postscore
      on AG-UI text; streaming moderation uses ``dragent_event_response_to_dome_chunk`` at the
      dome boundary.
    * **NAT chat** (``ChatRequest`` / ``ChatRequestOrMessage`` in, ``ChatResponse`` out): same
      guard pipeline with NAT message models instead of AG-UI.

    When no guards are configured (missing inline block and YAML file, or empty guard list),
    ``load_llm_moderation_pipeline`` returns ``None`` and this middleware is a no-op
    (``enabled`` is ``False``) without requiring DataRobot credentials.
    """

    def __init__(self, config: DataRobotModerationConfig, builder: Builder) -> None:  # noqa: ARG002
        super().__init__()
        self._config = config
        self._moderation = load_llm_moderation_pipeline(config)

    def _get_moderation(self) -> ModerationPipeline | None:
        """Return the moderation pipeline, retrying discovery if startup missed ``workflow.yaml``."""
        if self._moderation is None:
            publish_dragent_config_file_env()
            self._moderation = load_llm_moderation_pipeline(self._config)
        return self._moderation

    @property
    def enabled(self) -> bool:
        return self._get_moderation() is not None

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
        workflow_input = _workflow_input_from_args(context.original_args)
        if workflow_input is None:
            return None
        moderation = self._get_moderation()
        if moderation is None:
            return None

        pipeline = moderation._pipeline

        prompt_column_name = pipeline.get_input_column(GuardStage.PROMPT)
        prompt = moderation_prompt_from_workflow_input(workflow_input)

        # Step 1: Prescore via ``ModerationPipeline.evaluate_prompt_async`` (non-blocking).
        prompt_eval, prescore_latency, prescore_df = await moderation.evaluate_prompt_async(prompt)

        if prompt_eval.blocked:
            # If all prompts in the input are blocked, means history as well as the prompt
            # are not worthy to be sent to LLM. No invoke state: post_invoke / streaming never run.
            context.output = _dragent_event_response_from_blocked_prompt_eval(prompt_eval)
            return context

        prompt_sent, workflow_rewritten = _prompt_sent_after_prescore_replacement(
            workflow_input,
            original_prompt=prompt,
            prompt_eval=prompt_eval,
            prescore_df=prescore_df,
            prompt_column=prompt_column_name,
        )
        _set_moderation_invoke_state(
            prompt=prompt_sent,
            prescore_df=prescore_df,
            latency_so_far=prescore_latency,
        )
        # Return context only when workflow input was rewritten (signals modified_args to NAT).
        return context if workflow_rewritten else None

    async def post_invoke(self, context: InvocationContext) -> InvocationContext | None:
        """Post-invocation hook called after the function returns.

        Args:
            context: Invocation context containing function metadata, args, and output

        Returns:
            InvocationContext if modified, or None to pass through unchanged.
        """
        original_output = context.output
        moderation = self._get_moderation()
        if moderation is None:
            return None

        if isinstance(original_output, DRAgentEventResponse):
            if not _response_has_assistant_text_deltas(original_output):
                return None
            response_text = convert_dragent_event_response_to_str(original_output)
        elif isinstance(original_output, ChatResponse):
            if not original_output.choices:
                return None
            response_text = original_output.choices[0].message.content or ""
        elif isinstance(original_output, str):
            response_text = original_output
        else:
            return None

        if not response_text.strip():
            return None

        pipeline = moderation._pipeline
        state = _moderation_invoke_state_ctx.get()
        if state is None:
            return None

        # ==================================================================
        # Step 3: Postscore via ``ModerationPipeline.evaluate_response`` (same path as
        # ``_run_stage`` in dome) when response text is present.
        prompt_column_name = pipeline.get_input_column(GuardStage.PROMPT)
        response_eval, _, _postscore_df = await moderation.evaluate_response_async(
            response_text,
            prompt=state.prompt,
        )

        prompt_eval = _from_dataframe(state.prescore_df, prompt_column_name)

        if response_eval.blocked:
            response_message = response_eval.blocked_message or ""
            finish_reason = "content_filter"
        elif response_eval.replaced:
            response_message = response_eval.replacement or ""
            finish_reason = "content_filter"
        else:
            response_message = response_text
            finish_reason = "stop"

        if isinstance(original_output, DRAgentEventResponse):
            moderated_dr = _dragent_event_response_from_postscore_assistant_text(
                response_message,
                finish_reason,
                response_eval,
                upstream_model=_upstream_model_from_dragent_response(original_output),
                prompt_eval=prompt_eval,
            )
            preserve_ag_ui_envelope = (
                len(original_output.events) > 1
                and finish_reason == "stop"
                and not response_eval.blocked
                and not response_eval.replaced
                and response_message == response_text
            )
            if preserve_ag_ui_envelope:
                context.output = _merge_moderations_into_multi_event_response(
                    original_output, moderated_dr
                )
            else:
                context.output = moderated_dr
        elif isinstance(original_output, ChatResponse):
            context.output = _nat_chat_response_from_postscore_assistant_text(
                response_message,
                finish_reason,
                original_output,
                response_eval,
                prompt_eval=prompt_eval,
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
    ) -> AsyncIterator[DRAgentEventResponse]:
        """Execute middleware hooks around DRAgent streaming (``DRAgentEventResponse`` chunks).

        Pre-invoke runs once before streaming starts.
        Moderation is applied per-chunk as they stream through.

        Note: Framework checks ``enabled`` before calling this method.
        You do NOT need to check ``enabled`` yourself.

        Args:
            args: Positional arguments for the function (first arg is typically the input value).
            call_next: Callable to invoke next middleware or target stream.
            context: Static function metadata.
            kwargs: Keyword arguments for the function.

        Yields:
            Stream chunks with per-delta postscore via ``ModerationPipeline.stream_response_async``.

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
                async with contextlib.aclosing(
                    cast(
                        AsyncGenerator[DRAgentEventResponse, None],
                        call_next(*ctx.modified_args, **ctx.modified_kwargs),
                    )
                ) as upstream:
                    async for chunk in upstream:
                        yield chunk
                return

            moderation = self._get_moderation()
            assert moderation is not None

            async with contextlib.aclosing(
                _moderated_dragent_stream(
                    cast(
                        AsyncGenerator[DRAgentEventResponse, None],
                        call_next(*ctx.modified_args, **ctx.modified_kwargs),
                    ),
                    moderation=moderation,
                    stream_state=stream_state,
                )
            ) as moderated_stream:
                async for response in moderated_stream:
                    yield response
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
