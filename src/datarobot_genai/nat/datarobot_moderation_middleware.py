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
"""NAT middleware: DataRobot LLM guardrails (prescore/postscore) for DRAgent workflows."""

from __future__ import annotations

import asyncio
import logging
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
from datarobot_dome.chat_helper import get_response_message_and_finish_reason
from datarobot_dome.chat_helper import run_postscore_guards
from datarobot_dome.constants import CHAT_COMPLETION_OBJECT
from datarobot_dome.constants import GuardStage
from datarobot_dome.constants import TargetType
from datarobot_dome.otel_helpers import report_otel_evaluation_set_metric
from datarobot_dome.streaming import ModerationIterator
from datarobot_dome.streaming import StreamingContextBuilder
from datarobot_moderation_interface.drum_integration import _handle_result_df_error_cases
from datarobot_moderation_interface.drum_integration import _set_moderation_attribute_to_completion
from datarobot_moderation_interface.drum_integration import build_non_streaming_chat_completion
from datarobot_moderation_interface.drum_integration import build_predictions_df_from_completion
from datarobot_moderation_interface.drum_integration import filter_association_id
from datarobot_moderation_interface.drum_integration import filter_extra_body
from datarobot_moderation_interface.drum_integration import format_result_df
from datarobot_moderation_interface.drum_integration import get_chat_prompt
from datarobot_moderation_interface.drum_integration import moderation_pipeline_factory
from datarobot_moderation_interface.drum_integration import run_prescore_guards
from nat.builder.builder import Builder
from nat.cli.register_workflow import register_middleware
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
from datarobot_genai.dragent.frontends.request import DRAgentRunAgentInput
from datarobot_genai.dragent.frontends.response import DRAgentEventResponse

_logger = logging.getLogger(__name__)

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
        tool_calls=openai_tool_calls,  # type: ignore[arg-type]
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
    """Map NAT streaming chunk to OpenAI chat completion chunk (delta, not full message)."""
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
    """Rebuild AG-UI text events from one streaming OpenAI chunk (single delta)."""
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
                    tool_calls=event_tool_calls,  # type: ignore[arg-type]
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


def chat_completion_to_dragent_event_response(
    completion: ChatCompletion | ChatCompletionChunk,
    *,
    source_ag_ui_events: list[Any] | None = None,
) -> DRAgentEventResponse:
    """Convert OpenAI completion or streaming chunk back to DRAgentEventResponse."""
    if isinstance(completion, ChatCompletionChunk):
        chunk = _openai_chat_completion_chunk_to_nat_chat_response_chunk(completion)
        events = _streaming_text_events_from_openai_chunk(
            completion, source_ag_ui_events=source_ag_ui_events
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


def skip_event_type(event: Event) -> bool:
    return event.type not in {
        EventType.TEXT_MESSAGE_CONTENT,
        EventType.TEXT_MESSAGE_CHUNK,
    }


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
        if config.model_dir is not None:
            self._pipeline: Any = moderation_pipeline_factory(
                TargetType.AGENTIC_WORKFLOW, config.model_dir
            )
        else:
            self._pipeline = moderation_pipeline_factory(TargetType.AGENTIC_WORKFLOW)
        self.data = None
        self.prescore_df = None
        self.association_id = None
        self.assembled_response: list[str] = []
        self._prescore_blocked_completion: ChatCompletion | None = None

    @property
    def enabled(self) -> bool:
        return self._pipeline is not None

    async def pre_invoke(self, context: InvocationContext) -> InvocationContext | None:
        """Pre-invocation hook called before the function is invoked.

        Args:
            context: Invocation context containing function metadata and args

        Returns
        -------
            InvocationContext if modified, or None to pass through unchanged.
            When prescore blocks the prompt, returns None and sets
            ``_prescore_blocked_completion`` for the invoke/stream handlers to emit.
        """
        self._prescore_blocked_completion = None
        run_agent_input: DRAgentRunAgentInput | None = (
            context.original_args[0] if context.original_args else None
        )
        if run_agent_input is None:
            return None

        pipeline = self._pipeline._pipeline
        completion_create_params = run_agent_input_to_completion_dict(run_agent_input)

        # if association ID was included in extra_body, extract field name and value
        completion_create_params, eb_assoc_id_value = filter_association_id(
            completion_create_params
        )

        # extract any fields mentioned in "datarobot_metrics" to send as custom metrics later
        completion_create_params, chat_extra_body_params = filter_extra_body(
            completion_create_params
        )

        # define all pipeline-based and guard-based custom metrics (but not those from extra_body)
        # note: this is usually partially done at pipeline init; see delayed_custom_metric_creation
        pipeline.get_new_metrics_payload()

        # the chat request is not a dataframe, but we'll build a DF internally for moderation.
        prompt_column_name = pipeline.get_input_column(GuardStage.PROMPT)
        prompt = get_chat_prompt(completion_create_params)

        data = pd.DataFrame({prompt_column_name: [prompt]})
        # Association IDs: column must exist on deployment (pipeline association id name non-empty).
        # (Here: pipeline.get_association_id_column_name() "standard name" is not empty.)
        # There are 3 likely cases for association ID, and 1 corner case:
        # 1. ID value not provided (drum or extra_body) => no association ID column
        # 2. ID value provided by DRUM => new DF column with standard name and provided value
        # 3. ID defined in extra_body => new DF column with standard name and extra_body value
        # 4. ID in extra_body with empty value => no association ID column
        # Moderation no longer auto-generates association ID for chat; DRUM may.
        association_id_column_name = pipeline.get_association_id_column_name()
        association_id = eb_assoc_id_value  # or association_id  # TBD: drum passes this
        self.association_id = association_id
        if association_id_column_name:
            if association_id:
                data[association_id_column_name] = [association_id]

        # DRUM initializes the pipeline (which reads the deployment's list of custom metrics)
        # at start time.
        # If there are no extra_body fields (meaning no user-defined custom metrics to report),
        # then the list does not need to be reread.
        if chat_extra_body_params:
            pipeline.lookup_custom_metric_ids()

        # report any metrics from extra_body. They are not tied to a prompt or response phase.
        _logger.debug("Report extra_body params as custom metrics")
        pipeline.report_custom_metrics_from_extra_body(association_id, chat_extra_body_params)

        # ==================================================================
        # Step 1: Prescore Guards processing
        #
        prescore_df, filtered_df, prescore_latency = run_prescore_guards(pipeline, data)
        self.data = data
        self.input_df = data
        self.prescore_df = prescore_df
        self.latency_so_far = prescore_latency

        _logger.debug("After passing input through pre score guards")
        _logger.debug(filtered_df)
        _logger.debug(f"Pre Score Guard Latency: {prescore_latency} sec")

        blocked_prompt_column_name = f"blocked_{prompt_column_name}"
        if prescore_df.loc[0, blocked_prompt_column_name]:
            pipeline.report_custom_metrics(prescore_df)
            blocked_message_prompt_column_name = f"blocked_message_{prompt_column_name}"
            # If all prompts in the input are blocked, means history as well as the prompt
            # are not worthy to be sent to LLM
            chat_completion = build_non_streaming_chat_completion(
                prescore_df.loc[0, blocked_message_prompt_column_name],
                "content_filter",
            )
            result_df = _handle_result_df_error_cases(
                prompt_column_name, prescore_df, prescore_latency
            )
            completion = _set_moderation_attribute_to_completion(
                pipeline, chat_completion, result_df, association_id=association_id
            )
            report_otel_evaluation_set_metric(pipeline, result_df)
            self._prescore_blocked_completion = completion
            return None

        replaced_prompt_column_name = f"replaced_{prompt_column_name}"
        if (
            replaced_prompt_column_name in prescore_df.columns
            and prescore_df.loc[0, replaced_prompt_column_name]
        ):
            # PII kind of guard could have modified the prompt, so use that modified prompt
            # for the user chat function
            _modified_chat = context.modified_args[0].messages
            _modified_chat[-1]["content"] = filtered_df.loc[0, prompt_column_name]
            return context
        else:
            return None

    async def post_invoke(self, context: InvocationContext) -> InvocationContext | None:
        """Post-invocation hook called after the function returns.

        Args:
            context: Invocation context containing function metadata, args, and output

        Returns
        -------
            InvocationContext if modified, or None to pass through unchanged.
            Default implementation does nothing.
        """
        response: DRAgentEventResponse = context.output
        if skip_event_type(response.events[0]):
            return None

        chat_completion = dragent_event_response_to_chat_completion(response)

        pipeline = self._pipeline._pipeline

        # ==================================================================
        # Step 3: Postscore Guards processing
        #
        # Prompt column name is already part of data and gets included for
        # faithfulness calculation processing
        response_column_name = pipeline.get_input_column(GuardStage.RESPONSE)
        predictions_df, extra_attributes = build_predictions_df_from_completion(
            self.data, pipeline, chat_completion
        )
        response = predictions_df.loc[0, response_column_name]

        if response is not None:
            none_predictions_df = None
            postscore_df, _ = run_postscore_guards(pipeline, predictions_df)
        else:
            postscore_df, _ = pd.DataFrame(), 0
            none_predictions_df = predictions_df

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
        #     score_latency + prescore_latency + postscore_latency
        # ) / result_df.shape[0]

        response_message, finish_reason = get_response_message_and_finish_reason(
            pipeline, postscore_df
        )
        report_otel_evaluation_set_metric(pipeline, result_df)

        final_completion = build_non_streaming_chat_completion(
            response_message, finish_reason, extra_attributes
        )
        final_completion = _set_moderation_attribute_to_completion(
            pipeline, final_completion, result_df, association_id=self.association_id
        )
        context.output = chat_completion_to_dragent_event_response(final_completion)
        return context

    async def function_middleware_invoke(
        self,
        *args: Any,
        call_next: CallNext,
        context: FunctionMiddlewareContext,
        **kwargs: Any,
    ) -> Any:
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

        blocked = self._prescore_blocked_completion
        if blocked is not None:
            self._prescore_blocked_completion = None
            return chat_completion_to_dragent_event_response(blocked)

        ctx.output = await call_next(*ctx.modified_args, **ctx.modified_kwargs)

        result = await self.post_invoke(ctx)
        if result is not None:
            ctx = result

        return ctx.output

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

        Yields
        ------
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

        blocked = self._prescore_blocked_completion
        if blocked is not None:
            self._prescore_blocked_completion = None
            yield chat_completion_to_dragent_event_response(blocked)
            return

        streaming_context = (
            StreamingContextBuilder()
            .set_pipeline(self._pipeline._pipeline)
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

            while True:
                peek = await read_upstream()
                while peek is not None and (not peek.events or skip_event_type(peek.events[0])):
                    yield peek
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
                    worker_sentinel_received = True
                    if mod_errors:
                        raise mod_errors[0]
                    break

                ctx.output = chat_completion_to_dragent_event_response(
                    moderated, source_ag_ui_events=current_response.events
                )
                yield ctx.output
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
