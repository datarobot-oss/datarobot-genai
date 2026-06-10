# Copyright 2025 DataRobot, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import logging
import uuid
from typing import Any
from typing import Literal
from typing import cast

from ag_ui.core import CustomEvent
from ag_ui.core import RunAgentInput
from ag_ui.core import TextMessageChunkEvent
from ag_ui.core import TextMessageContentEvent
from ag_ui.core import ToolCallArgsEvent
from ag_ui.core import ToolCallChunkEvent
from ag_ui.core import ToolCallStartEvent
from langchain_core.messages import ToolMessage
from nat.data_models.api_server import ChatRequest
from nat.data_models.api_server import ChatRequestOrMessage
from nat.data_models.api_server import ChatResponseChunk
from nat.data_models.api_server import ChatResponseChunkChoice
from nat.data_models.api_server import ChoiceDelta
from nat.data_models.api_server import ChoiceDeltaToolCall
from nat.data_models.api_server import ChoiceDeltaToolCallFunction
from nat.data_models.api_server import Message
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice as OpenAIChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta as OpenAIChoiceDelta
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall as OpenAIChoiceDeltaToolCall
from openai.types.chat.chat_completion_chunk import (
    ChoiceDeltaToolCallFunction as OpenAIChoiceDeltaToolCallFunction,
)
from openai.types.completion_usage import CompletionUsage

from datarobot_genai.core.agents import default_usage_metrics
from datarobot_genai.core.chat.completions import convert_chat_completion_params_to_run_agent_input

from .request import DRAgentRunAgentInput
from .response import DRAgentEventResponse

logger = logging.getLogger(__name__)

# --- Input converters ---

## --- AG-UI -> NAT Chat Completions ---


def convert_dragent_run_agent_input_to_chat_request(input: DRAgentRunAgentInput) -> ChatRequest:
    # NAT's Message model only accepts user/assistant/system roles.
    # Convert tool results to assistant messages to preserve context.
    # Skip unsupported roles (e.g. tool, reasoning).
    messages = []
    for message in input.messages:
        if message.role in ("user", "assistant", "system"):
            if message.role == "assistant" and not message.content:
                continue
            messages.append(Message(role=message.role, content=message.content))
        elif message.role == "tool" and message.content:
            messages.append(Message(role="assistant", content=message.content))

    tools = []
    for tool in input.tools:
        tools.append(tool.model_dump())

    d = {}
    try:
        d = dict(input.forwarded_props)
    except Exception as e:
        logger.warning(f"Error converting forwarded props in RunAgentInput to ChatRequest: {e}")

    d["messages"] = messages
    # Always stream
    d["stream"] = True
    d["tools"] = tools

    return ChatRequest.model_validate(d)


def convert_dragent_run_agent_input_to_chat_request_or_message(
    input: DRAgentRunAgentInput,
) -> ChatRequestOrMessage:
    chat_request = convert_dragent_run_agent_input_to_chat_request(input)
    return ChatRequestOrMessage.model_validate(chat_request.model_dump())


def convert_run_agent_input_to_chat_request_or_message(
    input: RunAgentInput,
) -> ChatRequestOrMessage:
    """Bridge plain RunAgentInput to NAT chat completions for inner workflow agents.

    DRAgent registers converters for ``DRAgentRunAgentInput`` only. The DRUM
    ``NatAgent.invoke`` path produces plain ``RunAgentInput`` at the
    ``streaming_memory_agent`` passthrough boundary, where inner
    ``per_user_tool_calling_agent`` expects ``ChatRequestOrMessage``.
    """
    return convert_dragent_run_agent_input_to_chat_request_or_message(
        DRAgentRunAgentInput.model_validate(input.model_dump(by_alias=True))
    )


def _dragent_streaming_delta_from_events(
    events: list[Any],
) -> tuple[str | None, list[ChoiceDeltaToolCall]]:
    """Extract streaming delta fields from AG-UI events (shared NAT / OpenAI chunk builders)."""
    content_parts: list[str] = []
    tool_calls: list[ChoiceDeltaToolCall] = []
    for event in events:
        if isinstance(event, (TextMessageContentEvent, TextMessageChunkEvent)):
            content_parts.append(event.delta)
            continue
        # AG-UI tool-call events -> OpenAI streaming tool_calls deltas.
        # OpenAI's streaming format expects the *first* chunk of a tool call
        # to carry ``id``, ``type`` and ``function.name``; subsequent
        # argument chunks carry only ``function.arguments`` (with ``id``
        # null). NAT's tool-call stream emits a single tool call at a time,
        # so ``index=0`` is sufficient to correlate the shards downstream.
        if isinstance(event, ToolCallStartEvent):
            tool_calls.append(
                ChoiceDeltaToolCall(
                    index=0,
                    id=event.tool_call_id,
                    type="function",
                    function=ChoiceDeltaToolCallFunction(name=event.tool_call_name),
                )
            )
        elif isinstance(event, ToolCallArgsEvent):
            tool_calls.append(
                ChoiceDeltaToolCall(
                    index=0,
                    function=ChoiceDeltaToolCallFunction(arguments=event.delta),
                )
            )
        elif isinstance(event, ToolCallChunkEvent):
            function_kwargs: dict[str, str] = {}
            if event.tool_call_name:
                function_kwargs["name"] = event.tool_call_name
            if event.delta:
                function_kwargs["arguments"] = event.delta
            tool_calls.append(
                ChoiceDeltaToolCall(
                    index=0,
                    id=event.tool_call_id,
                    type="function" if event.tool_call_id else None,
                    function=ChoiceDeltaToolCallFunction(**function_kwargs)
                    if function_kwargs
                    else None,
                )
            )
        # ToolCallEndEvent / ToolCallResultEvent are intentionally skipped:
        # tool-call completion is signalled by ``finish_reason`` at the
        # outer choice level, and tool results are sent as separate
        # ``role="tool"`` messages rather than streaming deltas.
    if content_parts:
        content: str | None = "".join(content_parts)
    elif tool_calls:
        content = None
    else:
        content = ""
    return content, tool_calls


def _resolve_datarobot_moderations_for_chunk(
    chunk: ChatResponseChunk | None,
    from_response: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if from_response is not None:
        return from_response
    return getattr(chunk, "datarobot_moderations", None)


def _nat_chat_response_chunk_with_datarobot_moderations(
    chunk: ChatResponseChunk,
    datarobot_moderations: dict[str, Any] | None,
) -> ChatResponseChunk:
    if datarobot_moderations is None:
        return chunk
    if getattr(chunk, "datarobot_moderations", None) == datarobot_moderations:
        return chunk
    return chunk.model_copy(update={"datarobot_moderations": datarobot_moderations})


def _openai_chat_completion_chunk_with_datarobot_moderations(
    chunk: ChatCompletionChunk,
    datarobot_moderations: dict[str, Any] | None,
) -> ChatCompletionChunk:
    if datarobot_moderations is None:
        return chunk
    if getattr(chunk, "datarobot_moderations", None) == datarobot_moderations:
        return chunk
    return chunk.model_copy(update={"datarobot_moderations": datarobot_moderations})


def convert_dragent_event_response_to_chat_response_chunk(
    response: DRAgentEventResponse,
) -> ChatResponseChunk:
    if response.original_chunk is not None:
        chunk = response.original_chunk
    else:
        content, tool_calls = _dragent_streaming_delta_from_events(response.events)
        chunk = ChatResponseChunk(
            id=uuid.uuid4().hex,
            choices=[
                ChatResponseChunkChoice(
                    index=0,
                    delta=ChoiceDelta(content=content, tool_calls=tool_calls or None),
                )
            ],
            created=datetime.datetime.now(datetime.UTC),
        )
    moderations = _resolve_datarobot_moderations_for_chunk(chunk, response.datarobot_moderations)
    return _nat_chat_response_chunk_with_datarobot_moderations(chunk, moderations)


_FINISH_REASON = Literal["stop", "length", "tool_calls", "content_filter", "function_call"]


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


def convert_nat_chat_response_chunk_to_openai_chat_completion_chunk(
    chunk: ChatResponseChunk,
) -> ChatCompletionChunk:
    """Map NAT streaming chunk to OpenAI ``chat.completion.chunk``."""
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
        created = created.replace(tzinfo=datetime.UTC)
    created_ts = int(created.timestamp())
    usage_openai: CompletionUsage | None = None
    if chunk.usage is not None:
        u = chunk.usage
        usage_openai = CompletionUsage(
            prompt_tokens=u.prompt_tokens,
            completion_tokens=u.completion_tokens,
            total_tokens=u.total_tokens,
        )
    openai_chunk = ChatCompletionChunk(
        id=chunk.id,
        choices=[choice],
        created=created_ts,
        model=chunk.model,
        object="chat.completion.chunk",
        usage=usage_openai,
    )
    return _openai_chat_completion_chunk_with_datarobot_moderations(
        openai_chunk, getattr(chunk, "datarobot_moderations", None)
    )


def convert_dragent_event_response_to_openai_chat_completion_chunk(
    response: DRAgentEventResponse,
) -> ChatCompletionChunk:
    """Convert one DRAgent stream chunk to an OpenAI chunk (dome / moderation streaming)."""
    if response.original_chunk is not None:
        chunk = convert_nat_chat_response_chunk_to_openai_chat_completion_chunk(
            response.original_chunk
        )
    else:
        content, tool_calls = _dragent_streaming_delta_from_events(response.events)
        created_ts = int(datetime.datetime.now(datetime.UTC).timestamp())
        chunk = ChatCompletionChunk(
            id=uuid.uuid4().hex,
            choices=[
                OpenAIChunkChoice(
                    index=0,
                    delta=OpenAIChoiceDelta(content=content, tool_calls=tool_calls or None),
                    finish_reason=None,
                )
            ],
            created=created_ts,
            model=response.model or "unknown-model",
            object="chat.completion.chunk",
            usage=None,
        )
    moderations = _resolve_datarobot_moderations_for_chunk(
        response.original_chunk, response.datarobot_moderations
    )
    return _openai_chat_completion_chunk_with_datarobot_moderations(chunk, moderations)


## --- dragent Chat Completions -> AG-UI ---


def convert_chat_request_to_run_agent_input(request: ChatRequest) -> RunAgentInput:
    chat_request_dict = request.model_dump()
    return convert_chat_completion_params_to_run_agent_input(chat_request_dict)


# --- Output converters ---

## --- NAT chat completions -> dragent AG-UI ---


# When NAT native agent is used it returns a string with the response in streaming mode
# we don't need it: it is already returned from LLM events in StepAdaptor.
# So we return it as a custom event just to keep the interface consistent.
def convert_str_to_dragent_event_response(
    response: str,
) -> DRAgentEventResponse:
    return DRAgentEventResponse(
        usage_metrics=default_usage_metrics(),
        pipeline_interactions=None,
        events=[CustomEvent(name="DEFAULT_NAT_RESPONSE", value={"delta": response})],
    )


# --- Various converters ---


def convert_tool_message_to_str(message: ToolMessage) -> str:
    return message.content


def aggregate_dragent_event_responses(
    responses: list[DRAgentEventResponse],
) -> DRAgentEventResponse:
    all_events = [event for response in responses for event in response.events]
    datarobot_moderations: dict[str, Any] | None = None
    merged_usage_metrics: dict[str, int] | None = None
    model: str | None = None
    original_chunk: ChatResponseChunk | None = None
    for response in responses:
        if response.datarobot_moderations is not None:
            datarobot_moderations = response.datarobot_moderations
        if response.model is not None:
            model = response.model
        if response.original_chunk is not None:
            original_chunk = response.original_chunk
        if response.usage_metrics is not None:
            if merged_usage_metrics is None:
                merged_usage_metrics = dict(response.usage_metrics)
            else:
                for key, value in response.usage_metrics.items():
                    merged_usage_metrics[key] = merged_usage_metrics.get(key, 0) + value
    return DRAgentEventResponse(
        events=all_events,
        datarobot_moderations=datarobot_moderations,
        model=model,
        usage_metrics=merged_usage_metrics,
        original_chunk=original_chunk,
    )


def convert_dragent_event_response_to_str(response: DRAgentEventResponse) -> str:
    return "".join(
        event.delta
        for event in response.events
        if isinstance(event, (TextMessageContentEvent, TextMessageChunkEvent))
    )
