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


def convert_dragent_event_response_to_chat_response_chunk(
    response: DRAgentEventResponse,
) -> ChatResponseChunk:
    if response.original_chunk is not None:
        return response.original_chunk

    content_parts: list[str] = []
    tool_calls: list[ChoiceDeltaToolCall] = []

    for event in response.events:
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

    # Preserve the existing empty-string semantics for the "no events"
    # case while matching OpenAI's streaming convention of ``null`` content
    # whenever a chunk carries only tool_calls.
    if content_parts:
        content: str | None = "".join(content_parts)
    elif tool_calls:
        content = None
    else:
        content = ""

    return ChatResponseChunk(
        id=uuid.uuid4().hex,
        choices=[
            ChatResponseChunkChoice(
                index=0,
                delta=ChoiceDelta(content=content, tool_calls=tool_calls or None),
            )
        ],
        created=datetime.datetime.now(datetime.UTC),
    )


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
