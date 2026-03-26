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

import contextvars
import logging

from ag_ui.core import Event
from ag_ui.core import RunAgentInput
from ag_ui.core import TextMessageChunkEvent
from ag_ui.core import TextMessageContentEvent
from ag_ui.core import TextMessageStartEvent
from langchain_core.messages import ToolMessage
from nat.data_models.api_server import ChatRequest
from nat.data_models.api_server import ChatRequestOrMessage
from nat.data_models.api_server import Message

from datarobot_genai.core.agents import default_usage_metrics
from datarobot_genai.core.chat.completions import convert_chat_completion_params_to_run_agent_input

from .request import DRAgentRunAgentInput
from .response import DRAgentEventResponse

logger = logging.getLogger(__name__)

# --- Input converters ---

## --- AG-UI -> NAT Chat Completions ---


def convert_dragent_run_agent_input_to_chat_request(input: DRAgentRunAgentInput) -> ChatRequest:
    messages = []
    for message in input.messages:
        messages.append(Message(role=message.role, content=message.content))

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


## --- dragent Chat Completions -> AG-UI ---


def convert_chat_request_to_run_agent_input(request: ChatRequest) -> RunAgentInput:
    chat_request_dict = request.model_dump()
    return convert_chat_completion_params_to_run_agent_input(chat_request_dict)


# --- Output converters ---

## --- NAT chat completions -> dragent AG-UI ---


# When NAT native agent is used it returns a string with the response in streaming mode.
# NAT 1.5 native does not emit LLM intermediate events for the final answer, so the
# raw string stream is the only source of the assistant text message.  We emit a proper
# TextMessage lifecycle (Start on first chunk, Content on every chunk) so the frontend
# can persist the message across page refreshes.  RunFinishedEvent from the workflow
# acts as the implicit message end.
#
# ContextVar ensures per-request isolation in async/concurrent environments.
_text_message_started: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_text_message_started", default=False
)


def convert_str_to_dragent_event_response(
    response: str,
) -> DRAgentEventResponse:
    events: list[Event] = []
    if not _text_message_started.get():
        events.append(TextMessageStartEvent(message_id="default_nat_response"))
        _text_message_started.set(True)
    events.append(TextMessageContentEvent(message_id="default_nat_response", delta=response))
    return DRAgentEventResponse(
        usage_metrics=default_usage_metrics(),
        pipeline_interactions=None,
        events=events,
    )


# --- Various converters ---


def convert_tool_message_to_str(message: ToolMessage) -> str:
    return message.content


def aggregate_dragent_event_responses(
    responses: list[DRAgentEventResponse],
) -> DRAgentEventResponse:
    all_events = [event for response in responses for event in response.events]
    return DRAgentEventResponse(events=all_events)


def convert_dragent_event_response_to_str(response: DRAgentEventResponse) -> str:
    return "".join(
        event.delta
        for event in response.events
        if isinstance(event, (TextMessageContentEvent, TextMessageChunkEvent))
    )
