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

import logging
import uuid
from typing import TYPE_CHECKING

from ag_ui.core import Event
from ag_ui.core import RunAgentInput
from ag_ui.core import TextMessageChunkEvent
from langchain_core.messages import ToolMessage

if TYPE_CHECKING:
    from datarobot_genai.core.agents.base import BaseAgent
from nat.data_models.api_server import ChatRequest
from nat.data_models.api_server import ChatRequestOrMessage
from nat.data_models.api_server import Message

from datarobot_genai.core.agents import default_usage_metrics
from datarobot_genai.core.chat.completions import convert_chat_completion_params_to_run_agent_input
from datarobot_genai.dragent.request import DRAgentRunAgentInput
from datarobot_genai.dragent.response import DRAgentEventResponse

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


def convert_str_to_dragent_event_response(
    response: str,
) -> DRAgentEventResponse:
    return DRAgentEventResponse(
        usage_metrics=default_usage_metrics(),
        pipeline_interactions=None,
        events=[TextMessageChunkEvent(message_id=str(uuid.uuid4()), delta=response)],
    )


async def invoke_agent_to_dragent_event_response(
    agent: "BaseAgent",
    input_message: RunAgentInput,
) -> DRAgentEventResponse:
    """Invoke an agent and collect its event stream into a DRAgentEventResponse."""
    events: list[Event] = []
    metrics = default_usage_metrics()
    async for event, _, iteration_metrics in agent.invoke(input_message):
        metrics = iteration_metrics
        events.append(event)

    return DRAgentEventResponse(
        events=events,
        usage_metrics=metrics,
    )


# --- Various converters ---


def convert_tool_message_to_str(message: ToolMessage) -> str:
    return message.content
