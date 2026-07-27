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

import json
from typing import Any

from datarobot_dome.guards.agent_goal_accuracy import AIMessage as PipelineAIMessage
from datarobot_dome.guards.agent_goal_accuracy import HumanMessage as PipelineHumanMessage
from datarobot_dome.guards.agent_goal_accuracy import ToolCall as PipelineToolCall
from datarobot_dome.guards.agent_goal_accuracy import ToolMessage as PipelineToolMessage
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.messages import ToolMessage


def convert_to_moderations_messages(
    messages: list[Any],
) -> list[PipelineHumanMessage | PipelineAIMessage | PipelineToolMessage]:
    """Convert LangChain messages into pipeline-interaction messages.

    Ports the logic of the old ``ragas.integrations.langgraph.convert_to_ragas_messages``
    (metadata omitted): SystemMessages are skipped, and an AIMessage's tool calls are
    read out of ``additional_kwargs``. Content is assumed to be a plain string (the
    caller flattens list-form content beforehand).
    """
    converted: list[PipelineHumanMessage | PipelineAIMessage | PipelineToolMessage] = []
    for message in messages:
        if isinstance(message, SystemMessage):
            continue
        if isinstance(message, AIMessage):
            # Mirror ragas: only inspect tool calls when additional_kwargs is present;
            # a truthy-but-tool-call-free additional_kwargs yields an empty list, not None.
            if message.additional_kwargs:
                tool_calls: list[PipelineToolCall] | None = [
                    PipelineToolCall(
                        name=tc["function"]["name"],
                        args=json.loads(tc["function"]["arguments"]),
                    )
                    for tc in message.additional_kwargs.get("tool_calls", [])
                ]
            else:
                tool_calls = None
            converted.append(PipelineAIMessage(content=message.content, tool_calls=tool_calls))
        elif isinstance(message, HumanMessage):
            converted.append(PipelineHumanMessage(content=message.content))
        elif isinstance(message, ToolMessage):
            converted.append(PipelineToolMessage(content=message.content))
        else:
            raise ValueError(f"Unsupported message type: {type(message).__name__}")
    return converted
