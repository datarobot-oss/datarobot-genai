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

"""
Event listener utilities for CrewAI.

This module centralizes CrewAI event capture into Ragas-compatible messages so
agent templates can focus on business logic.
"""

from __future__ import annotations

import json
import logging
from typing import Any

try:
    # CrewAI >= 0.51
    from crewai.events import event_types as _crewai_events
    from crewai.events.base_event_listener import BaseEventListener
except Exception:  # pragma: no cover - fallback for older CrewAI
    # CrewAI < 0.51 fallback
    from crewai.utilities import events as _crewai_events
    from crewai.utilities.events.base_event_listener import BaseEventListener

from ragas.messages import AIMessage
from ragas.messages import HumanMessage
from ragas.messages import ToolCall
from ragas.messages import ToolMessage

AgentExecutionCompletedEvent = _crewai_events.AgentExecutionCompletedEvent
AgentExecutionStartedEvent = _crewai_events.AgentExecutionStartedEvent
CrewKickoffStartedEvent = _crewai_events.CrewKickoffStartedEvent
ToolUsageFinishedEvent = _crewai_events.ToolUsageFinishedEvent
ToolUsageStartedEvent = _crewai_events.ToolUsageStartedEvent


class CrewAIEventListener(BaseEventListener):
    """
    Captures CrewAI events and converts them to a sequence of Ragas messages.

    Collected messages are stored in ``self.messages`` for downstream
    conversion into pipeline interactions.
    """

    def __init__(self) -> None:
        super().__init__()
        self.messages: list[HumanMessage | AIMessage | ToolMessage] = []

    def setup_listeners(self, crewai_event_bus: Any) -> None:
        crewai_event_bus.on(CrewKickoffStartedEvent)(self.handle_crew_kickoff_started)
        crewai_event_bus.on(AgentExecutionStartedEvent)(self.handle_agent_execution_started)
        crewai_event_bus.on(AgentExecutionCompletedEvent)(self.handle_agent_execution_completed)
        crewai_event_bus.on(ToolUsageStartedEvent)(self.handle_tool_usage_started)
        crewai_event_bus.on(ToolUsageFinishedEvent)(self.handle_tool_usage_finished)

    def handle_crew_kickoff_started(self, _: Any, event: Any) -> None:
        self.messages.append(HumanMessage(content=f"Working on input '{json.dumps(event.inputs)}'"))

    def handle_agent_execution_started(self, _: Any, event: Any) -> None:
        self.messages.append(AIMessage(content=event.task_prompt, tool_calls=[]))

    def handle_agent_execution_completed(self, _: Any, event: Any) -> None:
        self.messages.append(AIMessage(content=event.output, tool_calls=[]))

    def handle_tool_usage_started(self, _: Any, event: Any) -> None:
        # It's a tool call - add tool call to last AIMessage
        if len(self.messages) == 0:
            logging.warning("Direct tool usage without agent invocation")
            return
        last_message = self.messages[-1]
        if not isinstance(last_message, AIMessage):
            logging.warning(
                "Tool call must be preceded by an AIMessage somewhere in the conversation."
            )
            return
        if isinstance(event.tool_args, (str, bytes, bytearray)):
            parsed_args: Any = json.loads(event.tool_args)
        else:
            parsed_args = event.tool_args
        tool_call = ToolCall(name=event.tool_name, args=parsed_args)
        if last_message.tool_calls is None:
            last_message.tool_calls = []
        last_message.tool_calls.append(tool_call)

    def handle_tool_usage_finished(self, _: Any, event: Any) -> None:
        if len(self.messages) == 0:
            logging.warning("Direct tool usage without agent invocation")
            return
        last_message = self.messages[-1]
        if not isinstance(last_message, AIMessage):
            logging.warning(
                "Tool call must be preceded by an AIMessage somewhere in the conversation."
            )
            return
        if not last_message.tool_calls:
            logging.warning("No previous tool calls found")
            return
        self.messages.append(ToolMessage(content=event.output))
