# Copyright 2025 DataRobot, Inc.
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
from __future__ import annotations

import importlib
import json
import logging
from typing import Any

from ragas.messages import AIMessage
from ragas.messages import HumanMessage
from ragas.messages import ToolCall
from ragas.messages import ToolMessage

# Resolve crewai symbols at runtime to avoid mypy issues with untyped packages
try:
    _events_mod = importlib.import_module("crewai.events.event_types")
    AgentExecutionCompletedEvent = getattr(_events_mod, "AgentExecutionCompletedEvent")
    AgentExecutionStartedEvent = getattr(_events_mod, "AgentExecutionStartedEvent")
    CrewKickoffStartedEvent = getattr(_events_mod, "CrewKickoffStartedEvent")
    ToolUsageFinishedEvent = getattr(_events_mod, "ToolUsageFinishedEvent")
    ToolUsageStartedEvent = getattr(_events_mod, "ToolUsageStartedEvent")

    _bus_mod = importlib.import_module("crewai.events.event_bus")
    CrewAIEventsBus = getattr(_bus_mod, "CrewAIEventsBus")

    _base_mod = importlib.import_module("crewai.events.base_event_listener")
    _RuntimeBaseEventListener = getattr(_base_mod, "BaseEventListener")
except Exception:
    try:  # pragma: no cover - compatibility for older crewai
        _events_mod = importlib.import_module("crewai.utilities.events")
        AgentExecutionCompletedEvent = getattr(_events_mod, "AgentExecutionCompletedEvent")
        AgentExecutionStartedEvent = getattr(_events_mod, "AgentExecutionStartedEvent")
        CrewKickoffStartedEvent = getattr(_events_mod, "CrewKickoffStartedEvent")
        ToolUsageFinishedEvent = getattr(_events_mod, "ToolUsageFinishedEvent")
        ToolUsageStartedEvent = getattr(_events_mod, "ToolUsageStartedEvent")

        _bus_mod = importlib.import_module("crewai.utilities.events")
        CrewAIEventsBus = getattr(_bus_mod, "CrewAIEventsBus")
        _base_mod = importlib.import_module("crewai.utilities.events.base_event_listener")
        _RuntimeBaseEventListener = getattr(_base_mod, "BaseEventListener")
    except Exception:
        raise ImportError(
            "CrewAI is required for datarobot_genai.crewai.* modules. "
            "Install with the CrewAI extra:\n"
            "  install 'datarobot-genai[crewai]'"
        )


class CrewAIEventListener:
    """Collects CrewAI events into Ragas messages for pipeline interactions."""

    def __init__(self) -> None:
        self.messages: list[HumanMessage | AIMessage | ToolMessage] = []

    def setup_listeners(self, crewai_event_bus: Any) -> None:
        @crewai_event_bus.on(CrewKickoffStartedEvent)
        def on_crew_execution_started(_: Any, event: Any) -> None:
            self.messages.append(
                HumanMessage(content=f"Working on input '{json.dumps(event.inputs)}'")
            )

        @crewai_event_bus.on(AgentExecutionStartedEvent)
        def on_agent_execution_started(_: Any, event: Any) -> None:
            self.messages.append(AIMessage(content=event.task_prompt, tool_calls=[]))

        @crewai_event_bus.on(AgentExecutionCompletedEvent)
        def on_agent_execution_completed(_: Any, event: Any) -> None:
            self.messages.append(AIMessage(content=event.output, tool_calls=[]))

        @crewai_event_bus.on(ToolUsageStartedEvent)
        def on_tool_usage_started(_: Any, event: Any) -> None:
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

        @crewai_event_bus.on(ToolUsageFinishedEvent)
        def on_tool_usage_finished(_: Any, event: Any) -> None:
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
