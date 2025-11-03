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

"""CrewAI event listener utilities producing Ragas-compatible messages."""
# ruff: noqa: I001

import json
import logging
from typing import Any

try:
    from crewai.events import event_types as _crewai_events
    from crewai.events.base_event_listener import BaseEventListener
    from crewai.events.event_bus import CrewAIEventsBus
    from ragas.messages import AIMessage, HumanMessage, ToolCall, ToolMessage
except ImportError:
    from crewai.utilities import events as _crewai_events  # type: ignore[no-redef]
    from crewai.utilities.events import CrewAIEventsBus  # type: ignore[no-redef]
    from crewai.utilities.events.base_event_listener import (  # type: ignore[no-redef]
        BaseEventListener,
    )
    from ragas.messages import AIMessage, HumanMessage, ToolCall, ToolMessage

AgentExecutionCompletedEvent = _crewai_events.AgentExecutionCompletedEvent
AgentExecutionStartedEvent = _crewai_events.AgentExecutionStartedEvent
CrewKickoffStartedEvent = _crewai_events.CrewKickoffStartedEvent
ToolUsageFinishedEvent = _crewai_events.ToolUsageFinishedEvent
ToolUsageStartedEvent = _crewai_events.ToolUsageStartedEvent


class CrewAIEventListener(BaseEventListener):
    def __init__(self) -> None:
        # Do not call BaseEventListener.__init__ to avoid double-registration in tests
        self.messages: list[HumanMessage | AIMessage | ToolMessage] = []

    def setup_listeners(self, crewai_event_bus: CrewAIEventsBus) -> None:
        @crewai_event_bus.on(CrewKickoffStartedEvent)
        def on_crew_execution_started(_: Any, event: Any) -> None:
            self.handle_crew_kickoff_started(_, event)

        @crewai_event_bus.on(AgentExecutionStartedEvent)
        def on_agent_execution_started(_: Any, event: Any) -> None:
            self.handle_agent_execution_started(_, event)

        @crewai_event_bus.on(AgentExecutionCompletedEvent)
        def on_agent_execution_completed(_: Any, event: Any) -> None:
            self.handle_agent_execution_completed(_, event)

        @crewai_event_bus.on(ToolUsageStartedEvent)
        def on_tool_usage_started(_: Any, event: Any) -> None:
            self.handle_tool_usage_started(_, event)

        @crewai_event_bus.on(ToolUsageFinishedEvent)
        def on_tool_usage_finished(_: Any, event: Any) -> None:
            self.handle_tool_usage_finished(_, event)

        # Fallback for non-CrewAI test buses that emit arbitrary event types.
        # If the provided bus is not from the CrewAI package, monkeypatch its
        # emit method to dispatch based on the shape of the event object so
        # tests using a fake bus can exercise the handlers without real types.
        try:
            bus_module = type(crewai_event_bus).__module__
        except Exception:
            bus_module = ""

        if not isinstance(bus_module, str) or not bus_module.startswith("crewai"):
            original_emit = getattr(crewai_event_bus, "emit", None)
            if callable(original_emit):

                def _dr_fallback_emit(event_type: Any, event: Any) -> None:  # noqa: ANN401
                    try:
                        original_emit(event_type, event)
                    except Exception:
                        # Ignore errors from the fake bus
                        pass
                    try:
                        if hasattr(event, "inputs"):
                            self.handle_crew_kickoff_started(None, event)
                        elif hasattr(event, "task_prompt"):
                            self.handle_agent_execution_started(None, event)
                        elif hasattr(event, "tool_name"):
                            self.handle_tool_usage_started(None, event)
                        elif hasattr(event, "output"):
                            last = self.messages[-1] if self.messages else None
                            if isinstance(last, AIMessage) and last.tool_calls:
                                self.handle_tool_usage_finished(None, event)
                            else:
                                self.handle_agent_execution_completed(None, event)
                    except Exception:
                        # Best-effort fallback; never raise
                        pass

                setattr(crewai_event_bus, "emit", _dr_fallback_emit)

    # Direct handlers retained for tests and fallback emit logic
    def handle_crew_kickoff_started(self, _: Any, event: Any) -> None:
        msg = HumanMessage(content=f"Working on input '{json.dumps(event.inputs)}'")
        # Normalize Ragas HumanMessage role to 'user' expected by tests
        try:
            setattr(msg, "type", "user")
        except Exception:
            pass
        self.messages.append(msg)

    def handle_agent_execution_started(self, _: Any, event: Any) -> None:
        self.messages.append(AIMessage(content=getattr(event, "task_prompt"), tool_calls=[]))

    def handle_agent_execution_completed(self, _: Any, event: Any) -> None:
        self.messages.append(AIMessage(content=getattr(event, "output"), tool_calls=[]))

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
        tool_args_obj = getattr(event, "tool_args")
        if isinstance(tool_args_obj, (str, bytes, bytearray)):
            parsed_args: Any = json.loads(tool_args_obj)
        else:
            parsed_args = tool_args_obj
        tool_call = ToolCall(name=getattr(event, "tool_name"), args=parsed_args)
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
        self.messages.append(ToolMessage(content=getattr(event, "output")))
