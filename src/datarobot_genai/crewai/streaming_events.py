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

from typing import Any

from crewai.events.event_bus import CrewAIEventsBus
from crewai.events.event_types import AgentReasoningCompletedEvent
from crewai.events.event_types import AgentReasoningFailedEvent
from crewai.events.event_types import AgentReasoningStartedEvent
from crewai.events.event_types import TaskCompletedEvent
from crewai.events.event_types import TaskFailedEvent
from crewai.events.event_types import TaskStartedEvent


class CrewAIStreamingEventListener:
    """Collects CrewAI events to distinguish the types of events currently happening."""

    def __init__(self) -> None:
        self.reasoning_event = False
        self.step_event = False

    def setup_listeners(self, crewai_event_bus: CrewAIEventsBus) -> None:

        @crewai_event_bus.on(AgentReasoningStartedEvent)
        def on_agent_reasoning_started(_: Any, event: Any) -> None:
            self.reasoning_event = True

        @crewai_event_bus.on(AgentReasoningCompletedEvent)
        def on_agent_reasoning_completed(_: Any, event: Any) -> None:
            self.reasoning_event = False

        @crewai_event_bus.on(AgentReasoningFailedEvent)
        def on_agent_reasoning_failed(_: Any, event: Any) -> None:
            self.reasoning_event = False

        @crewai_event_bus.on(TaskCompletedEvent)
        def on_agent_task_completed(_: Any, event: Any) -> None:
            self.step_event = False

        @crewai_event_bus.on(TaskFailedEvent)
        def on_agent_task_failed(_: Any, event: Any) -> None:
            self.step_event = False

        @crewai_event_bus.on(TaskStartedEvent)
        def on_agent_task_started(_: Any, event: Any) -> None:
            self.step_event = True
