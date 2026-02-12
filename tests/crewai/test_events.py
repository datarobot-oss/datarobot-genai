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

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ragas.messages import AIMessage
from ragas.messages import HumanMessage
from ragas.messages import ToolMessage

from datarobot_genai.crewai.events import AgentExecutionCompletedEvent
from datarobot_genai.crewai.events import AgentExecutionStartedEvent
from datarobot_genai.crewai.events import CrewAIRagasEventListener
from datarobot_genai.crewai.events import CrewKickoffStartedEvent
from datarobot_genai.crewai.events import ToolUsageFinishedEvent
from datarobot_genai.crewai.events import ToolUsageStartedEvent


class _FakeBus:
    def __init__(self) -> None:
        self.handlers: dict[Any, list[Callable[..., None]]] = {}

    def on(self, event_type: Any) -> Callable[[Callable[..., None]], Callable[..., None]]:
        def decorator(fn: Callable[..., None]) -> Callable[..., None]:
            self.handlers.setdefault(event_type, []).append(fn)
            return fn

        return decorator


@dataclass
class _Kickoff:
    inputs: dict[str, Any]


@dataclass
class _AgentStarted:
    task_prompt: str


@dataclass
class _AgentCompleted:
    output: str


@dataclass
class _ToolStarted:
    tool_name: str
    tool_args: Any


@dataclass
class _ToolFinished:
    output: str


def test_crewai_event_listener_accumulates_messages() -> None:
    bus = _FakeBus()
    listener = CrewAIRagasEventListener()
    listener.setup_listeners(bus)  # type: ignore[arg-type]

    # Fire kickoff (Human)
    for fn in bus.handlers.get(CrewKickoffStartedEvent, []):
        fn(None, _Kickoff(inputs={"x": 1}))

    # Fire agent started (AI with prompt)
    for fn in bus.handlers.get(AgentExecutionStartedEvent, []):
        fn(None, _AgentStarted(task_prompt="prompt"))

    # Fire tool started (adds ToolCall to last AI)
    for fn in bus.handlers.get(ToolUsageStartedEvent, []):
        fn(None, _ToolStarted(tool_name="t", tool_args={"a": 1}))

    # Fire tool finished (ToolMessage)
    for fn in bus.handlers.get(ToolUsageFinishedEvent, []):
        fn(None, _ToolFinished(output="out"))

    # Fire agent completed (AI with output)
    for fn in bus.handlers.get(AgentExecutionCompletedEvent, []):
        fn(None, _AgentCompleted(output="done"))

    # Validate message sequence
    assert isinstance(listener.messages[0], HumanMessage)
    assert isinstance(listener.messages[1], AIMessage)
    assert isinstance(listener.messages[2], ToolMessage)
    assert isinstance(listener.messages[3], AIMessage)
    # Tool call was attached to second message (AI)
    assert isinstance(listener.messages[1], AIMessage)
    assert listener.messages[1].tool_calls and listener.messages[1].tool_calls[0].name == "t"
