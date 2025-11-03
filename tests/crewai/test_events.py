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

from collections.abc import Callable
from typing import Any

import pytest
from ragas.messages import AIMessage
from ragas.messages import HumanMessage
from ragas.messages import ToolMessage

from datarobot_genai.crewai.events import CrewAIEventListener


class Obj:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


def test_handle_crew_kickoff_started_appends_human_message() -> None:
    listener = CrewAIEventListener()
    event = Obj(inputs={"topic": "t"})
    listener.handle_crew_kickoff_started(None, event)  # type: ignore[arg-type]
    assert listener.messages and listener.messages[0].type == "user"
    assert '"topic": "t"' in listener.messages[0].content


def test_handle_agent_execution_started_and_completed_append_ai_messages() -> None:
    listener = CrewAIEventListener()
    start_event = Obj(task_prompt="prompt")
    complete_event = Obj(output="final")

    listener.handle_agent_execution_started(None, start_event)  # type: ignore[arg-type]
    listener.handle_agent_execution_completed(None, complete_event)  # type: ignore[arg-type]

    assert len(listener.messages) == 2
    assert listener.messages[0].type == "ai" and listener.messages[0].content == "prompt"
    assert listener.messages[1].type == "ai" and listener.messages[1].content == "final"


def test_handle_tool_usage_started_adds_tool_call_from_json_string() -> None:
    listener = CrewAIEventListener()
    # add an AI message so tool call can attach
    listener.handle_agent_execution_started(None, Obj(task_prompt="p"))  # type: ignore[arg-type]

    evt = Obj(tool_name="t", tool_args='{"x":1}')
    listener.handle_tool_usage_started(None, evt)  # type: ignore[arg-type]

    last = listener.messages[-1]
    assert last.type == "ai"
    tool_calls = getattr(last, "tool_calls", [])
    assert tool_calls and tool_calls[0].name == "t"
    assert tool_calls[0].args == {"x": 1}


def test_handle_tool_usage_started_adds_tool_call_from_dict() -> None:
    listener = CrewAIEventListener()
    listener.handle_agent_execution_started(None, Obj(task_prompt="p"))  # type: ignore[arg-type]

    evt = Obj(tool_name="tool", tool_args={"a": 2})
    listener.handle_tool_usage_started(None, evt)  # type: ignore[arg-type]

    last = listener.messages[-1]
    tool_calls = getattr(last, "tool_calls", [])
    assert tool_calls and tool_calls[-1].name == "tool"
    assert tool_calls[-1].args == {"a": 2}


def test_handle_tool_usage_finished_appends_tool_message() -> None:
    listener = CrewAIEventListener()
    # prepare AI message with a tool call to satisfy preconditions
    listener.handle_agent_execution_started(None, Obj(task_prompt="p"))  # type: ignore[arg-type]
    listener.handle_tool_usage_started(None, Obj(tool_name="t", tool_args={}))  # type: ignore[arg-type]

    evt = Obj(output="ok")
    listener.handle_tool_usage_finished(None, evt)  # type: ignore[arg-type]

    assert listener.messages[-1].type == "tool"
    assert listener.messages[-1].content == "ok"


def test_handle_tool_usage_started_without_prior_ai_message_is_noop(
    caplog: pytest.LogCaptureFixture,
) -> None:
    listener = CrewAIEventListener()
    listener.handle_tool_usage_started(None, Obj(tool_name="t", tool_args={}))  # type: ignore[arg-type]
    # no messages added
    assert listener.messages == []
    # optional: a warning was logged
    assert any("Direct tool usage" in rec.getMessage() for rec in caplog.records)


class _FakeBus:
    def __init__(self) -> None:
        self._handlers: dict[type[Any], list[Callable[..., None]]] = {}

    def on(self, event_type: type[Any]) -> Callable[[Callable[..., None]], Callable[..., None]]:
        def decorator(fn: Callable[..., None]) -> Callable[..., None]:
            self._handlers.setdefault(event_type, []).append(fn)
            return fn

        return decorator

    def emit(self, event_type: type[Any], event: Any) -> None:
        for handler in self._handlers.get(event_type, []):
            handler(None, event)


class _Kickoff:
    def __init__(self, inputs: dict[str, Any]) -> None:
        self.inputs = inputs


class _AgentStart:
    def __init__(self, task_prompt: str) -> None:
        self.task_prompt = task_prompt


class _AgentDone:
    def __init__(self, output: str) -> None:
        self.output = output


class _ToolStart:
    def __init__(self, tool_name: str, tool_args: Any) -> None:
        self.tool_name = tool_name
        self.tool_args = tool_args


class _ToolDone:
    def __init__(self, output: str) -> None:
        self.output = output


def test_crewai_event_listener_collects_messages() -> None:
    # GIVEN a new listener and a fake bus
    listener = CrewAIEventListener()
    bus = _FakeBus()
    listener.setup_listeners(bus)  # type: ignore[arg-type]

    # WHEN emitting a sequence of events resembling a simple run
    bus.emit(type(_Kickoff({})), _Kickoff({"topic": "ai"}))
    bus.emit(type(_AgentStart("prompt")), _AgentStart("Plan something"))
    bus.emit(type(_ToolStart("search", {"q": "ai"})), _ToolStart("search", {"q": "ai"}))
    bus.emit(type(_ToolDone("result")), _ToolDone("result"))
    bus.emit(type(_AgentDone("final")), _AgentDone("final"))

    # THEN messages contain Human, AI (with tool call), ToolMessage, and final AI
    msgs = listener.messages
    assert any(isinstance(m, HumanMessage) for m in msgs)
    ai_messages: list[AIMessage] = [m for m in msgs if isinstance(m, AIMessage)]
    assert len(ai_messages) >= 2
    assert ai_messages[0].tool_calls and ai_messages[0].tool_calls[0].name == "search"
    assert any(isinstance(m, ToolMessage) for m in msgs)
