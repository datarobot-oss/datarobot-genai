# Copyright 2026 DataRobot, Inc. and its affiliates.
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

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

from datarobot_genai.crewai.logging_events import AgentExecutionStartedEvent
from datarobot_genai.crewai.logging_events import CrewAILoggingEventListener
from datarobot_genai.crewai.logging_events import TaskFailedEvent
from datarobot_genai.crewai.logging_events import TaskStartedEvent
from datarobot_genai.crewai.logging_events import ToolUsageErrorEvent
from datarobot_genai.crewai.logging_events import ToolUsageFinishedEvent
from datarobot_genai.crewai.logging_events import ToolUsageStartedEvent

LOGGER_NAME = "datarobot_genai.crewai.logging_events"


class _FakeBus:
    def __init__(self) -> None:
        self.handlers: dict[Any, list[Callable[..., None]]] = {}

    def on(self, event_type: Any) -> Callable[[Callable[..., None]], Callable[..., None]]:
        def decorator(fn: Callable[..., None]) -> Callable[..., None]:
            self.handlers.setdefault(event_type, []).append(fn)
            return fn

        return decorator

    def fire(self, event_type: Any, event: Any) -> None:
        for fn in self.handlers.get(event_type, []):
            fn(None, event)


@dataclass
class _ToolStarted:
    agent_role: str
    tool_name: str
    tool_args: Any
    run_attempts: int


@dataclass
class _ToolFinished:
    agent_role: str
    tool_name: str
    output: str


@dataclass
class _ToolError:
    agent_role: str
    tool_name: str
    error: str


@dataclass
class _TaskStarted:
    agent_role: str
    task_name: str


@dataclass
class _TaskFailed:
    agent_role: str
    error: str


@dataclass
class _AgentStarted:
    agent_role: str


def _bus_with_listener() -> _FakeBus:
    bus = _FakeBus()
    CrewAILoggingEventListener().setup_listeners(bus)  # type: ignore[arg-type]
    return bus


def test_tool_call_logs_name_args_result_and_role_at_info(caplog: pytest.LogCaptureFixture) -> None:
    # GIVEN a logging listener
    bus = _bus_with_listener()

    # WHEN a tool is invoked and finishes
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        bus.fire(
            ToolUsageStartedEvent,
            _ToolStarted(
                agent_role="Planner",
                tool_name="generate_objectid",
                tool_args={"type": "deployment"},
                run_attempts=2,
            ),
        )
        bus.fire(
            ToolUsageFinishedEvent,
            _ToolFinished(agent_role="Planner", tool_name="generate_objectid", output="69cbb737"),
        )

    text = caplog.text
    # THEN the tool call is logged at INFO with role, tool name, args, attempt, and result
    assert [r.levelname for r in caplog.records] == ["INFO", "INFO"]
    assert "Planner" in text
    assert "generate_objectid" in text
    assert "deployment" in text
    assert "attempt=2" in text  # run attempt surfaces the loop
    assert "69cbb737" in text


def test_failures_log_at_warning(caplog: pytest.LogCaptureFixture) -> None:
    # GIVEN a logging listener
    bus = _bus_with_listener()

    # WHEN a task fails and a tool errors
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        bus.fire(TaskFailedEvent, _TaskFailed(agent_role="Writer", error="boom"))
        bus.fire(
            ToolUsageErrorEvent,
            _ToolError(agent_role="Writer", tool_name="calculator", error="bad input"),
        )

    # THEN both are logged at WARNING
    assert {r.levelname for r in caplog.records} == {"WARNING"}
    assert "task failed" in caplog.text
    assert "calculator" in caplog.text


def test_lifecycle_events_log_at_info(caplog: pytest.LogCaptureFixture) -> None:
    # GIVEN a logging listener
    bus = _bus_with_listener()

    # WHEN an agent starts and a task starts
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        bus.fire(AgentExecutionStartedEvent, _AgentStarted(agent_role="Planner"))
        bus.fire(TaskStartedEvent, _TaskStarted(agent_role="Planner", task_name="plan"))

    assert [r.levelname for r in caplog.records] == ["INFO", "INFO"]
    assert "plan" in caplog.text


def test_handler_never_raises_on_bad_event(caplog: pytest.LogCaptureFixture) -> None:
    # GIVEN an event whose payload access raises (must not break the crew run)
    class _BadToolStarted:
        agent_role = "Planner"
        tool_name = "t"
        run_attempts = 1

        @property
        def tool_args(self) -> Any:
            raise RuntimeError("boom")

    bus = _bus_with_listener()

    # WHEN the handler runs, THEN it swallows the error instead of propagating
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        bus.fire(ToolUsageStartedEvent, _BadToolStarted())

    # THEN no INFO "tool call" line was emitted for the broken event
    assert not any(r.levelname == "INFO" for r in caplog.records)


@dataclass
class _Agent:
    role: str


@dataclass
class _Task:
    agent: Any


@dataclass
class _AgentStartedNoRole:
    agent_role: str
    agent: Any


@dataclass
class _TaskStartedNoRole:
    agent_role: str
    task_name: str
    task: Any


def test_role_falls_back_to_event_agent(caplog: pytest.LogCaptureFixture) -> None:
    # GIVEN CrewAI lifecycle events leave agent_role empty but attach the agent object
    bus = _bus_with_listener()

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        bus.fire(
            AgentExecutionStartedEvent,
            _AgentStartedNoRole(agent_role="", agent=_Agent(role="Planner")),
        )

    # THEN the role is resolved from event.agent.role, not left blank
    assert "[Planner] agent started" in caplog.text


def test_role_falls_back_to_task_agent(caplog: pytest.LogCaptureFixture) -> None:
    # GIVEN a task-started event with empty agent_role but a task carrying its agent
    bus = _bus_with_listener()

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        bus.fire(
            TaskStartedEvent,
            _TaskStartedNoRole(
                agent_role="", task_name="plan", task=_Task(agent=_Agent(role="Writer"))
            ),
        )

    # THEN the role is resolved from event.task.agent.role
    assert "[Writer]" in caplog.text


@dataclass
class _NamedTask:
    name: str
    agent: Any


def test_task_name_falls_back_to_event_task(caplog: pytest.LogCaptureFixture) -> None:
    # GIVEN CrewAI's TaskStartedEvent does not populate task_name; the name lives
    # on the attached task object instead
    bus = _bus_with_listener()

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        bus.fire(
            TaskStartedEvent,
            _TaskStartedNoRole(
                agent_role="",
                task_name="",
                task=_NamedTask(name="plan_the_post", agent=_Agent(role="Planner")),
            ),
        )

    # THEN the task name is resolved from event.task.name instead of logged blank
    assert "[Planner] task started: plan_the_post" in caplog.text
