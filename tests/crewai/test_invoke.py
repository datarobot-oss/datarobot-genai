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

"""Tests for emit_task_progress feature."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any
from typing import cast
from unittest.mock import MagicMock
from unittest.mock import patch

import datarobot_genai.crewai.base as base_mod
from datarobot_genai.core.agents.base import UsageMetrics
from datarobot_genai.crewai.base import CrewAIAgent


class _Crew:
    def kickoff(self, *, inputs: dict[str, Any]) -> Any:  # noqa: ARG002
        class Output:
            raw = "Agent response"
            token_usage = None

        return Output()


class _Agent(CrewAIAgent):
    @property
    def agents(self) -> list[Any]:
        return []

    @property
    def tasks(self) -> list[Any]:
        return []

    def build_crewai_workflow(self) -> Any:
        return _Crew()

    def make_kickoff_inputs(self, user_prompt_content: str) -> dict[str, Any]:
        return {}


async def test_crewai_emit_task_progress(mock_mcp_context: Any) -> None:
    """Verify task progress events are yielded when emit_task_progress is True.

    Tests that:
    - scoped_handlers() is used for proper handler cleanup
    - verbose handlers are re-registered inside the scope
    - task progress handlers are registered and fire correctly
    """
    from crewai.events.types.task_events import TaskCompletedEvent  # noqa: PLC0415
    from crewai.events.types.task_events import TaskStartedEvent  # noqa: PLC0415

    registered_handlers: dict[type, Any] = {}
    scoped_handlers_entered = False
    scoped_handlers_exited = False

    class MockEventBus:
        @staticmethod
        def scoped_handlers() -> Any:
            class ScopedCtx:
                def __enter__(self) -> None:
                    nonlocal scoped_handlers_entered
                    scoped_handlers_entered = True

                def __exit__(self, *args: Any) -> None:
                    nonlocal scoped_handlers_exited
                    scoped_handlers_exited = True

            return ScopedCtx()

        @staticmethod
        def on(event_type: type) -> Any:
            def decorator(fn: Any) -> Any:
                registered_handlers[event_type] = fn
                return fn

            return decorator

    class _CrewEmittingEvents:
        def kickoff(self, *, inputs: dict[str, Any]) -> Any:
            task = MagicMock()
            task.name = "Search Files"
            task.agent.role = "File Agent"

            if TaskStartedEvent in registered_handlers:
                event = MagicMock()
                event.task = task
                registered_handlers[TaskStartedEvent](None, event)

            if TaskCompletedEvent in registered_handlers:
                event = MagicMock()
                event.task = task
                registered_handlers[TaskCompletedEvent](None, event)

            class Output:
                raw = "Final response"
                token_usage = None

            return Output()

    class _AgentWithTaskProgress(_Agent):
        def __init__(self) -> None:
            super().__init__(emit_task_progress=True)

        def build_crewai_workflow(self) -> Any:
            return _CrewEmittingEvents()

    mock_event_listener = MagicMock()
    with (
        patch.object(base_mod, "crewai_event_bus", MockEventBus),
        patch.object(base_mod, "event_listener", mock_event_listener),
    ):
        agent = _AgentWithTaskProgress()
        gen = await agent.invoke(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "{}"}],
            }
        )
        chunks = [c async for c in cast(AsyncGenerator[tuple[str, Any, UsageMetrics], None], gen)]

    # Verify scoped_handlers was used for proper cleanup
    assert scoped_handlers_entered, "scoped_handlers() should be entered"
    assert scoped_handlers_exited, "scoped_handlers() should be exited for cleanup"

    # Verify verbose handlers were re-registered
    mock_event_listener.setup_listeners.assert_called_once()

    # Verify handlers were registered for correct event types
    assert TaskStartedEvent in registered_handlers
    assert TaskCompletedEvent in registered_handlers

    # Should yield: task_started, task_completed, final response
    assert len(chunks) == 3

    progress1 = json.loads(chunks[0][0])["task_progress"]
    assert progress1["type"] == "task_started"
    assert progress1["task_name"] == "Search Files"

    progress2 = json.loads(chunks[1][0])["task_progress"]
    assert progress2["type"] == "task_completed"

    assert chunks[2][0] == "Final response"
