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

from collections.abc import AsyncGenerator
from typing import Any
from typing import cast

import pytest
from ragas import MultiTurnSample
from ragas.messages import HumanMessage

import datarobot_genai.crewai.base as base_mod
from datarobot_genai.core.agents.base import UsageMetrics
from datarobot_genai.crewai.base import CrewAIAgent


@pytest.fixture
def mock_mcp_context(monkeypatch: Any) -> None:
    """Mock MCP tools context to return empty tools list."""
    class _Ctx:
        def __enter__(self) -> list[Any]:
            return []

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            pass

    def _ctx_factory(**_: Any) -> Any:
        return _Ctx()

    monkeypatch.setattr(base_mod, "mcp_tools_context", _ctx_factory, raising=True)


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


async def test_crewai_streaming_minimal(mock_mcp_context: Any) -> None:
    agent = _Agent()
    gen = await agent.invoke(
        {
            "model": "m",
            "stream": True,
            "messages": [{"role": "user", "content": "{}"}],
        }
    )

    chunks = [c async for c in cast(AsyncGenerator[tuple[str, Any, UsageMetrics], None], gen)]
    assert len(chunks) == 1
    text, interactions, usage = chunks[0]
    assert text == "Agent response"
    assert interactions is None
    assert usage == {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}


async def test_crewai_streaming_with_pipeline_interactions(mock_mcp_context: Any) -> None:
    class _TokenUsage:
        completion_tokens = 150
        prompt_tokens = 50
        total_tokens = 200

    class _CrewWithUsage:
        def kickoff(self, *, inputs: dict[str, Any]) -> Any:  # noqa: ARG002
            class Output:
                raw = "Agent response with usage"
                token_usage = _TokenUsage()

            return Output()

    # Create an agent with an event listener that has messages
    class _EventListener:
        def __init__(self) -> None:
            self.messages = [
                HumanMessage(content="Test question"),
            ]

    class _AgentWithListener(_Agent):
        def __init__(self) -> None:
            super().__init__()
            self.event_listener = _EventListener()

        def build_crewai_workflow(self) -> Any:
            return _CrewWithUsage()

    agent = _AgentWithListener()
    gen = await agent.invoke(
        {
            "model": "m",
            "stream": True,
            "messages": [{"role": "user", "content": "{}"}],
        }
    )

    chunks = [c async for c in cast(AsyncGenerator[tuple[str, Any, UsageMetrics], None], gen)]
    assert len(chunks) == 1
    text, interactions, usage = chunks[0]
    assert text == "Agent response with usage"
    assert interactions is not None
    assert isinstance(interactions, MultiTurnSample)
    assert len(interactions.user_input) == 1
    assert isinstance(interactions.user_input[0], HumanMessage)
    assert interactions.user_input[0].content == "Test question"
    assert usage == {"completion_tokens": 150, "prompt_tokens": 50, "total_tokens": 200}
