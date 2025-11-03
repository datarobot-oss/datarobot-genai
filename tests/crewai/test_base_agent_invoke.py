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

from typing import Any
from typing import cast

import pytest
from ragas import MultiTurnSample
from ragas.messages import HumanMessage

import datarobot_genai.crewai.base as base_mod
from datarobot_genai.core.agents.base import UsageMetrics
from datarobot_genai.crewai.base import CrewAIAgent


class DummyCrew:
    def __init__(self, output: Any) -> None:
        self._output = output

    def kickoff(self, *, inputs: dict[str, Any]) -> Any:  # noqa: ARG002
        return self._output


class DummyOutput:
    def __init__(self, raw: str, token_usage: Any | None = None) -> None:
        self.raw = raw
        self.token_usage = token_usage


class DummyTokens:
    def __init__(self, c: int, p: int, t: int) -> None:
        self.completion_tokens = c
        self.prompt_tokens = p
        self.total_tokens = t


class MyAgent(CrewAIAgent):
    @property
    def agents(self) -> list[Any]:
        return []

    @property
    def tasks(self) -> list[Any]:
        return []

    def __init__(self, crew_output: Any, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._crew_output = crew_output

    def build_crewai_workflow(self) -> Any:
        return DummyCrew(self._crew_output)


@pytest.mark.asyncio
async def test_invoke_appends_final_message_and_collects_usage(monkeypatch: Any) -> None:
    class _Holder:
        events: list[Any]

    holder = _Holder()
    holder.events = []

    def fake_create_pipeline_interactions(messages: list[Any]) -> str:
        holder.events = messages
        return "pipeline"

    monkeypatch.setattr(
        base_mod,
        "create_pipeline_interactions_from_messages",
        fake_create_pipeline_interactions,
        raising=True,
    )

    class DummyCtx:
        def __enter__(self: Any) -> list[Any]:
            return []

        def __exit__(self: Any, exc_type: Any, exc: Any, tb: Any) -> None:
            pass

    monkeypatch.setattr(base_mod, "mcp_tools_context", lambda **_: DummyCtx(), raising=True)

    out = DummyOutput("agent result", token_usage=DummyTokens(1, 2, 3))

    agent = MyAgent(out, api_base="https://x/", api_key="k", verbose=True)
    # Seed prior events to trigger append of final AIMessage
    agent.event_listener.messages = []  # type: ignore[attr-defined]
    agent.event_listener.messages.append(HumanMessage(content="hi"))  # type: ignore[attr-defined]

    # Act
    resp = await agent.invoke({"model": "m", "messages": [{"role": "user", "content": "{}"}]})
    resp_text, pipeline_interactions, usage = cast(
        tuple[str, MultiTurnSample | None, UsageMetrics], resp
    )

    # Assert
    assert resp_text == "agent result"
    assert pipeline_interactions == "pipeline"
    # events include the final AI message
    assert holder.events and getattr(holder.events[-1], "content", None) == "agent result"
    assert usage == {"completion_tokens": 1, "prompt_tokens": 2, "total_tokens": 3}


@pytest.mark.asyncio
async def test_invoke_when_no_events(monkeypatch: Any) -> None:
    # Arrange

    # No event bus path is default; ensure no registration is attempted.

    class _Holder:
        events: list[Any]

    holder = _Holder()
    holder.events = []

    def fake_create_pipeline_interactions(messages: list[Any]) -> None:
        holder.events = messages

    monkeypatch.setattr(
        base_mod,
        "create_pipeline_interactions_from_messages",
        fake_create_pipeline_interactions,
        raising=True,
    )

    class DummyCtx:
        def __enter__(self: Any) -> list[Any]:
            return []

        def __exit__(self: Any, exc_type: Any, exc: Any, tb: Any) -> None:
            pass

    monkeypatch.setattr(base_mod, "mcp_tools_context", lambda **_: DummyCtx(), raising=True)

    out = DummyOutput("ok", token_usage=None)
    agent = MyAgent(out, api_base="https://x/", api_key=None, verbose=False)
    # Force empty events list to hit the else: events = None path
    agent.event_listener.messages = []  # type: ignore[attr-defined]

    # Act
    resp = await agent.invoke({"model": "m", "messages": [{"role": "user", "content": "{}"}]})
    resp_text, pipeline_interactions, usage = cast(
        tuple[str, MultiTurnSample | None, UsageMetrics], resp
    )

    # Assert
    assert resp_text == "ok"
    assert pipeline_interactions is None
    assert holder.events == []
    # default usage metrics when token_usage is None
    assert usage == {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}
