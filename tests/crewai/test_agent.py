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

# ruff: noqa: I001
from typing import Any, cast

import pytest
from ragas import MultiTurnSample
from ragas.messages import AIMessage
from ragas.messages import HumanMessage

import datarobot_genai.crewai.agent as agent_mod
from datarobot_genai.core.agents.base import UsageMetrics
from datarobot_genai.crewai.agent import CrewAIAgent
from datarobot_genai.crewai.agent import create_pipeline_interactions_from_messages


# --- Test helpers ---


class _FakeCrewOutput:
    def __init__(self, raw: Any, completion_tokens: int = 3, prompt_tokens: int = 7):
        self.raw = raw
        self.token_usage = type(
            "TokenUsage",
            (),
            {
                "completion_tokens": completion_tokens,
                "prompt_tokens": prompt_tokens,
                "total_tokens": completion_tokens + prompt_tokens,
            },
        )()


class _FakeCrew:
    def __init__(self, *args: Any, **kwargs: Any):
        self.args = args
        self.kwargs = kwargs

    def kickoff(self, *, inputs: dict[str, Any]) -> _FakeCrewOutput:  # type: ignore[name-defined]
        return _FakeCrewOutput(raw="final-output")


class _Listener:
    def __init__(self, messages: list[Any] | None = None):
        self.messages = messages or []


class _TestAgent(CrewAIAgent):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        # Provide valid Ragas messages so interactions can be constructed
        self.event_listener = _Listener(
            messages=[HumanMessage(content="hi"), AIMessage(content="there")]
        )

    @property
    def agents(self) -> list[Any]:
        return [object()]

    @property
    def tasks(self) -> list[Any]:
        return [object()]

    def make_kickoff_inputs(self, user_prompt_content: str) -> dict[str, Any]:
        return {"topic": user_prompt_content}


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

    def crew(self) -> Any:
        return DummyCrew(self._crew_output)

    def make_kickoff_inputs(self, user_prompt_content: str) -> dict[str, Any]:
        return {"topic": str(user_prompt_content)}


# --- Tests for create_pipeline_interactions_from_messages ---


def test_create_pipeline_interactions_from_messages_returns_sample(monkeypatch: Any) -> None:
    assert create_pipeline_interactions_from_messages(None) is None

    msgs = [HumanMessage(content="hi")]
    sample = create_pipeline_interactions_from_messages(msgs)
    assert sample is not None
    assert sample.user_input == msgs


# --- Tests for CrewAIAgent invoke ---


@pytest.mark.asyncio
async def test_invoke_collects_usage_without_events(monkeypatch: Any) -> None:
    class DummyCtx:
        def __enter__(self: Any) -> list[Any]:
            return []

        def __exit__(self: Any, exc_type: Any, exc: Any, tb: Any) -> None:
            pass

    monkeypatch.setattr(agent_mod, "mcp_tools_context", lambda **_: DummyCtx(), raising=True)

    out = DummyOutput("agent result", token_usage=DummyTokens(1, 2, 3))

    agent = MyAgent(out, api_base="https://x/", api_key="k", verbose=True)

    # Act
    resp = await agent.invoke({"model": "m", "messages": [{"role": "user", "content": "{}"}]})
    resp_text, pipeline_interactions, usage = cast(
        tuple[str, MultiTurnSample | None, UsageMetrics], resp
    )

    # Assert
    assert resp_text == "agent result"
    assert pipeline_interactions is None
    assert usage == {"completion_tokens": 1, "prompt_tokens": 2, "total_tokens": 3}


@pytest.mark.asyncio
async def test_invoke_when_no_events(monkeypatch: Any) -> None:
    # Arrange

    # No event bus path is default; ensure no registration is attempted.

    class DummyCtx:
        def __enter__(self: Any) -> list[Any]:
            return []

        def __exit__(self: Any, exc_type: Any, exc: Any, tb: Any) -> None:
            pass

    monkeypatch.setattr(agent_mod, "mcp_tools_context", lambda **_: DummyCtx(), raising=True)

    out = DummyOutput("ok", token_usage=None)
    agent = MyAgent(out, api_base="https://x/", api_key=None, verbose=False)
    # No event listener used; pipeline interactions should be None

    # Act
    resp = await agent.invoke({"model": "m", "messages": [{"role": "user", "content": "{}"}]})
    resp_text, pipeline_interactions, usage = cast(
        tuple[str, MultiTurnSample | None, UsageMetrics], resp
    )

    # Assert
    assert resp_text == "ok"
    assert pipeline_interactions is None
    # default usage metrics when token_usage is None
    assert usage == {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}


@pytest.mark.asyncio
async def test_crewai_agent_non_streaming(
    monkeypatch: pytest.MonkeyPatch, mock_mcp_context: None
) -> None:
    # Patch Crew constructor in agent module
    monkeypatch.setattr(agent_mod, "Crew", _FakeCrew)

    agent = _TestAgent(api_key="k", api_base="https://x")

    completion_create_params: Any = {
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }

    result = await agent.invoke(cast(Any, completion_create_params))
    assert isinstance(result, tuple)
    response_text, interactions, usage = result

    assert response_text == "final-output"
    # interactions should be constructed from listener messages
    assert interactions is not None
    assert usage == {"completion_tokens": 3, "prompt_tokens": 7, "total_tokens": 10}


@pytest.mark.asyncio
async def test_crewai_agent_streaming(
    monkeypatch: pytest.MonkeyPatch, mock_mcp_context: None
) -> None:
    monkeypatch.setattr(agent_mod, "Crew", _FakeCrew)

    agent = _TestAgent(api_key="k", api_base="https://y")

    completion_create_params: Any = {
        "messages": [{"role": "user", "content": "world"}],
        "stream": True,
    }

    gen_or_tuple = await agent.invoke(cast(Any, completion_create_params))
    assert not isinstance(gen_or_tuple, tuple)
    gen = gen_or_tuple

    async def collect_all() -> list[tuple[str, Any | None, Any]]:
        out: list[tuple[str, Any | None, Any]] = []
        async for chunk in cast(Any, gen):
            out.append(chunk)
        return out

    chunks = await collect_all()
    # Streaming currently yields a single final chunk with empty text
    assert len(chunks) == 1
    text, interactions, usage = chunks[0]
    assert text == "final-output"
    assert interactions is not None
    assert usage == {"completion_tokens": 3, "prompt_tokens": 7, "total_tokens": 10}


@pytest.mark.asyncio
async def test_crewai_agent_passes_forwarded_headers_to_mcp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that CrewAI agent passes forwarded headers to MCP context."""
    # Patch Crew constructor in agent module
    monkeypatch.setattr(agent_mod, "Crew", _FakeCrew)

    # Track MCP context calls
    mcp_calls = []

    class _FakeCtx:
        def __init__(self, **kwargs: Any) -> None:
            mcp_calls.append(kwargs)
            self._tools = ["tool1"]

        def __enter__(self) -> list[str]:
            return self._tools

        def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
            return None

    monkeypatch.setattr(agent_mod, "mcp_tools_context", _FakeCtx)

    forwarded_headers = {
        "x-datarobot-api-key": "scoped-token-123",
    }
    agent = _TestAgent(api_key="k", api_base="https://x", forwarded_headers=forwarded_headers)

    completion_create_params: Any = {
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }

    await agent.invoke(cast(Any, completion_create_params))

    # Verify MCP context was called with forwarded headers
    assert len(mcp_calls) == 1
    assert mcp_calls[0]["forwarded_headers"] == forwarded_headers
