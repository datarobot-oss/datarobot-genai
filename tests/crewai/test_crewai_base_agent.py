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
from ragas.messages import AIMessage
from ragas.messages import HumanMessage

import datarobot_genai.crewai.base as base_mod
from datarobot_genai.crewai.base import CrewAIAgent


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


@pytest.mark.asyncio
async def test_crewai_agent_non_streaming(
    monkeypatch: pytest.MonkeyPatch, mock_mcp_context: None
) -> None:
    # Patch Crew constructor in base module
    monkeypatch.setattr(base_mod, "Crew", _FakeCrew)

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
    monkeypatch.setattr(base_mod, "Crew", _FakeCrew)

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
    # Patch Crew constructor in base module
    monkeypatch.setattr(base_mod, "Crew", _FakeCrew)

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

    monkeypatch.setattr(base_mod, "mcp_tools_context", _FakeCtx)

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
