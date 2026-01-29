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

from collections.abc import AsyncGenerator
from typing import Any
from typing import cast
from unittest.mock import MagicMock

import pytest

from datarobot_genai.core.agents.base import UsageMetrics
from datarobot_genai.llama_index import agent as agent_mod
from datarobot_genai.llama_index.agent import DataRobotLiteLLM
from datarobot_genai.llama_index.agent import LlamaIndexAgent
from datarobot_genai.llama_index.agent import create_pipeline_interactions_from_events

# --- Test helpers ---


class Handler:
    def __init__(self, events: list[Any], state: Any) -> None:
        self._events = events
        self._state = state

    async def stream_events(self) -> AsyncGenerator[Any, None]:
        for e in self._events:
            yield e

    @property
    def ctx(self) -> Any:
        class Ctx:
            async def get(self, key: str) -> Any:  # noqa: A003 - method name from API
                assert key == "state"
                return state

        state = self._state
        return Ctx()


class Workflow:
    def __init__(self, events: list[Any], state: Any) -> None:
        self._events = events
        self._state = state

    def run(self, *, user_msg: str) -> Handler:  # noqa: ARG002
        return Handler(self._events, self._state)


class MyLlamaAgent(LlamaIndexAgent):
    def __init__(self, workflow: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._wf = workflow

    def build_workflow(self) -> Any:
        return self._wf

    def extract_response_text(self, result_state: Any, events: list[Any]) -> str:
        return f"{result_state}:{len(events)}"


# --- Tests for DataRobotLiteLLM ---


def test_datarobot_litellm_metadata_properties() -> None:
    llm = DataRobotLiteLLM(model="dr/model", max_tokens=256)
    meta = llm.metadata

    assert meta.context_window == 128000
    assert meta.num_output == 256
    assert meta.is_chat_model is True
    assert meta.is_function_calling_model is True
    assert meta.model_name == "dr/model"


# --- Tests for create_pipeline_interactions_from_events ---


def test_create_pipeline_interactions_from_events_none() -> None:
    assert create_pipeline_interactions_from_events(None) is None


# --- Tests for LlamaIndexAgent ---


@pytest.mark.asyncio
async def test_llama_index_agent_invoke(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}
    captured["events"] = None

    def fake_create_pipeline_interactions(events: list[Any]) -> Any:  # noqa: ANN401
        captured["events"] = events
        return {"ok": True}

    # Mock load_mcp_tools to return empty list
    async def fake_load_mcp_tools(*args: Any, **kwargs: Any) -> list[Any]:  # noqa: ARG001, ANN401
        return []

    monkeypatch.setattr(
        agent_mod,
        "create_pipeline_interactions_from_events",
        fake_create_pipeline_interactions,
        raising=True,
    )
    monkeypatch.setattr(agent_mod, "load_mcp_tools", fake_load_mcp_tools, raising=True)

    workflow = Workflow(events=[{"e": 1}, {"e": 2}], state="S")
    agent = MyLlamaAgent(workflow)

    resp = await agent.invoke({"model": "m", "messages": [{"role": "user", "content": "{}"}]})
    response_text, interactions, usage = cast(tuple[str, object, UsageMetrics], resp)

    assert response_text == "S:2"
    assert interactions == {"ok": True}
    assert captured["events"] == [{"e": 1}, {"e": 2}]
    assert usage == {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}
    assert agent.mcp_tools == []


@pytest.mark.asyncio
async def test_llama_index_agent_invoke_with_mcp_tools(monkeypatch: Any) -> None:
    """Test that MCP tools are loaded and available via mcp_tools property."""
    mock_tools = [MagicMock(), MagicMock()]

    async def fake_load_mcp_tools(*args: Any, **kwargs: Any) -> list[Any]:  # noqa: ARG001, ANN401
        return mock_tools

    def fake_create_pipeline_interactions(events: list[Any]) -> Any:  # noqa: ANN401
        return {"ok": True}

    monkeypatch.setattr(
        agent_mod,
        "create_pipeline_interactions_from_events",
        fake_create_pipeline_interactions,
        raising=True,
    )
    monkeypatch.setattr(agent_mod, "load_mcp_tools", fake_load_mcp_tools, raising=True)

    workflow = Workflow(events=[], state="S")
    agent = MyLlamaAgent(workflow)

    await agent.invoke({"model": "m", "messages": [{"role": "user", "content": "{}"}]})

    # Verify MCP tools were loaded and are accessible
    assert agent.mcp_tools == mock_tools
    assert len(agent.mcp_tools) == 2


@pytest.mark.asyncio
async def test_llama_index_agent_passes_forwarded_headers_to_mcp(monkeypatch: Any) -> None:
    """Test that LlamaIndex agent passes forwarded headers to MCP tools loading."""
    mock_tools = [MagicMock()]

    mcp_calls = []

    async def fake_load_mcp_tools(*args: Any, **kwargs: Any) -> list[Any]:  # noqa: ARG001, ANN401
        mcp_calls.append(kwargs)
        return mock_tools

    def fake_create_pipeline_interactions(events: list[Any]) -> Any:  # noqa: ANN401
        return {"ok": True}

    monkeypatch.setattr(
        agent_mod,
        "create_pipeline_interactions_from_events",
        fake_create_pipeline_interactions,
        raising=True,
    )
    monkeypatch.setattr(agent_mod, "load_mcp_tools", fake_load_mcp_tools, raising=True)

    forwarded_headers = {
        "x-datarobot-api-key": "scoped-token-123",
    }

    workflow = Workflow(events=[], state="S")
    agent = MyLlamaAgent(workflow, forwarded_headers=forwarded_headers)

    await agent.invoke({"model": "m", "messages": [{"role": "user", "content": "{}"}]})

    # Verify load_mcp_tools was called with forwarded headers
    assert len(mcp_calls) == 1
    assert mcp_calls[0]["forwarded_headers"] == forwarded_headers


@pytest.mark.asyncio
async def test_llama_index_agent_invoke_branches(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}
    captured["events"] = None

    def fake_create_pipeline_interactions(events: list[Any]) -> Any:  # noqa: ANN401
        captured["events"] = events
        return {"ok": True}

    monkeypatch.setattr(
        agent_mod,
        "create_pipeline_interactions_from_events",
        fake_create_pipeline_interactions,
        raising=True,
    )

    # Build events to exercise additional non-streaming branches
    agent_input_cls = type("AgentInput", (), {})
    agent_output_cls = type("AgentOutput", (), {})
    tool_call_cls = type("ToolCall", (), {})
    tool_call_result_cls = type("ToolCallResult", (), {})

    ai = agent_input_cls()
    ai.input = "q"  # type: ignore[attr-defined]
    ao = agent_output_cls()
    ao.response = type("Resp", (), {"content": "c"})()  # type: ignore[attr-defined]
    # Include dict-style tool call to hit dict name extraction path
    ao.tool_calls = [{"tool_name": "t1"}]  # type: ignore[attr-defined]
    tc = tool_call_cls()
    tc.tool_name = "t2"
    tc.tool_kwargs = {"a": 1}  # type: ignore[attr-defined]
    tcr = tool_call_result_cls()
    tcr.tool_name = "t2"
    tcr.tool_output = "out"  # type: ignore[attr-defined]

    events: list[Any] = [
        type("Banner", (), {"current_agent_name": "beta"})(),
        type("Delta", (), {"delta": "x"})(),
        type("Text", (), {"text": "y"})(),
        ai,
        ao,
        tc,
        tcr,
    ]

    workflow = Workflow(events=events, state="S2")
    agent = MyLlamaAgent(workflow)

    resp = await agent.invoke({"model": "m", "messages": [{"role": "user", "content": "{}"}]})
    response_text, interactions, usage = cast(tuple[str, object, UsageMetrics], resp)

    assert response_text == "S2:7"
    assert interactions == {"ok": True}
    assert captured["events"] == events
    assert usage == {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}
