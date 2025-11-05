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

import pytest

from datarobot_genai.core.agents.base import UsageMetrics
from datarobot_genai.llama_index import base as base_mod
from datarobot_genai.llama_index.base import LlamaIndexAgent


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
    def __init__(self, workflow: Any) -> None:
        super().__init__()
        self._wf = workflow

    def build_workflow(self) -> Any:
        return self._wf

    def extract_response_text(self, result_state: Any, events: list[Any]) -> str:
        return f"{result_state}:{len(events)}"


@pytest.mark.asyncio
async def test_llama_index_base_invoke(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}
    captured["events"] = None

    def fake_create_pipeline_interactions(events: list[Any]) -> Any:  # noqa: ANN401
        captured["events"] = events
        return {"ok": True}

    monkeypatch.setattr(
        base_mod,
        "create_pipeline_interactions_from_events",
        fake_create_pipeline_interactions,
        raising=True,
    )

    workflow = Workflow(events=[{"e": 1}, {"e": 2}], state="S")
    agent = MyLlamaAgent(workflow)

    resp = await agent.invoke({"model": "m", "messages": [{"role": "user", "content": "{}"}]})
    response_text, interactions, usage = cast(tuple[str, object, UsageMetrics], resp)

    assert response_text == "S:2"
    assert interactions == {"ok": True}
    assert captured["events"] == [{"e": 1}, {"e": 2}]
    assert usage == {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}


@pytest.mark.asyncio
async def test_llama_index_base_invoke_branches(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}
    captured["events"] = None

    def fake_create_pipeline_interactions(events: list[Any]) -> Any:  # noqa: ANN401
        captured["events"] = events
        return {"ok": True}

    monkeypatch.setattr(
        base_mod,
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
