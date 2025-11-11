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
from types import SimpleNamespace
from typing import Any
from typing import cast

import pytest

import datarobot_genai.llama_index.base as base_mod
from datarobot_genai.core.agents.base import UsageMetrics
from datarobot_genai.llama_index.base import LlamaIndexAgent


class _Handler:
    def __init__(self, events: list[Any], ctx_get) -> None:  # noqa: ANN001
        self._events = events
        self._ctx_get = ctx_get

    async def stream_events(self) -> AsyncGenerator[Any]:  # noqa: ANN401
        for e in self._events:
            yield e

    @property
    def ctx(self) -> Any:  # noqa: ANN401
        class Ctx:
            def __init__(self, get_fn):  # noqa: ANN001
                self._get = get_fn

            def get(self, key: str) -> Any:  # noqa: ANN401, A003 - API name
                return self._get(key)

        return Ctx(self._ctx_get)


class _Workflow:
    def __init__(self, events: list[Any], ctx_get) -> None:  # noqa: ANN001
        self._events = events
        self._ctx_get = ctx_get

    def run(self, *, user_msg: str) -> _Handler:  # noqa: ARG002
        return _Handler(self._events, self._ctx_get)


class _Agent(LlamaIndexAgent):
    def __init__(self, wf: Any) -> None:  # noqa: ANN401
        super().__init__()
        self._wf = wf

    def build_workflow(self) -> Any:  # noqa: ANN401
        return self._wf

    def extract_response_text(self, result_state: Any, events: list[Any]) -> str:  # noqa: ANN401
        return f"{result_state}:{len(events)}"


@pytest.mark.asyncio
async def test_llama_index_streaming_yields_deltas_and_terminal_chunk(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {"events": None}

    # Avoid depending on real LlamaIndex events conversion
    def fake_cpie(events: list[Any]) -> Any:  # noqa: ANN401
        captured["events"] = events
        return {"ok": True}

    monkeypatch.setattr(
        base_mod,
        "create_pipeline_interactions_from_events",
        fake_cpie,
        raising=True,
    )

    # Events to trigger various branches: banner, delta, text, debug types
    agent_input_cls = type("AgentInput", (), {})
    agent_output_cls = type("AgentOutput", (), {})
    tool_call_cls = type("ToolCall", (), {})
    tool_call_result_cls = type("ToolCallResult", (), {})

    ai = agent_input_cls()
    ai.input = "q"  # type: ignore[attr-defined]
    ao = agent_output_cls()
    ao.response = SimpleNamespace(content="done")
    # Include a dict-style tool call to exercise name extraction path
    ao.tool_calls = [{"tool_name": "d"}]  # type: ignore[attr-defined]
    tc = tool_call_cls()
    tc.tool_name = "t"
    tc.tool_kwargs = {"a": 1}  # type: ignore[attr-defined]
    tcr = tool_call_result_cls()
    tcr.tool_name = "t"
    tcr.tool_output = "out"  # type: ignore[attr-defined]

    events: list[Any] = [
        SimpleNamespace(current_agent_name="alpha"),
        SimpleNamespace(delta="hello "),
        SimpleNamespace(text="world"),
        ai,
        tc,
        tcr,
        ao,
    ]

    async def _get_state(key: str) -> Any:  # noqa: ARG001, ANN401
        return {"state": 1}

    wf = _Workflow(events, ctx_get=_get_state)
    agent = _Agent(wf)

    gen = await agent.invoke(
        {
            "model": "m",
            "stream": True,
            "messages": [{"role": "user", "content": "{}"}],
        }
    )

    chunks = [c async for c in cast(AsyncGenerator[tuple[str, Any, UsageMetrics], None], gen)]

    # There should be at least one non-empty delta/banner chunk and a final empty terminal chunk
    assert any(text for text, _, _ in chunks[:-1])
    assert chunks[-1][0] == ""
    assert captured["events"] == events
    _, interactions, usage = chunks[-1]
    assert interactions == {"ok": True}
    assert usage == {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}
