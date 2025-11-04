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
