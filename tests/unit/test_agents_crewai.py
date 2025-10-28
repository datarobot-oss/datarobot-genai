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

from typing import Any

from ragas.messages import HumanMessage

import datarobot_genai.agents.crewai as crewai_mod
from datarobot_genai.agents.crewai import build_llm
from datarobot_genai.agents.crewai import create_pipeline_interactions_from_messages


def test_build_llm_uses_get_api_base_and_params(monkeypatch: Any) -> None:
    captured: list[Any] = []

    class Recorder:
        def __init__(self, *, model: str, api_base: str, api_key: str | None, timeout: int) -> None:
            self.model = model
            self.api_base = api_base
            self.api_key = api_key
            self.timeout = timeout
            captured.append(self)

    # Replace the LLM symbol inside the module under test to avoid depending on CrewAI internals
    monkeypatch.setattr(crewai_mod, "LLM", Recorder, raising=True)

    llm = build_llm(
        api_base="https://tenant.datarobot.com/api/v2",
        api_key="tok",
        model="mistral",
        deployment_id="dep-1",
        timeout=12,
    )

    assert captured and captured[0] is llm
    assert captured[0].model == "mistral"
    assert (
        captured[0].api_base
        == "https://tenant.datarobot.com/api/v2/deployments/dep-1/chat/completions"
    )
    assert captured[0].api_key == "tok"
    assert captured[0].timeout == 12


def test_create_pipeline_interactions_from_messages_returns_sample(monkeypatch: Any) -> None:
    assert create_pipeline_interactions_from_messages(None) is None

    msgs = [HumanMessage(content="hi")]  # type: ignore[call-arg]
    sample = create_pipeline_interactions_from_messages(msgs)
    assert sample is not None
    assert sample.user_input == msgs
