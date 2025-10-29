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

import pytest
from ragas.messages import AIMessage
from ragas.messages import HumanMessage

from datarobot_genai.agents.common import BaseAgent
from datarobot_genai.agents.common import create_pipeline_interactions_from_events_simple
from datarobot_genai.agents.common import extract_user_prompt_content
from datarobot_genai.agents.common import make_system_prompt


def test_base_agent_env_defaults_and_verbose(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATAROBOT_API_TOKEN", "env-token")
    monkeypatch.setenv("DATAROBOT_ENDPOINT", "https://app.datarobot.com/api/v2")

    agent = BaseAgent(verbose="false")

    assert agent.api_key == "env-token"
    # Stored as provided; normalization happens in litellm_api_base
    assert agent.api_base == "https://app.datarobot.com/api/v2"
    assert agent.verbose is False


def test_base_agent_litellm_api_base_normalization(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATAROBOT_API_TOKEN", "token")
    monkeypatch.setenv("DATAROBOT_ENDPOINT", "https://app.datarobot.com/api/v2")

    agent = BaseAgent()
    api = agent.litellm_api_base("dep-123")
    assert api == "https://app.datarobot.com/api/v2/deployments/dep-123/chat/completions"


def test_extract_user_prompt_content() -> None:
    params: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": "x"},
            {"role": "user", "content": {"foo": "bar"}},
            {"role": "user", "content": "ignored"},
        ]
    }
    assert extract_user_prompt_content(params) == {"foo": "bar"}
    assert extract_user_prompt_content({"messages": []}) == {}


def test_make_system_prompt() -> None:
    msg = make_system_prompt("TAIL")
    assert msg.endswith("\nTAIL")
    assert "helpful AI assistant" in msg


def test_create_pipeline_interactions_from_events_simple() -> None:
    assert create_pipeline_interactions_from_events_simple(None) is None
    msgs: list[Any] = [HumanMessage(content="hi"), AIMessage(content="ok")]
    sample = create_pipeline_interactions_from_events_simple(msgs)
    assert sample is not None
    assert sample.user_input == msgs
