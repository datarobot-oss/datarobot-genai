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
from openai.types import CompletionCreateParams
from ragas.messages import AIMessage
from ragas.messages import HumanMessage

from datarobot_genai.core.agents.base import BaseAgent
from datarobot_genai.core.agents.base import extract_user_prompt_content
from datarobot_genai.core.agents.base import make_system_prompt


class SimpleAgent(BaseAgent):
    async def invoke(
        self, completion_create_params: CompletionCreateParams
    ) -> tuple[str, Any | None, dict[str, int]]:
        return "ok", None, {}


def test_base_agent_env_defaults_and_verbose(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATAROBOT_API_TOKEN", "env-token")
    monkeypatch.setenv("DATAROBOT_ENDPOINT", "https://app.datarobot.com/api/v2")

    agent = SimpleAgent(verbose="false")

    assert agent.api_key == "env-token"
    # Stored as provided; normalization happens in litellm_api_base
    assert agent.api_base == "https://app.datarobot.com/api/v2"
    assert agent.verbose is False


def test_base_agent_litellm_api_base_normalization(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATAROBOT_API_TOKEN", "token")
    monkeypatch.setenv("DATAROBOT_ENDPOINT", "https://app.datarobot.com/api/v2")

    agent = SimpleAgent()
    api = agent.litellm_api_base("dep-123")
    assert api == "https://app.datarobot.com/api/v2/deployments/dep-123/chat/completions"


def test_extract_user_prompt_content_no_user_messages() -> None:
    # GIVEN a completion create params with no user messages
    params: dict[str, Any] = {"messages": []}
    # WHEN extracting the user prompt content
    user_prompt = extract_user_prompt_content(params)
    # THEN the user prompt is empty
    assert user_prompt == {}


def test_extract_user_prompt_content_the_last_user_message() -> None:
    # GIVEN a completion create params with multiple user messages
    params: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": "x"},
            {"role": "user", "content": "ignored"},
            {"role": "user", "content": {"foo": "bar"}},
        ]
    }
    # WHEN extracting the user prompt content
    user_prompt = extract_user_prompt_content(params)
    # THEN the user prompt is the last user message
    assert user_prompt == {"foo": "bar"}


def test_extract_user_prompt_content_the_last_user_message_is_a_json_string() -> None:
    # GIVEN a completion create params with a user message that is a json string
    params: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": "x"},
            {"role": "user", "content": "ignored"},
            {"role": "user", "content": '{"foo": "bar"}'},
        ]
    }
    # WHEN extracting the user prompt content
    user_prompt = extract_user_prompt_content(params)
    # THEN the user prompt is the last user message
    assert user_prompt == {"foo": "bar"}


def test_make_system_prompt() -> None:
    msg = make_system_prompt("TAIL")
    assert msg.endswith("\nTAIL")
    assert "helpful AI assistant" in msg


def test_create_pipeline_interactions_from_events_simple() -> None:
    assert BaseAgent.create_pipeline_interactions_from_events(None) is None
    msgs: list[Any] = [HumanMessage(content="hi"), AIMessage(content="ok")]
    sample = BaseAgent.create_pipeline_interactions_from_events(msgs)
    assert sample is not None
    assert sample.user_input == msgs


def test_base_agent_forwarded_headers_none() -> None:
    """Test BaseAgent with no forwarded headers."""
    agent = SimpleAgent(forwarded_headers=None)
    assert agent.forwarded_headers == {}


def test_base_agent_forwarded_headers_empty() -> None:
    """Test BaseAgent with empty forwarded headers."""
    agent = SimpleAgent(forwarded_headers={})
    assert agent.forwarded_headers == {}


def test_base_agent_forwarded_headers_with_scoped_token() -> None:
    """Test BaseAgent with forwarded headers including scoped token."""
    headers = {
        "x-datarobot-api-key": "scoped-token-123",
        "x-custom-header": "custom-value",
    }
    agent = SimpleAgent(forwarded_headers=headers)
    assert agent.forwarded_headers == headers
    assert agent.forwarded_headers["x-datarobot-api-key"] == "scoped-token-123"
    assert agent.forwarded_headers["x-custom-header"] == "custom-value"


def test_base_agent_forwarded_headers_with_bearer_token() -> None:
    """Test BaseAgent with forwarded headers containing Bearer token format."""
    headers = {
        "x-datarobot-api-key": "Bearer scoped-token-456",
        "authorization": "Bearer main-token",
    }
    agent = SimpleAgent(forwarded_headers=headers)
    assert agent.forwarded_headers == headers
    assert agent.forwarded_headers["x-datarobot-api-key"] == "Bearer scoped-token-456"
