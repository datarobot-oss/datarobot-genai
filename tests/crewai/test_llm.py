# Copyright 2026 DataRobot, Inc. and its affiliates.
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

"""Tests for :mod:`datarobot_genai.crewai.llm`."""

from __future__ import annotations

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from crewai import LLM

from datarobot_genai.core.config import LLMType
from datarobot_genai.crewai import llm as crewai_llm
from datarobot_genai.crewai.llm import LitellmStopWordLLM


@pytest.fixture(autouse=True)
def patched_crewai_llm_defaults() -> None:
    with (
        patch.object(crewai_llm, "default_api_key", return_value="sk-test-key"),
        patch.object(crewai_llm, "default_model_name", return_value="default-model"),
        patch.object(
            crewai_llm,
            "default_datarobot_llm_gateway_url",
            return_value="https://example.test/genai/llmgw",
        ),
        patch.object(
            crewai_llm,
            "default_deployment_url",
            side_effect=lambda deployment_id: (
                f"https://example.test/deployments/{deployment_id}/chat/completions"
            ),
        ),
    ):
        yield


def test_module_exports_llm_factory_functions() -> None:
    assert callable(crewai_llm.get_datarobot_gateway_llm)
    assert callable(crewai_llm.get_datarobot_deployment_llm)
    assert callable(crewai_llm.get_datarobot_nim_llm)


def test_get_datarobot_gateway_llm_returns_crewai_llm() -> None:
    llm = crewai_llm.get_datarobot_gateway_llm()
    assert isinstance(llm, LLM)
    assert llm.model == "datarobot/default-model"
    assert llm.api_base == "https://example.test/genai/llmgw"
    assert llm.api_key == "sk-test-key"
    assert llm.is_litellm is True
    assert llm.additional_params == {"stream_options": {"include_usage": True}}


def test_get_datarobot_gateway_llm_adds_datarobot_model_prefix_when_missing() -> None:
    llm = crewai_llm.get_datarobot_gateway_llm("azure/gpt-4")
    assert llm.model == "datarobot/azure/gpt-4"


def test_get_datarobot_gateway_llm_preserves_existing_datarobot_prefix() -> None:
    llm = crewai_llm.get_datarobot_gateway_llm("datarobot/azure/gpt-4")
    assert isinstance(llm, LLM)
    assert llm.is_litellm is True
    assert llm.model == "datarobot/azure/gpt-4"


def test_get_datarobot_gateway_llm_merges_parameters() -> None:
    llm = crewai_llm.get_datarobot_gateway_llm(parameters={"temperature": 0.25})
    assert isinstance(llm, LLM)
    assert llm.is_litellm is True
    assert llm.temperature == 0.25


def test_get_datarobot_deployment_llm_appends_chat_completions_to_api_base() -> None:
    llm = crewai_llm.get_datarobot_deployment_llm("dep-abc-123")
    assert isinstance(llm, LLM)
    assert llm.is_litellm is True
    assert llm.api_base == ("https://example.test/deployments/dep-abc-123/chat/completions")
    assert llm.additional_params == {
        "stream_options": {"include_usage": True},
    }


def test_get_datarobot_deployment_llm_merges_parameters() -> None:
    llm = crewai_llm.get_datarobot_deployment_llm(
        "dep-abc-123",
        parameters={"temperature": 0.4},
    )
    assert isinstance(llm, LLM)
    assert llm.is_litellm is True
    assert llm.temperature == 0.4


def test_get_datarobot_nim_llm_delegates_to_deployment_llm() -> None:
    with patch.object(
        crewai_llm,
        "get_datarobot_deployment_llm",
        wraps=crewai_llm.get_datarobot_deployment_llm,
    ) as spy:
        llm = crewai_llm.get_datarobot_nim_llm("nim-1", "m", {"max_tokens": 10})
    spy.assert_called_once_with("nim-1", "m", {"max_tokens": 10})
    assert isinstance(llm, LLM)
    assert llm.is_litellm is True


def test_get_external_llm_returns_crewai_llm() -> None:
    llm = crewai_llm.get_external_llm()
    assert isinstance(llm, LLM)
    assert llm.is_litellm is True
    assert llm.api_base is None
    assert llm.api_key is None
    assert llm.is_litellm is True
    assert llm.additional_params == {
        "stream_options": {"include_usage": True},
    }
    assert llm.model == "default-model"


def test_get_external_llm_strips_datarobot_prefix() -> None:
    llm = crewai_llm.get_external_llm("datarobot/azure/gpt-4")
    assert llm.model == "azure/gpt-4"


def test_get_external_llm_preserves_model_without_prefix() -> None:
    llm = crewai_llm.get_external_llm("azure/gpt-4")
    assert llm.model == "azure/gpt-4"


def test_get_external_llm_merges_parameters() -> None:
    llm = crewai_llm.get_external_llm(parameters={"temperature": 0.7})
    assert isinstance(llm, LLM)
    assert llm.is_litellm is True
    assert llm.temperature == 0.7


def test_get_llm_routes_to_gateway() -> None:
    config = MagicMock()
    config.get_llm_type.return_value = LLMType.GATEWAY
    with patch.object(crewai_llm, "Config", return_value=config):
        llm = crewai_llm.get_llm()
    assert isinstance(llm, LLM)
    assert llm.is_litellm is True
    assert llm.api_base == "https://example.test/genai/llmgw"


def test_get_llm_routes_to_deployment() -> None:
    config = MagicMock()
    config.get_llm_type.return_value = LLMType.DEPLOYMENT
    config.llm_deployment_id = "dep-123"
    with patch.object(crewai_llm, "Config", return_value=config):
        llm = crewai_llm.get_llm()
    assert isinstance(llm, LLM)
    assert llm.is_litellm is True
    assert llm.api_base == "https://example.test/deployments/dep-123/chat/completions"


def test_get_llm_routes_to_nim() -> None:
    config = MagicMock()
    config.get_llm_type.return_value = LLMType.NIM
    config.nim_deployment_id = "nim-456"
    with patch.object(crewai_llm, "Config", return_value=config):
        llm = crewai_llm.get_llm()
    assert isinstance(llm, LLM)
    assert llm.is_litellm is True
    assert llm.api_base == "https://example.test/deployments/nim-456/chat/completions"


def test_get_llm_routes_to_external() -> None:
    config = MagicMock()
    config.get_llm_type.return_value = LLMType.EXTERNAL
    with patch.object(crewai_llm, "Config", return_value=config):
        llm = crewai_llm.get_llm()
    assert isinstance(llm, LLM)
    assert llm.is_litellm is True
    assert llm.model == "default-model"


def test_get_llm_raises_on_unknown_type() -> None:
    config = MagicMock()
    config.get_llm_type.return_value = "unknown"
    with patch.object(crewai_llm, "Config", return_value=config):
        with pytest.raises(ValueError, match="Invalid LLM type"):
            crewai_llm.get_llm()


def test_get_llm_forwards_model_name_and_parameters() -> None:
    config = MagicMock()
    config.get_llm_type.return_value = LLMType.GATEWAY
    with patch.object(crewai_llm, "Config", return_value=config):
        llm = crewai_llm.get_llm(model_name="azure/gpt-4", parameters={"temperature": 0.5})
    assert llm.model == "datarobot/azure/gpt-4"
    assert llm.is_litellm is True
    assert llm.temperature == 0.5


# ---------------------------------------------------------------------------
# LitellmStopWordLLM – isolated unit tests
# ---------------------------------------------------------------------------


@pytest.fixture
def stop_word_llm() -> LitellmStopWordLLM:
    """Return a LitellmStopWordLLM instance with a single stop word configured."""
    return LitellmStopWordLLM(model="openai/gpt-4o", stop=["\nObservation:"])


def test_litellm_stop_word_llm_is_litellm_subclass() -> None:
    llm = LitellmStopWordLLM(model="openai/gpt-4o")
    assert isinstance(llm, LLM)
    assert llm.is_litellm is True


def test_litellm_stop_word_llm_call_applies_stop_words(
    stop_word_llm: LitellmStopWordLLM,
) -> None:
    """Hallucinated content after the stop word is truncated."""
    hallucinated = (
        "Thought: I need to search.\n"
        "Action: search\n"
        "Action Input: query\n"
        "Observation: fake result\n"
        "Final Answer: hallucinated"
    )
    with patch.object(LLM, "call", return_value=hallucinated):
        result = stop_word_llm.call("test message")
    assert result == "Thought: I need to search.\nAction: search\nAction Input: query"
    assert "Observation:" not in result
    assert "Final Answer:" not in result


def test_litellm_stop_word_llm_call_no_stop_words_returns_unchanged() -> None:
    """Without stop words configured, responses pass through unchanged."""
    llm = LitellmStopWordLLM(model="openai/gpt-4o")
    response = "Some text\nObservation: data\nFinal Answer: done"
    with patch.object(LLM, "call", return_value=response):
        result = llm.call("test message")
    assert result == response


def test_litellm_stop_word_llm_call_non_string_result_passes_through(
    stop_word_llm: LitellmStopWordLLM,
) -> None:
    """Non-string return values (e.g. tool call results) are not truncated."""
    tool_call_result = [{"function": {"name": "search", "arguments": "{}"}}]
    with patch.object(LLM, "call", return_value=tool_call_result):
        result = stop_word_llm.call("test message")
    assert result == tool_call_result


def test_litellm_stop_word_llm_call_stop_word_absent_returns_unchanged(
    stop_word_llm: LitellmStopWordLLM,
) -> None:
    """Stop words configured but not present in the response — unchanged."""
    clean_response = "Thought: I know the answer.\nFinal Answer: 42"
    with patch.object(LLM, "call", return_value=clean_response):
        result = stop_word_llm.call("test message")
    assert result == clean_response


def test_litellm_stop_word_llm_call_multiple_stop_words_truncates_at_earliest() -> None:
    """Multiple stop words: truncation happens at the earliest occurrence."""
    llm = LitellmStopWordLLM(model="openai/gpt-4o", stop=["\nObservation:", "\nFinal Answer:"])
    response = "Action: search\nObservation: found\nFinal Answer: done"
    with patch.object(LLM, "call", return_value=response):
        result = llm.call("test message")
    assert result == "Action: search"
