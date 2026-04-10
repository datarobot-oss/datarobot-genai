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

"""Tests for :mod:`datarobot_genai.langgraph.llm`."""

from __future__ import annotations

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from langchain_core.language_models import BaseChatModel

from datarobot_genai.core.config import LLMType
from datarobot_genai.langgraph import llm as langgraph_llm

pytestmark = pytest.mark.filterwarnings(
    "ignore:WARNING! api_base is not default parameter:UserWarning",
    "ignore:WARNING! stream_options is not default parameter:UserWarning",
)


@pytest.fixture
def patched_langgraph_llm_defaults() -> None:
    with (
        patch.object(langgraph_llm, "default_api_key", return_value="sk-test-key"),
        patch.object(langgraph_llm, "default_model_name", return_value="default-model"),
        patch.object(
            langgraph_llm,
            "default_datarobot_llm_gateway_url",
            return_value="https://example.test/genai/llmgw",
        ),
        patch.object(
            langgraph_llm,
            "default_deployment_url",
            side_effect=lambda deployment_id: (
                f"https://example.test/deployments/{deployment_id}/chat/completions"
            ),
        ),
    ):
        yield


def test_module_exports_llm_factory_functions() -> None:
    assert callable(langgraph_llm.get_datarobot_gateway_llm)
    assert callable(langgraph_llm.get_datarobot_deployment_llm)
    assert callable(langgraph_llm.get_datarobot_nim_llm)


def test_get_datarobot_gateway_llm_returns_chat_openai_subclass(
    patched_langgraph_llm_defaults: None,
) -> None:
    llm = langgraph_llm.get_datarobot_gateway_llm()
    assert isinstance(llm, BaseChatModel)
    assert llm.model == "datarobot/default-model"
    assert llm.api_base == "https://example.test/genai/llmgw"
    assert llm.stream_options == {"include_usage": True}


def test_get_datarobot_gateway_llm_strips_datarobot_model_prefix(
    patched_langgraph_llm_defaults: None,
) -> None:
    llm = langgraph_llm.get_datarobot_gateway_llm("datarobot/azure/gpt-4")
    assert llm.model == "datarobot/azure/gpt-4"


def test_get_datarobot_gateway_llm_merges_parameters(
    patched_langgraph_llm_defaults: None,
) -> None:
    llm = langgraph_llm.get_datarobot_gateway_llm(parameters={"temperature": 0.25})
    assert llm.temperature == 0.25


def test_get_datarobot_deployment_llm_sets_deployment_api_base(
    patched_langgraph_llm_defaults: None,
) -> None:
    llm = langgraph_llm.get_datarobot_deployment_llm("dep-abc-123")
    assert isinstance(llm, BaseChatModel)
    assert llm.api_base == "https://example.test/deployments/dep-abc-123/chat/completions"
    assert llm.stream_options == {"include_usage": True}


def test_get_datarobot_nim_llm_delegates_to_deployment_llm(
    patched_langgraph_llm_defaults: None,
) -> None:
    with patch.object(
        langgraph_llm,
        "get_datarobot_deployment_llm",
        wraps=langgraph_llm.get_datarobot_deployment_llm,
    ) as spy:
        llm = langgraph_llm.get_datarobot_nim_llm("nim-1", "m", {"max_tokens": 10})
    spy.assert_called_once_with("nim-1", "m", {"max_tokens": 10}, True)
    assert isinstance(llm, BaseChatModel)


def test_gateway_llm_factory_omits_stream_options_kwarg_when_not_streaming(
    patched_langgraph_llm_defaults: None,
) -> None:
    """ChatLiteLLM has no ``_get_request_payload``; assert our factory kwargs instead."""
    with patch("langchain_litellm.ChatLiteLLM") as mock_cls:
        mock_cls.return_value = MagicMock(spec=BaseChatModel)
        langgraph_llm.get_datarobot_gateway_llm(streaming=False)
        kwargs = mock_cls.call_args.kwargs
        assert kwargs.get("streaming") is False
        assert "stream_options" not in kwargs


def test_gateway_llm_factory_passes_stream_options_when_streaming(
    patched_langgraph_llm_defaults: None,
) -> None:
    with patch("langchain_litellm.ChatLiteLLM") as mock_cls:
        mock_cls.return_value = MagicMock(spec=BaseChatModel)
        langgraph_llm.get_datarobot_gateway_llm(streaming=True)
        kwargs = mock_cls.call_args.kwargs
        assert kwargs.get("streaming") is True
        assert kwargs.get("stream_options") == {"include_usage": True}


def test_get_external_llm_returns_base_chat_model(
    patched_langgraph_llm_defaults: None,
) -> None:
    llm = langgraph_llm.get_external_llm()
    assert isinstance(llm, BaseChatModel)
    assert llm.model == "default-model"


def test_get_external_llm_strips_datarobot_prefix(
    patched_langgraph_llm_defaults: None,
) -> None:
    llm = langgraph_llm.get_external_llm("datarobot/azure/gpt-4")
    assert llm.model == "azure/gpt-4"


def test_get_external_llm_preserves_model_without_prefix(
    patched_langgraph_llm_defaults: None,
) -> None:
    llm = langgraph_llm.get_external_llm("azure/gpt-4")
    assert llm.model == "azure/gpt-4"


def test_get_external_llm_merges_parameters(
    patched_langgraph_llm_defaults: None,
) -> None:
    llm = langgraph_llm.get_external_llm(parameters={"temperature": 0.7})
    assert llm.temperature == 0.7


def test_get_llm_routes_to_gateway(patched_langgraph_llm_defaults: None) -> None:
    config = MagicMock()
    config.get_llm_type.return_value = LLMType.GATEWAY
    with patch.object(langgraph_llm, "Config", return_value=config):
        llm = langgraph_llm.get_llm()
    assert isinstance(llm, BaseChatModel)
    assert llm.api_base == "https://example.test/genai/llmgw"


def test_get_llm_routes_to_deployment(patched_langgraph_llm_defaults: None) -> None:
    config = MagicMock()
    config.get_llm_type.return_value = LLMType.DEPLOYMENT
    config.llm_deployment_id = "dep-123"
    with patch.object(langgraph_llm, "Config", return_value=config):
        llm = langgraph_llm.get_llm()
    assert isinstance(llm, BaseChatModel)
    assert llm.api_base == "https://example.test/deployments/dep-123/chat/completions"


def test_get_llm_routes_to_nim(patched_langgraph_llm_defaults: None) -> None:
    config = MagicMock()
    config.get_llm_type.return_value = LLMType.NIM
    config.nim_deployment_id = "nim-456"
    with patch.object(langgraph_llm, "Config", return_value=config):
        llm = langgraph_llm.get_llm()
    assert isinstance(llm, BaseChatModel)
    assert llm.api_base == "https://example.test/deployments/nim-456/chat/completions"


def test_get_llm_routes_to_external(patched_langgraph_llm_defaults: None) -> None:
    config = MagicMock()
    config.get_llm_type.return_value = LLMType.EXTERNAL
    with patch.object(langgraph_llm, "Config", return_value=config):
        llm = langgraph_llm.get_llm()
    assert isinstance(llm, BaseChatModel)
    assert llm.model == "default-model"


def test_get_llm_raises_on_unknown_type(patched_langgraph_llm_defaults: None) -> None:
    config = MagicMock()
    config.get_llm_type.return_value = "unknown"
    with patch.object(langgraph_llm, "Config", return_value=config):
        with pytest.raises(ValueError, match="Invalid LLM type"):
            langgraph_llm.get_llm()


def test_get_llm_forwards_model_name_and_parameters(
    patched_langgraph_llm_defaults: None,
) -> None:
    config = MagicMock()
    config.get_llm_type.return_value = LLMType.GATEWAY
    with patch.object(langgraph_llm, "Config", return_value=config):
        llm = langgraph_llm.get_llm(model_name="azure/gpt-4", parameters={"temperature": 0.5})
    assert llm.model == "datarobot/azure/gpt-4"
    assert llm.temperature == 0.5


def test_get_llm_forwards_streaming_flag(patched_langgraph_llm_defaults: None) -> None:
    config = MagicMock()
    config.get_llm_type.return_value = LLMType.GATEWAY
    with (
        patch.object(langgraph_llm, "Config", return_value=config),
        patch("langchain_litellm.ChatLiteLLM") as mock_cls,
    ):
        mock_cls.return_value = MagicMock(spec=BaseChatModel)
        langgraph_llm.get_llm(streaming=False)
        kwargs = mock_cls.call_args.kwargs
        assert kwargs.get("streaming") is False
