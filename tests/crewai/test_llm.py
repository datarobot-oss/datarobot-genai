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

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from crewai import LLM

from datarobot_genai.core.config import LLMType
from datarobot_genai.crewai import llm as crewai_llm
from datarobot_genai.nat.datarobot_llm_providers import DataRobotLLMComponentModelConfig


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
# get_router_llm
# ---------------------------------------------------------------------------


def _make_component_config(deployment_id: str = "dep-id") -> DataRobotLLMComponentModelConfig:
    return DataRobotLLMComponentModelConfig(
        llm_deployment_id=deployment_id,
        api_key="test-key",
        use_datarobot_llm_gateway=False,
    )


def _make_mock_router(content: str = "response") -> MagicMock:
    choice = MagicMock()
    choice.message.content = content
    choice.message.tool_calls = None
    resp = MagicMock()
    resp.choices = [choice]
    mock_router = MagicMock()
    mock_router.completion = MagicMock(return_value=resp)
    mock_router.acompletion = AsyncMock(return_value=resp)
    return mock_router


def test_get_router_llm_returns_llm_instance() -> None:
    with patch("datarobot_genai.core.router.build_litellm_router") as mock_build, patch(
        "datarobot_genai.core.router._config_to_litellm_params",
        return_value={"model": "datarobot/gpt-4o", "api_base": "https://x/", "api_key": "k"},
    ):
        mock_build.return_value = _make_mock_router()
        llm = crewai_llm.get_router_llm(_make_component_config(), [_make_component_config("fb")])
    assert isinstance(llm, LLM)


def test_router_llm_call_accepts_available_functions_kwarg() -> None:
    """CrewAI passes available_functions to llm.call(); RouterLitellmOnlyLLM must accept it."""
    mock_router = _make_mock_router("hello")

    with patch("datarobot_genai.core.router.build_litellm_router", return_value=mock_router), patch(
        "datarobot_genai.core.router._config_to_litellm_params",
        return_value={"model": "datarobot/gpt-4o", "api_base": "https://x/", "api_key": "k"},
    ):
        llm = crewai_llm.get_router_llm(_make_component_config(), [_make_component_config("fb")])

    # This must not raise TypeError even though available_functions is not in the explicit signature.
    result = llm.call(
        [{"role": "user", "content": "hi"}],
        available_functions={"some_fn": lambda: None},
        response_model=None,
    )
    assert result == "hello"
    mock_router.completion.assert_called_once()


@pytest.mark.asyncio
async def test_router_llm_acall_accepts_available_functions_kwarg() -> None:
    """CrewAI may pass available_functions to llm.acall(); RouterLitellmOnlyLLM must accept it."""
    mock_router = _make_mock_router("async-hello")

    with patch("datarobot_genai.core.router.build_litellm_router", return_value=mock_router), patch(
        "datarobot_genai.core.router._config_to_litellm_params",
        return_value={"model": "datarobot/gpt-4o", "api_base": "https://x/", "api_key": "k"},
    ):
        llm = crewai_llm.get_router_llm(_make_component_config(), [_make_component_config("fb")])

    result = await llm.acall(
        [{"role": "user", "content": "hi"}],
        available_functions={"some_fn": lambda: None},
        response_model=None,
    )
    assert result == "async-hello"
    mock_router.acompletion.assert_called_once()
