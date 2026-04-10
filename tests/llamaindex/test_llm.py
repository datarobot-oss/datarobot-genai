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

"""Tests for :mod:`datarobot_genai.llama_index.llm`."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from llama_index.llms.litellm import LiteLLM

from datarobot_genai.llama_index import llm as llama_index_llm

pytestmark = pytest.mark.filterwarnings(
    "ignore:WARNING! api_base is not default parameter:UserWarning",
    "ignore:WARNING! stream_options is not default parameter:UserWarning",
)


@pytest.fixture
def patched_llama_index_llm_defaults() -> None:
    with (
        patch.object(llama_index_llm, "default_api_key", return_value="sk-test-key"),
        patch.object(llama_index_llm, "default_model_name", return_value="default-model"),
        patch.object(
            llama_index_llm,
            "default_datarobot_llm_gateway_url",
            return_value="https://example.test/genai/llmgw",
        ),
        patch.object(
            llama_index_llm,
            "default_deployment_url",
            side_effect=lambda deployment_id: f"https://example.test/deployments/{deployment_id}/chat/completions",
        ),
    ):
        yield


def test_module_exports_llm_factory_functions() -> None:
    assert callable(llama_index_llm.get_datarobot_gateway_llm)
    assert callable(llama_index_llm.get_datarobot_deployment_llm)
    assert callable(llama_index_llm.get_datarobot_nim_llm)


def test_get_datarobot_gateway_llm_returns_litellm_subclass(
    patched_llama_index_llm_defaults: None,
) -> None:
    llm = llama_index_llm.get_datarobot_gateway_llm()
    assert isinstance(llm, LiteLLM)
    assert type(llm).__name__ == "DataRobotLiteLLM"
    assert llm.model == "datarobot/default-model"
    extras = llm.additional_kwargs
    assert extras.get("api_base") == "https://example.test/genai/llmgw"
    assert extras.get("api_key") == "sk-test-key"


def test_get_datarobot_gateway_llm_adds_datarobot_model_prefix_when_missing(
    patched_llama_index_llm_defaults: None,
) -> None:
    llm = llama_index_llm.get_datarobot_gateway_llm("azure/gpt-4")
    assert llm.model == "datarobot/azure/gpt-4"


def test_get_datarobot_gateway_llm_preserves_existing_datarobot_prefix(
    patched_llama_index_llm_defaults: None,
) -> None:
    llm = llama_index_llm.get_datarobot_gateway_llm("datarobot/azure/gpt-4")
    assert llm.model == "datarobot/azure/gpt-4"


def test_get_datarobot_gateway_llm_merges_parameters(
    patched_llama_index_llm_defaults: None,
) -> None:
    llm = llama_index_llm.get_datarobot_gateway_llm(parameters={"temperature": 0.25})
    assert llm.temperature == 0.25


def test_get_datarobot_deployment_llm_appends_chat_completions_to_api_base(
    patched_llama_index_llm_defaults: None,
) -> None:
    llm = llama_index_llm.get_datarobot_deployment_llm("dep-abc-123")
    assert isinstance(llm, LiteLLM)
    extras = llm.additional_kwargs or {}
    assert extras.get("api_base") == (
        "https://example.test/deployments/dep-abc-123/chat/completions"
    )


def test_get_datarobot_nim_llm_delegates_to_deployment_llm(
    patched_llama_index_llm_defaults: None,
) -> None:
    with patch.object(
        llama_index_llm,
        "get_datarobot_deployment_llm",
        wraps=llama_index_llm.get_datarobot_deployment_llm,
    ) as spy:
        llm = llama_index_llm.get_datarobot_nim_llm("nim-1", "m", {"max_tokens": 10})
    spy.assert_called_once_with("nim-1", "m", {"max_tokens": 10})
    assert isinstance(llm, LiteLLM)


def test_datarobot_litellm_metadata_enables_chat_and_tooling(
    patched_llama_index_llm_defaults: None,
) -> None:
    llm = llama_index_llm.get_datarobot_gateway_llm()
    md = llm.metadata
    assert md.is_chat_model is True
    assert md.is_function_calling_model is True
    assert md.model_name == llm.model
    assert md.context_window == 128000
