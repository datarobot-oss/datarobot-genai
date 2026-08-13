# Copyright 2026 DataRobot, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from unittest import mock

import pytest

from datarobot_genai.dragent.eval.litellm_target import _secret_value
from datarobot_genai.dragent.eval.litellm_target import langchain_chat_model_to_litellm
from datarobot_genai.dragent.eval.litellm_target import wrap_langchain_judge_for_llamaindex


def test_secret_value_plain_string() -> None:
    assert _secret_value("token") == "token"


def test_secret_value_secret_str() -> None:
    secret = mock.Mock()
    secret.get_secret_value.return_value = "hidden"
    assert _secret_value(secret) == "hidden"


def test_secret_value_none() -> None:
    assert _secret_value(None) is None


def test_langchain_chat_model_to_litellm_openai() -> None:
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o-mini", api_key="secret")
    model, kwargs = langchain_chat_model_to_litellm(llm)
    assert model == "openai/gpt-4o-mini"
    assert kwargs["api_key"] == "secret"


def test_langchain_chat_model_to_litellm_chat_litellm() -> None:
    from langchain_litellm import ChatLiteLLM

    llm = ChatLiteLLM(
        model="datarobot/anthropic/claude-3",
        api_key="token",
        api_base="https://app.datarobot.com",
        extra_headers={"X-Custom": "1"},
        model_kwargs={"extra_body": {"reasoning": {"enabled": False}}},
    )
    model, kwargs = langchain_chat_model_to_litellm(llm)
    assert model == "datarobot/anthropic/claude-3"
    assert kwargs["api_key"] == "token"
    assert kwargs["api_base"] == "https://app.datarobot.com"
    assert kwargs["extra_headers"] == {"X-Custom": "1"}
    assert kwargs["extra_body"] == {"reasoning": {"enabled": False}}


def test_langchain_chat_model_to_litellm_azure() -> None:
    from langchain_openai import AzureChatOpenAI

    llm = AzureChatOpenAI(
        azure_deployment="my-deployment",
        api_version="2024-02-01",
        azure_endpoint="https://example.openai.azure.com",
        api_key="azure-key",
    )
    model, kwargs = langchain_chat_model_to_litellm(llm)
    assert model == "azure/my-deployment"
    assert kwargs["api_key"] == "azure-key"
    assert kwargs["api_base"] == "https://example.openai.azure.com"
    assert kwargs["api_version"] == "2024-02-01"


def test_langchain_chat_model_to_litellm_unsupported() -> None:
    with pytest.raises(ValueError, match="not supported"):
        langchain_chat_model_to_litellm(object())


def test_wrap_langchain_judge_for_llamaindex() -> None:
    pytest.importorskip("llama_index.llms.langchain")
    from langchain_core.language_models import BaseChatModel
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o-mini", api_key="secret")
    assert isinstance(llm, BaseChatModel)

    wrapped = wrap_langchain_judge_for_llamaindex(llm)
    assert wrapped is not None


def test_wrap_langchain_judge_for_llamaindex_rejects_non_chat_model() -> None:
    with pytest.raises(ValueError, match="not a LangChain chat model"):
        wrap_langchain_judge_for_llamaindex(object())
