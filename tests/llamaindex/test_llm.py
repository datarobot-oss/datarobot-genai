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

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from llama_index.llms.litellm import LiteLLM

from datarobot_genai.core.config import LLMType
from datarobot_genai.nat.datarobot_llm_providers import DataRobotLLMComponentModelConfig
from datarobot_genai.llama_index import llm as llama_index_llm

pytestmark = pytest.mark.filterwarnings(
    "ignore:WARNING! api_base is not default parameter:UserWarning",
    "ignore:WARNING! stream_options is not default parameter:UserWarning",
)


@pytest.fixture(autouse=True)
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
            side_effect=lambda deployment_id: (
                f"https://example.test/deployments/{deployment_id}/chat/completions"
            ),
        ),
    ):
        yield


def test_module_exports_llm_factory_functions() -> None:
    assert callable(llama_index_llm.get_datarobot_gateway_llm)
    assert callable(llama_index_llm.get_datarobot_deployment_llm)
    assert callable(llama_index_llm.get_datarobot_nim_llm)


def test_get_datarobot_gateway_llm_returns_litellm_subclass() -> None:
    llm = llama_index_llm.get_datarobot_gateway_llm()
    assert isinstance(llm, LiteLLM)
    assert type(llm).__name__ == "DataRobotLiteLLM"
    assert llm.model == "datarobot/default-model"
    extras = llm.additional_kwargs
    assert extras.get("api_base") == "https://example.test/genai/llmgw"
    assert extras.get("api_key") == "sk-test-key"


def test_get_datarobot_gateway_llm_adds_datarobot_model_prefix_when_missing() -> None:
    llm = llama_index_llm.get_datarobot_gateway_llm("azure/gpt-4")
    assert llm.model == "datarobot/azure/gpt-4"


def test_get_datarobot_gateway_llm_preserves_existing_datarobot_prefix() -> None:
    llm = llama_index_llm.get_datarobot_gateway_llm("datarobot/azure/gpt-4")
    assert llm.model == "datarobot/azure/gpt-4"


def test_get_datarobot_gateway_llm_merges_parameters() -> None:
    llm = llama_index_llm.get_datarobot_gateway_llm(parameters={"temperature": 0.25})
    assert llm.temperature == 0.25


def test_get_datarobot_deployment_llm_appends_chat_completions_to_api_base() -> None:
    llm = llama_index_llm.get_datarobot_deployment_llm("dep-abc-123")
    assert isinstance(llm, LiteLLM)
    extras = llm.additional_kwargs or {}
    assert extras.get("api_base") == (
        "https://example.test/deployments/dep-abc-123/chat/completions"
    )


def test_get_datarobot_nim_llm_delegates_to_deployment_llm() -> None:
    with patch.object(
        llama_index_llm,
        "get_datarobot_deployment_llm",
        wraps=llama_index_llm.get_datarobot_deployment_llm,
    ) as spy:
        llm = llama_index_llm.get_datarobot_nim_llm("nim-1", "m", {"max_tokens": 10})
    spy.assert_called_once_with("nim-1", "m", {"max_tokens": 10})
    assert isinstance(llm, LiteLLM)


def test_datarobot_litellm_metadata_enables_chat_and_tooling() -> None:
    llm = llama_index_llm.get_datarobot_gateway_llm()
    md = llm.metadata
    assert md.is_chat_model is True
    assert md.is_function_calling_model is True
    assert md.model_name == llm.model
    assert md.context_window == 128000


def test_get_external_llm_returns_litellm_subclass() -> None:
    llm = llama_index_llm.get_external_llm()
    assert isinstance(llm, LiteLLM)
    assert llm.model == "default-model"


def test_get_external_llm_strips_datarobot_prefix() -> None:
    llm = llama_index_llm.get_external_llm("datarobot/azure/gpt-4")
    assert llm.model == "azure/gpt-4"


def test_get_external_llm_preserves_model_without_prefix() -> None:
    llm = llama_index_llm.get_external_llm("azure/gpt-4")
    assert llm.model == "azure/gpt-4"


def test_get_external_llm_merges_parameters() -> None:
    llm = llama_index_llm.get_external_llm(parameters={"temperature": 0.7})
    assert llm.temperature == 0.7


def test_get_llm_routes_to_gateway() -> None:
    config = MagicMock()
    config.get_llm_type.return_value = LLMType.GATEWAY
    with patch.object(llama_index_llm, "Config", return_value=config):
        llm = llama_index_llm.get_llm()
    assert isinstance(llm, LiteLLM)
    extras = llm.additional_kwargs
    assert extras.get("api_base") == "https://example.test/genai/llmgw"


def test_get_llm_routes_to_deployment() -> None:
    config = MagicMock()
    config.get_llm_type.return_value = LLMType.DEPLOYMENT
    config.llm_deployment_id = "dep-123"
    with patch.object(llama_index_llm, "Config", return_value=config):
        llm = llama_index_llm.get_llm()
    assert isinstance(llm, LiteLLM)
    extras = llm.additional_kwargs
    assert extras.get("api_base") == "https://example.test/deployments/dep-123/chat/completions"


def test_get_llm_routes_to_nim() -> None:
    config = MagicMock()
    config.get_llm_type.return_value = LLMType.NIM
    config.nim_deployment_id = "nim-456"
    with patch.object(llama_index_llm, "Config", return_value=config):
        llm = llama_index_llm.get_llm()
    assert isinstance(llm, LiteLLM)
    extras = llm.additional_kwargs
    assert extras.get("api_base") == "https://example.test/deployments/nim-456/chat/completions"


def test_get_llm_routes_to_external() -> None:
    config = MagicMock()
    config.get_llm_type.return_value = LLMType.EXTERNAL
    with patch.object(llama_index_llm, "Config", return_value=config):
        llm = llama_index_llm.get_llm()
    assert isinstance(llm, LiteLLM)
    assert llm.model == "default-model"


def test_get_llm_raises_on_unknown_type() -> None:
    config = MagicMock()
    config.get_llm_type.return_value = "unknown"
    with patch.object(llama_index_llm, "Config", return_value=config):
        with pytest.raises(ValueError, match="Invalid LLM type"):
            llama_index_llm.get_llm()


def test_get_llm_forwards_model_name_and_parameters() -> None:
    config = MagicMock()
    config.get_llm_type.return_value = LLMType.GATEWAY
    with patch.object(llama_index_llm, "Config", return_value=config):
        llm = llama_index_llm.get_llm(model_name="azure/gpt-4", parameters={"temperature": 0.5})
    assert llm.model == "datarobot/azure/gpt-4"
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


def _make_mock_router(
    content: str = "response", tool_calls: list | None = None
) -> MagicMock:
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls or []
    resp = MagicMock()
    resp.choices = [MagicMock(message=message)]
    mock_router = MagicMock()
    mock_router.completion = MagicMock(return_value=resp)
    mock_router.acompletion = AsyncMock(return_value=resp)
    return mock_router


def _make_mock_tool_call(id: str, name: str, arguments: str) -> MagicMock:
    tc = MagicMock()
    tc.id = id
    tc.type = "function"
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


def test_get_router_llm_returns_litellm_instance() -> None:
    with patch("datarobot_genai.core.router.build_litellm_router") as mock_build, patch(
        "datarobot_genai.core.router._config_to_litellm_params",
        return_value={"model": "datarobot/gpt-4o", "api_base": "https://x/", "api_key": "k"},
    ):
        mock_build.return_value = _make_mock_router()
        llm = llama_index_llm.get_router_llm(
            _make_component_config(), [_make_component_config("fb")]
        )
    assert isinstance(llm, LiteLLM)


def test_router_llm_chat_returns_text_content() -> None:
    mock_router = _make_mock_router("hello from chat")

    with patch(
        "datarobot_genai.core.router.build_litellm_router", return_value=mock_router
    ), patch(
        "datarobot_genai.core.router._config_to_litellm_params",
        return_value={"model": "datarobot/gpt-4o", "api_base": "https://x/", "api_key": "k"},
    ):
        llm = llama_index_llm.get_router_llm(
            _make_component_config(), [_make_component_config("fb")]
        )

    from llama_index.core.base.llms.types import ChatMessage

    result = llm._chat([ChatMessage(role="user", content="hi")])
    assert result.message.content == "hello from chat"
    mock_router.completion.assert_called_once()


def test_router_llm_chat_includes_tool_calls_in_additional_kwargs() -> None:
    tc = _make_mock_tool_call("tc-1", "search", '{"query": "cats"}')
    mock_router = _make_mock_router(content="", tool_calls=[tc])

    with patch(
        "datarobot_genai.core.router.build_litellm_router", return_value=mock_router
    ), patch(
        "datarobot_genai.core.router._config_to_litellm_params",
        return_value={"model": "datarobot/gpt-4o", "api_base": "https://x/", "api_key": "k"},
    ):
        llm = llama_index_llm.get_router_llm(
            _make_component_config(), [_make_component_config("fb")]
        )

    from llama_index.core.base.llms.types import ChatMessage

    result = llm._chat([ChatMessage(role="user", content="search for cats")])
    tool_calls = result.message.additional_kwargs.get("tool_calls", [])
    assert len(tool_calls) == 1
    assert tool_calls[0]["id"] == "tc-1"
    assert tool_calls[0]["function"]["name"] == "search"
    assert tool_calls[0]["function"]["arguments"] == '{"query": "cats"}'


@pytest.mark.asyncio
async def test_router_llm_achat_returns_text_content() -> None:
    mock_router = _make_mock_router("async hello")

    with patch(
        "datarobot_genai.core.router.build_litellm_router", return_value=mock_router
    ), patch(
        "datarobot_genai.core.router._config_to_litellm_params",
        return_value={"model": "datarobot/gpt-4o", "api_base": "https://x/", "api_key": "k"},
    ):
        llm = llama_index_llm.get_router_llm(
            _make_component_config(), [_make_component_config("fb")]
        )

    from llama_index.core.base.llms.types import ChatMessage

    result = await llm._achat([ChatMessage(role="user", content="hi")])
    assert result.message.content == "async hello"
    mock_router.acompletion.assert_called_once()


@pytest.mark.asyncio
async def test_router_llm_achat_includes_tool_calls_in_additional_kwargs() -> None:
    tc = _make_mock_tool_call("tc-2", "lookup", '{"id": 42}')
    mock_router = _make_mock_router(content="", tool_calls=[tc])

    with patch(
        "datarobot_genai.core.router.build_litellm_router", return_value=mock_router
    ), patch(
        "datarobot_genai.core.router._config_to_litellm_params",
        return_value={"model": "datarobot/gpt-4o", "api_base": "https://x/", "api_key": "k"},
    ):
        llm = llama_index_llm.get_router_llm(
            _make_component_config(), [_make_component_config("fb")]
        )

    from llama_index.core.base.llms.types import ChatMessage

    result = await llm._achat([ChatMessage(role="user", content="look it up")])
    tool_calls = result.message.additional_kwargs.get("tool_calls", [])
    assert len(tool_calls) == 1
    assert tool_calls[0]["id"] == "tc-2"
    assert tool_calls[0]["function"]["name"] == "lookup"


def test_router_llm_stream_chat_yields_single_chunk_with_content() -> None:
    mock_router = _make_mock_router("streamed text")

    with patch(
        "datarobot_genai.core.router.build_litellm_router", return_value=mock_router
    ), patch(
        "datarobot_genai.core.router._config_to_litellm_params",
        return_value={"model": "datarobot/gpt-4o", "api_base": "https://x/", "api_key": "k"},
    ):
        llm = llama_index_llm.get_router_llm(
            _make_component_config(), [_make_component_config("fb")]
        )

    from llama_index.core.base.llms.types import ChatMessage

    chunks = list(llm._stream_chat([ChatMessage(role="user", content="hi")]))
    assert len(chunks) == 1
    assert chunks[0].message.content == "streamed text"
    assert chunks[0].delta == "streamed text"
    mock_router.completion.assert_called_once()


def test_router_llm_stream_chat_yields_tool_calls() -> None:
    tc = _make_mock_tool_call("tc-3", "fetch", '{"url": "https://example.com"}')
    mock_router = _make_mock_router(content="", tool_calls=[tc])

    with patch(
        "datarobot_genai.core.router.build_litellm_router", return_value=mock_router
    ), patch(
        "datarobot_genai.core.router._config_to_litellm_params",
        return_value={"model": "datarobot/gpt-4o", "api_base": "https://x/", "api_key": "k"},
    ):
        llm = llama_index_llm.get_router_llm(
            _make_component_config(), [_make_component_config("fb")]
        )

    from llama_index.core.base.llms.types import ChatMessage

    chunks = list(llm._stream_chat([ChatMessage(role="user", content="fetch it")]))
    assert len(chunks) == 1
    tool_calls = chunks[0].message.additional_kwargs.get("tool_calls", [])
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "fetch"


@pytest.mark.asyncio
async def test_router_llm_stream_achat_yields_single_chunk_with_content() -> None:
    mock_router = _make_mock_router("async streamed text")

    with patch(
        "datarobot_genai.core.router.build_litellm_router", return_value=mock_router
    ), patch(
        "datarobot_genai.core.router._config_to_litellm_params",
        return_value={"model": "datarobot/gpt-4o", "api_base": "https://x/", "api_key": "k"},
    ):
        llm = llama_index_llm.get_router_llm(
            _make_component_config(), [_make_component_config("fb")]
        )

    from llama_index.core.base.llms.types import ChatMessage

    chunks = [c async for c in llm._stream_achat([ChatMessage(role="user", content="hi")])]
    assert len(chunks) == 1
    assert chunks[0].message.content == "async streamed text"
    assert chunks[0].delta == "async streamed text"
    mock_router.acompletion.assert_called_once()


@pytest.mark.asyncio
async def test_router_llm_stream_achat_yields_tool_calls() -> None:
    tc = _make_mock_tool_call("tc-4", "analyze", '{"data": "cats"}')
    mock_router = _make_mock_router(content="", tool_calls=[tc])

    with patch(
        "datarobot_genai.core.router.build_litellm_router", return_value=mock_router
    ), patch(
        "datarobot_genai.core.router._config_to_litellm_params",
        return_value={"model": "datarobot/gpt-4o", "api_base": "https://x/", "api_key": "k"},
    ):
        llm = llama_index_llm.get_router_llm(
            _make_component_config(), [_make_component_config("fb")]
        )

    from llama_index.core.base.llms.types import ChatMessage

    chunks = [c async for c in llm._stream_achat([ChatMessage(role="user", content="analyze")])]
    assert len(chunks) == 1
    tool_calls = chunks[0].message.additional_kwargs.get("tool_calls", [])
    assert len(tool_calls) == 1
    assert tool_calls[0]["id"] == "tc-4"
    assert tool_calls[0]["function"]["name"] == "analyze"
