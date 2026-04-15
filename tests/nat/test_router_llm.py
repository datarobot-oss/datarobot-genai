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

from __future__ import annotations

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from crewai import LLM
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from llama_index.llms.litellm import LiteLLM
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.workflow_builder import WorkflowBuilder
from pydantic import ValidationError

import datarobot_genai.nat.datarobot_llm_clients  # noqa: F401
from datarobot_genai.core.router import _config_to_litellm_params
from datarobot_genai.core.router import build_litellm_router
from datarobot_genai.langgraph.router_llm import RouterChatModel
from datarobot_genai.nat.datarobot_llm_clients import _router_settings_from_config
from datarobot_genai.nat.datarobot_llm_providers import DataRobotLLMComponentModelConfig
from datarobot_genai.nat.datarobot_llm_providers import DataRobotLLMRouterConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_router_config(
    primary_id: str = "primary-deploy-id",
    fallback_id: str = "fallback-deploy-id",
    **router_kwargs,
) -> DataRobotLLMRouterConfig:
    return DataRobotLLMRouterConfig(
        primary=DataRobotLLMComponentModelConfig(
            llm_deployment_id=primary_id,
            api_key="test-key",
            use_datarobot_llm_gateway=False,
        ),
        fallbacks=[
            DataRobotLLMComponentModelConfig(
                llm_deployment_id=fallback_id,
                api_key="test-key",
                use_datarobot_llm_gateway=False,
            )
        ],
        **router_kwargs,
    )


def _make_litellm_response(content: str = "hello") -> MagicMock:
    """Build a minimal mock that looks like a litellm ModelResponse."""
    choice = MagicMock()
    choice.message.content = content
    choice.message.tool_calls = None
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ---------------------------------------------------------------------------
# Config round-trip
# ---------------------------------------------------------------------------


def test_router_config_round_trip():
    """DataRobotLLMRouterConfig instantiates and validates cleanly."""
    cfg = _make_router_config(allowed_fails=5, cooldown_time=30.0)
    assert cfg.primary.llm_deployment_id == "primary-deploy-id"
    assert len(cfg.fallbacks) == 1
    assert cfg.fallbacks[0].llm_deployment_id == "fallback-deploy-id"
    assert cfg.allowed_fails == 5
    assert cfg.cooldown_time == 30.0


def test_router_config_requires_at_least_one_fallback():
    with pytest.raises(ValidationError):
        DataRobotLLMRouterConfig(
            primary=DataRobotLLMComponentModelConfig(
                llm_deployment_id="abc",
                api_key="key",
                use_datarobot_llm_gateway=False,
            ),
            fallbacks=[],  # violates min_length=1
        )


def test_router_config_from_dict():
    """Config can be constructed from a plain dict (simulates YAML deserialization)."""
    data = {
        "primary": {
            "llm_deployment_id": "abc123",
            "api_key": "secret",
            "use_datarobot_llm_gateway": False,
        },
        "fallbacks": [
            {
                "llm_deployment_id": "def456",
                "api_key": "secret",
                "use_datarobot_llm_gateway": False,
            }
        ],
    }
    cfg = DataRobotLLMRouterConfig(**data)
    assert cfg.primary.llm_deployment_id == "abc123"
    assert cfg.fallbacks[0].llm_deployment_id == "def456"


# ---------------------------------------------------------------------------
# build_litellm_router structure
# ---------------------------------------------------------------------------


def test_build_litellm_router_model_list():
    """build_litellm_router produces the correct model_list and fallbacks."""
    primary = {"model": "datarobot/gpt-4o", "api_base": "https://primary/", "api_key": "k"}
    fallback0 = {"model": "datarobot/gpt-4o", "api_base": "https://fb0/", "api_key": "k"}
    fallback1 = {"model": "datarobot/gpt-4o", "api_base": "https://fb1/", "api_key": "k"}

    with patch("litellm.Router") as mock_router_cls:
        build_litellm_router(primary, [fallback0, fallback1])
        args, kwargs = mock_router_cls.call_args
        model_list = kwargs["model_list"]
        fallbacks = kwargs["fallbacks"]

    assert model_list[0] == {"model_name": "primary", "litellm_params": primary}
    assert model_list[1] == {"model_name": "fallback_0", "litellm_params": fallback0}
    assert model_list[2] == {"model_name": "fallback_1", "litellm_params": fallback1}
    assert fallbacks == [{"primary": ["fallback_0", "fallback_1"]}]


def test_build_litellm_router_passes_settings():
    primary = {"model": "datarobot/gpt-4o", "api_base": "https://x/", "api_key": "k"}
    fallback = {"model": "datarobot/gpt-4o", "api_base": "https://y/", "api_key": "k"}

    with patch("litellm.Router") as mock_router_cls:
        build_litellm_router(primary, [fallback], {"allowed_fails": 5, "cooldown_time": 10.0})
        _, kwargs = mock_router_cls.call_args

    assert kwargs["allowed_fails"] == 5
    assert kwargs["cooldown_time"] == 10.0


# ---------------------------------------------------------------------------
# _config_to_litellm_params
# ---------------------------------------------------------------------------


def test_config_to_litellm_params_deployment():
    cfg = DataRobotLLMComponentModelConfig(
        llm_deployment_id="my-deploy",
        api_key="mykey",
        use_datarobot_llm_gateway=False,
    )
    with patch(
        "datarobot_genai.core.config.Config",
        return_value=MagicMock(
            datarobot_endpoint="https://app.datarobot.com/api/v2",
            datarobot_api_token="mykey",
        ),
    ):
        params = _config_to_litellm_params(cfg)

    assert params["model"].startswith("datarobot/")
    assert "my-deploy" in params["api_base"]
    assert params["api_key"] == "mykey"


def test_config_to_litellm_params_gateway():
    cfg = DataRobotLLMComponentModelConfig(
        model_name="azure/gpt-4o",
        api_key="mykey",
        use_datarobot_llm_gateway=True,
    )
    with patch(
        "datarobot_genai.core.config.Config",
        return_value=MagicMock(
            datarobot_endpoint="https://app.datarobot.com/api/v2",
            datarobot_api_token="mykey",
        ),
    ):
        params = _config_to_litellm_params(cfg)

    assert params["model"] == "datarobot/azure/gpt-4o"
    assert "app.datarobot.com" in params["api_base"]


def test_config_to_litellm_params_nim():
    """Test NIM deployment type conversion."""
    cfg = DataRobotLLMComponentModelConfig(
        nim_deployment_id="nim-deploy",
        api_key="nimkey",
        use_datarobot_llm_gateway=False,
    )
    with patch(
        "datarobot_genai.core.config.Config",
        return_value=MagicMock(
            datarobot_endpoint="https://app.datarobot.com/api/v2",
            datarobot_api_token="fallback-token",
        ),
    ):
        params = _config_to_litellm_params(cfg)

    assert params["model"].startswith("datarobot/")
    assert "nim-deploy" in params["api_base"]
    assert params["api_key"] == "nimkey"


def test_config_to_litellm_params_external():
    """Test external LLM type conversion."""
    cfg = DataRobotLLMComponentModelConfig(
        model_name="openai/gpt-4o",
        api_key="ext-key",
        use_datarobot_llm_gateway=False,
    )
    with patch(
        "datarobot_genai.core.config.Config",
        return_value=MagicMock(datarobot_api_token="unused"),
    ):
        params = _config_to_litellm_params(cfg)

    assert params["model"] == "openai/gpt-4o"
    assert "api_base" not in params
    assert params["api_key"] == "ext-key"


# ---------------------------------------------------------------------------
# _router_settings_from_config
# ---------------------------------------------------------------------------


def test_router_settings_from_config_with_defaults():
    """Router settings respects defaults."""
    cfg = _make_router_config()
    settings = _router_settings_from_config(cfg)
    assert settings["allowed_fails"] == 3
    assert "retry_policy" not in settings
    assert "cooldown_time" not in settings


def test_router_settings_from_config_with_all_options():
    """Router settings includes all provided options."""
    cfg = _make_router_config(
        allowed_fails=5,
        retry_policy={"RateLimitErrorRetries": 2},
        cooldown_time=30.0,
    )
    settings = _router_settings_from_config(cfg)
    assert settings["allowed_fails"] == 5
    assert settings["retry_policy"] == {"RateLimitErrorRetries": 2}
    assert settings["cooldown_time"] == 30.0


# ---------------------------------------------------------------------------
# RouterChatModel — LangChain
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_router_chat_model_agenerate_success():
    """_agenerate delegates to router.acompletion and returns a ChatResult."""
    mock_router = MagicMock()
    mock_router.acompletion = AsyncMock(return_value=_make_litellm_response("world"))

    model = RouterChatModel(router=mock_router)
    result = await model._agenerate([HumanMessage(content="hello")])

    mock_router.acompletion.assert_called_once()
    call_kwargs = mock_router.acompletion.call_args
    assert call_kwargs[0][0] == "primary"
    assert result.generations[0].message.content == "world"


@pytest.mark.asyncio
async def test_router_chat_model_generate_success():
    """_generate delegates to router.completion and returns a ChatResult."""
    mock_router = MagicMock()
    mock_router.completion = MagicMock(return_value=_make_litellm_response("sync-reply"))

    model = RouterChatModel(router=mock_router)
    result = model._generate([HumanMessage(content="ping")])

    mock_router.completion.assert_called_once()
    assert result.generations[0].message.content == "sync-reply"


@pytest.mark.asyncio
async def test_router_chat_model_agenerate_propagates_error():
    """Errors from router.acompletion propagate out of RouterChatModel._agenerate.

    litellm.Router handles retries / fallbacks internally; RouterChatModel does not
    swallow exceptions — if the router exhausts all options it raises and we let it.
    """
    import litellm

    mock_router = MagicMock()
    mock_router.acompletion = AsyncMock(
        side_effect=litellm.RateLimitError(
            message="rate limit",
            llm_provider="openai",
            model="gpt-4o",
        )
    )

    model = RouterChatModel(router=mock_router)
    with pytest.raises(litellm.RateLimitError):
        await model._agenerate([HumanMessage(content="hi")])

    # The router was called exactly once — internal fallback is the router's job.
    mock_router.acompletion.assert_called_once()


@pytest.mark.asyncio
async def test_router_chat_model_agenerate_returns_fallback_result():
    """When the router returns a response (after internal fallback), RouterChatModel returns it."""
    mock_router = MagicMock()
    mock_router.acompletion = AsyncMock(return_value=_make_litellm_response("fallback-answer"))

    model = RouterChatModel(router=mock_router)
    result = await model._agenerate([HumanMessage(content="hi")])
    assert result.generations[0].message.content == "fallback-answer"


# ---------------------------------------------------------------------------
# NAT WorkflowBuilder smoke tests (all three frameworks)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_router_langchain_via_workflow_builder():
    cfg = _make_router_config()
    async with WorkflowBuilder() as builder:
        await builder.add_llm("datarobot_llm", cfg)
        llm = await builder.get_llm("datarobot_llm", wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    assert isinstance(llm, BaseChatModel)


@pytest.mark.asyncio
async def test_router_crewai_via_workflow_builder():
    cfg = _make_router_config()
    async with WorkflowBuilder() as builder:
        await builder.add_llm("datarobot_llm", cfg)
        llm = await builder.get_llm("datarobot_llm", wrapper_type=LLMFrameworkEnum.CREWAI)
    assert isinstance(llm, LLM)


@pytest.mark.asyncio
async def test_router_llamaindex_via_workflow_builder():
    cfg = _make_router_config()
    async with WorkflowBuilder() as builder:
        await builder.add_llm("datarobot_llm", cfg)
        llm = await builder.get_llm("datarobot_llm", wrapper_type=LLMFrameworkEnum.LLAMA_INDEX)
    assert isinstance(llm, LiteLLM)
