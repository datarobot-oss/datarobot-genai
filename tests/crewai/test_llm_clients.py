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

import os
from unittest.mock import patch

import pytest
from crewai import LLM
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.workflow_builder import WorkflowBuilder

import datarobot_genai.crewai.llm_clients  # noqa: F401
from datarobot_genai.core.config import LLMType
from datarobot_genai.dragent.plugins.llm_providers import DataRobotLitellmConfig
from datarobot_genai.dragent.plugins.llm_providers import DataRobotLLMComponentModelConfig
from datarobot_genai.dragent.plugins.llm_providers import DataRobotLLMDeploymentModelConfig
from datarobot_genai.dragent.plugins.llm_providers import DataRobotLLMGatewayModelConfig
from datarobot_genai.dragent.plugins.llm_providers import DataRobotNIMModelConfig


async def test_datarobot_llm_gateway_crewai():
    llm_config = DataRobotLLMGatewayModelConfig(
        model_name="azure/gpt-4o-2024-11-20", temperature=0.0, api_key="some_token"
    )
    async with WorkflowBuilder() as builder:
        await builder.add_llm("datarobot_llm", llm_config)
        llm = await builder.get_llm("datarobot_llm", wrapper_type=LLMFrameworkEnum.CREWAI)
        assert isinstance(llm, LLM)


async def test_datarobot_llm_deployment_crewai():
    llm_config = DataRobotLLMDeploymentModelConfig(
        llm_deployment_id="123", temperature=0.0, api_key="some_token"
    )
    async with WorkflowBuilder() as builder:
        await builder.add_llm("datarobot_llm", llm_config)
        llm = await builder.get_llm("datarobot_llm", wrapper_type=LLMFrameworkEnum.CREWAI)
        assert isinstance(llm, LLM)


async def test_datarobot_llm_deployment_crewai_with_identity_token():
    llm_config = DataRobotLLMDeploymentModelConfig(
        llm_deployment_id="123",
        temperature=0.0,
        api_key="some_token",
    )
    with patch(
        "datarobot_genai.crewai.llm_clients.extract_headers_from_context",
        return_value={"X-DataRobot-Identity-Token": "identity-token-123"},
    ):
        async with WorkflowBuilder() as builder:
            await builder.add_llm("datarobot_llm", llm_config)
            llm = await builder.get_llm("datarobot_llm", wrapper_type=LLMFrameworkEnum.CREWAI)
            assert isinstance(llm, LLM)
            assert llm.additional_params == {
                "max_retries": 10,
                "extra_headers": {"X-DataRobot-Identity-Token": "identity-token-123"},
                "stream_options": {"include_usage": True},
            }


async def test_datarobot_nim_crewai():
    llm_config = DataRobotNIMModelConfig(
        nim_deployment_id="123",
        model_name="azure/gpt-4o-2024-11-20",
        temperature=0.0,
        api_key="some_token",
    )
    async with WorkflowBuilder() as builder:
        await builder.add_llm("datarobot_llm", llm_config)
        llm = await builder.get_llm("datarobot_llm", wrapper_type=LLMFrameworkEnum.CREWAI)
        assert isinstance(llm, LLM)


@pytest.mark.parametrize(
    "use_datarobot_llm_gateway,llm_deployment_id,nim_deployment_id",
    [
        pytest.param(True, None, None, id="llm_gateway"),
        pytest.param(False, "123", None, id="llm_deployment"),
        pytest.param(False, None, "123", id="nim_deployment"),
        pytest.param(False, None, None, id="external"),
    ],
)
async def test_datarobot_llm_component_crewai(
    use_datarobot_llm_gateway, llm_deployment_id, nim_deployment_id
):
    with patch.dict(
        os.environ,
        {
            "DATAROBOT_ENDPOINT": "https://app.datarobot.com/api/v2",
            "DATAROBOT_API_TOKEN": "some_token",
        },
        clear=False,
    ):
        llm_config = DataRobotLLMComponentModelConfig(
            model_name="anthropic/claude-3",
            use_datarobot_llm_gateway=use_datarobot_llm_gateway,
            llm_deployment_id=llm_deployment_id,
            nim_deployment_id=nim_deployment_id,
            temperature=0.2,
        )
        identity_patch = patch(
            "datarobot_genai.crewai.llm_clients.extract_headers_from_context",
            return_value={"X-DataRobot-Identity-Token": "identity-token-123"},
        )
        with identity_patch:
            async with WorkflowBuilder() as builder:
                await builder.add_llm("datarobot_llm", llm_config)
                llm = await builder.get_llm("datarobot_llm", wrapper_type=LLMFrameworkEnum.CREWAI)

        assert isinstance(llm, LLM)
        assert llm.is_litellm is True
        assert llm.temperature == 0.2

        if llm_config.get_llm_type() == LLMType.GATEWAY:
            assert llm.model == "datarobot/anthropic/claude-3"
            assert llm.api_base == "https://app.datarobot.com"
            assert llm.api_key == "some_token"
            assert llm.additional_params == {
                "max_retries": 10,
                "stream_options": {"include_usage": True},
            }
        elif llm_config.get_llm_type() == LLMType.DEPLOYMENT:
            assert llm.model == "datarobot/anthropic/claude-3"
            assert (
                llm.api_base == "https://app.datarobot.com/api/v2/deployments/123/chat/completions"
            )
            assert llm.api_key == "some_token"
            assert llm.additional_params == {
                "max_retries": 10,
                "stream_options": {"include_usage": True},
                "extra_headers": {"X-DataRobot-Identity-Token": "identity-token-123"},
            }
        elif llm_config.get_llm_type() == LLMType.NIM:
            assert llm.model == "datarobot/anthropic/claude-3"
            assert (
                llm.api_base == "https://app.datarobot.com/api/v2/deployments/123/chat/completions"
            )
            assert llm.api_key == "some_token"
            assert llm.additional_params == {
                "max_retries": 10,
                "stream_options": {"include_usage": True},
            }
        elif llm_config.get_llm_type() == LLMType.EXTERNAL:
            assert llm.model == "anthropic/claude-3"
            assert llm.api_base is None
            assert llm.api_key is None
            assert llm.additional_params == {
                "max_retries": 10,
                "stream_options": {"include_usage": True},
            }
        else:
            raise ValueError(f"Invalid LLM type inferred from config: {llm_config.get_llm_type()}")


async def test_litellm_crewai():
    llm_config = DataRobotLitellmConfig(model_name="openai/gpt-4o", api_key="test-key")
    async with WorkflowBuilder() as builder:
        await builder.add_llm("ext_llm", llm_config)
        llm = await builder.get_llm("ext_llm", wrapper_type=LLMFrameworkEnum.CREWAI)
        assert isinstance(llm, LLM)
        assert llm.model == "openai/gpt-4o"


async def test_litellm_crewai_with_temperature():
    llm_config = DataRobotLitellmConfig(
        model_name="openai/gpt-4o", api_key="test-key", temperature=0.3
    )
    async with WorkflowBuilder() as builder:
        await builder.add_llm("ext_llm", llm_config)
        llm = await builder.get_llm("ext_llm", wrapper_type=LLMFrameworkEnum.CREWAI)
        assert isinstance(llm, LLM)
        assert llm.temperature == 0.3


async def test_litellm_crewai_only_passes_set_fields():
    """Ensure only explicitly set fields are forwarded (exclude_unset=True)."""
    llm_config = DataRobotLitellmConfig(model_name="anthropic/claude-3", api_key="key-abc")
    async with WorkflowBuilder() as builder:
        await builder.add_llm("ext_llm", llm_config)
        llm = await builder.get_llm("ext_llm", wrapper_type=LLMFrameworkEnum.CREWAI)
        assert isinstance(llm, LLM)
        assert llm.model == "anthropic/claude-3"
        assert llm.api_key == "key-abc"
