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

import pytest
from crewai import LLM
from langchain_openai import ChatOpenAI
from llama_index.llms.litellm import LiteLLM
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.workflow_builder import WorkflowBuilder

import datarobot_genai.nat.datarobot_llm_clients  # noqa: F401
from datarobot_genai.nat.datarobot_llm_providers import DataRobotLLMComponentModelConfig
from datarobot_genai.nat.datarobot_llm_providers import DataRobotLLMDeploymentModelConfig
from datarobot_genai.nat.datarobot_llm_providers import DataRobotLLMGatewayModelConfig
from datarobot_genai.nat.datarobot_llm_providers import DataRobotNIMModelConfig


async def test_datarobot_llm_gateway_langchain():
    llm_config = DataRobotLLMGatewayModelConfig(
        model_name="azure/gpt-4o-2024-11-20", temperature=0.0, api_key="some_token"
    )
    async with WorkflowBuilder() as builder:
        await builder.add_llm("datarobot_llm", llm_config)
        llm = await builder.get_llm("datarobot_llm", wrapper_type=LLMFrameworkEnum.LANGCHAIN)
        assert isinstance(llm, ChatOpenAI)


async def test_datarobot_llm_gateway_crewai():
    llm_config = DataRobotLLMGatewayModelConfig(
        model_name="azure/gpt-4o-2024-11-20", temperature=0.0, api_key="some_token"
    )
    async with WorkflowBuilder() as builder:
        await builder.add_llm("datarobot_llm", llm_config)
        llm = await builder.get_llm("datarobot_llm", wrapper_type=LLMFrameworkEnum.CREWAI)
        assert isinstance(llm, LLM)


async def test_datarobot_llm_gateway_llamaindex():
    llm_config = DataRobotLLMGatewayModelConfig(
        model_name="azure/gpt-4o-2024-11-20", temperature=0.0, api_key="some_token"
    )
    async with WorkflowBuilder() as builder:
        await builder.add_llm("datarobot_llm", llm_config)
        llm = await builder.get_llm("datarobot_llm", wrapper_type=LLMFrameworkEnum.LLAMA_INDEX)
        assert isinstance(llm, LiteLLM)


async def test_datarobot_llm_deployment_langchain():
    llm_config = DataRobotLLMDeploymentModelConfig(temperature=0.0, api_key="some_token")
    async with WorkflowBuilder() as builder:
        await builder.add_llm("datarobot_llm", llm_config)
        llm = await builder.get_llm("datarobot_llm", wrapper_type=LLMFrameworkEnum.LANGCHAIN)
        assert isinstance(llm, ChatOpenAI)


async def test_datarobot_llm_deployment_langchain_with_identity_token():
    llm_config = DataRobotLLMDeploymentModelConfig(
        temperature=0.0,
        api_key="some_token",
        headers={"X-DataRobot-Identity-Token": "identity-token-123"},
    )
    async with WorkflowBuilder() as builder:
        await builder.add_llm("datarobot_llm", llm_config)
        llm = await builder.get_llm("datarobot_llm", wrapper_type=LLMFrameworkEnum.LANGCHAIN)
        assert isinstance(llm, ChatOpenAI)
        assert llm.default_headers == {"X-DataRobot-Identity-Token": "identity-token-123"}


async def test_datarobot_llm_deployment_crewai():
    llm_config = DataRobotLLMDeploymentModelConfig(temperature=0.0, api_key="some_token")
    async with WorkflowBuilder() as builder:
        await builder.add_llm("datarobot_llm", llm_config)
        llm = await builder.get_llm("datarobot_llm", wrapper_type=LLMFrameworkEnum.CREWAI)
        assert isinstance(llm, LLM)


async def test_datarobot_llm_deployment_crewai_with_identity_token():
    llm_config = DataRobotLLMDeploymentModelConfig(
        temperature=0.0,
        api_key="some_token",
        headers={"X-DataRobot-Identity-Token": "identity-token-123"},
    )
    async with WorkflowBuilder() as builder:
        await builder.add_llm("datarobot_llm", llm_config)
        llm = await builder.get_llm("datarobot_llm", wrapper_type=LLMFrameworkEnum.CREWAI)
        assert isinstance(llm, LLM)
        assert llm.additional_params == {
            "max_retries": 10,
            "extra_headers": {"X-DataRobot-Identity-Token": "identity-token-123"},
        }


async def test_datarobot_llm_deployment_llamaindex():
    llm_config = DataRobotLLMDeploymentModelConfig(temperature=0.0, api_key="some_token")
    async with WorkflowBuilder() as builder:
        await builder.add_llm("datarobot_llm", llm_config)
        llm = await builder.get_llm("datarobot_llm", wrapper_type=LLMFrameworkEnum.LLAMA_INDEX)
        assert isinstance(llm, LiteLLM)


async def test_datarobot_llm_deployment_llamaindex_with_identity_token():
    llm_config = DataRobotLLMDeploymentModelConfig(
        temperature=0.0,
        api_key="some_token",
        headers={"X-DataRobot-Identity-Token": "identity-token-123"},
    )
    async with WorkflowBuilder() as builder:
        await builder.add_llm("datarobot_llm", llm_config)
        llm = await builder.get_llm("datarobot_llm", wrapper_type=LLMFrameworkEnum.LLAMA_INDEX)
        assert isinstance(llm, LiteLLM)
        assert llm.additional_kwargs == {
            "api_base": "https://app.datarobot.com/api/v2/deployments/None/chat/completions",
            "api_key": "some_token",
            "extra_headers": {"X-DataRobot-Identity-Token": "identity-token-123"},
        }


async def test_datarobot_nim_langchain():
    llm_config = DataRobotNIMModelConfig(temperature=0.0, api_key="some_token")
    async with WorkflowBuilder() as builder:
        await builder.add_llm("datarobot_llm", llm_config)
        llm = await builder.get_llm("datarobot_llm", wrapper_type=LLMFrameworkEnum.LANGCHAIN)
        assert isinstance(llm, ChatOpenAI)


async def test_datarobot_nim_crewai():
    llm_config = DataRobotNIMModelConfig(temperature=0.0, api_key="some_token")
    async with WorkflowBuilder() as builder:
        await builder.add_llm("datarobot_llm", llm_config)
        llm = await builder.get_llm("datarobot_llm", wrapper_type=LLMFrameworkEnum.CREWAI)
        assert isinstance(llm, LLM)


async def test_datarobot_nim_llamaindex():
    llm_config = DataRobotNIMModelConfig(temperature=0.0, api_key="some_token")
    async with WorkflowBuilder() as builder:
        await builder.add_llm("datarobot_llm", llm_config)
        llm = await builder.get_llm("datarobot_llm", wrapper_type=LLMFrameworkEnum.LLAMA_INDEX)
        assert isinstance(llm, LiteLLM)


async def test_datarobot_llm_component_langchain_use_gateway():
    llm_config = DataRobotLLMComponentModelConfig(api_key="some_token")
    async with WorkflowBuilder() as builder:
        await builder.add_llm("datarobot_llm", llm_config)
        llm = await builder.get_llm("datarobot_llm", wrapper_type=LLMFrameworkEnum.LANGCHAIN)
        assert isinstance(llm, ChatOpenAI)


@pytest.mark.parametrize("use_datarobot_llm_gateway", [True, False])
async def test_datarobot_llm_component_langchain_with_identity_token(use_datarobot_llm_gateway):
    llm_config = DataRobotLLMComponentModelConfig(
        api_key="some_token",
        use_datarobot_llm_gateway=use_datarobot_llm_gateway,
        headers={"X-DataRobot-Identity-Token": "identity-token-123"},
    )
    async with WorkflowBuilder() as builder:
        await builder.add_llm("datarobot_llm", llm_config)
        llm = await builder.get_llm("datarobot_llm", wrapper_type=LLMFrameworkEnum.LANGCHAIN)
        assert isinstance(llm, ChatOpenAI)

        if use_datarobot_llm_gateway:
            assert llm.default_headers is None
        else:
            assert llm.default_headers == {"X-DataRobot-Identity-Token": "identity-token-123"}


async def test_datarobot_llm_component_crewai():
    llm_config = DataRobotLLMComponentModelConfig(api_key="some_token")
    async with WorkflowBuilder() as builder:
        await builder.add_llm("datarobot_llm", llm_config)
        llm = await builder.get_llm("datarobot_llm", wrapper_type=LLMFrameworkEnum.CREWAI)
        assert isinstance(llm, LLM)


@pytest.mark.parametrize("use_datarobot_llm_gateway", [True, False])
async def test_datarobot_llm_component_crewai_with_identity_token(use_datarobot_llm_gateway):
    llm_config = DataRobotLLMComponentModelConfig(
        api_key="some_token",
        use_datarobot_llm_gateway=use_datarobot_llm_gateway,
        headers={"X-DataRobot-Identity-Token": "identity-token-123"},
    )
    async with WorkflowBuilder() as builder:
        await builder.add_llm("datarobot_llm", llm_config)
        llm = await builder.get_llm("datarobot_llm", wrapper_type=LLMFrameworkEnum.CREWAI)
        assert isinstance(llm, LLM)

        if use_datarobot_llm_gateway:
            assert llm.additional_params == {
                "max_retries": 10,
            }
        else:
            assert llm.additional_params == {
                "max_retries": 10,
                "extra_headers": {"X-DataRobot-Identity-Token": "identity-token-123"},
            }


async def test_datarobot_llm_component_llamaindex():
    llm_config = DataRobotLLMComponentModelConfig(api_key="some_token")
    async with WorkflowBuilder() as builder:
        await builder.add_llm("datarobot_llm", llm_config)
        llm = await builder.get_llm("datarobot_llm", wrapper_type=LLMFrameworkEnum.LLAMA_INDEX)
        assert isinstance(llm, LiteLLM)


@pytest.mark.parametrize("use_datarobot_llm_gateway", [True, False])
async def test_datarobot_llm_component_llamaindex_with_identity_token(use_datarobot_llm_gateway):
    llm_config = DataRobotLLMComponentModelConfig(
        api_key="some_token",
        use_datarobot_llm_gateway=use_datarobot_llm_gateway,
        headers={"X-DataRobot-Identity-Token": "identity-token-123"},
    )
    async with WorkflowBuilder() as builder:
        await builder.add_llm("datarobot_llm", llm_config)
        llm = await builder.get_llm("datarobot_llm", wrapper_type=LLMFrameworkEnum.LLAMA_INDEX)
        assert isinstance(llm, LiteLLM)

        if use_datarobot_llm_gateway:
            assert llm.additional_kwargs == {
                "api_base": "https://app.datarobot.com/api/v2/deployments/None",
                "api_key": "some_token",
            }
        else:
            assert llm.additional_kwargs == {
                "api_base": "https://app.datarobot.com/api/v2/deployments/None/chat/completions",
                "api_key": "some_token",
                "extra_headers": {"X-DataRobot-Identity-Token": "identity-token-123"},
            }
