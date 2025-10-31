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

# ruff: noqa: E402
import sys

import pytest

# Skip the entire module if the Python version is 3.10, nat is not available
pytestmark = pytest.mark.skipif(
    sys.version_info.major == 3 and sys.version_info.minor == 10,
    reason="NAT is not available for Python 3.10",
)

from crewai import LLM
from langchain_openai import ChatOpenAI
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.workflow_builder import WorkflowBuilder

from datarobot_genai.nat.datarobot_llm_clients import DataRobotLiteLLM
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
        assert isinstance(llm, DataRobotLiteLLM)


async def test_datarobot_llm_deployment_langchain():
    llm_config = DataRobotLLMDeploymentModelConfig(temperature=0.0, api_key="some_token")
    async with WorkflowBuilder() as builder:
        await builder.add_llm("datarobot_llm", llm_config)
        llm = await builder.get_llm("datarobot_llm", wrapper_type=LLMFrameworkEnum.LANGCHAIN)
        assert isinstance(llm, ChatOpenAI)


async def test_datarobot_llm_deployment_crewai():
    llm_config = DataRobotLLMDeploymentModelConfig(temperature=0.0, api_key="some_token")
    async with WorkflowBuilder() as builder:
        await builder.add_llm("datarobot_llm", llm_config)
        llm = await builder.get_llm("datarobot_llm", wrapper_type=LLMFrameworkEnum.CREWAI)
        assert isinstance(llm, LLM)


async def test_datarobot_llm_deployment_llamaindex():
    llm_config = DataRobotLLMDeploymentModelConfig(temperature=0.0, api_key="some_token")
    async with WorkflowBuilder() as builder:
        await builder.add_llm("datarobot_llm", llm_config)
        llm = await builder.get_llm("datarobot_llm", wrapper_type=LLMFrameworkEnum.LLAMA_INDEX)
        assert isinstance(llm, DataRobotLiteLLM)


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
        assert isinstance(llm, DataRobotLiteLLM)
