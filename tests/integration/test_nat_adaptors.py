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

from langchain_core.messages.ai import AIMessage
from langchain_core.prompts.chat import ChatPromptTemplate
from llama_index.core.llms import ChatMessage
from llama_index.core.llms import ChatResponse
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.workflow_builder import WorkflowBuilder

from datarobot_genai.nat_adaptors.datarobot_llm_clients import (
    datarobot_llm_gateway_langchain,  # noqa: F401
)
from datarobot_genai.nat_adaptors.datarobot_llm_providers import DataRobotLLMGatewayModelConfig


async def test_datarobot_llm_gateway_langchain():
    prompt = ChatPromptTemplate.from_messages(
        [("system", "You are a helpful AI assistant."), ("human", "{input}")]
    )

    llm_config = DataRobotLLMGatewayModelConfig(
        model_name="azure/gpt-4o-2024-11-20", temperature=0.0
    )

    async with WorkflowBuilder() as builder:
        await builder.add_llm("datarobot_llm", llm_config)
        llm = await builder.get_llm("datarobot_llm", wrapper_type=LLMFrameworkEnum.LANGCHAIN)
        agent = prompt | llm
        response = await agent.ainvoke({"input": "What is 1+2?"})
        assert isinstance(response, AIMessage)
        assert response.content is not None
        assert isinstance(response.content, str)
        assert "3" in response.content.lower()


async def test_datarobot_llm_gateway_crewai():
    input = "What is 1+2?"
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": f"{input}"},
    ]

    llm_config = DataRobotLLMGatewayModelConfig(
        model_name="azure/gpt-4o-2024-11-20", temperature=0.0
    )

    async with WorkflowBuilder() as builder:
        await builder.add_llm("datarobot_llm", llm_config)
        llm = await builder.get_llm("datarobot_llm", wrapper_type=LLMFrameworkEnum.CREWAI)
        response = llm.call(messages)
        assert isinstance(response, str)
        assert response is not None
        assert "3" in response


async def test_datarobot_llm_gateway_llamaindex():
    input = "What is 1+2?"
    messages = [
        ChatMessage.from_str("You are a helpful AI assistant.", "system"),
        ChatMessage.from_str(input, "user"),
    ]

    llm_config = DataRobotLLMGatewayModelConfig(
        model_name="azure/gpt-4o-2024-11-20", temperature=0.0
    )

    async with WorkflowBuilder() as builder:
        await builder.add_llm("datarobot_llm", llm_config)
        llm = await builder.get_llm("datarobot_llm", wrapper_type=LLMFrameworkEnum.LLAMA_INDEX)
        response = await llm.achat(messages)
        assert isinstance(response, ChatResponse)
        assert response is not None
        assert "3" in response.message.content
