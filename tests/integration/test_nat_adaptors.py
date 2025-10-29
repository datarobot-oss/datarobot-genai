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
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.workflow_builder import WorkflowBuilder

from datarobot_genai.nat_adaptors.datarobot_llm_providers import DataRobotLLMGatewayModelConfig


async def test_datarobot_langchain_agent():
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
