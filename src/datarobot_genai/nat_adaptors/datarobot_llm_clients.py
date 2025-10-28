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

from collections.abc import AsyncGenerator

from langchain_openai import ChatOpenAI
from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.cli.register_workflow import register_llm_client

from ..nat_adaptors.datarobot_llm_providers import DataRobotModelConfig


@register_llm_client(config_type=DataRobotModelConfig, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
async def datarobot_langchain(
    llm_config: DataRobotModelConfig, builder: Builder
) -> AsyncGenerator[ChatOpenAI]:
    yield ChatOpenAI(
        **llm_config.model_dump(exclude={"type", "thinking", "datarobot_endpoint"}, by_alias=True)
    )
