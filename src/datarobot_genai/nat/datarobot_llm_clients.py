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

from crewai import LLM
from langchain_openai import ChatOpenAI
from llama_index.core.base.llms.types import LLMMetadata
from llama_index.llms.litellm import LiteLLM
from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.cli.register_workflow import register_llm_client

from ..nat.datarobot_llm_providers import DataRobotLLMDeploymentModelConfig
from ..nat.datarobot_llm_providers import DataRobotLLMGatewayModelConfig
from ..nat.datarobot_llm_providers import DataRobotNIMModelConfig


class DataRobotLiteLLM(LiteLLM):  # type: ignore[misc]
    """DataRobotLiteLLM is a small LiteLLM wrapper class that makes all LiteLLM endpoints
    compatible with the LlamaIndex library.
    """

    @property
    def metadata(self) -> LLMMetadata:
        """Returns the metadata for the LLM.

        This is required to enable the is_chat_model and is_function_calling_model, which are
        mandatory for LlamaIndex agents. By default, LlamaIndex assumes these are false unless each
        individual model config in LiteLLM explicitly sets them to true. To use custom LLM
        endpoints with LlamaIndex agents, you must override this method to return the appropriate
        metadata.
        """
        return LLMMetadata(
            context_window=128000,
            num_output=self.max_tokens or -1,
            is_chat_model=True,
            is_function_calling_model=True,
            model_name=self.model,
        )


@register_llm_client(
    config_type=DataRobotLLMGatewayModelConfig, wrapper_type=LLMFrameworkEnum.LANGCHAIN
)
async def datarobot_llm_gateway_langchain(
    llm_config: DataRobotLLMGatewayModelConfig, builder: Builder
) -> AsyncGenerator[ChatOpenAI]:
    config = llm_config.model_dump(exclude={"type", "thinking"}, by_alias=True, exclude_none=True)
    config["base_url"] = config["base_url"] + "/genai/llmgw"
    yield ChatOpenAI(**config)


@register_llm_client(
    config_type=DataRobotLLMGatewayModelConfig, wrapper_type=LLMFrameworkEnum.CREWAI
)
async def datarobot_llm_gateway_crewai(
    llm_config: DataRobotLLMGatewayModelConfig, builder: Builder
) -> AsyncGenerator[LLM]:
    config = llm_config.model_dump(exclude={"type", "thinking"}, by_alias=True, exclude_none=True)
    config["model"] = "datarobot/" + config["model"]
    config["base_url"] = config["base_url"].removesuffix("/api/v2")
    yield LLM(**config)


@register_llm_client(
    config_type=DataRobotLLMGatewayModelConfig, wrapper_type=LLMFrameworkEnum.LLAMA_INDEX
)
async def datarobot_llm_gateway_llamaindex(
    llm_config: DataRobotLLMGatewayModelConfig, builder: Builder
) -> AsyncGenerator[LLM]:
    config = llm_config.model_dump(exclude={"type", "thinking"}, by_alias=True, exclude_none=True)
    config["model"] = "datarobot/" + config["model"]
    config["api_base"] = config.pop("base_url").removesuffix("/api/v2")
    yield DataRobotLiteLLM(**config)


@register_llm_client(
    config_type=DataRobotLLMDeploymentModelConfig, wrapper_type=LLMFrameworkEnum.LANGCHAIN
)
async def datarobot_llm_deployment_langchain(
    llm_config: DataRobotLLMDeploymentModelConfig, builder: Builder
) -> AsyncGenerator[ChatOpenAI]:
    yield ChatOpenAI(
        **llm_config.model_dump(
            exclude={"type", "thinking", "datarobot_endpoint", "llm_deployment_id"},
            by_alias=True,
            exclude_none=True,
        )
    )


@register_llm_client(
    config_type=DataRobotLLMDeploymentModelConfig, wrapper_type=LLMFrameworkEnum.CREWAI
)
async def datarobot_llm_deployment_crewai(
    llm_config: DataRobotLLMDeploymentModelConfig, builder: Builder
) -> AsyncGenerator[LLM]:
    config = llm_config.model_dump(
        exclude={"type", "thinking"},
        by_alias=True,
        exclude_none=True,
    )
    config["model"] = "datarobot/" + config["model"]
    config["api_base"] = config.pop("base_url") + "/chat/completions"
    yield LLM(**config)


@register_llm_client(
    config_type=DataRobotLLMDeploymentModelConfig, wrapper_type=LLMFrameworkEnum.LLAMA_INDEX
)
async def datarobot_llm_deployment_llamaindex(
    llm_config: DataRobotLLMDeploymentModelConfig, builder: Builder
) -> AsyncGenerator[LLM]:
    config = llm_config.model_dump(
        exclude={"type", "thinking"},
        by_alias=True,
        exclude_none=True,
    )
    config["model"] = "datarobot/" + config["model"]
    config["api_base"] = config.pop("base_url") + "/chat/completions"
    yield DataRobotLiteLLM(**config)


@register_llm_client(config_type=DataRobotNIMModelConfig, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
async def datarobot_nim_langchain(
    llm_config: DataRobotNIMModelConfig, builder: Builder
) -> AsyncGenerator[ChatOpenAI]:
    yield ChatOpenAI(
        **llm_config.model_dump(
            exclude={"type", "thinking"},
            by_alias=True,
            exclude_none=True,
        )
    )


@register_llm_client(config_type=DataRobotNIMModelConfig, wrapper_type=LLMFrameworkEnum.CREWAI)
async def datarobot_nim_crewai(
    llm_config: DataRobotNIMModelConfig, builder: Builder
) -> AsyncGenerator[LLM]:
    config = llm_config.model_dump(
        exclude={"type", "thinking", "max_retries"},
        by_alias=True,
        exclude_none=True,
    )
    config["model"] = "datarobot/" + config["model"]
    config["api_base"] = config.pop("base_url") + "/chat/completions"
    yield LLM(**config)


@register_llm_client(config_type=DataRobotNIMModelConfig, wrapper_type=LLMFrameworkEnum.LLAMA_INDEX)
async def datarobot_nim_llamaindex(
    llm_config: DataRobotNIMModelConfig, builder: Builder
) -> AsyncGenerator[LLM]:
    config = llm_config.model_dump(
        exclude={"type", "thinking"},
        by_alias=True,
        exclude_none=True,
    )
    config["model"] = "datarobot/" + config["model"]
    config["api_base"] = config.pop("base_url") + "/chat/completions"
    yield DataRobotLiteLLM(**config)
