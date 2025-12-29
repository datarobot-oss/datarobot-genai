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
from typing import Any
from typing import TypeVar

from crewai import LLM
from langchain_openai import ChatOpenAI
from llama_index.core.base.llms.types import LLMMetadata
from llama_index.llms.litellm import LiteLLM
from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.cli.register_workflow import register_llm_client
from nat.data_models.llm import LLMBaseConfig
from nat.data_models.retry_mixin import RetryMixin
from nat.plugins.langchain.llm import (
    _patch_llm_based_on_config as langchain_patch_llm_based_on_config,
)
from nat.utils.exception_handlers.automatic_retries import patch_with_retry

from ..nat.datarobot_llm_providers import DataRobotLLMComponentModelConfig
from ..nat.datarobot_llm_providers import DataRobotLLMDeploymentModelConfig
from ..nat.datarobot_llm_providers import DataRobotLLMGatewayModelConfig
from ..nat.datarobot_llm_providers import DataRobotNIMModelConfig

ModelType = TypeVar("ModelType")


def _patch_llm_based_on_config(client: ModelType, llm_config: LLMBaseConfig) -> ModelType:
    if isinstance(llm_config, RetryMixin):
        client = patch_with_retry(
            client,
            retries=llm_config.num_retries,
            retry_codes=llm_config.retry_on_status_codes,
            retry_on_messages=llm_config.retry_on_errors,
        )

    return client


class DataRobotChatOpenAI(ChatOpenAI):
    def _get_request_payload(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> dict:
        # We need to default to include_usage=True for streaming but we get 400 response
        # if stream_options is present for a non-streaming call.
        payload = super()._get_request_payload(*args, **kwargs)
        if not payload.get("stream"):
            payload.pop("stream_options", None)
        return payload


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
    config["stream_options"] = {"include_usage": True}
    config["model"] = config["model"].removeprefix("datarobot/")
    client = DataRobotChatOpenAI(**config)
    yield langchain_patch_llm_based_on_config(client, config)


@register_llm_client(
    config_type=DataRobotLLMGatewayModelConfig, wrapper_type=LLMFrameworkEnum.CREWAI
)
async def datarobot_llm_gateway_crewai(
    llm_config: DataRobotLLMGatewayModelConfig, builder: Builder
) -> AsyncGenerator[LLM]:
    config = llm_config.model_dump(exclude={"type", "thinking"}, by_alias=True, exclude_none=True)
    if not config["model"].startswith("datarobot/"):
        config["model"] = "datarobot/" + config["model"]
    config["base_url"] = config["base_url"].removesuffix("/api/v2")
    client = LLM(**config)
    yield _patch_llm_based_on_config(client, config)


@register_llm_client(
    config_type=DataRobotLLMGatewayModelConfig, wrapper_type=LLMFrameworkEnum.LLAMA_INDEX
)
async def datarobot_llm_gateway_llamaindex(
    llm_config: DataRobotLLMGatewayModelConfig, builder: Builder
) -> AsyncGenerator[LLM]:
    config = llm_config.model_dump(exclude={"type", "thinking"}, by_alias=True, exclude_none=True)
    if not config["model"].startswith("datarobot/"):
        config["model"] = "datarobot/" + config["model"]
    config["api_base"] = config.pop("base_url").removesuffix("/api/v2")
    client = DataRobotLiteLLM(**config)
    yield _patch_llm_based_on_config(client, config)


@register_llm_client(
    config_type=DataRobotLLMDeploymentModelConfig, wrapper_type=LLMFrameworkEnum.LANGCHAIN
)
async def datarobot_llm_deployment_langchain(
    llm_config: DataRobotLLMDeploymentModelConfig, builder: Builder
) -> AsyncGenerator[ChatOpenAI]:
    config = llm_config.model_dump(
        exclude={"type", "thinking"},
        by_alias=True,
        exclude_none=True,
    )
    config["stream_options"] = {"include_usage": True}
    config["model"] = config["model"].removeprefix("datarobot/")
    client = DataRobotChatOpenAI(**config)
    yield langchain_patch_llm_based_on_config(client, config)


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
    if not config["model"].startswith("datarobot/"):
        config["model"] = "datarobot/" + config["model"]
    config["api_base"] = config.pop("base_url") + "/chat/completions"
    client = LLM(**config)
    yield _patch_llm_based_on_config(client, config)


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
    if not config["model"].startswith("datarobot/"):
        config["model"] = "datarobot/" + config["model"]
    config["api_base"] = config.pop("base_url") + "/chat/completions"
    client = DataRobotLiteLLM(**config)
    yield _patch_llm_based_on_config(client, config)


@register_llm_client(config_type=DataRobotNIMModelConfig, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
async def datarobot_nim_langchain(
    llm_config: DataRobotNIMModelConfig, builder: Builder
) -> AsyncGenerator[ChatOpenAI]:
    config = llm_config.model_dump(
        exclude={"type", "thinking"},
        by_alias=True,
        exclude_none=True,
    )
    config["stream_options"] = {"include_usage": True}
    config["model"] = config["model"].removeprefix("datarobot/")
    client = DataRobotChatOpenAI(**config)
    yield langchain_patch_llm_based_on_config(client, config)


@register_llm_client(config_type=DataRobotNIMModelConfig, wrapper_type=LLMFrameworkEnum.CREWAI)
async def datarobot_nim_crewai(
    llm_config: DataRobotNIMModelConfig, builder: Builder
) -> AsyncGenerator[LLM]:
    config = llm_config.model_dump(
        exclude={"type", "thinking", "max_retries"},
        by_alias=True,
        exclude_none=True,
    )
    if not config["model"].startswith("datarobot/"):
        config["model"] = "datarobot/" + config["model"]
    config["api_base"] = config.pop("base_url") + "/chat/completions"
    client = LLM(**config)
    yield _patch_llm_based_on_config(client, config)


@register_llm_client(config_type=DataRobotNIMModelConfig, wrapper_type=LLMFrameworkEnum.LLAMA_INDEX)
async def datarobot_nim_llamaindex(
    llm_config: DataRobotNIMModelConfig, builder: Builder
) -> AsyncGenerator[LLM]:
    config = llm_config.model_dump(
        exclude={"type", "thinking"},
        by_alias=True,
        exclude_none=True,
    )
    if not config["model"].startswith("datarobot/"):
        config["model"] = "datarobot/" + config["model"]
    config["api_base"] = config.pop("base_url") + "/chat/completions"
    client = DataRobotLiteLLM(**config)
    yield _patch_llm_based_on_config(client, config)


@register_llm_client(
    config_type=DataRobotLLMComponentModelConfig, wrapper_type=LLMFrameworkEnum.LANGCHAIN
)
async def datarobot_llm_component_langchain(
    llm_config: DataRobotLLMComponentModelConfig, builder: Builder
) -> AsyncGenerator[ChatOpenAI]:
    config = llm_config.model_dump(exclude={"type", "thinking"}, by_alias=True, exclude_none=True)
    if config["use_datarobot_llm_gateway"]:
        config["base_url"] = config["base_url"] + "/genai/llmgw"
    config["stream_options"] = {"include_usage": True}
    config["model"] = config["model"].removeprefix("datarobot/")
    config.pop("use_datarobot_llm_gateway")
    client = DataRobotChatOpenAI(**config)
    yield langchain_patch_llm_based_on_config(client, config)


@register_llm_client(
    config_type=DataRobotLLMComponentModelConfig, wrapper_type=LLMFrameworkEnum.CREWAI
)
async def datarobot_llm_component_crewai(
    llm_config: DataRobotLLMComponentModelConfig, builder: Builder
) -> AsyncGenerator[LLM]:
    config = llm_config.model_dump(exclude={"type", "thinking"}, by_alias=True, exclude_none=True)
    if not config["model"].startswith("datarobot/"):
        config["model"] = "datarobot/" + config["model"]
    if config["use_datarobot_llm_gateway"]:
        config["base_url"] = config["base_url"].removesuffix("/api/v2")
    else:
        config["api_base"] = config.pop("base_url") + "/chat/completions"
    config.pop("use_datarobot_llm_gateway")
    client = LLM(**config)
    yield _patch_llm_based_on_config(client, config)


@register_llm_client(
    config_type=DataRobotLLMComponentModelConfig, wrapper_type=LLMFrameworkEnum.LLAMA_INDEX
)
async def datarobot_llm_component_llamaindex(
    llm_config: DataRobotLLMComponentModelConfig, builder: Builder
) -> AsyncGenerator[LLM]:
    config = llm_config.model_dump(exclude={"type", "thinking"}, by_alias=True, exclude_none=True)
    if not config["model"].startswith("datarobot/"):
        config["model"] = "datarobot/" + config["model"]
    if config["use_datarobot_llm_gateway"]:
        config["api_base"] = config.pop("base_url").removesuffix("/api/v2")
    else:
        config["api_base"] = config.pop("base_url") + "/chat/completions"
    config.pop("use_datarobot_llm_gateway")
    client = DataRobotLiteLLM(**config)
    yield _patch_llm_based_on_config(client, config)
