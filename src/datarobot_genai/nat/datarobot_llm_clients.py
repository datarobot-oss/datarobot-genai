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

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING
from typing import Any
from typing import TypeVar

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.cli.register_workflow import register_llm_client
from nat.data_models.llm import LLMBaseConfig
from nat.data_models.retry_mixin import RetryMixin
from nat.utils.exception_handlers.automatic_retries import patch_with_retry
from nat.utils.responses_api import validate_no_responses_api

from datarobot_genai.nat.helpers import extract_headers_from_context

from ..nat.datarobot_llm_providers import DataRobotLLMComponentModelConfig
from ..nat.datarobot_llm_providers import DataRobotLLMDeploymentModelConfig
from ..nat.datarobot_llm_providers import DataRobotLLMGatewayModelConfig
from ..nat.datarobot_llm_providers import DataRobotNIMModelConfig

if TYPE_CHECKING:
    from crewai import LLM
    from langchain_openai import ChatOpenAI
    from llama_index.llms.litellm import LiteLLM

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


def _create_datarobot_litellm(config: dict[str, Any]) -> Any:
    from llama_index.core.base.llms.types import LLMMetadata  # noqa: PLC0415
    from llama_index.llms.litellm import LiteLLM  # noqa: PLC0415

    class DataRobotLiteLLM(LiteLLM):  # type: ignore[misc]
        """DataRobotLiteLLM is a small LiteLLM wrapper class that makes all LiteLLM endpoints
        compatible with the LlamaIndex library.
        """

        @property
        def metadata(self) -> LLMMetadata:
            """Returns the metadata for the LLM.

            This is required to enable the is_chat_model and is_function_calling_model, which are
            mandatory for LlamaIndex agents. By default, LlamaIndex assumes these are false unless
            each individual model config in LiteLLM explicitly sets them to true. To use custom LLM
            endpoints with LlamaIndex agents, you must override this method to return the
            appropriate metadata.
            """
            return LLMMetadata(
                context_window=128000,
                num_output=self.max_tokens or -1,
                is_chat_model=True,
                is_function_calling_model=True,
                model_name=self.model,
            )

    return DataRobotLiteLLM(**config)


@register_llm_client(
    config_type=DataRobotLLMGatewayModelConfig, wrapper_type=LLMFrameworkEnum.LANGCHAIN
)
async def datarobot_llm_gateway_langchain(
    llm_config: DataRobotLLMGatewayModelConfig, builder: Builder
) -> AsyncGenerator[ChatOpenAI]:
    from nat.plugins.langchain.llm import (  # noqa: PLC0415
        _patch_llm_based_on_config as langchain_patch_llm_based_on_config,
    )

    from datarobot_genai.langgraph.llm import get_datarobot_gateway_llm

    validate_no_responses_api(llm_config, LLMFrameworkEnum.LANGCHAIN)

    config = llm_config.model_dump(
        exclude={"type", "thinking", "api_type"},
        by_alias=True,
        exclude_none=True,
    )
    client = get_datarobot_gateway_llm(config["model"], config)
    yield langchain_patch_llm_based_on_config(client, config)


@register_llm_client(
    config_type=DataRobotLLMGatewayModelConfig, wrapper_type=LLMFrameworkEnum.CREWAI
)
async def datarobot_llm_gateway_crewai(
    llm_config: DataRobotLLMGatewayModelConfig, builder: Builder
) -> AsyncGenerator[LLM]:
    from crewai import LLM  # noqa: PLC0415

    validate_no_responses_api(llm_config, LLMFrameworkEnum.CREWAI)

    config = llm_config.model_dump(
        exclude={"type", "thinking", "api_type"}, by_alias=True, exclude_none=True
    )
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
) -> AsyncGenerator[LiteLLM]:
    validate_no_responses_api(llm_config, LLMFrameworkEnum.LLAMA_INDEX)
    config = llm_config.model_dump(
        exclude={"type", "thinking", "api_type"}, by_alias=True, exclude_none=True
    )
    if not config["model"].startswith("datarobot/"):
        config["model"] = "datarobot/" + config["model"]
    config["api_base"] = config.pop("base_url").removesuffix("/api/v2")
    client = _create_datarobot_litellm(config)
    yield _patch_llm_based_on_config(client, config)


@register_llm_client(
    config_type=DataRobotLLMDeploymentModelConfig, wrapper_type=LLMFrameworkEnum.LANGCHAIN
)
async def datarobot_llm_deployment_langchain(
    llm_config: DataRobotLLMDeploymentModelConfig, builder: Builder
) -> AsyncGenerator[ChatOpenAI]:
    from nat.plugins.langchain.llm import (  # noqa: PLC0415
        _patch_llm_based_on_config as langchain_patch_llm_based_on_config,
    )

    from datarobot_genai.langgraph.llm import get_datarobot_deployment_llm

    validate_no_responses_api(llm_config, LLMFrameworkEnum.LANGCHAIN)

    config = llm_config.model_dump(
        exclude={"type", "thinking", "headers", "api_type", "llm_deployment_id", "model_name"},
        by_alias=True,
        exclude_none=True,
    )

    context_headers = extract_headers_from_context(["X-DataRobot-Identity-Token"])
    if llm_config.headers:
        context_headers = {**context_headers, **llm_config.headers}

    config["default_headers"] = context_headers

    client = get_datarobot_deployment_llm(
        llm_config.llm_deployment_id, llm_config.model_name, config
    )
    yield langchain_patch_llm_based_on_config(client, config)


@register_llm_client(
    config_type=DataRobotLLMDeploymentModelConfig, wrapper_type=LLMFrameworkEnum.CREWAI
)
async def datarobot_llm_deployment_crewai(
    llm_config: DataRobotLLMDeploymentModelConfig, builder: Builder
) -> AsyncGenerator[LLM]:
    from crewai import LLM  # noqa: PLC0415

    validate_no_responses_api(llm_config, LLMFrameworkEnum.CREWAI)

    config = llm_config.model_dump(
        exclude={"type", "thinking", "headers", "api_type"},
        by_alias=True,
        exclude_none=True,
    )
    if not config["model"].startswith("datarobot/"):
        config["model"] = "datarobot/" + config["model"]
    config["api_base"] = config.pop("base_url") + "/chat/completions"
    if llm_config.headers:
        config["extra_headers"] = llm_config.headers

    client = LLM(**config)
    yield _patch_llm_based_on_config(client, config)


@register_llm_client(
    config_type=DataRobotLLMDeploymentModelConfig, wrapper_type=LLMFrameworkEnum.LLAMA_INDEX
)
async def datarobot_llm_deployment_llamaindex(
    llm_config: DataRobotLLMDeploymentModelConfig, builder: Builder
) -> AsyncGenerator[LiteLLM]:
    validate_no_responses_api(llm_config, LLMFrameworkEnum.LLAMA_INDEX)
    config = llm_config.model_dump(
        exclude={"type", "thinking", "headers", "api_type"},
        by_alias=True,
        exclude_none=True,
    )
    if not config["model"].startswith("datarobot/"):
        config["model"] = "datarobot/" + config["model"]
    config["api_base"] = config.pop("base_url") + "/chat/completions"
    if llm_config.headers:
        config["additional_kwargs"] = {"extra_headers": llm_config.headers}

    client = _create_datarobot_litellm(config)
    yield _patch_llm_based_on_config(client, config)


@register_llm_client(config_type=DataRobotNIMModelConfig, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
async def datarobot_nim_langchain(
    llm_config: DataRobotNIMModelConfig, builder: Builder
) -> AsyncGenerator[ChatOpenAI]:
    from nat.plugins.langchain.llm import (  # noqa: PLC0415
        _patch_llm_based_on_config as langchain_patch_llm_based_on_config,
    )

    from datarobot_genai.langgraph.llm import get_datarobot_nim_llm

    validate_no_responses_api(llm_config, LLMFrameworkEnum.LANGCHAIN)

    config = llm_config.model_dump(
        exclude={"type", "thinking", "api_type", "nim_deployment_id", "model_name"},
        by_alias=True,
        exclude_none=True,
    )
    client = get_datarobot_nim_llm(llm_config.nim_deployment_id, llm_config.model_name, config)
    yield langchain_patch_llm_based_on_config(client, config)


@register_llm_client(config_type=DataRobotNIMModelConfig, wrapper_type=LLMFrameworkEnum.CREWAI)
async def datarobot_nim_crewai(
    llm_config: DataRobotNIMModelConfig, builder: Builder
) -> AsyncGenerator[LLM]:
    from crewai import LLM  # noqa: PLC0415

    validate_no_responses_api(llm_config, LLMFrameworkEnum.CREWAI)

    config = llm_config.model_dump(
        exclude={"type", "thinking", "max_retries", "api_type"},
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
) -> AsyncGenerator[LiteLLM]:
    validate_no_responses_api(llm_config, LLMFrameworkEnum.LLAMA_INDEX)
    config = llm_config.model_dump(
        exclude={"type", "thinking", "api_type"},
        by_alias=True,
        exclude_none=True,
    )
    if not config["model"].startswith("datarobot/"):
        config["model"] = "datarobot/" + config["model"]
    config["api_base"] = config.pop("base_url") + "/chat/completions"
    client = _create_datarobot_litellm(config)
    yield _patch_llm_based_on_config(client, config)


@register_llm_client(
    config_type=DataRobotLLMComponentModelConfig, wrapper_type=LLMFrameworkEnum.LANGCHAIN
)
async def datarobot_llm_component_langchain(
    llm_config: DataRobotLLMComponentModelConfig, builder: Builder
) -> AsyncGenerator[ChatOpenAI]:
    from nat.plugins.langchain.llm import (  # noqa: PLC0415
        _patch_llm_based_on_config as langchain_patch_llm_based_on_config,
    )

    from datarobot_genai.langgraph.llm import get_datarobot_deployment_llm
    from datarobot_genai.langgraph.llm import get_datarobot_gateway_llm
    from datarobot_genai.langgraph.llm import get_datarobot_nim_llm

    validate_no_responses_api(llm_config, LLMFrameworkEnum.LANGCHAIN)

    config = llm_config.model_dump(
        exclude={"type", "thinking", "headers", "api_type", "use_datarobot_llm_gateway"},
        by_alias=True,
        exclude_none=True,
    )
    if llm_config.use_datarobot_llm_gateway:
        client = get_datarobot_gateway_llm(llm_config.model_name, config)
    elif llm_config.llm_deployment_id:
        if llm_config.headers:
            config["default_headers"] = llm_config.headers
        client = get_datarobot_deployment_llm(
            llm_config.llm_deployment_id, llm_config.model_name, config
        )
    elif llm_config.nim_deployment_id:
        client = get_datarobot_nim_llm(llm_config.nim_deployment_id, llm_config.model_name, config)
    else:
        raise ValueError(
            "Either use_datarobot_llm_gateway, llm_deployment_id, or nim_deployment_id must be "
            "provided"
        )

    yield langchain_patch_llm_based_on_config(client, config)


@register_llm_client(
    config_type=DataRobotLLMComponentModelConfig, wrapper_type=LLMFrameworkEnum.CREWAI
)
async def datarobot_llm_component_crewai(
    llm_config: DataRobotLLMComponentModelConfig, builder: Builder
) -> AsyncGenerator[LLM]:
    from crewai import LLM  # noqa: PLC0415

    validate_no_responses_api(llm_config, LLMFrameworkEnum.CREWAI)

    config = llm_config.model_dump(
        exclude={"type", "thinking", "headers", "api_type"}, by_alias=True, exclude_none=True
    )
    if not config["model"].startswith("datarobot/"):
        config["model"] = "datarobot/" + config["model"]
    if config["use_datarobot_llm_gateway"]:
        config["base_url"] = config["base_url"].removesuffix("/api/v2")
    else:
        config["api_base"] = config.pop("base_url") + "/chat/completions"
        if llm_config.headers:
            config["extra_headers"] = llm_config.headers
    config.pop("use_datarobot_llm_gateway")
    client = LLM(**config)
    yield _patch_llm_based_on_config(client, config)


@register_llm_client(
    config_type=DataRobotLLMComponentModelConfig, wrapper_type=LLMFrameworkEnum.LLAMA_INDEX
)
async def datarobot_llm_component_llamaindex(
    llm_config: DataRobotLLMComponentModelConfig, builder: Builder
) -> AsyncGenerator[LiteLLM]:
    validate_no_responses_api(llm_config, LLMFrameworkEnum.LLAMA_INDEX)
    config = llm_config.model_dump(
        exclude={"type", "thinking", "headers", "api_type"}, by_alias=True, exclude_none=True
    )
    if not config["model"].startswith("datarobot/"):
        config["model"] = "datarobot/" + config["model"]
    if config["use_datarobot_llm_gateway"]:
        config["api_base"] = config.pop("base_url").removesuffix("/api/v2")
    else:
        config["api_base"] = config.pop("base_url") + "/chat/completions"
        if llm_config.headers:
            config["additional_kwargs"] = {"extra_headers": llm_config.headers}
    config.pop("use_datarobot_llm_gateway")
    client = _create_datarobot_litellm(config)
    yield _patch_llm_based_on_config(client, config)
