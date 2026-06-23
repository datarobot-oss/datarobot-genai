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
from typing import TypeVar

from langchain_openai import ChatOpenAI
from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.cli.register_workflow import register_llm_client
from nat.data_models.llm import LLMBaseConfig
from nat.data_models.retry_mixin import RetryMixin
from nat.utils.exception_handlers.automatic_retries import patch_with_retry
from nat.utils.responses_api import validate_no_responses_api

from datarobot_genai.core.config import LLMType
from datarobot_genai.dragent.context import extract_headers_from_context

from .llm_providers import DataRobotLitellmConfig
from .llm_providers import DataRobotLLMComponentModelConfig
from .llm_providers import DataRobotLLMDeploymentModelConfig
from .llm_providers import DataRobotLLMGatewayModelConfig
from .llm_providers import DataRobotLLMRouterConfig
from .llm_providers import DataRobotNIMModelConfig

ModelType = TypeVar("ModelType")

EXCLUDE_FIELDS = {
    "type",
    "thinking",
    "headers",
    "api_type",
    "llm_deployment_id",
    "nim_deployment_id",
    "use_datarobot_llm_gateway",
    # Fields inherited from LLMConfig that are not framework-level LLM kwargs:
    "datarobot_api_token",
    "datarobot_endpoint",
    "llm_default_model",
}


def patch_llm_based_on_config(client: ModelType, llm_config: LLMBaseConfig) -> ModelType:
    if isinstance(llm_config, RetryMixin):
        client = patch_with_retry(
            client,
            retries=llm_config.num_retries,
            retry_codes=llm_config.retry_on_status_codes,
            retry_on_messages=llm_config.retry_on_errors,
        )

    return client


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
        exclude=EXCLUDE_FIELDS,
        by_alias=True,
        exclude_none=True,
    )
    client = get_datarobot_gateway_llm(config["model"], parameters=config)
    yield langchain_patch_llm_based_on_config(client, config)


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
        exclude=EXCLUDE_FIELDS,
        by_alias=True,
        exclude_none=True,
    )

    context_headers = extract_headers_from_context(["X-DataRobot-Identity-Token"])
    if llm_config.headers:
        context_headers = {**context_headers, **llm_config.headers}

    config["extra_headers"] = context_headers

    client = get_datarobot_deployment_llm(
        llm_config.llm_deployment_id, llm_config.model_name, parameters=config
    )
    yield langchain_patch_llm_based_on_config(client, config)


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
        exclude=EXCLUDE_FIELDS,
        by_alias=True,
        exclude_none=True,
    )
    client = get_datarobot_nim_llm(
        llm_config.nim_deployment_id, llm_config.model_name, parameters=config
    )
    yield langchain_patch_llm_based_on_config(client, config)


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
    from datarobot_genai.langgraph.llm import get_external_llm

    validate_no_responses_api(llm_config, LLMFrameworkEnum.LANGCHAIN)
    config = llm_config.model_dump(
        exclude=EXCLUDE_FIELDS,
        by_alias=True,
        exclude_none=True,
    )
    llm_type = llm_config.get_llm_type()
    if llm_type == LLMType.GATEWAY:
        client = get_datarobot_gateway_llm(llm_config.model_name, config)
    elif llm_type == LLMType.DEPLOYMENT:
        if llm_config.headers:
            config["extra_headers"] = llm_config.headers
        client = get_datarobot_deployment_llm(
            llm_config.llm_deployment_id,  # type: ignore[arg-type]
            llm_config.model_name,
            config,
        )
    elif llm_type == LLMType.NIM:
        client = get_datarobot_nim_llm(llm_config.nim_deployment_id, llm_config.model_name, config)  # type: ignore[arg-type]
    elif llm_type == LLMType.EXTERNAL:
        client = get_external_llm(llm_config.model_name, config)
    else:
        raise ValueError(f"Invalid LLM type inferred from config: {llm_type}, config: {llm_config}")

    yield langchain_patch_llm_based_on_config(client, config)


@register_llm_client(config_type=DataRobotLitellmConfig, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
async def litellm_langchain_internal(
    llm_config: DataRobotLitellmConfig, _builder: Builder
) -> AsyncGenerator[ChatOpenAI]:
    from datarobot_genai.langgraph.llm import get_external_llm

    validate_no_responses_api(llm_config, LLMFrameworkEnum.LANGCHAIN)

    client = get_external_llm(
        llm_config.model_name,
        llm_config.model_dump(
            exclude={"type", "thinking", "api_type"},
            by_alias=True,
            exclude_none=True,
            exclude_unset=True,
        ),
    )

    yield patch_llm_based_on_config(client, llm_config)


def router_settings_from_config(llm_config: DataRobotLLMRouterConfig) -> dict:
    return {"num_retries": llm_config.num_retries}


@register_llm_client(config_type=DataRobotLLMRouterConfig, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
async def datarobot_llm_router_langchain(
    llm_config: DataRobotLLMRouterConfig, builder: Builder
) -> AsyncGenerator[ChatOpenAI]:
    from nat.plugins.langchain.llm import (  # noqa: PLC0415
        _patch_llm_based_on_config as langchain_patch_llm_based_on_config,
    )

    from datarobot_genai.langgraph.llm import get_router_llm

    validate_no_responses_api(llm_config, LLMFrameworkEnum.LANGCHAIN)

    client = get_router_llm(
        llm_config.primary,
        llm_config.fallbacks,
        router_settings_from_config(llm_config),
    )
    yield langchain_patch_llm_based_on_config(client, llm_config)
