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
from typing import TypeVar

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.cli.register_workflow import register_llm_client
from nat.utils.responses_api import validate_no_responses_api

from datarobot_genai.core.config import LLMType
from datarobot_genai.dragent.context import extract_headers_from_context
from datarobot_genai.dragent.plugins.llm_clients import patch_llm_based_on_config
from datarobot_genai.dragent.plugins.llm_clients import prepare_llm_parameters
from datarobot_genai.dragent.plugins.llm_clients import router_settings_from_config
from datarobot_genai.dragent.plugins.llm_providers import DataRobotLitellmConfig
from datarobot_genai.dragent.plugins.llm_providers import DataRobotLLMComponentModelConfig
from datarobot_genai.dragent.plugins.llm_providers import DataRobotLLMDeploymentModelConfig
from datarobot_genai.dragent.plugins.llm_providers import DataRobotLLMGatewayModelConfig
from datarobot_genai.dragent.plugins.llm_providers import DataRobotLLMRouterConfig
from datarobot_genai.dragent.plugins.llm_providers import DataRobotNIMModelConfig

ModelType = TypeVar("ModelType")

if TYPE_CHECKING:
    from llama_index.llms.litellm import LiteLLM


@register_llm_client(
    config_type=DataRobotLLMGatewayModelConfig, wrapper_type=LLMFrameworkEnum.LLAMA_INDEX
)
async def datarobot_llm_gateway_llamaindex(
    llm_config: DataRobotLLMGatewayModelConfig, builder: Builder
) -> AsyncGenerator[LiteLLM]:

    from datarobot_genai.llama_index.llm import get_datarobot_gateway_llm

    validate_no_responses_api(llm_config, LLMFrameworkEnum.LLAMA_INDEX)

    config = prepare_llm_parameters(llm_config)

    client = get_datarobot_gateway_llm(config["model"], config)
    yield patch_llm_based_on_config(client, config)


@register_llm_client(
    config_type=DataRobotLLMDeploymentModelConfig, wrapper_type=LLMFrameworkEnum.LLAMA_INDEX
)
async def datarobot_llm_deployment_llamaindex(
    llm_config: DataRobotLLMDeploymentModelConfig, builder: Builder
) -> AsyncGenerator[LiteLLM]:

    from datarobot_genai.llama_index.llm import get_datarobot_deployment_llm

    validate_no_responses_api(llm_config, LLMFrameworkEnum.LLAMA_INDEX)
    config = prepare_llm_parameters(llm_config)
    additional_kwargs = dict(config.get("additional_kwargs") or {})
    additional_kwargs["extra_headers"] = extract_headers_from_context(
        ["X-DataRobot-Identity-Token"]
    )
    config["additional_kwargs"] = additional_kwargs

    client = get_datarobot_deployment_llm(
        llm_config.llm_deployment_id, llm_config.model_name, config
    )
    yield patch_llm_based_on_config(client, config)


@register_llm_client(config_type=DataRobotNIMModelConfig, wrapper_type=LLMFrameworkEnum.LLAMA_INDEX)
async def datarobot_nim_llamaindex(
    llm_config: DataRobotNIMModelConfig, builder: Builder
) -> AsyncGenerator[LiteLLM]:
    from datarobot_genai.llama_index.llm import get_datarobot_nim_llm

    validate_no_responses_api(llm_config, LLMFrameworkEnum.LLAMA_INDEX)

    config = prepare_llm_parameters(llm_config)
    if not llm_config.nim_deployment_id:
        raise ValueError("nim_deployment_id is required")
    client = get_datarobot_nim_llm(llm_config.nim_deployment_id, llm_config.model_name, config)
    yield patch_llm_based_on_config(client, config)


@register_llm_client(
    config_type=DataRobotLLMComponentModelConfig, wrapper_type=LLMFrameworkEnum.LLAMA_INDEX
)
async def datarobot_llm_component_llamaindex(
    llm_config: DataRobotLLMComponentModelConfig, builder: Builder
) -> AsyncGenerator[LiteLLM]:
    from datarobot_genai.llama_index.llm import get_datarobot_deployment_llm
    from datarobot_genai.llama_index.llm import get_datarobot_gateway_llm
    from datarobot_genai.llama_index.llm import get_datarobot_nim_llm
    from datarobot_genai.llama_index.llm import get_external_llm

    validate_no_responses_api(llm_config, LLMFrameworkEnum.LLAMA_INDEX)

    config = prepare_llm_parameters(llm_config)

    llm_type = llm_config.get_llm_type()
    if llm_type == LLMType.GATEWAY:
        client = get_datarobot_gateway_llm(llm_config.model_name, config)
    elif llm_type == LLMType.DEPLOYMENT:
        additional_kwargs = dict(config.get("additional_kwargs") or {})
        additional_kwargs["extra_headers"] = extract_headers_from_context(
            ["X-DataRobot-Identity-Token"]
        )
        config["additional_kwargs"] = additional_kwargs
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
    yield patch_llm_based_on_config(client, config)


@register_llm_client(config_type=DataRobotLitellmConfig, wrapper_type=LLMFrameworkEnum.LLAMA_INDEX)
async def litellm_llamaindex_internal(
    llm_config: DataRobotLitellmConfig, _builder: Builder
) -> AsyncGenerator[LiteLLM]:
    from datarobot_genai.llama_index.llm import get_external_llm

    validate_no_responses_api(llm_config, LLMFrameworkEnum.LLAMA_INDEX)

    llm = get_external_llm(
        llm_config.model_name,
        prepare_llm_parameters(llm_config, exclude_unset=True),
    )

    yield patch_llm_based_on_config(llm, llm_config)


@register_llm_client(
    config_type=DataRobotLLMRouterConfig, wrapper_type=LLMFrameworkEnum.LLAMA_INDEX
)
async def datarobot_llm_router_llamaindex(
    llm_config: DataRobotLLMRouterConfig, builder: Builder
) -> AsyncGenerator[LiteLLM]:
    from nat.plugins.llama_index.llm import (  # noqa: PLC0415
        _patch_llm_based_on_config as llama_index_patch_llm_based_on_config,
    )

    from datarobot_genai.llama_index.llm import get_router_llm

    validate_no_responses_api(llm_config, LLMFrameworkEnum.LLAMA_INDEX)

    client = get_router_llm(
        llm_config.primary,
        llm_config.fallbacks,
        router_settings_from_config(llm_config),
    )
    yield llama_index_patch_llm_based_on_config(client, llm_config)
