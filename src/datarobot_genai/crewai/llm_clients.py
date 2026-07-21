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
    from crewai import LLM


@register_llm_client(
    config_type=DataRobotLLMGatewayModelConfig, wrapper_type=LLMFrameworkEnum.CREWAI
)
async def datarobot_llm_gateway_crewai(
    llm_config: DataRobotLLMGatewayModelConfig, builder: Builder
) -> AsyncGenerator[LLM]:
    from datarobot_genai.crewai.llm import get_datarobot_gateway_llm

    validate_no_responses_api(llm_config, LLMFrameworkEnum.CREWAI)

    config = prepare_llm_parameters(llm_config)

    client = get_datarobot_gateway_llm(config["model"], config)
    yield patch_llm_based_on_config(client, config)


@register_llm_client(
    config_type=DataRobotLLMDeploymentModelConfig, wrapper_type=LLMFrameworkEnum.CREWAI
)
async def datarobot_llm_deployment_crewai(
    llm_config: DataRobotLLMDeploymentModelConfig, builder: Builder
) -> AsyncGenerator[LLM]:
    from datarobot_genai.crewai.llm import get_datarobot_deployment_llm

    validate_no_responses_api(llm_config, LLMFrameworkEnum.CREWAI)

    config = prepare_llm_parameters(llm_config)
    config["extra_headers"] = extract_headers_from_context(["X-DataRobot-Identity-Token"])

    client = get_datarobot_deployment_llm(
        llm_config.llm_deployment_id, llm_config.model_name, config
    )
    yield patch_llm_based_on_config(client, config)


@register_llm_client(config_type=DataRobotNIMModelConfig, wrapper_type=LLMFrameworkEnum.CREWAI)
async def datarobot_nim_crewai(
    llm_config: DataRobotNIMModelConfig, builder: Builder
) -> AsyncGenerator[LLM]:
    from datarobot_genai.crewai.llm import get_datarobot_nim_llm

    validate_no_responses_api(llm_config, LLMFrameworkEnum.CREWAI)

    config = prepare_llm_parameters(llm_config)
    client = get_datarobot_nim_llm(llm_config.llm_nim_deployment_id, llm_config.model_name, config)
    yield patch_llm_based_on_config(client, config)


@register_llm_client(
    config_type=DataRobotLLMComponentModelConfig, wrapper_type=LLMFrameworkEnum.CREWAI
)
async def datarobot_llm_component_crewai(
    llm_config: DataRobotLLMComponentModelConfig, builder: Builder
) -> AsyncGenerator[LLM]:
    from datarobot_genai.crewai.llm import get_datarobot_deployment_llm
    from datarobot_genai.crewai.llm import get_datarobot_gateway_llm
    from datarobot_genai.crewai.llm import get_datarobot_nim_llm
    from datarobot_genai.crewai.llm import get_external_llm

    validate_no_responses_api(llm_config, LLMFrameworkEnum.CREWAI)

    config = prepare_llm_parameters(llm_config)
    llm_type = llm_config.get_llm_type()

    if llm_type == LLMType.GATEWAY:
        client = get_datarobot_gateway_llm(llm_config.model_name, config)
    elif llm_type == LLMType.DEPLOYMENT:
        config["extra_headers"] = extract_headers_from_context(["X-DataRobot-Identity-Token"])
        client = get_datarobot_deployment_llm(
            llm_config.llm_deployment_id,  # type: ignore[arg-type]
            llm_config.model_name,
            config,
        )
    elif llm_type == LLMType.NIM:
        client = get_datarobot_nim_llm(
            llm_config.llm_nim_deployment_id,  # type: ignore[arg-type]
            llm_config.model_name,
            config,
        )
    elif llm_type == LLMType.EXTERNAL:
        client = get_external_llm(llm_config.model_name, config)
    else:
        raise ValueError(f"Invalid LLM type inferred from config: {llm_type}, config: {llm_config}")
    yield patch_llm_based_on_config(client, config)


@register_llm_client(config_type=DataRobotLitellmConfig, wrapper_type=LLMFrameworkEnum.CREWAI)
async def litellm_crewai_internal(
    llm_config: DataRobotLitellmConfig, _builder: Builder
) -> AsyncGenerator[LLM]:
    from datarobot_genai.crewai.llm import get_external_llm

    validate_no_responses_api(llm_config, LLMFrameworkEnum.CREWAI)

    client = get_external_llm(
        llm_config.model_name,
        prepare_llm_parameters(llm_config, exclude_unset=True),
    )

    yield patch_llm_based_on_config(client, llm_config)


@register_llm_client(config_type=DataRobotLLMRouterConfig, wrapper_type=LLMFrameworkEnum.CREWAI)
async def datarobot_llm_router_crewai(
    llm_config: DataRobotLLMRouterConfig, builder: Builder
) -> AsyncGenerator[LLM]:
    from nat.plugins.crewai.llm import (  # noqa: PLC0415
        _patch_llm_based_on_config as crewai_patch_llm_based_on_config,
    )

    from datarobot_genai.crewai.llm import get_router_llm

    validate_no_responses_api(llm_config, LLMFrameworkEnum.CREWAI)

    client = get_router_llm(
        llm_config.primary,
        llm_config.fallbacks,
        router_settings_from_config(llm_config),
    )
    yield crewai_patch_llm_based_on_config(client, llm_config)
