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
from typing import Protocol
from typing import TypeVar

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.cli.register_workflow import register_llm_client
from nat.data_models.llm import LLMBaseConfig
from nat.data_models.retry_mixin import RetryMixin
from nat.utils.exception_handlers.automatic_retries import patch_with_retry
from nat.utils.responses_api import validate_no_responses_api

from datarobot_genai.core.config import LLMType
from datarobot_genai.core.config import default_model_name
from datarobot_genai.core.llm.reasoning import apply_reasoning_to_parameters
from datarobot_genai.dragent.context import extract_headers_from_context

from .llm_providers import DataRobotLitellmConfig
from .llm_providers import DataRobotLLMComponentModelConfig
from .llm_providers import DataRobotLLMDeploymentModelConfig
from .llm_providers import DataRobotLLMGatewayModelConfig
from .llm_providers import DataRobotLLMRouterConfig
from .llm_providers import DataRobotNIMModelConfig

ModelType = TypeVar("ModelType")

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI


class SupportsReasoningConfig(Protocol):
    reasoning: bool
    model_fields_set: set[str]

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]: ...


EXCLUDE_FIELDS = {
    "type",
    "thinking",
    "reasoning",
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


def _resolve_model_name(
    llm_config: SupportsReasoningConfig,
    config: dict[str, Any],
) -> str | None:
    return (
        getattr(llm_config, "model_name", None)
        or config.get("model")
        or getattr(llm_config, "llm_default_model", None)
        or default_model_name()
    )


def apply_reasoning_config(
    config: dict[str, Any],
    llm_config: SupportsReasoningConfig,
) -> dict[str, Any]:
    """Map workflow ``reasoning`` to provider ``extra_body`` and temperature."""
    return apply_reasoning_to_parameters(
        config,
        reasoning=llm_config.reasoning,
        model_name=_resolve_model_name(llm_config, config),
        explicit_extra_body="extra_body" in llm_config.model_fields_set,
    )


def prepare_llm_parameters(
    llm_config: SupportsReasoningConfig,
    *,
    exclude_unset: bool = False,
) -> dict[str, Any]:
    """Dump LLM config kwargs and apply ``reasoning`` / ``extra_body`` mapping."""
    dumped = llm_config.model_dump(
        exclude=EXCLUDE_FIELDS,
        by_alias=True,
        exclude_none=True,
        exclude_unset=exclude_unset,
    )
    return apply_reasoning_config(dumped, llm_config)


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

    config = prepare_llm_parameters(llm_config)
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

    config = prepare_llm_parameters(llm_config)

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

    config = prepare_llm_parameters(llm_config)
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
    config = prepare_llm_parameters(llm_config)
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

    config = prepare_llm_parameters(llm_config, exclude_unset=True)
    client = get_external_llm(
        llm_config.model_name,
        config,
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
