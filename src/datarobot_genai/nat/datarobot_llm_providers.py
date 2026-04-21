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

from nat.builder.builder import Builder
from nat.builder.llm import LLMProviderInfo
from nat.cli.register_workflow import register_llm_provider
from nat.llm.litellm_llm import LiteLlmModelConfig
from nat.llm.nim_llm import NIMModelConfig
from nat.llm.openai_llm import OpenAIModelConfig
from pydantic import AliasChoices
from pydantic import Field

from datarobot_genai.core.config import LLMConfig
from datarobot_genai.core.config import default_llm_deployment_id
from datarobot_genai.core.config import default_model_name
from datarobot_genai.core.config import default_nim_deployment_id
from datarobot_genai.core.config import default_use_datarobot_llm_gateway


class DataRobotLLMComponentModelConfig(LLMConfig, OpenAIModelConfig, name="datarobot-llm-component"):  # type: ignore[call-arg]
    """A DataRobot LLM provider to be used with an LLM client."""

    model_name: str = Field(
        validation_alias=AliasChoices("model_name", "model"),
        serialization_alias="model",
        description="The model name.",
        default_factory=default_model_name,
    )
    use_datarobot_llm_gateway: bool = Field(
        default_factory=default_use_datarobot_llm_gateway,
        description="Whether to use the DataRobot LLM gateway.",
    )
    llm_deployment_id: str | None = Field(
        description="The LLM deployment ID.",
        default_factory=default_llm_deployment_id,
    )
    nim_deployment_id: str | None = Field(
        description="The NIM deployment ID.",
        default_factory=default_nim_deployment_id,
    )
    headers: dict[str, str] | None = Field(
        description="Additional headers send to LLM deployment.",
        default=None,
    )


@register_llm_provider(config_type=DataRobotLLMComponentModelConfig)
async def datarobot_llm_component(
    config: DataRobotLLMComponentModelConfig, _builder: Builder
) -> LLMProviderInfo:
    yield LLMProviderInfo(
        config=config, description="DataRobot LLM Component for use with an LLM client."
    )


class DataRobotLLMGatewayModelConfig(OpenAIModelConfig, name="datarobot-llm-gateway"):  # type: ignore[call-arg]
    """A DataRobot LLM provider to be used with an LLM client."""


@register_llm_provider(config_type=DataRobotLLMGatewayModelConfig)
async def datarobot_llm_gateway(
    config: DataRobotLLMGatewayModelConfig, _builder: Builder
) -> LLMProviderInfo:
    yield LLMProviderInfo(
        config=config, description="DataRobot LLM Gateway for use with an LLM client."
    )


class DataRobotLLMDeploymentModelConfig(OpenAIModelConfig, name="datarobot-llm-deployment"):  # type: ignore[call-arg]
    """A DataRobot LLM provider to be used with an LLM client."""

    model_name: str = Field(
        validation_alias=AliasChoices("model_name", "model"),
        serialization_alias="model",
        description="The model name to pass through to the deployment.",
        default="datarobot-deployed-llm",
    )
    llm_deployment_id: str = Field(
        description="The LLM deployment ID.",
        default_factory=default_llm_deployment_id,
    )
    headers: dict[str, str] | None = Field(
        description="Additional headers send to LLM deployment.",
        default=None,
    )


@register_llm_provider(config_type=DataRobotLLMDeploymentModelConfig)
async def datarobot_llm_deployment(
    config: DataRobotLLMDeploymentModelConfig, _builder: Builder
) -> LLMProviderInfo:
    yield LLMProviderInfo(
        config=config, description="DataRobot LLM deployment for use with an LLM client."
    )


class DataRobotNIMModelConfig(NIMModelConfig, name="datarobot-nim"):  # type: ignore[call-arg]
    """A DataRobot NIM LLM provider to be used with an LLM client."""

    model_name: str = Field(
        validation_alias=AliasChoices("model_name", "model"),
        serialization_alias="model",
        description="The model name to pass through to the deployment.",
        default="datarobot-deployed-llm",
    )
    nim_deployment_id: str = Field(
        description="The LLM deployment ID.",
        default_factory=default_nim_deployment_id,
    )


@register_llm_provider(config_type=DataRobotNIMModelConfig)
async def datarobot_nim(config: DataRobotNIMModelConfig, _builder: Builder) -> LLMProviderInfo:
    yield LLMProviderInfo(
        config=config, description="DataRobot NIM deployment for use with an LLM client."
    )


class DataRobotLitellmConfig(LiteLlmModelConfig, name="datarobot-litellm"):  # type: ignore[call-arg]
    """A DataRobot Litellm provider to be used with an LLM client."""

    model_name: str = Field(
        validation_alias=AliasChoices("model_name", "model"),
        serialization_alias="model",
        description="The model name.",
        default_factory=default_model_name,
    )


@register_llm_provider(config_type=DataRobotLitellmConfig)
async def datarobot_litellm(config: DataRobotLitellmConfig, _builder: Builder) -> LLMProviderInfo:
    yield LLMProviderInfo(
        config=config, description="DataRobot Litellm provider for use with an LLM client."
    )


class DataRobotLLMRouterConfig(OpenAIModelConfig, name="datarobot-llm-router"):  # type: ignore[call-arg]
    """Primary + one-or-more fallback LLMs with automatic failover via LiteLLM Router.

    Example workflow YAML::

        datarobot_llm:
          _type: datarobot-llm-router
          primary:
            llm_deployment_id: "abc123"
            use_datarobot_llm_gateway: false
          fallbacks:
            - llm_deployment_id: "def456"
              use_datarobot_llm_gateway: false
          allowed_fails: 3
    """

    model_name: str = Field(
        validation_alias=AliasChoices("model_name", "model"),
        serialization_alias="model",
        description="Placeholder model name (not used for routing; each sub-config has its own).",
        default="datarobot-router",
    )
    primary: LLMConfig = Field(
        description="Primary LLM configuration."
    )
    fallbacks: list[LLMConfig] = Field(
        description="Ordered list of fallback LLM configurations (at least one required).",
        min_length=1,
    )
    allowed_fails: int = Field(
        default=3,
        description="Number of failures allowed before a deployment enters cooldown.",
    )
    retry_policy: dict[str, int] | None = Field(
        default=None,
        description="Per-exception retry counts, e.g. {'RateLimitErrorRetries': 2}.",
    )
    cooldown_time: float | None = Field(
        default=None,
        description="Seconds a failed deployment stays in cooldown before being retried.",
    )


@register_llm_provider(config_type=DataRobotLLMRouterConfig)
async def datarobot_llm_router(
    config: DataRobotLLMRouterConfig, _builder: Builder
) -> LLMProviderInfo:
    yield LLMProviderInfo(
        config=config,
        description="DataRobot LLM Router with automatic failover via LiteLLM Router.",
    )
