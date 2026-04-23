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

from nat.builder.builder import Builder
from nat.builder.llm import LLMProviderInfo
from nat.cli.register_workflow import register_llm_provider
from nat.llm.litellm_llm import LiteLlmModelConfig
from nat.llm.nim_llm import NIMModelConfig
from nat.llm.openai_llm import OpenAIModelConfig
from pydantic import AliasChoices
from pydantic import Field

from datarobot_genai.core.config import DEFAULT_MODEL_NAME_FOR_DEPLOYED_LLM
from datarobot_genai.core.config import LLMType
from datarobot_genai.core.config import default_llm_deployment_id
from datarobot_genai.core.config import default_nim_deployment_id
from datarobot_genai.core.config import default_use_datarobot_llm_gateway


class DataRobotLLMComponentModelConfig(OpenAIModelConfig, name="datarobot-llm-component"):  # type: ignore[call-arg]
    """A DataRobot LLM provider to be used with an LLM client."""

    model_name: str | None = Field(
        validation_alias=AliasChoices("model_name", "model"),
        serialization_alias="model",
        description=(
            "The model name (required for gateway, NIM, and external; optional for deployment)."
        ),
        default=None,
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

    def get_llm_type(self) -> LLMType:
        if self.use_datarobot_llm_gateway:
            return LLMType.GATEWAY
        elif self.llm_deployment_id:
            return LLMType.DEPLOYMENT
        elif self.nim_deployment_id:
            return LLMType.NIM
        else:
            return LLMType.EXTERNAL


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
        default=DEFAULT_MODEL_NAME_FOR_DEPLOYED_LLM,
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

    model_name: str | None = Field(
        validation_alias=AliasChoices("model_name", "model"),
        serialization_alias="model",
        description="The model name to pass through to the NIM deployment.",
        default=None,
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

    model_name: str | None = Field(
        validation_alias=AliasChoices("model_name", "model"),
        serialization_alias="model",
        description="The model name.",
        default=None,
    )


@register_llm_provider(config_type=DataRobotLitellmConfig)
async def datarobot_litellm(config: DataRobotLitellmConfig, _builder: Builder) -> LLMProviderInfo:
    yield LLMProviderInfo(
        config=config, description="DataRobot Litellm provider for use with an LLM client."
    )
