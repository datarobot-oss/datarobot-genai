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
from nat.data_models.common import OptionalSecretStr
from nat.llm.openai_llm import OpenAIModelConfig
from pydantic import AliasChoices
from pydantic import Field

from datarobot_genai.core.config import Config
from datarobot_genai.core.config import default_api_key
from datarobot_genai.core.config import default_base_url
from datarobot_genai.core.config import default_model_name

class DataRobotLLMComponentModelConfig(OpenAIModelConfig, name="datarobot-llm-component"):  # type: ignore[call-arg]
    """A DataRobot LLM provider to be used with an LLM client."""

    api_key: OptionalSecretStr = Field(
        default_factory=default_api_key,
        description="DataRobot API key.",
    )
    base_url: str | None = Field(
        default_factory=default_base_url,
        description="DataRobot LLM URL.",
    )
    model_name: str = Field(
        validation_alias=AliasChoices("model_name", "model"),
        serialization_alias="model",
        description="The model name.",
        default_factory=default_model_name,
    )
    use_datarobot_llm_gateway: bool = Field(
        default_factory=lambda: Config().use_datarobot_llm_gateway,
        description="Whether to use the DataRobot LLM gateway.",
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

    api_key: OptionalSecretStr = Field(
        default_factory=default_api_key,
        description="DataRobot API key.",
    )
    base_url: str | None = Field(
        default_factory=lambda: Config().datarobot_endpoint.rstrip("/"),
        description="DataRobot LLM gateway URL.",
    )


@register_llm_provider(config_type=DataRobotLLMGatewayModelConfig)
async def datarobot_llm_gateway(
    config: DataRobotLLMGatewayModelConfig, _builder: Builder
) -> LLMProviderInfo:
    yield LLMProviderInfo(
        config=config, description="DataRobot LLM Gateway for use with an LLM client."
    )


def default_llm_deployment_url() -> str:
    config = Config()
    return f"{config.datarobot_endpoint}/deployments/{config.llm_deployment_id}"


class DataRobotLLMDeploymentModelConfig(OpenAIModelConfig, name="datarobot-llm-deployment"):  # type: ignore[call-arg]
    """A DataRobot LLM provider to be used with an LLM client."""

    api_key: OptionalSecretStr = Field(
        default_factory=default_api_key,
        description="DataRobot API key.",
    )
    base_url: str | None = Field(
        default_factory=default_llm_deployment_url,
        description="DataRobot LLM deployment URL.",
    )
    model_name: str = Field(
        validation_alias=AliasChoices("model_name", "model"),
        serialization_alias="model",
        description="The model name to pass through to the deployment.",
        default="datarobot-deployed-llm",
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


def default_nim_deployment_url() -> str:
    config = Config()
    return f"{config.datarobot_endpoint}/deployments/{config.nim_deployment_id}"


class DataRobotNIMModelConfig(DataRobotLLMDeploymentModelConfig, name="datarobot-nim"):  # type: ignore[call-arg]
    """A DataRobot NIM LLM provider to be used with an LLM client."""

    base_url: str | None = Field(
        default_factory=default_nim_deployment_url,
        description="DataRobot NIM deployment URL.",
    )


@register_llm_provider(config_type=DataRobotNIMModelConfig)
async def datarobot_nim(config: DataRobotNIMModelConfig, _builder: Builder) -> LLMProviderInfo:
    yield LLMProviderInfo(
        config=config, description="DataRobot NIM deployment for use with an LLM client."
    )
