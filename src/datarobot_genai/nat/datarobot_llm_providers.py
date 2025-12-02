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

from datarobot.core.config import DataRobotAppFrameworkBaseSettings
from nat.builder.builder import Builder
from nat.builder.llm import LLMProviderInfo
from nat.cli.register_workflow import register_llm_provider
from nat.llm.openai_llm import OpenAIModelConfig
from pydantic import AliasChoices
from pydantic import Field


class Config(DataRobotAppFrameworkBaseSettings):
    """
    Finds variables in the priority order of: env
    variables (including Runtime Parameters), .env, file_secrets, then
    Pulumi output variables.
    """

    datarobot_endpoint: str = "https://app.datarobot.com/api/v2"
    datarobot_api_token: str | None = None
    llm_deployment_id: str | None = None
    nim_deployment_id: str | None = None
    use_datarobot_llm_gateway: bool = False
    llm_default_model: str | None = None


config = Config()


class DataRobotLLMComponentModelConfig(OpenAIModelConfig, name="datarobot-llm-component"):  # type: ignore[call-arg]
    """A DataRobot LLM provider to be used with an LLM client."""

    api_key: str | None = Field(
        default=config.datarobot_api_token, description="DataRobot API key."
    )
    base_url: str | None = Field(
        default=config.datarobot_endpoint.rstrip("/")
        if config.use_datarobot_llm_gateway
        else config.datarobot_endpoint + f"/deployments/{config.llm_deployment_id}",
        description="DataRobot LLM URL.",
    )
    model_name: str = Field(
        validation_alias=AliasChoices("model_name", "model"),
        serialization_alias="model",
        description="The model name.",
        default=config.llm_default_model or "datarobot-deployed-llm",
    )
    use_datarobot_llm_gateway: bool = config.use_datarobot_llm_gateway


@register_llm_provider(config_type=DataRobotLLMComponentModelConfig)
async def datarobot_llm_component(
    config: DataRobotLLMComponentModelConfig, _builder: Builder
) -> LLMProviderInfo:
    yield LLMProviderInfo(
        config=config, description="DataRobot LLM Component for use with an LLM client."
    )


class DataRobotLLMGatewayModelConfig(OpenAIModelConfig, name="datarobot-llm-gateway"):  # type: ignore[call-arg]
    """A DataRobot LLM provider to be used with an LLM client."""

    api_key: str | None = Field(
        default=config.datarobot_api_token, description="DataRobot API key."
    )
    base_url: str | None = Field(
        default=config.datarobot_endpoint.rstrip("/"), description="DataRobot LLM gateway URL."
    )


@register_llm_provider(config_type=DataRobotLLMGatewayModelConfig)
async def datarobot_llm_gateway(
    config: DataRobotLLMGatewayModelConfig, _builder: Builder
) -> LLMProviderInfo:
    yield LLMProviderInfo(
        config=config, description="DataRobot LLM Gateway for use with an LLM client."
    )


class DataRobotLLMDeploymentModelConfig(OpenAIModelConfig, name="datarobot-llm-deployment"):  # type: ignore[call-arg]
    """A DataRobot LLM provider to be used with an LLM client."""

    api_key: str | None = Field(
        default=config.datarobot_api_token, description="DataRobot API key."
    )
    base_url: str | None = Field(
        default=config.datarobot_endpoint + f"/deployments/{config.llm_deployment_id}"
    )
    model_name: str = Field(
        validation_alias=AliasChoices("model_name", "model"),
        serialization_alias="model",
        description="The model name to pass through to the deployment.",
        default="datarobot-deployed-llm",
    )


@register_llm_provider(config_type=DataRobotLLMDeploymentModelConfig)
async def datarobot_llm_deployment(
    config: DataRobotLLMDeploymentModelConfig, _builder: Builder
) -> LLMProviderInfo:
    yield LLMProviderInfo(
        config=config, description="DataRobot LLM deployment for use with an LLM client."
    )


class DataRobotNIMModelConfig(DataRobotLLMDeploymentModelConfig, name="datarobot-nim"):  # type: ignore[call-arg]
    """A DataRobot NIM LLM provider to be used with an LLM client."""

    base_url: str | None = Field(
        default=config.datarobot_endpoint + f"/deployments/{config.nim_deployment_id}"
    )


@register_llm_provider(config_type=DataRobotNIMModelConfig)
async def datarobot_nim(config: DataRobotNIMModelConfig, _builder: Builder) -> LLMProviderInfo:
    yield LLMProviderInfo(
        config=config, description="DataRobot NIM deployment for use with an LLM client."
    )
