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

import os

from dotenv import load_dotenv
from nat.builder.builder import Builder
from nat.builder.llm import LLMProviderInfo
from nat.cli.register_workflow import register_llm_provider
from nat.llm.openai_llm import OpenAIModelConfig
from pydantic import Field
from pydantic import model_validator

load_dotenv()


class DataRobotLLMGatewayModelConfig(OpenAIModelConfig, name="datarobot-llm-gateway"):  # type: ignore[call-arg]
    """A DataRobot LLM provider to be used with an LLM client."""

    api_key: str | None = Field(
        default=os.getenv("DATAROBOT_API_TOKEN"), description="DataRobot API key."
    )
    datarobot_endpoint: str | None = Field(
        default=os.getenv("DATAROBOT_ENDPOINT"), description="DataRobot endpoint URL."
    )

    @model_validator(mode="after")  # type: ignore[misc]
    def set_base_url(self) -> None:
        if self.datarobot_endpoint and not self.base_url:  # type: ignore[has-type]
            self.base_url = self.datarobot_endpoint + "/genai/llmgw"


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
        default=os.getenv("DATAROBOT_API_TOKEN"), description="DataRobot API key."
    )
    datarobot_endpoint: str | None = Field(
        default=os.getenv("DATAROBOT_ENDPOINT"), description="DataRobot endpoint URL."
    )
    llm_deployment_id: str | None = Field(
        default=os.getenv("LLM_DEPLOYMENT_ID"), description="DataRobot LLM deployment ID."
    )

    @model_validator(mode="after")  # type: ignore[misc]
    def set_base_url(self) -> None:
        if self.datarobot_endpoint and self.llm_deployment_id and not self.base_url:  # type: ignore[has-type]
            self.base_url = (
                self.datarobot_endpoint + f"/deployments/{self.llm_deployment_id}/chat/completions"
            )


@register_llm_provider(config_type=DataRobotLLMDeploymentModelConfig)
async def datarobot_llm_deployment(
    config: DataRobotLLMDeploymentModelConfig, _builder: Builder
) -> LLMProviderInfo:
    yield LLMProviderInfo(
        config=config, description="DataRobot LLM deployment for use with an LLM client."
    )
