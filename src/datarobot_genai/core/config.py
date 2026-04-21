# Copyright 2026 DataRobot, Inc. and its affiliates.
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

from enum import StrEnum

from datarobot.core.config import DataRobotAppFrameworkBaseSettings
from pydantic import BaseModel
from pydantic import Field

DEFAULT_MAX_HISTORY_MESSAGES = 20
DEFAULT_MODEL_NAME_FOR_DEPLOYED_LLM = "datarobot/datarobot-deployed-llm"


class LLMType(StrEnum):
    GATEWAY = "gateway"
    DEPLOYMENT = "deployment"
    NIM = "nim"
    EXTERNAL = "external"


def _with_datarobot_prefix(model_name: str) -> str:
    return model_name if model_name.startswith("datarobot/") else "datarobot/" + model_name


class LLMConfig(BaseModel):
    """Pure LLM connection parameters — no env-var machinery, no NAT base class.

    Used as a base for ``Config`` (adds env reading) and
    ``DataRobotLLMComponentModelConfig`` (adds NAT schema), and as the
    primary/fallback type accepted by the router so ``core`` has no NAT
    dependency.
    """

    datarobot_endpoint: str | None = None
    datarobot_api_token: str | None = None
    llm_deployment_id: str | None = None
    nim_deployment_id: str | None = None
    use_datarobot_llm_gateway: bool = True
    llm_default_model: str | None = None

    def get_llm_type(self) -> LLMType:
        if self.use_datarobot_llm_gateway:
            return LLMType.GATEWAY
        elif self.llm_deployment_id:
            return LLMType.DEPLOYMENT
        elif self.nim_deployment_id:
            return LLMType.NIM
        else:
            return LLMType.EXTERNAL

    def to_litellm_params(self) -> dict:
        """Return a litellm_params dict suitable for ``litellm.Router``'s model_list.

        Falls back to env-loaded ``Config`` values for endpoint and api_key
        when they are not set on this instance directly.
        """
        env = Config()
        api_key = self.datarobot_api_token if self.datarobot_api_token is not None else env.datarobot_api_token
        endpoint = self.datarobot_endpoint if self.datarobot_endpoint is not None else env.datarobot_endpoint
        model_name = (
            getattr(self, "model_name", None)
            or self.llm_default_model
            or env.llm_default_model
            or "datarobot-deployed-llm"
        )
        llm_type = self.get_llm_type()

        if llm_type == LLMType.GATEWAY:
            return {
                "model": _with_datarobot_prefix(model_name),
                "api_base": llm_gateway_url(endpoint),
                "api_key": api_key,
            }
        elif llm_type == LLMType.DEPLOYMENT:
            return {
                "model": _with_datarobot_prefix(model_name),
                "api_base": deployment_url(self.llm_deployment_id, endpoint),  # type: ignore[arg-type]
                "api_key": api_key,
            }
        elif llm_type == LLMType.NIM:
            return {
                "model": _with_datarobot_prefix(model_name),
                "api_base": deployment_url(self.nim_deployment_id, endpoint),  # type: ignore[arg-type]
                "api_key": api_key,
            }
        else:  # EXTERNAL
            return {
                "model": model_name.removeprefix("datarobot/"),
                "api_key": api_key,
            }


class Config(LLMConfig, DataRobotAppFrameworkBaseSettings):
    """
    Finds variables in the priority order of: env
    variables (including Runtime Parameters), .env, file_secrets, then
    Pulumi output variables.
    """

    datarobot_endpoint: str = "https://app.datarobot.com/api/v2"

    max_history_messages: int = Field(
        default=DEFAULT_MAX_HISTORY_MESSAGES, ge=0, alias="datarobot_genai_max_history_messages"
    )


def get_max_history_messages_default() -> int:
    """Return the default maximum number of history messages.

    This can be overridden globally via the
    ``DATAROBOT_GENAI_MAX_HISTORY_MESSAGES`` environment variable.
    Invalid values fall back to the built-in default. Negative values are
    treated as 0 (disable history).
    """
    return max(Config().max_history_messages, 0)


def default_api_key() -> str | None:
    config = Config()
    return config.datarobot_api_token if config.datarobot_api_token else None


def default_model_name() -> str | None:
    config = Config()
    return config.llm_default_model


def default_use_datarobot_llm_gateway() -> bool:
    config = Config()
    return config.use_datarobot_llm_gateway


def deployment_url(deployment_id: str, datarobot_endpoint: str) -> str:
    return f"{datarobot_endpoint}/deployments/{deployment_id}/chat/completions"


def default_deployment_url(deployment_id: str | None = None) -> str:
    config = Config()
    default_deployment_id = deployment_id or config.llm_deployment_id
    if default_deployment_id is None:
        raise ValueError("Neither deployment ID nor default deployment ID is set")

    return deployment_url(default_deployment_id, config.datarobot_endpoint)


def llm_gateway_url(datarobot_endpoint: str) -> str:
    return datarobot_endpoint.removesuffix("/api/v2")


def default_datarobot_llm_gateway_url() -> str:
    config = Config()
    return llm_gateway_url(config.datarobot_endpoint)


def default_llm_deployment_id() -> str | None:
    config = Config()
    return config.llm_deployment_id


def default_nim_deployment_id() -> str | None:
    config = Config()
    return config.nim_deployment_id
