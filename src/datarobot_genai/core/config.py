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

from datarobot.core.config import DataRobotAppFrameworkBaseSettings

DEFAULT_MAX_HISTORY_MESSAGES = 20


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
    use_datarobot_llm_gateway: bool = True
    llm_default_model: str | None = None
    max_history_messages: int = DEFAULT_MAX_HISTORY_MESSAGES


def get_max_history_messages_default() -> int:
    """Return the default maximum number of history messages.

    This can be overridden globally via the
    ``DATAROBOT_GENAI_MAX_HISTORY_MESSAGES`` environment variable.
    Invalid values fall back to the built-in default. Negative values are
    treated as 0 (disable history).
    """
    return Config().max_history_messages


def default_base_url() -> str:
    config = Config()
    return (
        config.datarobot_endpoint.rstrip("/")
        if config.use_datarobot_llm_gateway
        else config.datarobot_endpoint + f"/deployments/{config.llm_deployment_id}"
    )


def default_api_key() -> str | None:
    config = Config()
    return config.datarobot_api_token if config.datarobot_api_token else None


def default_model_name() -> str:
    config = Config()
    return config.llm_default_model or "datarobot-deployed-llm"


def default_use_datarobot_llm_gateway() -> bool:
    config = Config()
    return config.use_datarobot_llm_gateway


def default_deployment_url(deployment_id: str | None = None) -> str:
    config = Config()
    deployment_id = deployment_id or config.llm_deployment_id
    return f"{config.datarobot_endpoint}/deployments/{deployment_id}"


def default_datarobot_llm_gateway_url() -> str:
    config = Config()
    return f"{config.datarobot_endpoint}/genai/llmgw"


def default_llm_deployment_id() -> str | None:
    config = Config()
    return config.llm_deployment_id


def default_nim_deployment_id() -> str | None:
    config = Config()
    return config.nim_deployment_id
