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

import os

from datarobot.core.config import DataRobotAppFrameworkBaseSettings
from nat.data_models.common import SecretStr

DEFAULT_MAX_HISTORY_MESSAGES = 20


def get_max_history_messages_default() -> int:
    """Return the default maximum number of history messages.

    This can be overridden globally via the
    ``DATAROBOT_GENAI_MAX_HISTORY_MESSAGES`` environment variable.
    Invalid values fall back to the built-in default. Negative values are
    treated as 0 (disable history).
    """
    raw = os.getenv("DATAROBOT_GENAI_MAX_HISTORY_MESSAGES")
    if not raw:
        return DEFAULT_MAX_HISTORY_MESSAGES

    try:
        value = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_MAX_HISTORY_MESSAGES

    # 0 means "no history". Clamp negatives to 0 to allow "disable history"
    # semantics via env var while preventing unbounded/undefined behavior.
    return max(0, value)


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
    streamming: bool = False


def default_api_key() -> SecretStr | None:
    config = Config()
    return SecretStr(config.datarobot_api_token) if config.datarobot_api_token else None


def default_llm_deployment_url() -> str:
    config = Config()
    return f"{config.datarobot_endpoint}/deployments/{config.llm_deployment_id}"


def default_base_url() -> str:
    config = Config()
    return (
        config.datarobot_endpoint.rstrip("/")
        if config.use_datarobot_llm_gateway
        else config.datarobot_endpoint + f"/deployments/{config.llm_deployment_id}"
    )


def default_model_name() -> str:
    config = Config()
    return config.llm_default_model or "datarobot-deployed-llm"
