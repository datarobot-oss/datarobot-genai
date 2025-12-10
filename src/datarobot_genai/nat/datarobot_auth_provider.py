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
from nat.authentication.api_key.api_key_auth_provider import APIKeyAuthProvider
from nat.authentication.api_key.api_key_auth_provider_config import APIKeyAuthProviderConfig
from nat.builder.builder import Builder
from nat.cli.register_workflow import register_auth_provider
from pydantic import Field


class Config(DataRobotAppFrameworkBaseSettings):
    """
    Finds variables in the priority order of: env
    variables (including Runtime Parameters), .env, file_secrets, then
    Pulumi output variables.
    """

    datarobot_api_token: str


config = Config()


class DataRobotAPIKeyAuthProviderConfig(APIKeyAuthProviderConfig, name="datarobot_api_key"):  # type: ignore[call-arg]
    raw_key: str = Field(
        description=(
            "Raw API token or credential to be injected into the request parameter. "
            "Used for 'bearer','x-api-key','custom', and other schemes. "
        ),
        default=config.datarobot_api_token,
    )
    default_user_id: str | None = Field(default="default-user", description="Default user ID")
    allow_default_user_id_for_tool_calls: bool = Field(
        default=True, description="Allow default user ID for tool calls"
    )


@register_auth_provider(config_type=DataRobotAPIKeyAuthProviderConfig)
async def datarobot_api_key_client(
    config: DataRobotAPIKeyAuthProviderConfig, builder: Builder
) -> APIKeyAuthProviderConfig:
    yield APIKeyAuthProvider(config=config)
