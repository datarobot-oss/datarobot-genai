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
from collections.abc import AsyncGenerator
from typing import Any

from datarobot.core.config import DataRobotAppFrameworkBaseSettings
from nat.authentication.api_key.api_key_auth_provider import APIKeyAuthProvider
from nat.authentication.api_key.api_key_auth_provider_config import APIKeyAuthProviderConfig
from nat.authentication.interfaces import AuthProviderBase
from nat.builder.builder import Builder
from nat.cli.register_workflow import register_auth_provider
from nat.data_models.authentication import AuthProviderBaseConfig
from nat.data_models.authentication import AuthResult
from nat.data_models.authentication import HeaderCred
from pydantic import Field

from datarobot_genai.core.mcp.common import MCPConfig


class Config(DataRobotAppFrameworkBaseSettings):
    """
    Finds variables in the priority order of: env
    variables (including Runtime Parameters), .env, file_secrets, then
    Pulumi output variables.
    """

    datarobot_api_token: str | None = None


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
) -> AsyncGenerator[APIKeyAuthProvider]:
    yield APIKeyAuthProvider(config=config)


mcp_config = MCPConfig().server_config


class DataRobotMCPAuthProviderConfig(AuthProviderBaseConfig, name="datarobot_mcp_auth"):  # type: ignore[call-arg]
    headers: dict[str, str] | None = Field(
        description=("Headers to be used for authentication. "),
        default=mcp_config["headers"] if mcp_config else None,
    )
    default_user_id: str | None = Field(default="default-user", description="Default user ID")
    allow_default_user_id_for_tool_calls: bool = Field(
        default=True, description="Allow default user ID for tool calls"
    )


class DataRobotMCPAuthProvider(AuthProviderBase[DataRobotMCPAuthProviderConfig]):
    def __init__(
        self, config: DataRobotMCPAuthProviderConfig, config_name: str | None = None
    ) -> None:
        assert isinstance(config, DataRobotMCPAuthProviderConfig), (
            "Config is not DataRobotMCPAuthProviderConfig"
        )
        super().__init__(config)

    async def authenticate(self, user_id: str | None = None, **kwargs: Any) -> AuthResult | None:
        """
        Authenticate the user using the API key credentials.

        Args:
            user_id (str): The user ID to authenticate.

        Returns
        -------
            AuthenticatedContext: The authenticated context containing headers
        """
        return AuthResult(
            credentials=[
                HeaderCred(name=name, value=value) for name, value in self.config.headers.items()
            ]
        )


@register_auth_provider(config_type=DataRobotMCPAuthProviderConfig)
async def datarobot_mcp_auth_provider(
    config: DataRobotMCPAuthProviderConfig, builder: Builder
) -> AsyncGenerator[DataRobotMCPAuthProvider]:
    yield DataRobotMCPAuthProvider(config=config)
