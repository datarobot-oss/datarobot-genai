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

from nat.authentication.api_key.api_key_auth_provider import APIKeyAuthProvider
from nat.authentication.api_key.api_key_auth_provider_config import APIKeyAuthProviderConfig
from nat.authentication.interfaces import AuthProviderBase
from nat.builder.builder import Builder
from nat.cli.register_workflow import register_auth_provider
from nat.data_models.authentication import AuthProviderBaseConfig
from nat.data_models.authentication import AuthResult
from nat.data_models.authentication import HeaderCred
from nat.data_models.common import SerializableSecretStr
from pydantic import Field
from pydantic import SecretStr

from datarobot_genai.core.config import default_api_key
from datarobot_genai.core.mcp.target import build_headers
from datarobot_genai.dragent.context import extract_authorization_from_context
from datarobot_genai.dragent.context import extract_datarobot_headers_from_context


def _get_default_api_token() -> SecretStr | None:
    """Get the default API token, wrapped in SecretStr for proper serialization.

    Resolved off the global app config (``resolve_config().resolve_datarobot_api_token()``)
    rather than a local settings class, so an app that registers its own config
    supplies the token here too.
    """
    if datarobot_api_token := default_api_key():
        return SecretStr(datarobot_api_token)
    return None


class DataRobotAPIKeyAuthProviderConfig(APIKeyAuthProviderConfig, name="datarobot_api_key"):  # type: ignore[call-arg]
    raw_key: SerializableSecretStr = Field(
        description=(
            "Raw API token or credential to be injected into the request parameter. "
            "Used for 'bearer','x-api-key','custom', and other schemes. "
        ),
        default_factory=_get_default_api_token,
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


class DataRobotMCPAuthProviderConfig(AuthProviderBaseConfig, name="datarobot_mcp_auth"):  # type: ignore[call-arg]
    headers: dict[str, str] | None = Field(
        default=None,
        description=(
            "Extra headers, merged LAST so they override the resolved ones. The only way "
            "to attach a static header to a DataRobot-hosted server."
        ),
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
        Build the credentials for one MCP server.

        Args:
            user_id (str): The user ID to authenticate.
            target (MCPTarget): The server being called, passed by the caller's auth
                adapter. Required.

        Returns
        -------
            AuthenticatedContext: The authenticated context containing headers
        """
        # NAT shares ONE provider instance across every block naming it
        # (workflow_builder.get_auth_provider returns self._auth_providers[name].instance),
        # so the target MUST arrive with the call. Storing it on `self` would mean the
        # last block to build wins and every block got that block's credentials -- the
        # same defect this replaces, relocated into a different object.
        target = kwargs.get("target")
        if target is None:
            raise ValueError(
                "datarobot_mcp_auth requires a resolved MCPTarget passed as `target=`. It "
                "must never fall back to reading the environment: which credentials a "
                "server receives depends on that server's kind, so a fleet would get one "
                "server's credentials for all of them."
            )

        # In dragent the forwarded headers and authorization context come from Context;
        # in drum a custom loader writes self.config.headers instead.
        auth_headers = build_headers(
            target,
            forwarded=extract_datarobot_headers_from_context(),
            auth_context=extract_authorization_from_context(),
            extra=self.config.headers,  # merged last: an explicit override wins
        )

        return AuthResult(
            credentials=[HeaderCred(name=name, value=value) for name, value in auth_headers.items()]
        )


@register_auth_provider(config_type=DataRobotMCPAuthProviderConfig)
async def datarobot_mcp_auth_provider(
    config: DataRobotMCPAuthProviderConfig, builder: Builder
) -> AsyncGenerator[DataRobotMCPAuthProvider]:
    yield DataRobotMCPAuthProvider(config=config)
