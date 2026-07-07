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
import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

import httpx
from nat.authentication.interfaces import AuthProviderBase
from nat.builder.context import Context
from nat.cli.register_workflow import register_auth_provider
from nat.data_models.authentication import AuthProviderBaseConfig
from nat.data_models.authentication import AuthResult
from nat.data_models.authentication import BearerTokenCred
from nat.data_models.authentication import HeaderCred
from nat.data_models.common import OptionalSecretStr
from pydantic import Field
from pydantic import SecretStr

from datarobot_genai.dragent.http_client import get_retriable_async_http_client
from datarobot_genai.dragent.plugins.okta_a2a_auth import (
    OAuth2CrossApplicationAccessAuthProviderConfig,
)
from datarobot_genai.dragent.plugins.okta_a2a_auth import _CrossAppFlowParams
from datarobot_genai.dragent.plugins.okta_a2a_auth import _get_default_principal_id
from datarobot_genai.dragent.plugins.okta_a2a_auth import _get_default_private_jwk
from datarobot_genai.dragent.plugins.okta_a2a_auth import get_token_exchange

logger = logging.getLogger(__name__)


@dataclass()
class XAAStepOneTokenExchangeParams:
    trusted_issuer: str
    exchange_audience: str

    @classmethod
    def create_from_mcp_auth_server_metadata(
        cls, metadata: dict[str, Any]
    ) -> "XAAStepOneTokenExchangeParams":
        return cls(metadata["trustedIssuer"], metadata["audience"])


@dataclass()
class XAAStepTwoTokenRequestParams:
    token_url: str
    target_audience: str
    id_jag_scopes: list[str]

    @classmethod
    def create_from_mcp_auth_server_metadata(
        cls, metadata: dict[str, Any]
    ) -> "XAAStepTwoTokenRequestParams":
        return cls(metadata["tokenUrl"], metadata["audience"], metadata["scopes"])


@dataclass
class MCPXAAParams:
    step_one_token_exchange_params: XAAStepOneTokenExchangeParams
    step_two_token_request_params: XAAStepTwoTokenRequestParams

    @staticmethod
    async def create_from_mcp_auth_server_metadata(mcp_auth_server_url: str) -> "MCPXAAParams":
        return await get_xaa_param_from_mcp_auth_server_metadata(mcp_auth_server_url)


def parse_xaa_params_from_mcp_auth_server_metadata(
    mcp_auth_server_metadata: dict[str, Any],
) -> MCPXAAParams:
    xaa_metadata = mcp_auth_server_metadata["urn:datarobot:nat_mcp_xaa_client"]

    return MCPXAAParams(
        XAAStepOneTokenExchangeParams.create_from_mcp_auth_server_metadata(
            xaa_metadata["tokenExchange"]
        ),
        XAAStepTwoTokenRequestParams.create_from_mcp_auth_server_metadata(
            xaa_metadata["tokenRequest"]
        ),
    )


async def get_xaa_param_from_mcp_auth_server_metadata(
    mcp_auth_server_metadata_url: str,
) -> MCPXAAParams:
    async with get_retriable_async_http_client() as http_client:
        try:
            resp = await http_client.get(mcp_auth_server_metadata_url)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(
                "Failed to fetch MCP auth server metadata from "
                f"{mcp_auth_server_metadata_url}: {exc}"
            )

    return parse_xaa_params_from_mcp_auth_server_metadata(resp.json())


def extract_token_value_from_bearer_or_non_bearer_header(http_header_value: str) -> str:
    if http_header_value.startswith("bearer "):
        return http_header_value[len("bearer ") :]
    elif http_header_value.startswith("Bearer "):
        return http_header_value[len("Bearer ") :]
    else:
        return http_header_value


class MCPXAAAuthProviderConfig(
    AuthProviderBaseConfig,
    name="mcp_xaa_auth_provider_config",
):  # type: ignore[call-arg]
    okta_token_header: str = Field(
        default="x-datarobot-external-access-token",
        description=(
            "Incoming header carrying the caller's access token. "
            "Used as ``subject_token`` in Step 1 of the XAA exchange. "
            "Matched case-insensitively."
        ),
    )
    fallback_token_headers: list[str] = Field(
        default=["authorization"],
        description=(
            "Fallback headers to try (in order) when ``okta_token_header`` is absent. "
            "If the value starts with 'Bearer ', the prefix is stripped automatically."
        ),
    )
    principal_id: str | None = Field(
        default_factory=_get_default_principal_id,
        description=(
            "Okta AI agent principal ID (env: ``IDP_AGENT_ID``). "
            "Used as ``iss``/``sub`` in the JWT client assertion."
        ),
    )
    private_jwk: OptionalSecretStr = Field(
        default_factory=_get_default_private_jwk,
        description=(
            "Base64-encoded or raw-JSON RSA private JWK (env: ``IDP_AGENT_PRIVATE_KEY_JWK``)."
        ),
    )


class MCPXAAAuthProvider(AuthProviderBase[MCPXAAAuthProviderConfig]):
    def __init__(self, config: MCPXAAAuthProviderConfig) -> None:
        super().__init__(config)
        self._xaa_params: MCPXAAParams | None = None

    def set_xaa_params(self, params: MCPXAAParams) -> None:
        self._xaa_params = params
        logger.info(
            "MCP XAA params set: trusted_issuer=%s, exchange_audience=%s, "
            "target_audience=%s, token_url=%s, scopes=%s",
            params.step_one_token_exchange_params.trusted_issuer,
            params.step_one_token_exchange_params.exchange_audience,
            params.step_two_token_request_params.target_audience,
            params.step_two_token_request_params.token_url,
            params.step_two_token_request_params.id_jag_scopes,
        )

    def get_cross_app_flow_params(self) -> _CrossAppFlowParams:
        if self._xaa_params is None:
            raise RuntimeError("This method shall be called after set_xaa_params() is called.")

        return _CrossAppFlowParams(
            token_url=self._xaa_params.step_two_token_request_params.token_url,
            trusted_issuer=self._xaa_params.step_one_token_exchange_params.trusted_issuer,
            exchange_audience=self._xaa_params.step_one_token_exchange_params.exchange_audience,
            target_audience=self._xaa_params.step_two_token_request_params.target_audience,
            token_endpoint_auth_method="private_key_jwt",
            id_jag_scopes=self._xaa_params.step_two_token_request_params.id_jag_scopes,
        )

    def get_oauth2_cross_app_access_auth_provider_config(
        self,
    ) -> OAuth2CrossApplicationAccessAuthProviderConfig:
        return OAuth2CrossApplicationAccessAuthProviderConfig(
            okta_token_header=self.config.okta_token_header,
            fallback_token_headers=self.config.fallback_token_headers,
            principal_id=self.config.principal_id,
            private_jwk=self.config.private_jwk,
        )

    def extract_subject_token_from_inbound_request(self, headers: dict[str, str]) -> str:
        header_keys_sorted_by_descending_priority = [self.config.okta_token_header.lower()]
        header_keys_sorted_by_descending_priority.extend(
            [header.lower() for header in self.config.fallback_token_headers]
        )
        for header_key in header_keys_sorted_by_descending_priority:
            header_value = headers.get(header_key)
            if header_value:
                token = extract_token_value_from_bearer_or_non_bearer_header(header_value)
                if token:
                    return token

        raise RuntimeError(
            f"Header '{self.config.okta_token_header}' not found in request context "
            f"(also tried fallbacks: {self.config.fallback_token_headers}). "
            "The access token must be forwarded with every agent call."
        )

    def get_non_forwardable_header_keys(self) -> set[str]:
        excluded_header_keys = {self.config.okta_token_header.lower()}
        excluded_header_keys.update(header.lower() for header in self.config.fallback_token_headers)
        return excluded_header_keys

    def get_forwardable_headers_from_inbound_request(
        self,
        headers: dict[str, str],
    ) -> list[HeaderCred]:
        return [
            HeaderCred(name=header_key, value=SecretStr(header_value))
            for header_key, header_value in headers.items()
            if header_key not in self.get_non_forwardable_header_keys() and header_value is not None
        ]

    async def get_exchanged_token(
        self,
        headers: dict[str, str],
    ) -> BearerTokenCred:
        token_exchange_impl = get_token_exchange(
            self.get_oauth2_cross_app_access_auth_provider_config()
        )
        flow_params = self.get_cross_app_flow_params()
        subject_token = self.extract_subject_token_from_inbound_request(headers)
        exchanged_token = await token_exchange_impl.exchange_token(flow_params, subject_token)
        return BearerTokenCred(token=exchanged_token)

    async def authenticate(self, user_id: str | None = None, **kwargs: Any) -> AuthResult | None:
        if self._xaa_params is None:
            raise RuntimeError("authenticate() shall be called after set_xaa_params() is called.")

        headers: dict[str, str] = Context.get().metadata.headers or {}
        forwardable_header_creds = self.get_forwardable_headers_from_inbound_request(headers)
        bearer_token_cred = await self.get_exchanged_token(headers)
        return AuthResult(credentials=[*forwardable_header_creds, bearer_token_cred])


@register_auth_provider(config_type=MCPXAAAuthProviderConfig)
async def mcp_xaa_auth_provider(
    config: MCPXAAAuthProviderConfig,
) -> AsyncGenerator[MCPXAAAuthProvider, None]:
    """NAT auth provider factory for MCP through XAA."""
    yield MCPXAAAuthProvider(config=config)
