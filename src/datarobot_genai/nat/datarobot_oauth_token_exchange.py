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

"""RFC 8693 OAuth 2.0 Token Exchange auth provider for DataRobot NAT agents."""

import logging
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

import httpx
from nat.authentication.interfaces import AuthProviderBase
from nat.builder.builder import Builder
from nat.cli.register_workflow import register_auth_provider
from nat.data_models.authentication import AuthProviderBaseConfig, AuthResult, BearerTokenCred
from nat.data_models.common import SerializableSecretStr
from pydantic import Field, SecretStr, field_validator, model_validator

logger = logging.getLogger(__name__)

# RFC 8693 constants
GRANT_TYPE_TOKEN_EXCHANGE = "urn:ietf:params:oauth:grant-type:token-exchange"
SUBJECT_TOKEN_TYPE_ACCESS_TOKEN = "urn:ietf:params:oauth:token-type:access_token"


class OAuth2TokenExchangeConfig(AuthProviderBaseConfig, name="oauth2_token_exchange"):  # type: ignore[call-arg]
    """RFC 8693 OAuth 2.0 Token Exchange configuration.

    Used to exchange an existing subject token (e.g. the caller's access token)
    for a new token scoped/audienced for a downstream service via the
    authorization server's token endpoint.

    Reference: https://datatracker.ietf.org/doc/html/rfc8693
    """

    # --- Token endpoint resolution (explicit URL or OIDC discovery) ---

    token_url: str | None = Field(
        default=None,
        description=(
            "Token endpoint of the authorization server. "
            "If not set, it will be resolved via `discovery_url` or derived from `issuer_url`."
        ),
    )
    discovery_url: str | None = Field(
        default=None,
        description=(
            "OIDC discovery metadata URL (e.g. https://issuer/.well-known/openid-configuration). "
            "Used to resolve `token_url` automatically when it is not explicitly provided."
        ),
    )
    issuer_url: str | None = Field(
        default=None,
        description=(
            "Authorization server issuer identifier. Used as a last-resort fallback "
            "to derive the discovery URL (<issuer_url>/.well-known/openid-configuration) "
            "when neither `token_url` nor `discovery_url` are provided."
        ),
    )

    # --- Client credentials (for authenticating *this* app to the token endpoint) ---

    client_id: str = Field(
        description="OAuth2 client ID for authenticating to the token endpoint.",
    )
    client_secret: SerializableSecretStr = Field(
        description="OAuth2 client secret for authenticating to the token endpoint.",
    )

    # --- RFC 8693 exchange parameters ---

    audience: str | None = Field(
        default=None,
        description="Logical name or URI of the target service the exchanged token is intended for.",
    )
    scopes: list[str] = Field(
        default_factory=list,
        description="Scopes to request on the exchanged token.",
    )

    # --- Validators ---

    @staticmethod
    def _is_https_or_localhost(url: str) -> bool:
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return False
            if parsed.scheme == "https":
                return True
            return parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
        except Exception:
            return False

    @field_validator("token_url", "discovery_url", "issuer_url")
    @classmethod
    def _require_valid_url(cls, value: str | None, info: Any) -> str | None:
        if value is None:
            return value
        if not cls._is_https_or_localhost(value):
            raise ValueError(
                f"{info.field_name} must be HTTPS (http allowed only for localhost). Got: {value}"
            )
        return value

    @model_validator(mode="after")
    def _ensure_token_endpoint_resolvable(self) -> "OAuth2TokenExchangeConfig":
        """At least one of token_url, discovery_url, or issuer_url must be provided."""
        if not (self.token_url or self.discovery_url or self.issuer_url):
            raise ValueError(
                "Cannot resolve the token endpoint. Provide at least one of: "
                "token_url, discovery_url, or issuer_url."
            )
        return self


class OAuth2TokenExchangeProvider(AuthProviderBase[OAuth2TokenExchangeConfig]):
    """Auth provider that performs RFC 8693 token exchange."""

    def __init__(self, config: OAuth2TokenExchangeConfig, config_name: str | None = None) -> None:
        super().__init__(config)
        self._resolved_token_url: str | None = config.token_url

    async def _resolve_token_url(self) -> str:
        """Resolve the token endpoint via OIDC discovery if not explicitly configured."""
        if self._resolved_token_url:
            return self._resolved_token_url

        discovery_url = self.config.discovery_url
        if not discovery_url and self.config.issuer_url:
            discovery_url = f"{self.config.issuer_url.rstrip('/')}/.well-known/openid-configuration"

        if not discovery_url:
            raise ValueError("Cannot resolve token endpoint: no discovery_url or issuer_url configured.")

        async with httpx.AsyncClient() as client:
            resp = await client.get(discovery_url)
            resp.raise_for_status()
            metadata = resp.json()

        token_endpoint = metadata.get("token_endpoint")
        if not token_endpoint:
            raise ValueError(f"OIDC discovery metadata at {discovery_url} does not contain 'token_endpoint'.")

        self._resolved_token_url = token_endpoint
        logger.info("Resolved token_endpoint via OIDC discovery: %s", token_endpoint)
        return token_endpoint

    async def authenticate(
        self,
        user_id: str | None = None,
        *,
        subject_token: str | None = None,
        **kwargs: Any,
    ) -> AuthResult:
        """Exchange a subject token for a new token via RFC 8693.

        Parameters
        ----------
        user_id
            Optional user identifier (unused in this flow but required by interface).
        subject_token
            The existing token to exchange. Must be provided.
        """
        if not subject_token:
            raise ValueError("subject_token is required for RFC 8693 token exchange.")

        token_url = await self._resolve_token_url()

        # Build the token exchange request body per RFC 8693 §2.1
        data: dict[str, str] = {
            "grant_type": GRANT_TYPE_TOKEN_EXCHANGE,
            "subject_token": subject_token,
            "subject_token_type": SUBJECT_TOKEN_TYPE_ACCESS_TOKEN,
        }
        if self.config.audience:
            data["audience"] = self.config.audience
        if self.config.scopes:
            data["scope"] = " ".join(self.config.scopes)

        # Authenticate to the token endpoint using client credentials (HTTP Basic)
        auth = httpx.BasicAuth(
            username=self.config.client_id,
            password=self.config.client_secret.get_secret_value(),
        )

        async with httpx.AsyncClient() as client:
            resp = await client.post(token_url, data=data, auth=auth)
            resp.raise_for_status()
            token_data = resp.json()

        access_token = token_data.get("access_token")
        if not access_token:
            raise ValueError(f"Token exchange response missing 'access_token': {token_data}")

        expires_at = None
        if "expires_in" in token_data:
            expires_at = datetime.fromtimestamp(
                datetime.now(UTC).timestamp() + token_data["expires_in"], tz=UTC
            )

        return AuthResult(
            credentials=[BearerTokenCred(token=SecretStr(access_token))],
            token_expires_at=expires_at,
            raw=token_data,
        )


@register_auth_provider(config_type=OAuth2TokenExchangeConfig)
async def datarobot_oauth2_token_exchange_provider(
    config: OAuth2TokenExchangeConfig, builder: Builder
) -> AsyncGenerator[OAuth2TokenExchangeProvider]:
    yield OAuth2TokenExchangeProvider(config=config)
