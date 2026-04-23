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

"""Okta cross-app token exchange auth provider for DataRobot NAT agents.

Implements the two-step exchange flow used by DataRobot agents:

1. **Access token → ID-JAG token** — exchanges an Okta access token for an
   Identity-based Just-in-time Access Grant (ID-JAG) token scoped to the
   authorisation server audience.
2. **ID-JAG token → Authorization server token** — exchanges the ID-JAG token
   for a final authorization-server-scoped token that downstream services
   (e.g. MCP server, Agent) can consume.

The provider is registered as a NAT authentication plugin and can be enabled
via ``workflow.yaml``::

    authentication:
      okta_exchange:
        _type: okta_token_exchange
        okta_domain: https://your-org.okta.com
        client_id: ...
        client_secret: ...
        authorization_server_id: default
        principal_id: ...
        private_jwk: <base64-encoded-or-json>
        scope: read_data

Requires the ``auth-okta`` extra::

    pip install datarobot-genai[auth-okta]
"""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from dataclasses import field
from typing import Any

from nat.authentication.interfaces import AuthProviderBase
from nat.builder.builder import Builder
from nat.cli.register_workflow import register_auth_provider
from nat.data_models.authentication import AuthProviderBaseConfig
from nat.data_models.authentication import AuthResult
from nat.data_models.authentication import BearerTokenCred
from nat.data_models.common import SerializableSecretStr
from pydantic import Field
from pydantic import SecretStr

from datarobot_genai.nat.helpers import parse_private_jwk

logger = logging.getLogger(__name__)

# Default header name used to pass the Okta access token into the agent.
# Mirrors the convention in dell-okta-datarobot-application.
DEFAULT_SOURCE_HEADER = "x-datarobot-okta-access-token"


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------


class OktaTokenExchangeError(Exception):
    """Raised when any step of the Okta token exchange fails."""


# ---------------------------------------------------------------------------
# Lazy Okta AI SDK import
# ---------------------------------------------------------------------------


def _lazy_okta_sdk():  # type: ignore[no-untyped-def]
    """Lazy-import Okta AI SDK so the module can be imported without the extra installed."""
    try:
        from okta_ai_sdk import OktaAIConfig
        from okta_ai_sdk import OktaAISDK
        from okta_ai_sdk.types import AuthServerTokenRequest
    except ImportError as exc:
        raise ImportError(
            "The 'okta-ai-sdk-proto' package is required for Okta token exchange. "
            "Install the auth-okta extra: pip install datarobot-genai[auth-okta]"
        ) from exc
    return OktaAISDK, OktaAIConfig, AuthServerTokenRequest


# ---------------------------------------------------------------------------
# Result dataclass (public, framework-agnostic)
# ---------------------------------------------------------------------------


@dataclass
class OktaTokenExchangeResult:
    """Result of a full two-step Okta token exchange.

    Attributes
    ----------
    auth_server_token : str
        The final authorization-server token ready for downstream use.
    id_jag_token : str
        The intermediate ID-JAG token (kept for debugging / audit).
    refresh_token : str | None
        Optional refresh token returned by the authorization server.
    """

    auth_server_token: str
    id_jag_token: str
    refresh_token: str | None = None


# ---------------------------------------------------------------------------
# Agent Card discovery — extract Step 2 parameters from a target agent's card
# ---------------------------------------------------------------------------

# Parses the authorization server ID from an Okta token URL:
#   https://{domain}/oauth2/{authorization_server_id}/v1/token
_AUTH_SERVER_ID_RE = re.compile(r"/oauth2/([^/]+)/")


@dataclass
class AgentCardAuthParams:
    """OAuth2 parameters discovered from a target agent's ``authSchemes``.

    Attributes
    ----------
    token_url : str
        Token endpoint URL from the oauth2 auth scheme.
    authorization_server_id : str
        Okta authorization server ID parsed from ``token_url``.
    scopes : list[str]
        OAuth scopes advertised by the target agent.
    audience : str | None
        Resource-server audience (``service_identifier``) if present.
    """

    token_url: str
    authorization_server_id: str
    scopes: list[str] = field(default_factory=list)
    audience: str | None = None


async def fetch_agent_card_auth_params(agent_card_url: str) -> AgentCardAuthParams:
    """Fetch a target agent's card and extract OAuth2 parameters for Step 2.

    Expects the agent card to contain an ``authSchemes`` array with at least
    one entry of ``scheme: oauth2``, e.g.::

        {
          "authSchemes": [
            {
              "scheme": "oauth2",
              "tokenUrl": "https://org.okta.com/oauth2/<auth_server_id>/v1/token",
              "scopes": ["blog:write"],
              "service_identifier": "https://app.example.com/resource"
            }
          ]
        }

    Parameters
    ----------
    agent_card_url : str
        Full URL to the target agent's card JSON.

    Returns
    -------
    AgentCardAuthParams
        Discovered OAuth2 parameters.

    Raises
    ------
    OktaTokenExchangeError
        If the agent card cannot be fetched, contains no oauth2 auth scheme,
        or the ``tokenUrl`` cannot be parsed.
    """
    import httpx

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            response = await client.get(agent_card_url)
            response.raise_for_status()
            card_data = response.json()
    except httpx.HTTPError as exc:
        raise OktaTokenExchangeError(f"Failed to fetch agent card from {agent_card_url}") from exc

    auth_schemes = card_data.get("authSchemes") or []
    oauth2_scheme = next(
        (s for s in auth_schemes if isinstance(s, dict) and s.get("scheme") == "oauth2"),
        None,
    )

    if not oauth2_scheme:
        raise OktaTokenExchangeError(
            f"No oauth2 auth scheme found in agent card at {agent_card_url}"
        )

    token_url = oauth2_scheme.get("tokenUrl")
    if not token_url:
        raise OktaTokenExchangeError(
            f"oauth2 auth scheme in agent card at {agent_card_url} has no tokenUrl"
        )

    m = _AUTH_SERVER_ID_RE.search(token_url)
    if not m:
        raise OktaTokenExchangeError(
            f"Cannot parse authorization_server_id from tokenUrl: {token_url}"
        )
    authorization_server_id = m.group(1)

    scopes = oauth2_scheme.get("scopes") or []
    audience = oauth2_scheme.get("service_identifier")

    params = AgentCardAuthParams(
        token_url=token_url,
        authorization_server_id=authorization_server_id,
        scopes=scopes,
        audience=audience,
    )
    logger.info(
        "Discovered OAuth2 params from agent card %s (auth_server=%s, scopes=%s, audience=%s)",
        agent_card_url,
        params.authorization_server_id,
        params.scopes,
        params.audience,
    )
    return params


# ---------------------------------------------------------------------------
# Core exchange client (framework-agnostic, re-usable outside NAT)
# ---------------------------------------------------------------------------


class OktaTokenExchangeClient:
    """Client that performs the Okta two-step cross-app token exchange.

    Instantiate once per set of credentials and re-use across requests.
    """

    def __init__(
        self,
        *,
        okta_domain: str,
        client_id: str,
        client_secret: str,
        authorization_server_id: str,
        principal_id: str,
        private_jwk: dict,
        resource_server_audience: str | None = None,
        default_scope: str = "read_data",
    ) -> None:
        okta_ai_sdk_cls, okta_ai_config_cls, _ = _lazy_okta_sdk()

        self._private_jwk = private_jwk
        self._default_scope = default_scope
        self._authorization_server_id = authorization_server_id
        self._principal_id = principal_id
        self._resource_server_audience = resource_server_audience

        okta_ai_config = okta_ai_config_cls(
            okta_domain=okta_domain,
            client_id=client_id,
            client_secret=client_secret,
            authorization_server_id=authorization_server_id,
            principal_id=principal_id,
            private_jwk=private_jwk,
        )
        self._sdk = okta_ai_sdk_cls(okta_ai_config)
        self._id_jag_audience = f"{okta_domain}/oauth2/{authorization_server_id}"

    # ------------------------------------------------------------------
    # Step 1: access token → ID-JAG token
    # ------------------------------------------------------------------

    def exchange_access_token_for_id_jag_token(
        self,
        access_token: str,
        *,
        scope: str | None = None,
    ) -> str:
        """Exchange an Okta access token for an ID-JAG token.

        Parameters
        ----------
        access_token : str
            The caller's Okta access token.
        scope : str | None
            OAuth scope to request.  Defaults to ``self._default_scope``.

        Returns
        -------
        str
            The ID-JAG token.

        Raises
        ------
        OktaTokenExchangeError
            If the exchange or subsequent verification fails.
        """
        scope = scope or self._default_scope

        logger.info("Exchanging access token for ID-JAG token")
        try:
            id_jag_result = self._sdk.cross_app_access.exchange_token(
                token=access_token,
                token_type="access_token",
                audience=self._id_jag_audience,
                scope=scope,
            )
        except Exception as exc:
            raise OktaTokenExchangeError(
                "Failed to exchange access token for ID-JAG token"
            ) from exc

        id_jag_token = id_jag_result.access_token

        # Verify the ID-JAG token
        verification = self._sdk.cross_app_access.verify_id_jag_token(
            token=id_jag_token,
            audience=self._id_jag_audience,
        )
        if not verification.valid:
            raise OktaTokenExchangeError(f"ID-JAG token verification failed: {verification.error}")

        logger.debug("ID-JAG token verified (exp=%s)", getattr(verification, "exp", None))
        return id_jag_token

    # ------------------------------------------------------------------
    # Step 2: ID-JAG token → authorization-server token
    # ------------------------------------------------------------------

    def exchange_id_jag_token_for_auth_server_token(self, id_jag_token: str) -> str:
        """Exchange an ID-JAG token for an authorization-server token.

        Parameters
        ----------
        id_jag_token : str
            A verified ID-JAG token obtained from step 1.

        Returns
        -------
        str
            The authorization-server access token.

        Raises
        ------
        OktaTokenExchangeError
            If the exchange fails.
        """
        _, _, auth_server_token_request_cls = _lazy_okta_sdk()

        logger.info("Exchanging ID-JAG token for authorization server token")
        try:
            auth_server_request = auth_server_token_request_cls(
                id_jag_token=id_jag_token,
                authorization_server_id=self._authorization_server_id,
                principal_id=self._principal_id,
                private_jwk=self._private_jwk,
            )
            result = self._sdk.cross_app_access.exchange_id_jag_for_auth_server_token(
                auth_server_request
            )
        except Exception as exc:
            raise OktaTokenExchangeError(
                "Failed to exchange ID-JAG token for authorization server token"
            ) from exc

        logger.debug("Obtained authorization server token")
        return result.access_token

    # ------------------------------------------------------------------
    # Full two-step exchange (convenience)
    # ------------------------------------------------------------------

    def exchange(
        self,
        access_token: str,
        *,
        scope: str | None = None,
        verify_result: bool = False,
    ) -> OktaTokenExchangeResult:
        """Run the full two-step exchange: access_token → ID-JAG → auth-server token.

        Parameters
        ----------
        access_token : str
            The caller's Okta access token.
        scope : str | None
            OAuth scope for step 1.
        verify_result : bool
            If *True*, verify the final authorization-server token before
            returning.

        Returns
        -------
        OktaTokenExchangeResult
            Container with the final ``auth_server_token`` and the
            intermediate ``id_jag_token``.
        """
        id_jag_token = self.exchange_access_token_for_id_jag_token(access_token, scope=scope)
        auth_server_token = self.exchange_id_jag_token_for_auth_server_token(id_jag_token)

        if verify_result:
            self.verify_auth_server_token(auth_server_token)

        return OktaTokenExchangeResult(
            auth_server_token=auth_server_token,
            id_jag_token=id_jag_token,
        )

    # ------------------------------------------------------------------
    # Optional verification
    # ------------------------------------------------------------------

    def verify_auth_server_token(self, auth_server_token: str) -> None:
        """Verify an authorization-server token against the resource-server audience.

        Parameters
        ----------
        auth_server_token : str
            The token to verify.

        Raises
        ------
        OktaTokenExchangeError
            If verification fails or ``resource_server_audience`` is not configured.
        """
        if not self._resource_server_audience:
            raise OktaTokenExchangeError(
                "Cannot verify auth server token: resource_server_audience is not configured."
            )

        verification = self._sdk.cross_app_access.verify_auth_server_token(
            token=auth_server_token,
            authorization_server_id=self._authorization_server_id,
            audience=self._resource_server_audience,
        )
        if not verification.valid:
            raise OktaTokenExchangeError(
                f"Authorization server token verification failed: {verification.error}"
            )

        logger.debug(
            "Authorization server token verified (exp=%s, scopes=%s)",
            getattr(verification, "exp", None),
            getattr(verification, "scope", None),
        )

    # # ------------------------------------------------------------------
    # # Dynamic exchange — Step 2 uses agent-card-discovered parameters
    # # ------------------------------------------------------------------

    # def exchange_for_agent(
    #     self,
    #     access_token: str,
    #     agent_card_params: AgentCardAuthParams,
    #     *,
    #     scope: str | None = None,
    #     verify_result: bool = False,
    # ) -> OktaTokenExchangeResult:
    #     """Run the two-step exchange with Step 2 parameters from an agent card.

    #     Step 1 uses the locally configured authorization server (same as
    #     ``exchange()``).  Step 2 uses the ``authorization_server_id``
    #     discovered from the target agent's card.

    #     Parameters
    #     ----------
    #     access_token : str
    #         The caller's Okta access token.
    #     agent_card_params : AgentCardAuthParams
    #         OAuth2 parameters discovered from the target agent's card.
    #     scope : str | None
    #         OAuth scope for Step 1.
    #     verify_result : bool
    #         If *True*, verify the final token using the agent card's
    #         audience.

    #     Returns
    #     -------
    #     OktaTokenExchangeResult
    #     """
    #     id_jag_token = self.exchange_access_token_for_id_jag_token(
    #         access_token, scope=scope,
    #     )
    #     auth_server_token = self.exchange_id_jag_token_for_auth_server_token(
    #         id_jag_token,
    #         authorization_server_id=agent_card_params.authorization_server_id,
    #     )

    #     if verify_result:
    #         self.verify_auth_server_token(
    #             auth_server_token,
    #             authorization_server_id=agent_card_params.authorization_server_id,
    #             audience=agent_card_params.audience,
    #         )

    #     return OktaTokenExchangeResult(
    #         auth_server_token=auth_server_token,
    #         id_jag_token=id_jag_token,
    #     )


# ---------------------------------------------------------------------------
# NAT plugin: Config (workflow.yaml _type: okta_token_exchange)
# ---------------------------------------------------------------------------


class OktaTokenExchangeProviderConfig(AuthProviderBaseConfig, name="okta_token_exchange"):  # type: ignore[call-arg]
    """NAT auth-provider configuration for Okta cross-app token exchange.

    Workflow.yaml example::

        authentication:
          okta_exchange:
            _type: okta_token_exchange
            okta_domain: https://your-org.okta.com
            client_id: 0oa...
            client_secret: ...
            authorization_server_id: default
            principal_id: ...
            private_jwk: <base64-encoded-or-json>
            scope: read_data
    """

    okta_domain: str = Field(
        description="Okta organisation URL (e.g. https://your-org.okta.com).",
    )
    client_id: str = Field(
        description="OAuth client ID registered in Okta.",
    )
    client_secret: SerializableSecretStr = Field(
        description="OAuth client secret.",
    )
    authorization_server_id: str = Field(
        description="Custom authorisation server ID (e.g. 'default').",
    )
    principal_id: str = Field(
        description="Service principal identifier used for the ID-JAG exchange.",
    )
    private_jwk: SerializableSecretStr = Field(
        description=(
            "Private JWK for signing assertions. "
            "Accepts a base64-encoded JSON string or a plain JSON string."
        ),
    )
    scope: str = Field(
        description="OAuth scope requested during the access-token → ID-JAG exchange.",
    )


# ---------------------------------------------------------------------------
# NAT plugin: Provider
# ---------------------------------------------------------------------------


class OktaTokenExchangeProvider(AuthProviderBase[OktaTokenExchangeProviderConfig]):
    """NAT auth provider that performs the Okta two-step cross-app token exchange."""

    def __init__(
        self,
        config: OktaTokenExchangeProviderConfig,
        config_name: str | None = None,
    ) -> None:
        super().__init__(config)

        private_jwk = parse_private_jwk(config.private_jwk.get_secret_value())

        self._client = OktaTokenExchangeClient(
            okta_domain=config.okta_domain,
            client_id=config.client_id,
            client_secret=config.client_secret.get_secret_value(),
            authorization_server_id=config.authorization_server_id,
            principal_id=config.principal_id,
            private_jwk=private_jwk,
            resource_server_audience=config.resource_server_audience,
            default_scope=config.scope,
        )
        self._agent_card_params: AgentCardAuthParams | None = None

    def _get_subject_token_from_context(self) -> str | None:
        """Extract the Okta access token from the NAT runtime context headers."""
        from datarobot_genai.nat.helpers import extract_headers_from_context

        header_name = self.config.source_header_name
        headers = extract_headers_from_context([header_name])
        return headers.get(header_name)

    async def _get_agent_card_params(self) -> AgentCardAuthParams | None:
        """Return cached agent card params, fetching on first call if configured."""
        if self._agent_card_params is not None:
            return self._agent_card_params

        if not self.config.agent_card_url:
            return None

        self._agent_card_params = await fetch_agent_card_auth_params(
            self.config.agent_card_url,
        )
        return self._agent_card_params

    async def authenticate(
        self,
        user_id: str | None = None,
        *,
        subject_token: str | None = None,
        **kwargs: Any,
    ) -> AuthResult:
        """Perform the two-step Okta token exchange.

        Parameters
        ----------
        user_id
            Optional user identifier (unused in this flow but required by interface).
        subject_token
            The caller's Okta access token.  When *None*, the provider attempts
            to read it from the incoming request header configured via
            ``source_header_name``.
        """
        if not subject_token:
            subject_token = self._get_subject_token_from_context()

        if not subject_token:
            raise OktaTokenExchangeError(
                f"No Okta access token available. "
                f"Pass subject_token or ensure the '{self.config.source_header_name}' "
                f"header is forwarded."
            )

        agent_card_params = await self._get_agent_card_params()

        if agent_card_params:
            # Dynamic mode — Step 2 uses agent-card-discovered parameters.
            result = self._client.exchange_for_agent(
                access_token=subject_token,
                agent_card_params=agent_card_params,
            )
        else:
            # Static mode — Step 2 uses config authorization_server_id.
            result = self._client.exchange(access_token=subject_token)

        return AuthResult(
            credentials=[BearerTokenCred(token=SecretStr(result.auth_server_token))],
            raw={
                "id_jag_token": result.id_jag_token,
                "auth_server_token": result.auth_server_token,
            },
        )


# ---------------------------------------------------------------------------
# NAT plugin registration
# ---------------------------------------------------------------------------


@register_auth_provider(config_type=OktaTokenExchangeProviderConfig)
async def okta_token_exchange_provider(
    config: OktaTokenExchangeProviderConfig, builder: Builder
) -> AsyncGenerator[OktaTokenExchangeProvider]:
    """Create an Okta token exchange provider for NAT's plugin discovery system."""
    yield OktaTokenExchangeProvider(config=config)
