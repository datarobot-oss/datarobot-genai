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

"""Okta token exchange auth provider for A2A agent communication.

Registered as ``_type: okta_cross_app_access`` in workflow YAML.

Discovery phase: forwards the incoming ``x-datarobot-external-access-token`` header
directly to the agent card endpoint as a Bearer token.

Call phase: executes a two-step Okta Cross App Access (XAA) token exchange:

  Step 1 — exchange the incoming Okta access token for an ID-JAG token via the
  org authorization server (``{trusted_issuer}/v1/token``).

  Step 2 — exchange the ID-JAG token for a scoped agent token via the custom
  authorization server token endpoint published in the agent card's
  ``securitySchemes`` / RFC 8693 extension.

Credentials (``principal_id``, ``private_jwk``) are loaded automatically from
environment variables / Runtime Parameters / ``.env`` / ``file_secrets`` via
:class:`_OktaSettings`.  They do **not** need to be specified in
``workflow.yaml`` — the minimal YAML is::

    authentication:
      okta_auth:
        _type: okta_cross_app_access

    function_groups:
      remote_agent:
        _type: authenticated_a2a_client
        url: "https://..."
        auth_provider: okta_auth

Environment variables consumed (map to the reference ``Config`` field names):

* ``PRINCIPAL_ID``     — Okta AI agent principal ID
* ``PRIVATE_JWK``      — base64-encoded or raw-JSON private JWK
"""

import base64
import json
import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

from a2a.types import AgentCard
from datarobot.core.config import DataRobotAppFrameworkBaseSettings
from nat.authentication.interfaces import AuthProviderBase
from nat.builder.builder import Builder
from nat.builder.context import Context
from nat.cli.register_workflow import register_auth_provider
from nat.data_models.authentication import AuthProviderBaseConfig
from nat.data_models.authentication import AuthResult
from nat.data_models.authentication import BearerTokenCred
from pydantic import Field
from pydantic import SecretStr

from datarobot_genai.dragent.plugins.auth_a2a_client import A2ADiscoveryAuthMixin

try:
    from okta_client.authfoundation import LocalKeyProvider
    from okta_client.authfoundation import OAuth2Client
    from okta_client.authfoundation import OAuth2ClientConfiguration
    from okta_client.authfoundation.oauth2.client_authorization import ClientAssertionAuthorization
    from okta_client.authfoundation.oauth2.jwt_bearer_claims import JWTBearerClaims
    from okta_client.oauth2auth import CrossAppAccessFlow
    from okta_client.oauth2auth import CrossAppAccessTarget
except ImportError:
    LocalKeyProvider = None  # type: ignore[assignment,misc]
    OAuth2Client = None  # type: ignore[assignment,misc]
    OAuth2ClientConfiguration = None  # type: ignore[assignment,misc]
    ClientAssertionAuthorization = None  # type: ignore[assignment,misc]
    JWTBearerClaims = None  # type: ignore[assignment,misc]
    CrossAppAccessFlow = None  # type: ignore[assignment,misc]
    CrossAppAccessTarget = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Environment / Runtime Parameter settings
# ---------------------------------------------------------------------------


class _OktaSettings(DataRobotAppFrameworkBaseSettings):
    """Reads Okta credentials from env vars, Runtime Parameters, .env, or file_secrets.

    Field names map to the reference application ``Config`` naming convention so
    that the same Runtime Parameters work across both the application and the
    DataRobot agent template.
    """

    principal_id: str | None = None
    """Okta AI agent principal ID (``PRINCIPAL_ID``)."""

    private_jwk: str | None = None
    """Base64-encoded or raw-JSON private JWK (``PRIVATE_JWK``)."""


def _get_default_principal_id() -> str | None:
    return _OktaSettings().principal_id


def _get_default_private_jwk() -> SecretStr | None:
    if jwk := _OktaSettings().private_jwk:
        return SecretStr(jwk)
    return None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_JWT_BEARER_GRANT_TYPE_URI = "urn:ietf:params:oauth:grant-type:jwt-bearer"


# ---------------------------------------------------------------------------
# _CrossAppFlowParams — populated from the fetched AgentCard
# ---------------------------------------------------------------------------


@dataclass
class _CrossAppFlowParams:
    """Agent card parameters needed for the Okta XAA flow.

    Populated by :meth:`OAuth2CrossApplicationAccessOAuth2AuthProvider.set_agent_card` from
    ``securitySchemes`` (OpenAPI layer) and ``capabilities.extensions`` (A2A layer).
    """

    token_url: str
    """From ``securitySchemes.oauth2.flows.clientCredentials.tokenUrl``.
    Stored for reference; the SDK discovers the endpoint from ``exchange_audience``."""

    trusted_issuer: str
    """Org-level AS issuer (``token_exchange.trusted_issuer``); runs Step 1 (RFC 8693).
    Passed to ``OAuth2ClientConfiguration(issuer=...)``.  JWT assertion ``aud`` is
    derived as ``trusted_issuer + "/oauth2/v1/token"``."""

    exchange_audience: str
    """Resource AS issuer (``token_exchange.audience``); dual role in the SDK:
    ``CrossAppAccessTarget(issuer=...)`` for Step 2 and ``flow.start(audience=...)``
    so the originating AS embeds the right audience in the ID-JAG."""

    target_audience: str
    """Final resource identifier for the agent (``params.target_audience``).
    A2A metadata; not passed to the Okta SDK."""

    token_endpoint_auth_method: str
    """Client auth method (``params.token_endpoint_auth_method``).
    ``"private_key_jwt"`` triggers ``ClientAssertionAuthorization`` with ``LocalKeyProvider``."""

    id_jag_scopes: list[str]
    """Step 1 scopes; sourced from provider config, not the agent card."""


# ---------------------------------------------------------------------------
# OAuth2CrossApplicationAccessAuthProviderConfig
# ---------------------------------------------------------------------------


class OAuth2CrossApplicationAccessAuthProviderConfig(
    AuthProviderBaseConfig,
    name="okta_cross_app_access",
):  # type: ignore[call-arg]
    """Configuration for :class:`OAuth2CrossApplicationAccessOAuth2AuthProvider`.

    ``principal_id`` and ``private_jwk`` are auto-loaded from env vars /
    Runtime Parameters / ``.env`` / ``file_secrets`` — no ``workflow.yaml``
    entries needed for credentials.
    """

    okta_token_header: str = Field(
        default="x-datarobot-external-access-token",
        description=(
            "Incoming header carrying the caller's Okta access token. "
            "Forwarded as Bearer for discovery; used as subject token in Step 1. "
            "Matched case-insensitively."
        ),
    )
    fallback_token_headers: list[str] = Field(
        default=["authorization"],
        description=(
            "Fallback headers to try (in order) when ``okta_token_header`` is absent. "
            "Useful for local development without an API gateway that remaps "
            "``Authorization`` → ``x-datarobot-external-access-token``. "
            "If the value starts with 'Bearer ', the prefix is stripped automatically."
        ),
    )
    principal_id: str | None = Field(
        default_factory=_get_default_principal_id,
        description=(
            "Okta AI agent principal ID (env: ``PRINCIPAL_ID``). "
            "Used as ``iss``/``sub`` in the JWT client assertion."
        ),
    )
    private_jwk: SecretStr | None = Field(
        default_factory=_get_default_private_jwk,
        description=(
            "Base64-encoded or raw-JSON private JWK (env: ``PRIVATE_JWK``). "
            "Signs JWT client assertions."
        ),
    )
    id_jag_scopes: list[str] = Field(
        default=["read_data"],
        description=(
            "Scopes for the Step 1 ID-JAG request. ``['read_data']`` matches "
            "Okta XAA reference implementations."
        ),
    )


# ---------------------------------------------------------------------------
# OAuth2CrossApplicationAccessOAuth2AuthProvider
# ---------------------------------------------------------------------------


class OAuth2CrossApplicationAccessOAuth2AuthProvider(
    A2ADiscoveryAuthMixin,
    AuthProviderBase[OAuth2CrossApplicationAccessAuthProviderConfig],
):
    """Auth provider for Okta XAA A2A calls.

    * **Discovery** — forwards the incoming Okta bearer token as ``Authorization: Bearer``.
    * **Call** — two-step XAA token exchange (RFC 8693 → RFC 7523) via ``okta-client-python``.
    """

    def __init__(self, config: OAuth2CrossApplicationAccessAuthProviderConfig) -> None:
        super().__init__(config)
        self._flow_params: _CrossAppFlowParams | None = None

    # ------------------------------------------------------------------
    # A2ADiscoveryAuthMixin
    # ------------------------------------------------------------------

    async def authenticate_for_discovery(self, user_id: str | None = None) -> dict[str, str]:
        """Return the incoming Okta token as ``Authorization: Bearer`` headers.

        Raises ``RuntimeError`` if ``okta_token_header`` is absent from the request context.
        """
        token = self._extract_okta_token()
        logger.info(
            "Forwarding Okta token from header '%s' for agent card discovery",
            self.config.okta_token_header,
        )
        return {"Authorization": f"Bearer {token}"}

    # ------------------------------------------------------------------
    # Agent card injection (called by _AuthenticatedA2ABaseClient after discovery)
    # ------------------------------------------------------------------

    def set_agent_card(self, card: AgentCard) -> None:
        """Parse the agent card and store :class:`_CrossAppFlowParams`.

        Called before ``authenticate()``.  Raises ``ValueError`` if security
        schemes or the JWT Bearer capability extension are missing.
        """
        self._flow_params = _parse_cross_app_params(card, id_jag_scopes=self.config.id_jag_scopes)
        logger.info(
            "Agent card parsed: trusted_issuer=%s, exchange_audience=%s, "
            "target_audience=%s, token_endpoint_auth_method=%s",
            self._flow_params.trusted_issuer,
            self._flow_params.exchange_audience,
            self._flow_params.target_audience,
            self._flow_params.token_endpoint_auth_method,
        )

    # ------------------------------------------------------------------
    # AuthProviderBase
    # ------------------------------------------------------------------

    async def authenticate(self, user_id: str | None = None, **kwargs: Any) -> AuthResult | None:
        """Obtain a scoped agent token via the two-step Okta XAA token exchange.

        Requires :meth:`set_agent_card` to have been called first.
        """
        if self._flow_params is None:
            raise RuntimeError(
                "OAuth2CrossApplicationAccessOAuth2AuthProvider.authenticate() called "
                "before set_agent_card(). Ensure the provider is used with "
                "authenticated_a2a_client."
            )

        if not self.config.principal_id:
            raise ValueError("principal_id is required for the Okta cross-app access flow")

        private_jwk = self._parse_private_jwk()
        if not private_jwk:
            raise ValueError("private_jwk is required for the Okta cross-app access flow")

        access_token = self._extract_okta_token()
        flow = self._build_cross_app_flow(private_jwk=private_jwk)

        logger.info(
            "Step 1: exchanging access token for ID-JAG (trusted_issuer=%s, "
            "exchange_audience=%s, user_id=%s)",
            self._flow_params.trusted_issuer,
            self._flow_params.exchange_audience,
            user_id,
        )
        await flow.start(token=access_token, audience=self._flow_params.exchange_audience)

        logger.info(
            "Step 2: exchanging ID-JAG for scoped agent token (target_audience=%s)",
            self._flow_params.target_audience,
        )
        token_result = await flow.resume()

        logger.info("Cross-app access flow completed successfully")
        return AuthResult(credentials=[BearerTokenCred(token=token_result.access_token)])

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_okta_token(self) -> str:
        """Extract the Okta access token from NAT request context headers.

        Tries ``okta_token_header`` first, then each entry in
        ``fallback_token_headers`` (stripping ``Bearer `` prefix if present).
        This supports local development where no API gateway remaps
        ``Authorization`` into the canonical ``x-datarobot-external-access-token``.
        """
        context_headers: dict[str, str] = Context.get().metadata.headers or {}

        # Primary header (already a raw token value, no prefix expected)
        token = context_headers.get(self.config.okta_token_header.lower())
        if token:
            return token

        # Fallback headers (e.g. Authorization: Bearer <token>)
        for fallback in self.config.fallback_token_headers:
            value = context_headers.get(fallback.lower())
            if value:
                # Strip "Bearer " prefix if present (case-insensitive)
                if value.lower().startswith("bearer "):
                    value = value[len("bearer ") :]
                logger.debug(
                    "Primary header '%s' absent; using fallback '%s'",
                    self.config.okta_token_header,
                    fallback,
                )
                return value

        raise RuntimeError(
            f"Header '{self.config.okta_token_header}' not found in the request context "
            f"(also tried fallbacks: {self.config.fallback_token_headers}). "
            "The Okta access token must be forwarded with every agent call."
        )

    def _parse_private_jwk(self) -> dict[str, Any] | None:
        """Decode and parse the private JWK from config (supports base64 and raw JSON)."""
        if not self.config.private_jwk:
            return None
        raw = self.config.private_jwk.get_secret_value()
        # Try base64-decode first (new format)
        try:
            return json.loads(base64.b64decode(raw).decode())
        except Exception:
            pass
        # Try direct JSON
        try:
            return json.loads(raw)
        except Exception:
            pass
        raise ValueError(
            "Could not parse private_jwk: expected base64-encoded JSON or raw JSON string."
        )

    def _build_cross_app_flow(self, private_jwk: dict[str, Any]) -> Any:
        """Construct a :class:`CrossAppAccessFlow` from the Okta SDK.

        ``trusted_issuer`` → Step 1 AS (``OAuth2ClientConfiguration(issuer=...)``).
        ``exchange_audience`` → Step 2 AS (``CrossAppAccessTarget(issuer=...)``) and
        the ``audience`` arg for ``flow.start()``.
        """
        if CrossAppAccessFlow is None:
            raise ImportError(
                "okta-client-python is required for the Okta cross-app access flow. "
                "Install it with: pip install datarobot-genai[auth]"
            )

        assert self._flow_params is not None  # validated by authenticate() before this call

        client_authorization = None
        if self._flow_params.token_endpoint_auth_method == "private_key_jwt":
            key_provider = LocalKeyProvider.from_jwk(private_jwk, algorithm="RS256")
            token_endpoint = self._flow_params.trusted_issuer.rstrip("/") + "/oauth2/v1/token"
            client_authorization = ClientAssertionAuthorization(
                assertion_claims=JWTBearerClaims(
                    issuer=self.config.principal_id,
                    subject=self.config.principal_id,
                    audience=token_endpoint,
                    expires_in=60,
                ),
                key_provider=key_provider,
            )

        config = OAuth2ClientConfiguration(
            issuer=self._flow_params.trusted_issuer,
            scope=self._flow_params.id_jag_scopes,
            client_authorization=client_authorization,
        )

        oauth_client = OAuth2Client(configuration=config)
        target = CrossAppAccessTarget(issuer=self._flow_params.exchange_audience)
        return CrossAppAccessFlow(client=oauth_client, target=target)


# ---------------------------------------------------------------------------
# Agent card parsing helpers
# ---------------------------------------------------------------------------


def _parse_cross_app_params(
    card: AgentCard,
    id_jag_scopes: list[str] | None = None,
) -> _CrossAppFlowParams:
    """Extract :class:`_CrossAppFlowParams` from a fetched :class:`AgentCard`.

    Combines ``securitySchemes.oauth2.flows.clientCredentials`` (``tokenUrl``) and
    ``capabilities.extensions`` (JWT Bearer entry) into a single params object.
    """
    token_url, _ = _parse_security_schemes(card)
    ext = _parse_cross_app_extension(card)

    return _CrossAppFlowParams(
        token_url=token_url,
        trusted_issuer=ext.trusted_issuer,
        exchange_audience=ext.exchange_audience,
        target_audience=ext.target_audience,
        token_endpoint_auth_method=ext.token_endpoint_auth_method,
        id_jag_scopes=id_jag_scopes if id_jag_scopes is not None else ["read_data"],
    )


def _parse_security_schemes(card: AgentCard) -> tuple[str, list[str]]:
    """Return ``(agent_token_url, scopes)`` from the agent card security schemes."""
    if not card.security_schemes:
        raise ValueError("Agent card has no securitySchemes — cannot extract token_url.")

    oauth2 = card.security_schemes.get("oauth2")
    if not oauth2:
        raise ValueError("Agent card securitySchemes has no 'oauth2' entry.")

    flows = getattr(oauth2.root, "flows", None)
    if not flows:
        raise ValueError("Agent card oauth2 security scheme has no flows.")

    cc_flow = getattr(flows, "client_credentials", None)
    if not cc_flow:
        raise ValueError("Agent card oauth2 flows has no clientCredentials flow.")

    token_url: str = str(cc_flow.token_url)
    scopes: list[str] = list(cc_flow.scopes.keys()) if cc_flow.scopes else []

    return token_url, scopes


@dataclass
class _CrossAppExtensionFields:
    """Raw fields extracted from the JWT Bearer capability extension."""

    trusted_issuer: str
    exchange_audience: str
    target_audience: str
    token_endpoint_auth_method: str


def _parse_cross_app_extension(card: AgentCard) -> _CrossAppExtensionFields:
    """Extract Cross-Application Access fields from the JWT Bearer capability extension.

    Finds the extension with ``uri == _JWT_BEARER_GRANT_TYPE_URI`` and validates
    required fields.  ``token_request.grant_type`` is validated but not returned.

    Raises ``ValueError`` if the extension is missing or any required field is absent.
    """
    extensions = (
        card.capabilities.extensions if card.capabilities and card.capabilities.extensions else []
    )
    for ext in extensions:
        if ext.uri == _JWT_BEARER_GRANT_TYPE_URI:
            params = ext.params or {}
            token_exchange = params.get("token_exchange") or {}
            token_request = params.get("token_request") or {}

            token_endpoint_auth_method = params.get("token_endpoint_auth_method")
            trusted_issuer = token_exchange.get("trusted_issuer")
            exchange_audience = token_exchange.get("audience")
            grant_type = token_request.get("grant_type")
            target_audience = token_request.get("audience")

            missing = [
                name
                for name, val in [
                    ("token_endpoint_auth_method", token_endpoint_auth_method),
                    ("token_exchange.trusted_issuer", trusted_issuer),
                    ("token_exchange.audience", exchange_audience),
                    ("token_request.grant_type", grant_type),
                    ("token_request.audience", target_audience),
                ]
                if not val
            ]
            if missing:
                raise ValueError(
                    f"Agent card Cross-Application Access extension is missing required "
                    f"fields: {missing}"
                )

            return _CrossAppExtensionFields(
                trusted_issuer=trusted_issuer,  # type: ignore[arg-type]
                exchange_audience=exchange_audience,  # type: ignore[arg-type]
                target_audience=target_audience,  # type: ignore[arg-type]
                token_endpoint_auth_method=token_endpoint_auth_method,  # type: ignore[arg-type]
            )

    raise ValueError(
        f"Agent card capabilities.extensions has no entry with uri='{_JWT_BEARER_GRANT_TYPE_URI}'. "
        "The remote agent must advertise the Cross-Application Access extension."
    )


# ---------------------------------------------------------------------------
# NAT registration
# ---------------------------------------------------------------------------


@register_auth_provider(config_type=OAuth2CrossApplicationAccessAuthProviderConfig)
async def okta_cross_app_access_auth_provider(
    config: OAuth2CrossApplicationAccessAuthProviderConfig, builder: Builder
) -> AsyncGenerator[OAuth2CrossApplicationAccessOAuth2AuthProvider, None]:
    """NAT auth provider factory for Okta Cross-Application Access."""
    yield OAuth2CrossApplicationAccessOAuth2AuthProvider(config=config)


# ---------------------------------------------------------------------------
# Backward-compatible aliases (deprecated — use the OAuth2* names)
# ---------------------------------------------------------------------------

OktaCrossApplicationAccessAuthProviderConfig = OAuth2CrossApplicationAccessAuthProviderConfig
OktaCrossApplicationAccessAuthProvider = OAuth2CrossApplicationAccessOAuth2AuthProvider
