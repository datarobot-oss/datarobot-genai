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

Registered as ``_type: okta_token_exchange`` in workflow YAML.

Discovery phase: forwards the incoming ``x-datarobot-okta-access-token`` header
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
        _type: okta_token_exchange

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
    """Full configuration context parsed from the agent card for the Okta XAA flow.

    Populated by :meth:`OktaTokenExchangeAuthProvider.set_agent_card` from
    the agent card's ``securitySchemes`` (OpenAPI layer) and JWT Bearer
    ``capabilities.extensions`` (A2A layer).

    Every field used by the Okta SDK during the two-step Cross-Application
    Access flow is extracted here so that the authentication handler never
    needs to hardcode or derive values at runtime.
    """

    token_url: str
    """Token endpoint URL from
    ``securitySchemes.oauth2.flows.clientCredentials.tokenUrl``.
    Represents the resource AS token endpoint.  Stored for reference; the SDK
    discovers the endpoint automatically from ``exchange_audience``."""

    trusted_issuer: str
    """**Originating** authorization server issuer
    (``capabilities.extensions[].params.token_exchange.trusted_issuer``).
    Passed to ``OAuth2ClientConfiguration(issuer=...)`` — this is the AS
    that runs Step 1 (RFC 8693 token exchange) and validates the JWT client
    assertion.  The assertion ``audience`` claim is derived as
    ``trusted_issuer + "/oauth2/v1/token"``."""

    exchange_audience: str
    """**Resource** authorization server issuer
    (``capabilities.extensions[].params.token_exchange.audience``).
    Serves a dual role in the SDK:

    * ``CrossAppAccessTarget(issuer=...)`` — structural config that tells
      ``resume()`` which server to talk to for Step 2 (JWT bearer grant).
    * ``flow.start(audience=...)`` — logical parameter that tells the
      originating AS what audience to embed in the ID-JAG.

    Per Okta SDK docs these are the same issuer URL in the common case."""

    target_audience: str
    """Final resource identifier for the agent being called
    (``capabilities.extensions[].params.target_audience``).
    A2A-level metadata; not passed to the Okta SDK directly."""

    token_endpoint_auth_method: str
    """Client authentication method
    (``capabilities.extensions[].params.token_endpoint_auth_method``).
    When ``"private_key_jwt"``, :class:`ClientAssertionAuthorization` with
    :class:`LocalKeyProvider` is initialized for JWT client assertions."""

    id_jag_scopes: list[str]
    """OAuth scopes for Step 1 (ID-JAG request). Defaults to ``["read_data"]``.
    Sourced from the provider config, not the agent card."""


# ---------------------------------------------------------------------------
# OktaTokenExchangeAuthProviderConfig
# ---------------------------------------------------------------------------


class OktaTokenExchangeAuthProviderConfig(
    AuthProviderBaseConfig,
    name="okta_token_exchange",
):  # type: ignore[call-arg]
    """Configuration for :class:`OktaTokenExchangeAuthProvider`.

    Credential fields (``principal_id``, ``private_jwk``) are populated
    automatically from environment variables / Runtime Parameters / ``.env`` /
    ``file_secrets`` via :class:`_OktaSettings` and do **not** need to appear
    in ``workflow.yaml``.

    Override any field explicitly in the YAML when you need to supply a value
    that is not in the environment (e.g. in unit tests).
    """

    okta_token_header: str = Field(
        default="x-datarobot-okta-access-token",
        description=(
            "Name of the incoming request header that carries the caller's Okta access "
            "token. Used for agent card discovery and as the subject token in Step 1. "
            "Header names are matched case-insensitively."
        ),
    )
    principal_id: str | None = Field(
        default_factory=_get_default_principal_id,
        description=(
            "Okta AI agent principal ID (env: ``PRINCIPAL_ID``). Used as ``iss`` and "
            "``sub`` claims in the JWT client assertion when "
            "``token_endpoint_auth_method`` is ``private_key_jwt``."
        ),
    )
    private_jwk: SecretStr | None = Field(
        default_factory=_get_default_private_jwk,
        description=(
            "Base64-encoded private JWK or raw JSON string (env: ``PRIVATE_JWK``). "
            "Used to sign JWT client assertions when ``token_endpoint_auth_method`` "
            "is ``private_key_jwt``."
        ),
    )
    id_jag_scopes: list[str] = Field(
        default=["read_data"],
        description=(
            "OAuth scopes to request in Step 1 of the token exchange (ID-JAG request). "
            "Defaults to ``['read_data']`` which matches Okta XAA reference implementations. "
            "Override when the authorization server requires a different scope set."
        ),
    )


# ---------------------------------------------------------------------------
# OktaTokenExchangeAuthProvider
# ---------------------------------------------------------------------------


class OktaTokenExchangeAuthProvider(
    A2ADiscoveryAuthMixin,
    AuthProviderBase[OktaTokenExchangeAuthProviderConfig],
):
    """Auth provider for Okta-authenticated A2A agent communication.

    Implements :class:`~datarobot_genai.dragent.plugins.auth_a2a_client.A2ADiscoveryAuthMixin`
    so that agent card discovery and A2A RPC calls use separate credentials:

    * **Discovery** — forwards the incoming Okta bearer token from the request
      context as ``Authorization: Bearer <token>``.
    * **Call** — executes the two-step Okta XAA token exchange via the
      ``okta-client-python`` SDK using parameters parsed from the agent card.
    """

    def __init__(self, config: OktaTokenExchangeAuthProviderConfig) -> None:
        super().__init__(config)
        self._flow_params: _CrossAppFlowParams | None = None

    # ------------------------------------------------------------------
    # A2ADiscoveryAuthMixin
    # ------------------------------------------------------------------

    async def authenticate_for_discovery(self, user_id: str | None = None) -> dict[str, str]:
        """Return the incoming Okta token as ``Authorization: Bearer`` headers.

        Reads the header named by :attr:`~OktaTokenExchangeAuthProviderConfig.okta_token_header`
        from the current NAT request context (case-insensitive).

        Raises
        ------
        RuntimeError
            If the expected header is absent from the request context.
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

        Called by
        :class:`~datarobot_genai.dragent.plugins.auth_a2a_client._AuthenticatedA2ABaseClient`
        immediately after the card is resolved, before ``authenticate()`` is invoked.

        Raises
        ------
        ValueError
            If the card does not contain the expected security schemes or
            JWT Bearer capability extension.
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

        Returns
        -------
        AuthResult
            Contains a single :class:`~nat.data_models.authentication.BearerTokenCred`
            with the final scoped agent token.
        """
        if self._flow_params is None:
            raise RuntimeError(
                "OktaTokenExchangeAuthProvider.authenticate() called before set_agent_card(). "
                "Ensure the provider is used with authenticated_a2a_client."
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
        """Extract the Okta access token from NAT request context headers."""
        context_headers: dict[str, str] = Context.get().metadata.headers or {}
        token = context_headers.get(self.config.okta_token_header.lower())
        if not token:
            raise RuntimeError(
                f"Header '{self.config.okta_token_header}' not found in the request context. "
                "The Okta access token must be forwarded with every agent call."
            )
        return token

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
        """Construct a :class:`CrossAppAccessFlow` from the Okta Client SDK.

        Mapping from :attr:`_flow_params` (agent card) to SDK constructor args
        follows Okta's ``target`` vs ``audience`` distinction:

        * ``trusted_issuer`` → ``OAuth2ClientConfiguration(issuer=...)`` — the
          **originating** authorization server that runs Step 1 (RFC 8693 token
          exchange).  The JWT client assertion ``audience`` claim targets this
          server's token endpoint (``trusted_issuer + "/oauth2/v1/token"``).
        * ``token_endpoint_auth_method`` → when ``"private_key_jwt"``,
          ``ClientAssertionAuthorization`` with ``LocalKeyProvider`` is used.
        * ``exchange_audience`` → ``CrossAppAccessTarget(issuer=...)`` — the
          **resource** authorization server that ``resume()`` talks to (Step 2).
          The SDK uses its issuer to discover the token endpoint and rewrite the
          client assertion ``aud`` claim for the JWT bearer exchange.
        * ``exchange_audience`` is also the ``audience`` arg in
          ``flow.start()`` — tells the originating AS what audience the ID-JAG
          should carry so the resource AS will accept it.

        Parameters
        ----------
        private_jwk:
            Parsed private JWK dict used to sign JWT client assertions.
        """
        if CrossAppAccessFlow is None:
            raise ImportError(
                "okta-client-python is required for the Okta cross-app access flow. "
                "Install it with: pip install datarobot-genai[auth-okta]"
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

    Combines fields from two locations in the card:

    * **OpenAPI layer** — ``securitySchemes.oauth2.flows.clientCredentials``
      provides the ``tokenUrl`` (needed to initialize the ``OAuth2Client``).
    * **Extension layer** — ``capabilities.extensions`` (JWT Bearer entry)
      provides the remaining negotiation parameters: ``trusted_issuer``,
      ``exchange_audience``, ``target_audience``, and
      ``token_endpoint_auth_method``.

    Parameters
    ----------
    card:
        Fetched agent card.
    id_jag_scopes:
        Scopes to request in Step 1.  When *None*, defaults to ``["read_data"]``.
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
    """Extract all Cross-Application Access fields from the JWT Bearer extension.

    Reads the ``capabilities.extensions`` entry whose
    ``uri == "urn:ietf:params:oauth:grant-type:jwt-bearer"`` and returns a
    :class:`_CrossAppExtensionFields` containing every parameter required by
    the Okta Cross-Application Access SDK:

    - ``params.target_audience`` — final resource identifier.
    - ``params.token_endpoint_auth_method`` — client auth method.
    - ``params.token_exchange.trusted_issuer`` — org-level AS for the ID-JAG.
    - ``params.token_exchange.audience`` — Step 1 AS destination for
      ``flow.start(audience=...)``.
    - ``params.token_request.grant_type`` — validated present for card
      well-formedness but not returned (always
      ``urn:ietf:params:oauth:grant-type:jwt-bearer``).

    Returns
    -------
    _CrossAppExtensionFields
        All extracted extension parameters needed to configure the SDK.

    Raises
    ------
    ValueError
        If the extension is absent or any required field is missing.
    """
    extensions = (
        card.capabilities.extensions if card.capabilities and card.capabilities.extensions else []
    )
    for ext in extensions:
        if ext.uri == _JWT_BEARER_GRANT_TYPE_URI:
            params = ext.params or {}
            token_exchange = params.get("token_exchange") or {}
            token_request = params.get("token_request") or {}

            target_audience = params.get("target_audience")
            token_endpoint_auth_method = params.get("token_endpoint_auth_method")
            trusted_issuer = token_exchange.get("trusted_issuer")
            exchange_audience = token_exchange.get("audience")
            grant_type = token_request.get("grant_type")

            missing = [
                name
                for name, val in [
                    ("target_audience", target_audience),
                    ("token_endpoint_auth_method", token_endpoint_auth_method),
                    ("token_exchange.trusted_issuer", trusted_issuer),
                    ("token_exchange.audience", exchange_audience),
                    ("token_request.grant_type", grant_type),
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


@register_auth_provider(config_type=OktaTokenExchangeAuthProviderConfig)
async def okta_token_exchange_auth_provider(
    config: OktaTokenExchangeAuthProviderConfig, builder: Builder
) -> AsyncGenerator[OktaTokenExchangeAuthProvider, None]:
    """NAT auth provider factory for Okta XAA token exchange."""
    yield OktaTokenExchangeAuthProvider(config=config)
