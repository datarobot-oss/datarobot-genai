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

"""Cross-Application Access (XAA) auth provider for A2A agent communication.

Registered as ``_type: okta_cross_app_access`` in workflow YAML.

Overview
--------
Implements the **Identity Assertion Authorization Grant (ID-JAG)** flow for
secure cross-application access between AI agents.  The flow involves two
authorization servers:

* **Org Authorization Server** — issues and validates user identity; handles
  the initial token exchange (Step 1).
* **Custom Authorization Server** — protects the target resource/API; issues
  the final scoped access token with custom audience, scopes, claims, and
  policies required by that resource (Step 2).

Discovery phase
~~~~~~~~~~~~~~~
Forwards the incoming ``x-datarobot-external-access-token`` header directly
to the agent card endpoint as a Bearer token.

Call phase
~~~~~~~~~~
Executes a two-step XAA token exchange:

  **Step 1 — RFC 8693 Token Exchange** (Org AS):
  Exchange the caller's access token for an ID-JAG token.  The ID-JAG is a
  short-lived, single-use identity assertion that encodes *who* (human user)
  is acting through *which* AI agent (workload principal).

  **Step 2 — RFC 7523 JWT Bearer Grant** (Custom AS):
  Exchange the ID-JAG for a scoped access token at the custom authorization
  server.  The resulting token contains the ``sub`` (human user), ``cid``
  (AI agent workload principal), and requested scopes.

Both steps authenticate the client using ``private_key_jwt`` — a signed JWT
client assertion built from the agent's RSA private key (``PRIVATE_JWK``)
and principal ID (``PRINCIPAL_ID``).

Two implementations are available, selected via ``XAA_TOKEN_EXCHANGE_IMPL``:

* ``okta_sdk`` (default) — delegates to ``okta-client-python``'s
  ``CrossAppAccessFlow`` which handles both steps internally.
* ``http`` — direct HTTP calls via ``httpx`` + ``PyJWT``, giving full
  control over request parameters.

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

Environment variables
~~~~~~~~~~~~~~~~~~~~~
* ``PRINCIPAL_ID``              — Okta AI agent workload principal ID
* ``PRIVATE_JWK``               — base64-encoded or raw-JSON RSA private JWK
* ``XAA_TOKEN_EXCHANGE_IMPL``   — ``okta_sdk`` (default) or ``http``
"""

import base64
import json
import logging
import os
import time
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from enum import StrEnum
from typing import Any
from typing import Protocol

import httpx
import jwt
from a2a.types import AgentCard
from datarobot.core.config import DataRobotAppFrameworkBaseSettings
from jwt.algorithms import RSAAlgorithm
from nat.authentication.interfaces import AuthProviderBase
from nat.builder.builder import Builder
from nat.builder.context import Context
from nat.cli.register_workflow import register_auth_provider
from nat.data_models.authentication import AuthProviderBaseConfig
from nat.data_models.authentication import AuthResult
from nat.data_models.authentication import BearerTokenCred
from nat.data_models.common import OptionalSecretStr
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

    _HAS_OKTA_SDK = True
except ImportError:
    _HAS_OKTA_SDK = False

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

_TOKEN_EXCHANGE_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:token-exchange"
_JWT_BEARER_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:jwt-bearer"
_CLIENT_ASSERTION_TYPE = "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"
_SUBJECT_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:access_token"
_REQUESTED_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:id-jag"

_TOKEN_EXCHANGE_TIMEOUT = 30  # seconds per token HTTP request


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _raise_token_error(resp: httpx.Response, step: int, url: str) -> None:
    """Raise a descriptive ``RuntimeError`` if *resp* is an HTTP error.

    Extracts Okta-style JSON error fields (``error_description``, ``error``)
    so operators see the actual rejection reason rather than a bare HTTP status.
    """
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        try:
            body = exc.response.json()
            detail = body.get("error_description") or body.get("error") or exc.response.text
        except Exception:
            detail = exc.response.text or str(exc)
        raise RuntimeError(
            f"Token exchange Step {step} failed at {url}: "
            f"HTTP {exc.response.status_code} — {detail}"
        ) from exc


def _make_client_assertion(principal_id: str, token_url: str, private_jwk: dict[str, Any]) -> str:
    """Build and sign a ``private_key_jwt`` client assertion (RS256).

    The assertion authenticates the AI agent workload principal to the
    authorization server.  Claims: ``iss``/``sub`` = *principal_id*,
    ``aud`` = *token_url*, ``exp`` = now + 60s, ``jti`` = random UUID.

    Error handling deliberately uses ``from None`` to prevent RSA key
    material from leaking into traceback frames that may surface in logs.
    """
    try:
        private_key = RSAAlgorithm.from_jwk(json.dumps(private_jwk))
    except Exception:
        raise ValueError(
            "Failed to load RSA private key from JWK (check PRIVATE_JWK format and key type)"
        ) from None

    now = int(time.time())
    try:
        return jwt.encode(
            {
                "iss": principal_id,
                "sub": principal_id,
                "aud": token_url,
                "iat": now,
                "exp": now + 60,
                "jti": str(uuid.uuid4()),
            },
            private_key,
            algorithm="RS256",
            headers={"kid": private_jwk.get("kid"), "typ": "JWT"},
        )
    except Exception:
        raise ValueError("Failed to sign client assertion JWT") from None


def _parse_private_jwk(raw: str) -> dict[str, Any]:
    """Decode a private JWK from *raw* (base64-encoded JSON or raw JSON string).

    Raises ``ValueError`` if neither format can be parsed.
    """
    try:
        return json.loads(base64.b64decode(raw).decode())
    except Exception:
        pass
    try:
        return json.loads(raw)
    except Exception:
        pass
    raise ValueError(
        "Could not parse private_jwk: expected base64-encoded JSON or raw JSON string."
    )


# ---------------------------------------------------------------------------
# _CrossAppFlowParams — populated from the fetched AgentCard
# ---------------------------------------------------------------------------


@dataclass
class _CrossAppFlowParams:
    """Parameters extracted from the agent card for the XAA flow.

    These come from two parts of the agent card:

    * ``securitySchemes.oauth2.flows.clientCredentials`` — provides the
      custom AS ``token_url`` and ``scopes``.
    * ``capabilities.extensions[uri=jwt-bearer]`` — provides the org AS
      ``trusted_issuer``, ``exchange_audience``, ``target_audience``, and
      ``token_endpoint_auth_method``.
    """

    token_url: str
    """Custom AS token endpoint (from ``clientCredentials.tokenUrl``).
    Used in Step 2 (JWT Bearer grant) and as ``aud`` for the Step 2 client assertion."""

    trusted_issuer: str
    """Org AS issuer (e.g. ``https://your-domain.okta.com``).
    Step 1 token endpoint is ``{trusted_issuer}/oauth2/v1/token``."""

    exchange_audience: str
    """Custom AS issuer (e.g. ``https://domain.okta.com/oauth2/{as_id}``).
    Passed as ``audience`` in Step 1 so the org AS issues an ID-JAG scoped
    to this authorization server."""

    target_audience: str
    """Final resource identifier (e.g. ``https://api.example.com/``).
    Passed as ``resource`` in Step 1 (HTTP impl only; SDK derives from AS policy)."""

    token_endpoint_auth_method: str
    """Client auth method — ``"private_key_jwt"`` triggers signed JWT assertions."""

    id_jag_scopes: list[str]
    """Scopes from ``clientCredentials.scopes`` (e.g. ``["dr.impersonation"]``)."""


# ---------------------------------------------------------------------------
# OAuth2CrossApplicationAccessAuthProviderConfig
# ---------------------------------------------------------------------------


class OAuth2CrossApplicationAccessAuthProviderConfig(
    AuthProviderBaseConfig,
    name="okta_cross_app_access",
):  # type: ignore[call-arg]
    """Configuration for the XAA auth provider.

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
    private_jwk: OptionalSecretStr = Field(
        default_factory=_get_default_private_jwk,
        description="Base64-encoded or raw-JSON RSA private JWK (env: ``PRIVATE_JWK``).",
    )


# ---------------------------------------------------------------------------
# Token exchange implementations
# ---------------------------------------------------------------------------


class XAATokenExchangeImpl(StrEnum):
    """Allowed values for ``XAA_TOKEN_EXCHANGE_IMPL``."""

    HTTP = "http"
    OKTA_SDK = "okta_sdk"


_IMPL_ENV_VAR = "XAA_TOKEN_EXCHANGE_IMPL"
_IMPL_DEFAULT = XAATokenExchangeImpl.OKTA_SDK


class XAATokenExchange(Protocol):
    """Protocol for XAA token exchange implementations."""

    config: OAuth2CrossApplicationAccessAuthProviderConfig

    async def exchange_token(self, params: _CrossAppFlowParams, subject_token: str) -> str:
        """Execute the full two-step XAA flow and return the final access token."""
        ...


class OktaTokenExchange(XAATokenExchange):
    """Token exchange via the ``okta-client-python`` SDK.

    Delegates both XAA steps to ``CrossAppAccessFlow`` which internally:

    1. Calls ``flow.start(token, audience, scope)`` — builds a ``private_key_jwt``
       client assertion, posts RFC 8693 token exchange to the org AS, and returns
       the ID-JAG token.
    2. Calls ``flow.resume()`` — builds a second client assertion for the custom AS,
       posts RFC 7523 JWT Bearer grant, and returns the final scoped access token.

    The SDK handles assertion signing, endpoint discovery, and request formatting.
    This implementation maps our ``_CrossAppFlowParams`` to the SDK's configuration:

    * ``OAuth2ClientConfiguration(issuer=trusted_issuer)`` — org AS domain
    * ``JWTBearerClaims(audience=org_token_url)`` — ``aud`` for client assertion
    * ``CrossAppAccessTarget(issuer=exchange_audience)`` — custom AS issuer
    * ``flow.start(audience=exchange_audience, scope=id_jag_scopes)`` — ID-JAG params

    Requires ``pip install okta-client-python``.
    """

    def __init__(self, config: OAuth2CrossApplicationAccessAuthProviderConfig) -> None:
        self.config = config

    async def exchange_token(self, params: _CrossAppFlowParams, subject_token: str) -> str:
        if not _HAS_OKTA_SDK:
            raise RuntimeError(
                "okta-client-python is not installed. "
                "Install it (`pip install okta-client-python`) or "
                "set XAA_TOKEN_EXCHANGE_IMPL=http to use the HTTP implementation."
            )

        if not self.config.principal_id:
            raise ValueError("principal_id is required for the XAA flow")
        if not self.config.private_jwk:
            raise ValueError("private_jwk is required for the XAA flow")

        private_jwk = _parse_private_jwk(self.config.private_jwk.get_secret_value())
        org_token_url = params.trusted_issuer.rstrip("/") + "/oauth2/v1/token"

        key_provider = LocalKeyProvider(
            key=private_jwk,
            algorithm=private_jwk.get("alg", "RS256"),
            key_id=private_jwk.get("kid"),
        )
        claims = JWTBearerClaims(
            issuer=self.config.principal_id,
            subject=self.config.principal_id,
            audience=org_token_url,
            expires_in=300,
        )
        client = OAuth2Client(
            configuration=OAuth2ClientConfiguration(
                issuer=params.trusted_issuer,
                client_authorization=ClientAssertionAuthorization(
                    assertion_claims=claims,
                    key_provider=key_provider,
                ),
            )
        )
        target = CrossAppAccessTarget(issuer=params.exchange_audience)
        flow = CrossAppAccessFlow(client=client, target=target)

        logger.info(
            "OktaTokenExchange: starting XAA flow (trusted_issuer=%s, exchange_audience=%s)",
            params.trusted_issuer,
            params.exchange_audience,
        )

        result = await flow.start(
            token=subject_token,
            audience=params.exchange_audience,
            scope=params.id_jag_scopes,
        )
        if result.resume_assertion_claims is not None:
            logger.info("OktaTokenExchange: SDK requires resume step")
        token = await flow.resume()

        logger.info("OktaTokenExchange: XAA flow completed successfully")
        return token.access_token


class ApiTokenExchange(XAATokenExchange):
    """Token exchange via direct HTTP calls (``httpx`` + ``PyJWT``).

    Implements both XAA steps explicitly, giving full control over request
    parameters (unlike the SDK which derives some from AS policy):

    **Step 1 — RFC 8693 Token Exchange** at ``{trusted_issuer}/oauth2/v1/token``:
      Sends ``subject_token`` (caller's access token), ``audience`` (custom AS
      issuer), ``resource`` (target API), and ``scope`` to obtain an ID-JAG.
      The ID-JAG is a short-lived, single-use JWT with ``typ: oauth-id-jag+jwt``
      that encodes the user identity and agent principal.

    **Step 2 — RFC 7523 JWT Bearer Grant** at ``token_url`` (custom AS):
      Sends the ID-JAG as ``assertion`` to obtain the final scoped access token.
      The token carries ``sub`` (human user) and ``cid`` (agent workload principal).

    Both steps use a ``private_key_jwt`` client assertion signed with the
    agent's RSA private key.
    """

    def __init__(self, config: OAuth2CrossApplicationAccessAuthProviderConfig) -> None:
        self.config = config

    async def exchange_token(self, params: _CrossAppFlowParams, subject_token: str) -> str:
        if not self.config.principal_id:
            raise ValueError("principal_id is required for the XAA flow")
        if not self.config.private_jwk:
            raise ValueError("private_jwk is required for the XAA flow")

        private_jwk = _parse_private_jwk(self.config.private_jwk.get_secret_value())

        org_as_token_url = params.trusted_issuer.rstrip("/") + "/oauth2/v1/token"
        custom_as_token_url = params.token_url

        # Build both client assertions up front — pure crypto, no I/O
        assertion1 = _make_client_assertion(
            principal_id=self.config.principal_id,
            token_url=org_as_token_url,
            private_jwk=private_jwk,
        )
        assertion2 = _make_client_assertion(
            principal_id=self.config.principal_id,
            token_url=custom_as_token_url,
            private_jwk=private_jwk,
        )

        async with httpx.AsyncClient() as http:
            # Step 1: RFC 8693 token exchange → ID-JAG
            logger.info(
                "ApiTokenExchange Step 1: token exchange → ID-JAG (org_as=%s, audience=%s)",
                org_as_token_url,
                params.exchange_audience,
            )
            resp1 = await http.post(
                org_as_token_url,
                data={
                    "grant_type": _TOKEN_EXCHANGE_GRANT_TYPE,
                    "subject_token": subject_token,
                    "subject_token_type": _SUBJECT_TOKEN_TYPE,
                    "requested_token_type": _REQUESTED_TOKEN_TYPE,
                    "audience": params.exchange_audience,
                    "resource": params.target_audience,
                    "scope": " ".join(params.id_jag_scopes),
                    "client_assertion_type": _CLIENT_ASSERTION_TYPE,
                    "client_assertion": assertion1,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=_TOKEN_EXCHANGE_TIMEOUT,
            )
            _raise_token_error(resp1, step=1, url=org_as_token_url)
            id_jag = resp1.json()["access_token"]

            # Step 2: RFC 7523 JWT Bearer grant → scoped agent token
            logger.info(
                "ApiTokenExchange Step 2: JWT Bearer grant → scoped token "
                "(custom_as=%s, target=%s)",
                custom_as_token_url,
                params.target_audience,
            )
            resp2 = await http.post(
                custom_as_token_url,
                data={
                    "grant_type": _JWT_BEARER_GRANT_TYPE,
                    "assertion": id_jag,
                    "client_assertion_type": _CLIENT_ASSERTION_TYPE,
                    "client_assertion": assertion2,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=_TOKEN_EXCHANGE_TIMEOUT,
            )
            _raise_token_error(resp2, step=2, url=custom_as_token_url)
            return resp2.json()["access_token"]


def _get_token_exchange_impl() -> XAATokenExchangeImpl:
    """Read ``XAA_TOKEN_EXCHANGE_IMPL`` from env. Defaults to ``okta_sdk``."""
    raw = (os.getenv(_IMPL_ENV_VAR) or _IMPL_DEFAULT).strip().lower()
    try:
        return XAATokenExchangeImpl(raw)
    except ValueError as exc:
        allowed = ", ".join(m.value for m in XAATokenExchangeImpl)
        raise ValueError(f"Invalid {_IMPL_ENV_VAR}='{raw}'. Allowed values: {allowed}") from exc


def get_token_exchange(
    config: OAuth2CrossApplicationAccessAuthProviderConfig,
) -> XAATokenExchange:
    """Return the configured :class:`XAATokenExchange` implementation."""
    if _get_token_exchange_impl() is XAATokenExchangeImpl.HTTP:
        return ApiTokenExchange(config)
    return OktaTokenExchange(config)


# ---------------------------------------------------------------------------
# OAuth2CrossApplicationAccessOAuth2AuthProvider
# ---------------------------------------------------------------------------


class OAuth2CrossApplicationAccessOAuth2AuthProvider(
    A2ADiscoveryAuthMixin,
    AuthProviderBase[OAuth2CrossApplicationAccessAuthProviderConfig],
):
    """Auth provider for XAA A2A calls.

    * **Discovery** — forwards the incoming bearer token as ``Authorization: Bearer``.
    * **Call** — delegates to :func:`get_token_exchange` which returns either
      :class:`OktaTokenExchange` (SDK) or :class:`ApiTokenExchange` (HTTP)
      based on ``XAA_TOKEN_EXCHANGE_IMPL``.
    """

    def __init__(self, config: OAuth2CrossApplicationAccessAuthProviderConfig) -> None:
        super().__init__(config)
        self._flow_params: _CrossAppFlowParams | None = None

    async def authenticate_for_discovery(self, user_id: str | None = None) -> dict[str, str]:
        token = self._extract_token()
        logger.info("Forwarding token from '%s' for discovery", self.config.okta_token_header)
        return {"Authorization": f"Bearer {token}"}

    def set_agent_card(self, card: AgentCard) -> None:
        """Parse the agent card and store :class:`_CrossAppFlowParams`.

        Called by ``_AuthenticatedA2ABaseClient`` after discovery, before
        ``authenticate()``.
        """
        self._flow_params = _parse_cross_app_params(card)
        logger.info(
            "Agent card parsed: trusted_issuer=%s, exchange_audience=%s, "
            "target_audience=%s, auth_method=%s, scopes=%s",
            self._flow_params.trusted_issuer,
            self._flow_params.exchange_audience,
            self._flow_params.target_audience,
            self._flow_params.token_endpoint_auth_method,
            self._flow_params.id_jag_scopes,
        )

    async def authenticate(self, user_id: str | None = None, **kwargs: Any) -> AuthResult | None:
        """Obtain a scoped agent token via the XAA flow.

        Delegates to the configured :class:`XAATokenExchange` implementation.
        Requires :meth:`set_agent_card` to have been called first.
        """
        if self._flow_params is None:
            raise RuntimeError(
                "authenticate() called before set_agent_card(). "
                "Ensure the provider is used with authenticated_a2a_client."
            )

        subject_token = self._extract_token()
        impl = get_token_exchange(self.config)
        exchanged_token = await impl.exchange_token(self._flow_params, subject_token)
        return AuthResult(credentials=[BearerTokenCred(token=exchanged_token)])

    def _extract_token(self) -> str:
        """Extract the access token from NAT request context headers.

        Tries ``okta_token_header`` first, then each ``fallback_token_headers``
        entry (stripping ``Bearer `` prefix if present).
        """
        headers: dict[str, str] = Context.get().metadata.headers or {}

        token = headers.get(self.config.okta_token_header.lower())
        if token:
            return token

        for fallback in self.config.fallback_token_headers:
            value = headers.get(fallback.lower())
            if value:
                if value.lower().startswith("bearer "):
                    value = value[len("bearer ") :]
                logger.debug("Using fallback header '%s'", fallback)
                return value

        raise RuntimeError(
            f"Header '{self.config.okta_token_header}' not found in request context "
            f"(also tried fallbacks: {self.config.fallback_token_headers}). "
            "The access token must be forwarded with every agent call."
        )


# ---------------------------------------------------------------------------
# Agent card parsing helpers
# ---------------------------------------------------------------------------


def _parse_cross_app_params(card: AgentCard) -> _CrossAppFlowParams:
    """Extract :class:`_CrossAppFlowParams` from a fetched :class:`AgentCard`.

    Combines ``securitySchemes.oauth2.flows.clientCredentials`` (``tokenUrl``,
    ``scopes``) and ``capabilities.extensions`` (JWT Bearer entry) into a single
    params object.
    """
    token_url, scopes = _parse_security_schemes(card)
    ext = _parse_cross_app_extension(card)

    return _CrossAppFlowParams(
        token_url=token_url,
        trusted_issuer=ext.trusted_issuer,
        exchange_audience=ext.exchange_audience,
        target_audience=ext.target_audience,
        token_endpoint_auth_method=ext.token_endpoint_auth_method,
        id_jag_scopes=scopes,
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
    """NAT auth provider factory for Cross-Application Access."""
    yield OAuth2CrossApplicationAccessOAuth2AuthProvider(config=config)


# ---------------------------------------------------------------------------
# Backward-compatible aliases (deprecated)
# ---------------------------------------------------------------------------

OktaCrossApplicationAccessAuthProviderConfig = OAuth2CrossApplicationAccessAuthProviderConfig
OktaCrossApplicationAccessAuthProvider = OAuth2CrossApplicationAccessOAuth2AuthProvider
