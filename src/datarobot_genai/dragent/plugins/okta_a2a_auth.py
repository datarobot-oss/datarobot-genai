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
  org authorization server (``{trusted_issuer}/oauth2/v1/token``).

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
    """Raise a descriptive ``RuntimeError`` if *resp* is an HTTP error response.

    Extracts Okta-style JSON error fields (``error_description``, ``error``) so
    operators see the actual Okta rejection reason rather than a bare HTTP status.
    Falls back to the raw response text when the body is not JSON.

    Parameters
    ----------
    resp:
        The httpx response to check.
    step:
        Token exchange step number (1 or 2) for the error message.
    url:
        Token endpoint URL included in the error message for quick diagnosis.
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


def _make_client_assertion(
    principal_id: str, token_url: str, private_jwk: dict[str, Any]
) -> str:
    """Build and sign a JWT client assertion (RS256) for OAuth token requests.

    Parameters
    ----------
    principal_id:
        ``iss`` / ``sub`` claims — the Okta AI agent principal ID.
    token_url:
        Token endpoint URL set as the JWT ``aud`` claim.
    private_jwk:
        Parsed RSA private JWK dict (``kty``, ``n``, ``e``, ``d``, …).

    Returns
    -------
    str
        Compact-serialized signed JWT.
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
    """Agent card parameters needed for the Okta XAA flow.

    Populated by :meth:`OAuth2CrossApplicationAccessOAuth2AuthProvider.set_agent_card` from
    ``securitySchemes`` (OpenAPI layer) and ``capabilities.extensions`` (A2A layer).
    """

    token_url: str
    """From ``securitySchemes.oauth2.flows.clientCredentials.tokenUrl``.
    Used as the Step 2 (JWT Bearer grant) token endpoint."""

    trusted_issuer: str
    """Org-level AS issuer (``token_exchange.trusted_issuer``); runs Step 1 (RFC 8693).
    The Step 1 token endpoint is derived as ``trusted_issuer + "/oauth2/v1/token"``.
    Also used as the JWT client assertion ``aud`` for Step 1."""

    exchange_audience: str
    """Resource AS issuer (``token_exchange.audience``); embedded as ``audience`` in
    the Step 1 token-exchange request so the originating AS issues an ID-JAG scoped
    to this audience."""

    target_audience: str
    """Final resource identifier for the agent (``token_request.audience``).
    Passed as ``resource`` in the Step 1 token-exchange request."""

    token_endpoint_auth_method: str
    """Client auth method (``params.token_endpoint_auth_method``).
    ``"private_key_jwt"`` triggers a signed JWT client assertion on both steps."""

    id_jag_scopes: list[str]
    """Scopes; sourced from ``securitySchemes.oauth2.flows.clientCredentials.scopes``
    in the agent card."""


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
    private_jwk: OptionalSecretStr = Field(
        default_factory=_get_default_private_jwk,
        description=(
            "Base64-encoded or raw-JSON private JWK (env: ``PRIVATE_JWK``). "
            "Signs JWT client assertions."
        ),
    )


# ---------------------------------------------------------------------------
# Token exchange implementations
# ---------------------------------------------------------------------------


class XAATokenExchangeImpl(StrEnum):
    """Allowed values for ``XAA_TOKEN_EXCHANGE_IMPL``.

    Set the ``XAA_TOKEN_EXCHANGE_IMPL`` environment variable to select the
    token exchange implementation used by :func:`get_token_exchange`.
    """

    HTTP = "http"
    OKTA_SDK = "okta_sdk"


_IMPL_ENV_VAR = "XAA_TOKEN_EXCHANGE_IMPL"
_IMPL_DEFAULT = XAATokenExchangeImpl.OKTA_SDK


class XAATokenExchange(Protocol):
    """Protocol for XAA token exchange implementations."""

    config: OAuth2CrossApplicationAccessAuthProviderConfig

    async def exchange_token(self, params: _CrossAppFlowParams, subject_token: str) -> str:
        """Execute the full two-step XAA flow and return the final access token.

        Parameters
        ----------
        params:
            Flow parameters extracted from the remote agent card.
        subject_token:
            The incoming Okta access token to exchange (Step 1 subject_token).
        """
        ...


class OktaTokenExchange(XAATokenExchange):
    """Token exchange via the ``okta-client-python`` SDK.

    Delegates both XAA steps to :class:`CrossAppAccessFlow` which handles
    the RFC 8693 token exchange and RFC 7523 JWT Bearer grant internally.

    Requires the ``okta-client-python`` package (``pip install okta-client-python``).
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
            raise ValueError("principal_id is required for the Okta cross-app access flow")
        if not self.config.private_jwk:
            raise ValueError("private_jwk is required for the Okta cross-app access flow")

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
        client_config = OAuth2ClientConfiguration(
            issuer=params.trusted_issuer,
            client_authorization=ClientAssertionAuthorization(
                assertion_claims=claims,
                key_provider=key_provider,
            ),
        )
        client = OAuth2Client(configuration=client_config)
        target = CrossAppAccessTarget(issuer=params.exchange_audience)
        flow = CrossAppAccessFlow(client=client, target=target)

        logger.info(
            "OktaTokenExchange: starting SDK cross-app access flow "
            "(trusted_issuer=%s, exchange_audience=%s)",
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

        logger.info("OktaTokenExchange: cross-app access flow completed successfully")
        return token.access_token


class ApiTokenExchange(XAATokenExchange):
    """Token exchange via direct HTTP calls (``httpx`` + ``PyJWT``).

    Implements the full two-step Okta Cross App Access (XAA) flow:

    * **Step 1** — RFC 8693 token exchange at the org authorization server
      (``{trusted_issuer}/oauth2/v1/token``) to obtain an ID-JAG token.
    * **Step 2** — RFC 7523 JWT Bearer grant at the custom authorization server
      (from agent card ``securitySchemes``) to obtain the final scoped agent token.

    Both steps use a signed ``private_key_jwt`` client assertion built from
    ``config.principal_id`` and ``config.private_jwk``.
    """

    def __init__(self, config: OAuth2CrossApplicationAccessAuthProviderConfig) -> None:
        self.config = config

    async def exchange_token(self, params: _CrossAppFlowParams, subject_token: str) -> str:
        """Execute the full two-step XAA flow and return the final access token.

        Parameters
        ----------
        params:
            Flow parameters extracted from the remote agent card via
            :func:`_parse_cross_app_params`.
        subject_token:
            The caller's incoming Okta access token used as ``subject_token``
            in the Step 1 RFC 8693 token exchange request.

        Returns
        -------
        str
            The final scoped agent access token issued by the custom AS.

        Raises
        ------
        ValueError
            If ``principal_id`` or ``private_jwk`` are not set in config, or if
            the JWK cannot be parsed.
        RuntimeError
            If either token-exchange HTTP request returns an HTTP error status.
        """
        if not self.config.principal_id:
            raise ValueError("principal_id is required for the Okta cross-app access flow")

        if not self.config.private_jwk:
            raise ValueError("private_jwk is required for the Okta cross-app access flow")

        private_jwk = _parse_private_jwk(self.config.private_jwk.get_secret_value())

        # Derive token endpoint URLs from the card parameters
        org_as_token_url = params.trusted_issuer.rstrip("/") + "/oauth2/v1/token"
        custom_as_token_url = params.token_url

        # Build both JWT client assertions up front — pure crypto, no I/O
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
            # ------------------------------------------------------------------
            # Step 1: RFC 8693 token exchange → ID-JAG
            # ------------------------------------------------------------------
            logger.info(
                "ApiTokenExchange Step 1: exchanging access token for ID-JAG "
                "(org_as_token_url=%s, exchange_audience=%s)",
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

            # ------------------------------------------------------------------
            # Step 2: RFC 7523 JWT Bearer grant → scoped agent token
            # ------------------------------------------------------------------
            logger.info(
                "ApiTokenExchange Step 2: exchanging ID-JAG for scoped agent token "
                "(custom_as_token_url=%s, target_audience=%s)",
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
            exchanged_token = resp2.json()["access_token"]

        logger.info("ApiTokenExchange cross-app access flow completed successfully")
        return exchanged_token


def _get_token_exchange_impl() -> XAATokenExchangeImpl:
    """Read and validate ``XAA_TOKEN_EXCHANGE_IMPL`` from the environment.

    Returns the default (``okta_sdk``) when the variable is unset.
    Raises ``ValueError`` with the list of allowed values on an invalid input.
    """
    raw = (os.getenv(_IMPL_ENV_VAR) or _IMPL_DEFAULT).strip().lower()
    try:
        return XAATokenExchangeImpl(raw)
    except ValueError as exc:
        allowed = ", ".join(m.value for m in XAATokenExchangeImpl)
        raise ValueError(f"Invalid {_IMPL_ENV_VAR}='{raw}'. Allowed values: {allowed}") from exc


async def get_token_exchange(
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
    """Auth provider for Okta XAA A2A calls.

    * **Discovery** — forwards the incoming Okta bearer token as ``Authorization: Bearer``.
    * **Call** — two-step XAA token exchange (RFC 8693 → RFC 7523) via direct HTTP
      calls (``httpx`` + ``PyJWT``).
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
        self._flow_params = _parse_cross_app_params(card)
        logger.info(
            "Agent card parsed: trusted_issuer=%s, exchange_audience=%s, "
            "target_audience=%s, token_endpoint_auth_method=%s, id_jag_scopes=%s",
            self._flow_params.trusted_issuer,
            self._flow_params.exchange_audience,
            self._flow_params.target_audience,
            self._flow_params.token_endpoint_auth_method,
            self._flow_params.id_jag_scopes,
        )

    # ------------------------------------------------------------------
    # AuthProviderBase
    # ------------------------------------------------------------------

    async def authenticate(self, user_id: str | None = None, **kwargs: Any) -> AuthResult | None:
        """Obtain a scoped agent token via the two-step Okta XAA token exchange.

        Executes two sequential HTTP token requests:

        * **Step 1** — RFC 8693 token exchange at the org AS to obtain an ID-JAG.
        * **Step 2** — RFC 7523 JWT Bearer grant at the custom AS to obtain the
          final scoped agent token.

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

        # Derive Step 1 token URL from trusted_issuer (org AS)
        org_as_token_url = self._flow_params.trusted_issuer.rstrip("/") + "/oauth2/v1/token"
        # Step 2 token URL comes directly from the agent card securitySchemes
        custom_as_token_url = self._flow_params.token_url

        # Build both assertions before opening the HTTP client — both are pure
        # crypto operations that do not depend on each other or on network state.
        assertion1 = self._make_client_assertion(
            token_url=org_as_token_url, private_jwk=private_jwk
        )
        assertion2 = self._make_client_assertion(
            token_url=custom_as_token_url, private_jwk=private_jwk
        )

        async with httpx.AsyncClient() as http:
            # ------------------------------------------------------------------
            # Step 1: RFC 8693 token exchange → ID-JAG
            # ------------------------------------------------------------------
            logger.info(
                "Step 1: exchanging access token for ID-JAG "
                "(org_as_token_url=%s, exchange_audience=%s, user_id=%s)",
                org_as_token_url,
                self._flow_params.exchange_audience,
                user_id,
            )
            resp1 = await http.post(
                org_as_token_url,
                data={
                    "grant_type": _TOKEN_EXCHANGE_GRANT_TYPE,
                    "subject_token": access_token,
                    "subject_token_type": _SUBJECT_TOKEN_TYPE,
                    "requested_token_type": _REQUESTED_TOKEN_TYPE,
                    "audience": self._flow_params.exchange_audience,
                    "resource": self._flow_params.target_audience,
                    "scope": " ".join(self._flow_params.id_jag_scopes),
                    "client_assertion_type": _CLIENT_ASSERTION_TYPE,
                    "client_assertion": assertion1,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=_TOKEN_EXCHANGE_TIMEOUT,
            )
            _raise_token_error(resp1, step=1, url=org_as_token_url)
            id_jag = resp1.json()["access_token"]

            # ------------------------------------------------------------------
            # Step 2: RFC 7523 JWT Bearer grant → scoped agent token
            # ------------------------------------------------------------------
            logger.info(
                "Step 2: exchanging ID-JAG for scoped agent token "
                "(custom_as_token_url=%s, target_audience=%s)",
                custom_as_token_url,
                self._flow_params.target_audience,
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
            exchanged_token = resp2.json()["access_token"]

        logger.info("Cross-app access flow completed successfully")
        return AuthResult(credentials=[BearerTokenCred(token=exchanged_token)])

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

    def _make_client_assertion(self, token_url: str, private_jwk: dict[str, Any]) -> str:
        """Build and sign a JWT client assertion for use in token requests.

        The assertion is signed with the RSA private key from ``private_jwk`` using
        RS256.  ``iss`` / ``sub`` are set to ``principal_id``; ``aud`` is ``token_url``;
        ``exp`` is 60 seconds from now; ``jti`` is a random UUID.

        Parameters
        ----------
        token_url:
            The token endpoint URL to set as the JWT ``aud`` claim.
        private_jwk:
            The parsed RSA private JWK dict (``kty``, ``n``, ``e``, ``d``, …).

        Returns
        -------
        str
            A compact serialized signed JWT string.
        """
        try:
            private_key = RSAAlgorithm.from_jwk(json.dumps(private_jwk))
        except Exception:
            # Suppress the original exception so the JWK material is not
            # included in the traceback frame that may surface in logs.
            raise ValueError(
                "Failed to load RSA private key from JWK (check PRIVATE_JWK format and key type)"
            ) from None

        now = int(time.time())
        try:
            assertion = jwt.encode(
                {
                    "iss": self.config.principal_id,
                    "sub": self.config.principal_id,
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
        return assertion


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
    """NAT auth provider factory for Okta Cross-Application Access."""
    yield OAuth2CrossApplicationAccessOAuth2AuthProvider(config=config)


# ---------------------------------------------------------------------------
# Backward-compatible aliases (deprecated — use the OAuth2* names)
# ---------------------------------------------------------------------------

OktaCrossApplicationAccessAuthProviderConfig = OAuth2CrossApplicationAccessAuthProviderConfig
OktaCrossApplicationAccessAuthProvider = OAuth2CrossApplicationAccessOAuth2AuthProvider
