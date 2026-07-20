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

"""A2A server helpers for DataRobot-hosted agents.

This module owns the A2A protocol layer: agent card construction, OAuth2
security scheme assembly, Cross-Application Access capability extensions,
and endpoint URL resolution.  The FastAPI framework glue lives in
:mod:`~datarobot_genai.dragent.frontends.fastapi`.
"""

import logging
from contextvars import ContextVar

import httpx
from a2a.server.apps import A2AStarletteApplication
from a2a.server.context import ServerCallContext
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import BasePushNotificationSender
from a2a.server.tasks import InMemoryPushNotificationConfigStore
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities
from a2a.types import AgentCard
from a2a.types import AgentExtension
from a2a.types import AgentSkill
from a2a.types import AuthorizationCodeOAuthFlow
from a2a.types import ClientCredentialsOAuthFlow
from a2a.types import InvalidParamsError
from a2a.types import OAuth2SecurityScheme
from a2a.types import OAuthFlows
from a2a.types import SecurityScheme
from a2a.utils.errors import ServerError
from nat.authentication.oauth2.oauth2_resource_server_config import OAuth2ResourceServerConfig
from nat.data_models.user_info import UserInfo
from nat.plugins.a2a.server.agent_executor_adapter import NATWorkflowAgentExecutor
from nat.plugins.a2a.server.front_end_config import A2AFrontEndConfig
from nat.plugins.a2a.server.front_end_plugin_worker import A2AFrontEndPluginWorker

from datarobot_genai.core.runtime import get_deployment_id
from datarobot_genai.core.runtime import get_workload_id
from datarobot_genai.dragent.cross_app_access_config import CrossApplicationAccessConfig
from datarobot_genai.dragent.deployment_urls import build_deployment_a2a_url
from datarobot_genai.dragent.deployment_urls import build_workload_a2a_url
from datarobot_genai.dragent.deployment_urls import resolve_datarobot_endpoint

from .register import DRAgentA2AExternalConfig
from .session import _auth_handler

logger = logging.getLogger(__name__)

_AUTH_CONTEXT_HEADER = "x-datarobot-authorization-context"
_GATEWAY_USER_ID_HEADER = "x-datarobot-user-id"
_INVALID_AUTH_CONTEXT_MSG = (
    "X-DataRobot-Authorization-Context header is present but invalid or expired"
)

# Populated by :class:`DRAgentA2AStarletteApplication` before the SDK card_modifier runs.
_agent_card_request_headers: ContextVar[dict[str, str] | None] = ContextVar(
    "_agent_card_request_headers", default=None
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

A2A_MOUNT_PATH = "a2a"

OAUTH2_SECURITY_DESCRIPTION_WITH_TOKEN_EXCHANGE = (
    "OAuth 2.0 authorization utilizing RFC 7523 JWT Bearer Grant. Requires a prerequisite "
    "identity assertion via RFC 8693 Token Exchange. Refer to the capabilities.extensions "
    "block for strict execution parameters and routing."
)

# Extension URI for the RFC 7523 JWT Bearer Grant (outer grant type for the hybrid flow).
JWT_BEARER_GRANT_TYPE_URI = "urn:ietf:params:oauth:grant-type:jwt-bearer"

# IETF URNs injected by the generator into the token_exchange block.
TOKEN_EXCHANGE_GRANT_TYPE_URI = "urn:ietf:params:oauth:grant-type:token-exchange"
TOKEN_EXCHANGE_REQUESTED_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:id-jag"

CROSS_APP_EXTENSION_DESCRIPTION = (
    "Two-Step Cross-Application Access execution parameters. "
    "Step 1: RFC 8693 Token Exchange prerequisite. "
    "Step 2: RFC 7523 JWT Bearer Grant."
)

# Binding references linking the extension to the OpenAPI security scheme.
CROSS_APP_SECURITY_SCHEME_REF = "oauth2"
CROSS_APP_SECURITY_SCHEME_FLOW_REF = "clientCredentials"

INTERNAL_IDENTITY_URI = "urn:datarobot:agent:identity:internal"
INTERNAL_IDENTITY_DESCRIPTION = "Internal DataRobot routing and system identifiers."

EXTERNAL_IDENTITY_URI = "urn:datarobot:agent:identity:external"
EXTERNAL_IDENTITY_DESCRIPTION = (
    "Customer-provided external agent identifiers for catalog discovery."
)

_IDENTITY_EXTENSION_URIS = frozenset({INTERNAL_IDENTITY_URI, EXTERNAL_IDENTITY_URI})


# ---------------------------------------------------------------------------
# Endpoint URL
# ---------------------------------------------------------------------------


def get_a2a_endpoint_url(host: str, port: int) -> str:
    """Construct the A2A endpoint URL for the running server.

    In a DataRobot deployment (``MLOPS_DEPLOYMENT_ID`` is set), uses the
    deployment's direct-access URL built from ``DATAROBOT_PUBLIC_API_ENDPOINT``
    / ``DATAROBOT_ENDPOINT``.  Otherwise falls back to the local
    ``http://{host}:{port}/a2a/`` URL.
    """
    deployment_id = get_deployment_id()
    workload_id = get_workload_id()

    if not (deployment_id or workload_id):
        return f"http://{host}:{port}/{A2A_MOUNT_PATH}/"

    datarobot_endpoint = resolve_datarobot_endpoint(require=True)
    assert datarobot_endpoint is not None  # guaranteed by require=True

    if deployment_id:
        return build_deployment_a2a_url(datarobot_endpoint, deployment_id)
    assert workload_id is not None  # non-None guaranteed by the early-return above
    return build_workload_a2a_url(datarobot_endpoint, workload_id)


# ---------------------------------------------------------------------------
# OAuth2 / security scheme helpers
# ---------------------------------------------------------------------------


async def resolve_oauth_endpoints(
    server_auth_config: OAuth2ResourceServerConfig,
) -> tuple[str, str]:
    """Resolve ``(authorization_url, token_url)`` from an OAuth2ResourceServerConfig.

    Uses OIDC discovery when ``discovery_url`` is set, otherwise derives from ``issuer_url``.
    """
    if server_auth_config.discovery_url:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(server_auth_config.discovery_url, timeout=5.0)
                response.raise_for_status()
                metadata = response.json()
                auth_url = metadata.get("authorization_endpoint")
                token_url = metadata.get("token_endpoint")
                if auth_url and token_url:
                    logger.info(
                        "Resolved OAuth endpoints via discovery: %s",
                        server_auth_config.discovery_url,
                    )
                    return auth_url, token_url
        except Exception as e:
            logger.warning("Failed to discover OAuth endpoints: %s", e)

    issuer = server_auth_config.issuer_url.rstrip("/")
    auth_url = f"{issuer}/oauth/authorize"
    token_url = f"{issuer}/oauth/token"
    logger.info("Using derived OAuth endpoints from issuer: %s", issuer)
    return auth_url, token_url


async def build_oauth_flow_from_server_auth(
    server_auth: OAuth2ResourceServerConfig,
) -> tuple[AuthorizationCodeOAuthFlow, list[str]]:
    """Build the authorization_code OAuth2 flow and scopes from a NAT server_auth config."""
    auth_url, token_url = await resolve_oauth_endpoints(server_auth)
    flow = AuthorizationCodeOAuthFlow(
        authorization_url=auth_url,
        token_url=token_url,
        scopes={scope: f"Permission: {scope}" for scope in server_auth.scopes},
    )
    return flow, list(server_auth.scopes)


def build_oauth_flow_from_cross_app_access(
    config: CrossApplicationAccessConfig,
) -> tuple[ClientCredentialsOAuthFlow, list[str]]:
    """Build the client_credentials flow and scopes from a CrossApplicationAccessConfig.

    Extracts the OpenAPI-standard fields (``token_url``, ``scopes``) only.
    Cross-Application Access extension parameters are handled separately by
    :func:`build_cross_app_capability_extension` and MUST NOT appear here.
    """
    flow = ClientCredentialsOAuthFlow(
        token_url=config.token_request.token_url,
        scopes={scope: f"Permission: {scope}" for scope in config.token_request.scopes},
    )
    return flow, list(config.token_request.scopes)


def build_cross_app_capability_extension(
    config: CrossApplicationAccessConfig,
) -> list[AgentExtension]:
    """Build the Cross-Application Access extension entry for ``capabilities.extensions``.

    Only extension-bound fields go in ``params``; ``token_url`` and ``scopes``
    are intentionally omitted — they belong to OpenAPI ``securitySchemes``.
    """
    params: dict = {
        "ref": {
            "scheme": CROSS_APP_SECURITY_SCHEME_REF,
            "flow": CROSS_APP_SECURITY_SCHEME_FLOW_REF,
        },
        "tokenEndpointAuthMethod": config.token_endpoint_auth_method,
        "tokenExchange": {
            "grantType": TOKEN_EXCHANGE_GRANT_TYPE_URI,
            "requestedTokenType": TOKEN_EXCHANGE_REQUESTED_TOKEN_TYPE,
            "trustedIssuer": config.token_exchange.trusted_issuer,
            "audience": config.token_exchange.audience,
        },
        "tokenRequest": {
            "grantType": JWT_BEARER_GRANT_TYPE_URI,
            "audience": config.token_request.audience,
        },
    }
    return [
        AgentExtension(
            uri=JWT_BEARER_GRANT_TYPE_URI,
            description=CROSS_APP_EXTENSION_DESCRIPTION,
            params=params,
        )
    ]


def build_internal_identity_extension() -> AgentExtension | None:
    """Build the internal identity extension for the current runtime, or None in local dev.

    In a deployment container (``MLOPS_DEPLOYMENT_ID``) the params carry
    ``deployment_id``; in a workload container (``WORKLOAD_ID``) they carry
    ``workload_id``.  Returns *None* when neither identity is present.
    """
    if dep_id := get_deployment_id():
        params = {"deployment_id": dep_id}
    elif wl_id := get_workload_id():
        params = {"workload_id": wl_id}
    else:
        return None

    return AgentExtension(
        uri=INTERNAL_IDENTITY_URI,
        description=INTERNAL_IDENTITY_DESCRIPTION,
        required=True,
        params=params,
    )


def build_external_identity_extension(external_id: str) -> AgentExtension:
    """Build the external identity extension for catalog discovery."""
    return AgentExtension(
        uri=EXTERNAL_IDENTITY_URI,
        description=EXTERNAL_IDENTITY_DESCRIPTION,
        required=False,
        params={"id": external_id},
    )


def _collect_extensions(
    cross_app_access: CrossApplicationAccessConfig | None,
    external: DRAgentA2AExternalConfig | None,
) -> list[AgentExtension] | None:
    """Assemble all agent card extensions from the configured sources."""
    extensions: list[AgentExtension] = []
    if cross_app_access:
        extensions.extend(build_cross_app_capability_extension(cross_app_access))
    if internal := build_internal_identity_extension():
        extensions.append(internal)
    if external and external.id:
        extensions.append(build_external_identity_extension(external.id))
    return extensions or None


def _resolve_url(
    frontend_config: A2AFrontEndConfig,
    external: DRAgentA2AExternalConfig | None,
) -> str:
    """Return the agent card URL, preferring ``external.url`` when provided."""
    if external and external.url:
        return external.url
    return get_a2a_endpoint_url(frontend_config.host, frontend_config.port)


async def build_security_schemes(
    frontend_config: A2AFrontEndConfig,
    cross_app_access: CrossApplicationAccessConfig | None,
) -> tuple[
    dict[str, SecurityScheme] | None,
    list[dict[str, list[str]]] | None,
]:
    """Assemble A2A security schemes, merging up to two auth sources.

    * ``server_auth`` → authorization_code flow.
    * ``cross_app_access`` → client_credentials flow.

    Returns ``(security_schemes, security_requirements)``, both ``None``
    when neither source is configured.
    """
    server_auth = frontend_config.server_auth

    if not server_auth and not cross_app_access:
        return None, None

    auth_code_flow, server_auth_scopes = (
        await build_oauth_flow_from_server_auth(server_auth) if server_auth else (None, [])
    )
    client_creds_flow, cross_app_scopes = (
        build_oauth_flow_from_cross_app_access(cross_app_access) if cross_app_access else (None, [])
    )

    all_scopes = list(dict.fromkeys(server_auth_scopes + cross_app_scopes))
    security_schemes = {
        "oauth2": SecurityScheme(
            root=OAuth2SecurityScheme(
                type="oauth2",
                description=OAUTH2_SECURITY_DESCRIPTION_WITH_TOKEN_EXCHANGE,
                flows=OAuthFlows(
                    authorization_code=auth_code_flow,
                    client_credentials=client_creds_flow,
                ),
            )
        )
    }
    return security_schemes, [{"oauth2": all_scopes}]


# ---------------------------------------------------------------------------
# Agent card factory
# ---------------------------------------------------------------------------


async def create_agent_card(
    frontend_config: A2AFrontEndConfig,
    cross_app_access: CrossApplicationAccessConfig | None,
    skills: list[AgentSkill],
    external: DRAgentA2AExternalConfig | None = None,
) -> AgentCard:
    """Build an :class:`~a2a.types.AgentCard` for a DataRobot-hosted A2A agent.

    When ``skills`` is empty, a single default skill is generated from
    ``frontend_config.name`` / ``frontend_config.description``.
    """
    security_schemes, security = await build_security_schemes(frontend_config, cross_app_access)
    extensions = _collect_extensions(cross_app_access, external)

    resolved_skills = skills or [
        AgentSkill(
            id="call",
            name=frontend_config.name,
            description=frontend_config.description,
            tags=[],
            examples=[],
        )
    ]

    url = _resolve_url(frontend_config, external)

    return AgentCard(
        name=frontend_config.name,
        description=frontend_config.description,
        url=url,
        version=frontend_config.version,
        default_input_modes=frontend_config.default_input_modes,
        default_output_modes=frontend_config.default_output_modes,
        capabilities=AgentCapabilities(
            streaming=frontend_config.capabilities.streaming,
            push_notifications=frontend_config.capabilities.push_notifications,
            extensions=extensions,
        ),
        skills=resolved_skills,
        security_schemes=security_schemes or None,
        security=security or None,
        supports_authenticated_extended_card=True,
    )


# ---------------------------------------------------------------------------
# Per-request agent card selection (public redacted vs authenticated extended)
# ---------------------------------------------------------------------------


def _normalise_headers(headers: dict[str, str] | None) -> dict[str, str] | None:
    if not headers:
        return None
    return {k.lower(): v for k, v in headers.items()}


def resolve_identity_from_headers(headers: dict[str, str] | None) -> str | None:
    """Extract gateway-validated user identity from A2A-forwarded headers.

    Resolution order (first match wins):

    1. ``X-DataRobot-Authorization-Context`` -- signed JWT forwarded by
       components in the agent application template.  Decoded via
       :data:`_auth_handler` and hashed through
       ``UserInfo._from_session_cookie`` to produce the same UUID5 workflow
       key as the AG-UI path.  When this header is present but validation
       fails, raises :class:`~a2a.utils.errors.ServerError` with
       :class:`~a2a.types.InvalidParamsError` (no fall-through to other headers
       or ``context_id``).
    2. ``X-DataRobot-User-Id`` -- raw DataRobot user ID injected by the API
       gateway, tied to the API-key owner.  Used only when the auth-context
       header is absent.  Same ``_from_session_cookie`` transform is applied
       for key-format consistency.
    3. ``None`` -- no gateway-provided identity (local dev).

    Returns ``None`` when *headers* are absent or contain no recognised
    identity header.
    """
    if not headers:
        return None

    if _AUTH_CONTEXT_HEADER in headers:
        try:
            auth_ctx = _auth_handler.get_context(headers)
        except Exception:
            logger.warning("Failed to decode auth-context header", exc_info=True)
            auth_ctx = None
        if auth_ctx is None:
            raise ServerError(error=InvalidParamsError(message=_INVALID_AUTH_CONTEXT_MSG))
        return UserInfo._from_session_cookie(auth_ctx.user.id).get_user_id()

    raw_user_id = headers.get(_GATEWAY_USER_ID_HEADER)
    if raw_user_id:
        return UserInfo._from_session_cookie(raw_user_id).get_user_id()

    return None


def redact_agent_card(card: AgentCard) -> AgentCard:
    """Return a public-safe view of an agent card.

    Strips advertised skills and removes internal/external identity extensions
    while preserving auth and cross-application-access metadata needed for
    anonymous discovery.
    """
    extensions = card.capabilities.extensions
    filtered_extensions = None
    if extensions:
        filtered = [ext for ext in extensions if ext.uri not in _IDENTITY_EXTENSION_URIS]
        filtered_extensions = filtered or None

    return card.model_copy(
        update={
            "skills": [],
            "capabilities": card.capabilities.model_copy(
                update={"extensions": filtered_extensions}
            ),
        }
    )


def _public_card_modifier(card: AgentCard) -> AgentCard:
    """Serve the extended card to authenticated callers, redacted otherwise."""
    headers = _agent_card_request_headers.get()
    if resolve_identity_from_headers(headers) is not None:
        return card
    return redact_agent_card(card)


def _extended_card_modifier(card: AgentCard, context: ServerCallContext) -> AgentCard:
    """Serve the extended card for ``agent/getAuthenticatedExtendedCard`` callers."""
    raw_headers = context.state.get("headers") if context.state else None
    headers = _normalise_headers(raw_headers) if isinstance(raw_headers, dict) else None
    if resolve_identity_from_headers(headers) is None:
        raise ServerError(
            error=InvalidParamsError(
                message="Authenticated identity required for extended agent card"
            )
        )
    return card


class DRAgentA2AStarletteApplication(A2AStarletteApplication):
    """A2A server that selects redacted vs extended agent cards per request."""

    async def _handle_get_agent_card(self, request):  # type: ignore[no-untyped-def]
        headers = _normalise_headers(dict(request.headers))
        if headers and _AUTH_CONTEXT_HEADER in headers:
            try:
                resolve_identity_from_headers(headers)
            except ServerError:
                from starlette.responses import JSONResponse

                return JSONResponse(
                    {"error": _INVALID_AUTH_CONTEXT_MSG},
                    status_code=403,
                )

        token = _agent_card_request_headers.set(headers)
        try:
            return await super()._handle_get_agent_card(request)
        finally:
            _agent_card_request_headers.reset(token)


def create_dr_a2a_server(
    a2a_worker: A2AFrontEndPluginWorker,
    agent_card: AgentCard,
    agent_executor: NATWorkflowAgentExecutor,
) -> DRAgentA2AStarletteApplication:
    """Create an A2A server with per-request agent card selection.

    Mirrors NAT's :meth:`A2AFrontEndPluginWorker.create_a2a_server` but wires
    ``card_modifier`` / ``extended_agent_card`` / ``extended_card_modifier`` so
    anonymous callers receive a redacted public card while same-tenant
    authenticated callers receive the full card.
    """
    a2a_worker._httpx_client = httpx.AsyncClient()

    push_config_store = InMemoryPushNotificationConfigStore()
    push_sender = BasePushNotificationSender(
        httpx_client=a2a_worker._httpx_client,
        config_store=push_config_store,
    )
    request_handler = DefaultRequestHandler(
        agent_executor=agent_executor,
        task_store=InMemoryTaskStore(),
        push_config_store=push_config_store,
        push_sender=push_sender,
    )

    server = DRAgentA2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
        extended_agent_card=agent_card,
        card_modifier=_public_card_modifier,
        extended_card_modifier=_extended_card_modifier,
    )
    logger.info("Created A2A server with per-request agent card selection")
    return server
