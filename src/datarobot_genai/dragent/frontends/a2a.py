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
from collections.abc import Awaitable
from collections.abc import Callable

import httpx
from a2a.server.apps import A2AStarletteApplication
from a2a.server.apps.jsonrpc.jsonrpc_app import CallContextBuilder
from a2a.server.context import ServerCallContext
from a2a.server.request_handlers.request_handler import RequestHandler
from a2a.types import AgentCapabilities
from a2a.types import AgentCard
from a2a.types import AgentExtension
from a2a.types import AgentSkill
from a2a.types import AuthorizationCodeOAuthFlow
from a2a.types import ClientCredentialsOAuthFlow
from a2a.types import HTTPAuthSecurityScheme
from a2a.types import InvalidParamsError
from a2a.types import OAuth2SecurityScheme
from a2a.types import OAuthFlows
from a2a.types import SecurityScheme
from a2a.utils.errors import ServerError
from nat.authentication.oauth2.oauth2_resource_server_config import OAuth2ResourceServerConfig
from nat.plugins.a2a.server.agent_executor_adapter import NATWorkflowAgentExecutor
from nat.plugins.a2a.server.front_end_config import A2AFrontEndConfig
from nat.plugins.a2a.server.front_end_plugin_worker import A2AFrontEndPluginWorker
from starlette.responses import JSONResponse

from datarobot_genai.core.runtime import get_deployment_id
from datarobot_genai.core.runtime import get_workload_id
from datarobot_genai.dragent.constants import A2A_MOUNT_PATH
from datarobot_genai.dragent.cross_app_access_config import CrossApplicationAccessConfig
from datarobot_genai.dragent.deployment_urls import build_deployment_a2a_url
from datarobot_genai.dragent.deployment_urls import build_workload_a2a_url
from datarobot_genai.dragent.deployment_urls import join_mount_path
from datarobot_genai.dragent.deployment_urls import resolve_datarobot_endpoint
from datarobot_genai.dragent.deployment_urls import resolve_external_workload_base

from .register import DRAgentA2AExternalConfig
from .session import _a2a_headers
from .session import headers_from_a2a_state
from .session import normalise_headers
from .session import resolve_identity_from_headers

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


OAUTH2_SECURITY_DESCRIPTION_WITH_TOKEN_EXCHANGE = (
    "OAuth 2.0 authorization utilizing RFC 7523 JWT Bearer Grant. Requires a prerequisite "
    "identity assertion via RFC 8693 Token Exchange. Refer to the capabilities.extensions "
    "block for strict execution parameters and routing."
)

BEARER_SECURITY_SCHEME_NAME = "bearerAuth"

BEARER_SECURITY_DESCRIPTION = "DataRobot API token supplied as an Authorization Bearer header."

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

# Body served for every negative agent-card case, including an agent that has not opted in to
# unauthenticated discovery.  A distinguishable response is an oracle: a 401 confirms to an
# anonymous scanner that the agent exists, and naming the opt-in flag hands it the knob that
# unlocks the card.  A not-opted-in agent must look exactly like one that does not exist.
AGENT_CARD_NOT_FOUND_BODY = {"detail": "Not Found"}


# ---------------------------------------------------------------------------
# Endpoint URL
# ---------------------------------------------------------------------------


def get_a2a_endpoint_url(host: str, port: int, mount_path: str = A2A_MOUNT_PATH) -> str:
    """Construct the A2A endpoint URL for the running server.

    Three tiers, most specific first:

    1. Behind an Envoy API gateway (``DR_WORKLOAD_EXTERNAL_URL_HOST`` and
       ``DR_WORKLOAD_EXTERNAL_URL_PREFIX`` are both injected), the gateway's own
       route is the only externally reachable one, so it wins in every hosting
       mode.
    2. In a DataRobot deployment (``MLOPS_DEPLOYMENT_ID`` is set) or workload
       (``WORKLOAD_ID``), composes the platform URL from
       ``DATAROBOT_PUBLIC_API_ENDPOINT`` / ``DATAROBOT_ENDPOINT``.
    3. Otherwise falls back to the local ``http://{host}:{port}/a2a/`` URL.

    ``mount_path`` is the suffix A2A is actually mounted under inside the container.
    Every tier forwards the full prefixed path through to the app, so the suffix has
    to appear in the advertised URL or clients would address a route that does not
    exist. ``DRAgentA2AConfig`` never passes an empty ``mount_path`` (it rejects
    mounting A2A at the application root), but this function stays a generic URL
    composer rather than re-asserting that policy: an empty value here just yields
    the tier's own base with a single trailing slash.
    """
    if external_base := resolve_external_workload_base():
        return join_mount_path(external_base, mount_path)

    deployment_id = get_deployment_id()
    workload_id = get_workload_id()

    if not (deployment_id or workload_id):
        return join_mount_path(f"http://{host}:{port}", mount_path)

    datarobot_endpoint = resolve_datarobot_endpoint(require=True)
    assert datarobot_endpoint is not None  # guaranteed by require=True

    if deployment_id:
        return build_deployment_a2a_url(datarobot_endpoint, deployment_id, mount_path)
    assert workload_id is not None  # non-None guaranteed by the early-return above
    return build_workload_a2a_url(datarobot_endpoint, workload_id, mount_path)


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
    mount_path: str = A2A_MOUNT_PATH,
) -> str:
    """Return the agent card URL, preferring ``external.url`` when provided.

    ``mount_path`` only shapes the derived URL — an explicit ``external.url`` is
    already the caller's final answer and is passed through untouched.
    """
    if external and external.url:
        return external.url
    return get_a2a_endpoint_url(frontend_config.host, frontend_config.port, mount_path)


def build_default_bearer_security_schemes() -> tuple[
    dict[str, SecurityScheme], list[dict[str, list[str]]]
]:
    """Return the default HTTP Bearer scheme for agents without explicit OAuth config."""
    security_schemes = {
        BEARER_SECURITY_SCHEME_NAME: SecurityScheme(
            root=HTTPAuthSecurityScheme(
                type="http",
                scheme="bearer",
                description=BEARER_SECURITY_DESCRIPTION,
            )
        )
    }
    return security_schemes, [{BEARER_SECURITY_SCHEME_NAME: []}]


async def build_security_schemes(
    frontend_config: A2AFrontEndConfig,
    cross_app_access: CrossApplicationAccessConfig | None,
) -> tuple[dict[str, SecurityScheme], list[dict[str, list[str]]]]:
    """Assemble A2A security schemes, merging up to two auth sources.

    * ``server_auth`` → authorization_code flow.
    * ``cross_app_access`` → client_credentials flow.
    * Neither configured → HTTP Bearer scheme for DataRobot API tokens.

    Always returns a populated ``securitySchemes`` map.
    """
    server_auth = frontend_config.server_auth

    if not server_auth and not cross_app_access:
        return build_default_bearer_security_schemes()

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
    mount_path: str = A2A_MOUNT_PATH,
) -> AgentCard:
    """Build an :class:`~a2a.types.AgentCard` for a DataRobot-hosted A2A agent.

    When ``skills`` is empty, a single default skill is generated from
    ``frontend_config.name`` / ``frontend_config.description``.

    ``mount_path`` must match where the A2A app is actually mounted, so the card's
    ``url`` points at the live RPC endpoint rather than the default ``/a2a/``.
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

    url = _resolve_url(frontend_config, external, mount_path)

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
        security_schemes=security_schemes,
        security=security,
        supports_authenticated_extended_card=True,
    )


# ---------------------------------------------------------------------------
# Agent card selection (public GET) and authenticated extended card
# ---------------------------------------------------------------------------


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
    headers = _a2a_headers.get()
    if resolve_identity_from_headers(headers, on_invalid_auth_context="none") is not None:
        return card
    return redact_agent_card(card)


def _extended_card_modifier(card: AgentCard, context: ServerCallContext) -> AgentCard:
    """Serve the extended card for ``agent/getAuthenticatedExtendedCard`` callers."""
    headers = headers_from_a2a_state(context.state)
    if resolve_identity_from_headers(headers) is None:
        raise ServerError(
            error=InvalidParamsError(
                message="Authenticated identity required for extended agent card"
            )
        )
    return card


class DRAgentA2AStarletteApplication(A2AStarletteApplication):
    """A2A server that gates public agent-card access on per-agent developer opt-in.

    Unauthenticated access also depends on platform-level opt-in per cluster to
    route anonymous traffic to the agent; this class enforces only the agent-side
    policy once a request reaches the process.
    """

    def __init__(
        self,
        agent_card: AgentCard,
        http_handler: RequestHandler,
        extended_agent_card: AgentCard | None = None,
        context_builder: CallContextBuilder | None = None,
        card_modifier: Callable[[AgentCard], Awaitable[AgentCard] | AgentCard] | None = None,
        extended_card_modifier: Callable[
            [AgentCard, ServerCallContext], Awaitable[AgentCard] | AgentCard
        ]
        | None = None,
        max_content_length: int | None = 10 * 1024 * 1024,
        *,
        enable_unauthenticated_well_known_route: bool = False,
    ) -> None:
        self._enable_unauthenticated_well_known_route = enable_unauthenticated_well_known_route
        super().__init__(
            agent_card=agent_card,
            http_handler=http_handler,
            extended_agent_card=extended_agent_card,
            context_builder=context_builder,
            card_modifier=card_modifier,
            extended_card_modifier=extended_card_modifier,
            max_content_length=max_content_length,
        )

    async def _handle_get_agent_card(self, request):  # type: ignore[no-untyped-def]
        headers = normalise_headers(dict(request.headers))
        token = _a2a_headers.set(headers)
        try:
            if (
                resolve_identity_from_headers(headers, on_invalid_auth_context="none") is None
                and not self._enable_unauthenticated_well_known_route
            ):
                # The reason belongs in the log, not the response: the caller is anonymous and
                # telling it why it was refused is what leaks the agent's existence.
                logger.info(
                    "Serving 404 for unauthenticated agent-card request: "
                    "enable_unauthenticated_well_known_route is disabled for this agent "
                    "(set it to true in workflow.yaml to allow anonymous access; also "
                    "requires platform-level opt-in per cluster)"
                )
                return JSONResponse(status_code=404, content=AGENT_CARD_NOT_FOUND_BODY)
            return await super()._handle_get_agent_card(request)
        finally:
            _a2a_headers.reset(token)


class DRAgentA2AFrontEndPluginWorker(A2AFrontEndPluginWorker):
    """A2A worker with identity-keyed public cards and an authenticated extended card."""

    def create_a2a_server(
        self,
        agent_card: AgentCard,
        agent_executor: NATWorkflowAgentExecutor,
        *,
        enable_unauthenticated_well_known_route: bool = False,
    ) -> DRAgentA2AStarletteApplication:
        """Create an A2A server with identity-keyed public and extended agent cards.

        Unauthenticated ``GET /.well-known/agent-card.json`` access requires opt-in
        at two levels: platform administrators must enable unauthenticated routing
        per cluster, and ``enable_unauthenticated_well_known_route`` must be set
        in the agent's ``workflow.yaml``. This method enforces the agent-side
        flag only.

        When the agent flag is disabled (default), unauthenticated callers receive
        the generic 404 -- indistinguishable from a nonexistent agent, so the
        refusal is not an existence oracle. When enabled, they receive a redacted
        card. Authenticated callers always receive the full card.
        ``extended_agent_card`` is also wired for
        ``agent/getAuthenticatedExtendedCard`` clients.
        """
        base_server = super().create_a2a_server(agent_card, agent_executor)
        server = DRAgentA2AStarletteApplication(
            agent_card=base_server.agent_card,
            http_handler=base_server.handler.request_handler,
            extended_agent_card=agent_card,
            card_modifier=_public_card_modifier,
            extended_card_modifier=_extended_card_modifier,
            enable_unauthenticated_well_known_route=enable_unauthenticated_well_known_route,
        )
        logger.info("Created A2A server with identity-keyed public agent card")
        return server
