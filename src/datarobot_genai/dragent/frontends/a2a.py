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
security scheme assembly, RFC 8693 capability extensions, endpoint URL
resolution, and the per-user executor adapter.  The FastAPI framework glue
lives in :mod:`~datarobot_genai.dragent.frontends.fastapi`.
"""

import logging

import httpx
from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from a2a.types import AgentCapabilities
from a2a.types import AgentCard
from a2a.types import AgentExtension
from a2a.types import AgentSkill
from a2a.types import AuthorizationCodeOAuthFlow
from a2a.types import ClientCredentialsOAuthFlow
from a2a.types import OAuth2SecurityScheme
from a2a.types import OAuthFlows
from a2a.types import SecurityScheme
from nat.authentication.oauth2.oauth2_resource_server_config import OAuth2ResourceServerConfig
from nat.front_ends.fastapi.fastapi_front_end_plugin_worker import SessionManager
from nat.plugins.a2a.server.agent_executor_adapter import NATWorkflowAgentExecutor
from nat.plugins.a2a.server.front_end_config import A2AFrontEndConfig

from datarobot_genai.dragent.deployment_urls import build_deployment_a2a_url
from datarobot_genai.dragent.deployment_urls import resolve_datarobot_endpoint

from .server_auth import OAuth2TokenExchangeConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

A2A_MOUNT_PATH = "a2a"

OAUTH2_SECURITY_DESCRIPTION_WITH_TOKEN_EXCHANGE = (
    "OAuth 2.0 authorization utilizing RFC 8693 Token Exchange. Clients must "
    "supply a valid internal passport JWT as the subject token. Refer to the "
    "capabilities.extensions block for strict token exchange parameters and "
    "audience specifications."
)

# The extension explicitly references the security scheme key so SDKs can resolve
# the RFC 8693 Token Exchange override without ambiguity.
RFC8693_GRANT_TYPE_URI = "urn:ietf:params:oauth:grant-type:token-exchange"
RFC8693_SECURITY_SCHEME_REF = "oauth2"
RFC8693_SECURITY_SCHEME_FLOW_REF = "clientCredentials"
RFC8693_TOKEN_EXCHANGE_EXTENSION_DESCRIPTION = (
    "Two-Step RFC 8693 Token Exchange execution parameters."
)


# ---------------------------------------------------------------------------
# Per-user executor adapter
# ---------------------------------------------------------------------------


class PerUserCompatibleAgentExecutor(NATWorkflowAgentExecutor):
    """Subclass of NATWorkflowAgentExecutor that supports per-user workflows.

    Two problems with the parent class for per-user workflows:

    1. ``__init__`` accesses ``session_manager.workflow`` which raises ``ValueError``
       for per-user workflows.  We bypass it and log via ``config.workflow.type`` instead.

    2. ``execute`` calls ``self.session_manager.session()`` with no ``user_id``, which
       raises ``ValueError`` for per-user workflows.  We override it to pass the A2A
       ``context_id`` as the ``user_id``, giving each conversation its own isolated
       per-user workflow instance.
    """

    def __init__(self, session_manager: SessionManager) -> None:
        # Bypass parent __init__ to avoid session_manager.workflow access,
        # which raises ValueError for per-user workflows.
        self.session_manager = session_manager
        logger.info(
            "Initialized NATWorkflowAgentExecutor (message-only) for workflow: %s",
            session_manager.config.workflow.type,
        )

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:  # type: ignore[override]
        # Inject the A2A context_id as user_id before delegating to the parent execute.
        # The parent calls self.session_manager.session() with no user_id, which raises
        # ValueError for per-user workflows.  Setting the context var here means the
        # SessionManager's _get_user_id_from_context() will find it automatically.
        token = None
        if context.context_id:
            token = self.session_manager._context_state.user_id.set(context.context_id)
        try:
            await super().execute(context, event_queue)
        finally:
            if token is not None:
                self.session_manager._context_state.user_id.reset(token)


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
    import os

    mlops_deployment_id = os.getenv("MLOPS_DEPLOYMENT_ID", "")
    if mlops_deployment_id:
        datarobot_endpoint = resolve_datarobot_endpoint(require=True)
        return build_deployment_a2a_url(datarobot_endpoint, mlops_deployment_id)
    return f"http://{host}:{port}/{A2A_MOUNT_PATH}/"


# ---------------------------------------------------------------------------
# OAuth2 / RFC 8693 security scheme helpers
# ---------------------------------------------------------------------------


async def resolve_oauth_endpoints(
    server_auth_config: OAuth2ResourceServerConfig,
) -> tuple[str, str]:
    """Resolve authorization and token URLs from an OAuth2ResourceServerConfig.

    Uses OIDC discovery when ``discovery_url`` is set, otherwise derives URLs
    from ``issuer_url``.

    Returns
    -------
    tuple[str, str]
        ``(authorization_url, token_url)``
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


def build_oauth_flow_from_token_exchange(
    config: OAuth2TokenExchangeConfig,
) -> tuple[ClientCredentialsOAuthFlow, list[str]]:
    """Build the client_credentials flow and scopes from an OAuth2TokenExchangeConfig.

    Token URL and scopes live only here (OpenAPI-compatible). RFC 8693 second-phase
    requirements are signaled separately via
    :func:`build_token_exchange_capability_extension`.
    """
    flow = ClientCredentialsOAuthFlow(
        token_url=config.token_url,
        scopes={scope: f"Permission: {scope}" for scope in config.scopes},
    )
    return flow, list(config.scopes)


def build_token_exchange_capability_extension(
    config: OAuth2TokenExchangeConfig,
) -> list[AgentExtension]:
    """Build the RFC 8693 agent card extension for OAuth2 token exchange.

    OpenAPI ``token_url`` / ``scopes`` remain on ``securitySchemes.oauth2.flows``.
    ``params`` carries ``subject_token_constraints``, ``token_exchange_request``,
    and a ``ref`` binding to the client-credentials flow.
    """
    params = {
        "ref": {
            "scheme": RFC8693_SECURITY_SCHEME_REF,
            "flow": RFC8693_SECURITY_SCHEME_FLOW_REF,
        },
        "subject_token_constraints": config.subject_token_constraints.model_dump(),
        "token_exchange_request": config.token_exchange_request.model_dump(),
    }
    return [
        AgentExtension(
            uri=RFC8693_GRANT_TYPE_URI,
            description=RFC8693_TOKEN_EXCHANGE_EXTENSION_DESCRIPTION,
            params=params,
        )
    ]


async def build_security_schemes(
    frontend_config: A2AFrontEndConfig,
    token_exchange: OAuth2TokenExchangeConfig | None,
) -> tuple[
    dict[str, SecurityScheme] | None,
    list[dict[str, list[str]]] | None,
    list[AgentExtension] | None,
]:
    """Assemble A2A security schemes from a frontend configuration.

    Supports two independent auth sources merged into a single ``oauth2``
    security scheme with separate flows:

    * ``server_auth`` (OAuth2ResourceServerConfig) → authorization_code flow.
    * ``token_exchange`` (OAuth2TokenExchangeConfig) → client_credentials flow
      + RFC 8693 capability extension.

    Returns
    -------
    tuple
        ``(security_schemes, security_requirements, capability_extensions)`` —
        all ``None`` when neither auth source is configured.
    """
    server_auth = frontend_config.server_auth

    if not server_auth and not token_exchange:
        return None, None, None

    auth_code_flow, server_auth_scopes = (
        await build_oauth_flow_from_server_auth(server_auth) if server_auth else (None, [])
    )
    client_creds_flow, token_exchange_scopes = (
        build_oauth_flow_from_token_exchange(token_exchange) if token_exchange else (None, [])
    )
    extensions = (
        build_token_exchange_capability_extension(token_exchange) if token_exchange else None
    )

    all_scopes = list(dict.fromkeys(server_auth_scopes + token_exchange_scopes))
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
    return security_schemes, [{"oauth2": all_scopes}], extensions


# ---------------------------------------------------------------------------
# Agent card factory
# ---------------------------------------------------------------------------


async def create_agent_card(
    frontend_config: A2AFrontEndConfig,
    token_exchange: OAuth2TokenExchangeConfig | None,
    skills: list[AgentSkill],
) -> AgentCard:
    """Build an :class:`~a2a.types.AgentCard` for a DataRobot-hosted A2A agent.

    Parameters
    ----------
    frontend_config:
        NAT A2A frontend configuration (name, description, capabilities, etc.).
    token_exchange:
        Optional DR OAuth2 token exchange configuration for RFC 8693.
    skills:
        Skills to advertise. When empty, a single default skill is generated
        from ``frontend_config.name`` and ``frontend_config.description``.
    """
    security_schemes, security, extensions = await build_security_schemes(
        frontend_config, token_exchange
    )

    resolved_skills = skills or [
        AgentSkill(
            id="call",
            name=frontend_config.name,
            description=frontend_config.description,
            tags=[],
            examples=[],
        )
    ]

    return AgentCard(
        name=frontend_config.name,
        description=frontend_config.description,
        url=get_a2a_endpoint_url(frontend_config.host, frontend_config.port),
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
    )
