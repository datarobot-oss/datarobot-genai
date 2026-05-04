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
import os

import httpx
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
from nat.plugins.a2a.server.front_end_config import A2AFrontEndConfig

from datarobot_genai.dragent.deployment_urls import build_deployment_a2a_url
from datarobot_genai.dragent.deployment_urls import resolve_datarobot_endpoint

from .server_auth import CrossApplicationAccessConfig

logger = logging.getLogger(__name__)

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

# Binding references linking the extension to the OpenAPI security scheme.
CROSS_APP_SECURITY_SCHEME_REF = "oauth2"
CROSS_APP_SECURITY_SCHEME_FLOW_REF = "clientCredentials"

CROSS_APP_EXTENSION_DESCRIPTION = (
    "Two-Step Cross-Application Access execution parameters. "
    "Step 1: RFC 8693 Token Exchange prerequisite. "
    "Step 2: RFC 7523 JWT Bearer Grant."
)


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
    mlops_deployment_id = os.getenv("MLOPS_DEPLOYMENT_ID", "")
    if mlops_deployment_id:
        datarobot_endpoint = resolve_datarobot_endpoint(require=True)
        assert datarobot_endpoint is not None  # guaranteed by require=True
        return build_deployment_a2a_url(datarobot_endpoint, mlops_deployment_id)
    return f"http://{host}:{port}/{A2A_MOUNT_PATH}/"


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
        token_url=config.token_url,
        scopes={scope: f"Permission: {scope}" for scope in config.scopes},
    )
    return flow, list(config.scopes)


def build_cross_app_capability_extension(
    config: CrossApplicationAccessConfig,
) -> list[AgentExtension]:
    """Build the JWT Bearer Grant extension entry for ``capabilities.extensions``.

    Only extension-bound fields go in ``params``; ``token_url`` and ``scopes``
    are intentionally omitted — they belong to OpenAPI ``securitySchemes``.
    """
    params: dict = {
        "ref": {
            "scheme": CROSS_APP_SECURITY_SCHEME_REF,
            "flow": CROSS_APP_SECURITY_SCHEME_FLOW_REF,
        },
        "target_audience": config.target_audience,
        "token_endpoint_auth_method": config.token_endpoint_auth_method,
        "token_exchange": config.token_exchange.model_dump(),
        "token_request": config.token_request.model_dump(),
    }
    return [
        AgentExtension(
            uri=JWT_BEARER_GRANT_TYPE_URI,
            description=CROSS_APP_EXTENSION_DESCRIPTION,
            params=params,
        )
    ]


async def build_security_schemes(
    frontend_config: A2AFrontEndConfig,
    cross_app_access: CrossApplicationAccessConfig | None,
) -> tuple[
    dict[str, SecurityScheme] | None,
    list[dict[str, list[str]]] | None,
    list[AgentExtension] | None,
]:
    """Assemble A2A security schemes, merging up to two auth sources.

    * ``server_auth`` → authorization_code flow.
    * ``cross_app_access`` → client_credentials flow + JWT Bearer capability extension.

    Returns ``(security_schemes, security_requirements, extensions)``, all ``None``
    when neither source is configured.
    """
    server_auth = frontend_config.server_auth

    if not server_auth and not cross_app_access:
        return None, None, None

    auth_code_flow, server_auth_scopes = (
        await build_oauth_flow_from_server_auth(server_auth) if server_auth else (None, [])
    )
    client_creds_flow, cross_app_scopes = (
        build_oauth_flow_from_cross_app_access(cross_app_access) if cross_app_access else (None, [])
    )
    extensions = (
        build_cross_app_capability_extension(cross_app_access) if cross_app_access else None
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
    return security_schemes, [{"oauth2": all_scopes}], extensions


# ---------------------------------------------------------------------------
# Agent card factory
# ---------------------------------------------------------------------------


async def create_agent_card(
    frontend_config: A2AFrontEndConfig,
    cross_app_access: CrossApplicationAccessConfig | None,
    skills: list[AgentSkill],
) -> AgentCard:
    """Build an :class:`~a2a.types.AgentCard` for a DataRobot-hosted A2A agent.

    When ``skills`` is empty, a single default skill is generated from
    ``frontend_config.name`` / ``frontend_config.description``.
    """
    security_schemes, security, extensions = await build_security_schemes(
        frontend_config, cross_app_access
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
