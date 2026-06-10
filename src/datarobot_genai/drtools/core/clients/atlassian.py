# Copyright 2025 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Atlassian API client utilities for OAuth Bearer and API token Basic auth."""

import base64
import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Any
from typing import Literal

import httpx

from datarobot_genai.drtools.core.auth import get_oauth_access_token_with_header_fallback
from datarobot_genai.drtools.core.credentials import AuthResolutionStrategy
from datarobot_genai.drtools.core.credentials import get_credentials
from datarobot_genai.drtools.core.exceptions import ToolError
from datarobot_genai.drtools.core.exceptions import ToolErrorKind

logger = logging.getLogger(__name__)

# Atlassian Cloud API base URL
ATLASSIAN_API_BASE = "https://api.atlassian.com"

# API endpoint paths
OAUTH_ACCESSIBLE_RESOURCES_PATH = "/oauth/token/accessible-resources"
TENANT_INFO_PATH = "/_edge/tenant_info"

# Supported Atlassian service types
AtlassianServiceType = Literal["jira", "confluence"]


class AtlassianAuthMode(StrEnum):
    """How Atlassian REST calls are authenticated."""

    OAUTH_BEARER = "oauth_bearer"
    API_TOKEN_BASIC = "api_token_basic"


@dataclass(frozen=True)
class AtlassianAuth:
    """Resolved Atlassian credentials for Jira/Confluence clients."""

    mode: AtlassianAuthMode
    token: str
    email: str = ""
    site_url: str = ""
    cloud_id: str = ""

    @classmethod
    def oauth_bearer(cls, token: str, *, cloud_id: str = "") -> "AtlassianAuth":
        return cls(mode=AtlassianAuthMode.OAUTH_BEARER, token=token, cloud_id=cloud_id)

    @classmethod
    def api_token_basic(
        cls,
        *,
        email: str,
        api_token: str,
        site_url: str,
        cloud_id: str = "",
    ) -> "AtlassianAuth":
        return cls(
            mode=AtlassianAuthMode.API_TOKEN_BASIC,
            token=api_token,
            email=email,
            site_url=site_url,
            cloud_id=cloud_id,
        )

    def authorization_header(self) -> str:
        if self.mode == AtlassianAuthMode.API_TOKEN_BASIC:
            raw = f"{self.email}:{self.token}".encode()
            encoded = base64.b64encode(raw).decode("ascii")
            return f"Basic {encoded}"
        return f"Bearer {self.token}"

    def http_headers(self) -> dict[str, str]:
        return {
            "Authorization": self.authorization_header(),
            "Accept": "application/json",
            "Content-Type": "application/json",
        }


def normalize_atlassian_site_url(site_url: str) -> str:
    """Normalize a Jira/Confluence Cloud site URL (``https://acme.atlassian.net``)."""
    url = site_url.strip()
    if not url:
        return ""
    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"
    return url.rstrip("/")


def resolve_config_atlassian_auth() -> AtlassianAuth | ToolError:
    """Build Atlassian auth from config/env when ``auth_resolution_strategy=config``."""
    creds = get_credentials()
    api_token = creds.atlassian_api_token.strip()
    if not api_token:
        return ToolError(
            "No Atlassian API token configured. Set ATLASSIAN_API_TOKEN.",
            kind=ToolErrorKind.AUTHENTICATION,
        )

    email = creds.atlassian_email.strip()
    if email:
        site_url = normalize_atlassian_site_url(creds.atlassian_site_url)
        if not site_url:
            return ToolError(
                "ATLASSIAN_SITE_URL is required when ATLASSIAN_EMAIL is set "
                "(API token Basic auth). Example: https://your-domain.atlassian.net",
                kind=ToolErrorKind.AUTHENTICATION,
            )
        return AtlassianAuth.api_token_basic(
            email=email,
            api_token=api_token,
            site_url=site_url,
        )

    return AtlassianAuth.oauth_bearer(api_token)


async def get_jira_access_token() -> AtlassianAuth | ToolError:
    """
    Resolve Jira Cloud credentials (OAuth Bearer or API token Basic in config mode).

    HTTP strategy: DataRobot OBO for provider ``jira``, then
    ``x-datarobot-jira-access-token`` header fallback.

    Config strategy: ``ATLASSIAN_API_TOKEN`` with optional ``ATLASSIAN_EMAIL`` /
    ``ATLASSIAN_SITE_URL`` for API token Basic auth; token alone is treated as a
    static OAuth access token (Bearer).
    """
    creds = get_credentials()
    if creds.auth_resolution_strategy == AuthResolutionStrategy.CONFIG:
        return resolve_config_atlassian_auth()

    token = await get_oauth_access_token_with_header_fallback(
        "jira",
        display_name="Jira",
        access_token_header_segment="jira",
    )
    if isinstance(token, ToolError):
        return token
    return AtlassianAuth.oauth_bearer(token)


async def get_confluence_access_token() -> AtlassianAuth | ToolError:
    """
    Resolve Confluence Cloud credentials (OAuth Bearer or API token Basic in config mode).

    HTTP strategy: DataRobot OBO for provider ``confluence``, then
    ``x-datarobot-confluence-access-token`` header fallback.

    Config strategy: same as :func:`get_jira_access_token` (shared Atlassian site credentials).
    """
    creds = get_credentials()
    if creds.auth_resolution_strategy == AuthResolutionStrategy.CONFIG:
        return resolve_config_atlassian_auth()

    token = await get_oauth_access_token_with_header_fallback(
        "confluence",
        display_name="Confluence",
        access_token_header_segment="confluence",
    )
    if isinstance(token, ToolError):
        return token
    return AtlassianAuth.oauth_bearer(token)


def _find_resource_by_service(
    resources: list[dict[str, Any]], service_type: str
) -> dict[str, Any] | None:
    """
    Find a resource that matches the specified service type.

    Args:
        resources: List of accessible resources from Atlassian API
        service_type: Service type to filter by (e.g., "jira", "confluence")

    Returns
    -------
        Resource dictionary if found, None otherwise
    """
    service_lower = service_type.lower()
    for resource in resources:
        if not resource.get("id"):
            continue
        scopes = resource.get("scopes", [])
        if any(service_lower in scope.lower() for scope in scopes):
            return resource
    return None


def _find_first_resource_with_id(
    resources: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """
    Find the first resource that has an ID.

    Args:
        resources: List of accessible resources from Atlassian API

    Returns
    -------
        Resource dictionary if found, None otherwise
    """
    for resource in resources:
        if resource.get("id"):
            return resource
    return None


async def get_atlassian_cloud_id_from_site(
    client: httpx.AsyncClient,
    site_url: str,
) -> str:
    """
    Resolve cloud ID for API token auth via ``{site}/_edge/tenant_info``.

    Args:
        client: HTTP client with Basic auth headers configured
        site_url: Normalized site URL (e.g. ``https://acme.atlassian.net``)

    Returns
    -------
        Cloud ID string for the Atlassian Cloud site

    Raises
    ------
        ValueError: If cloud ID cannot be retrieved
    """
    url = f"{site_url.rstrip('/')}{TENANT_INFO_PATH}"

    try:
        response = await client.get(url)
        response.raise_for_status()
        data = response.json()
        cloud_id = data.get("cloudId") if isinstance(data, dict) else None
        if not cloud_id:
            raise ValueError("No cloudId in tenant_info response")
        logger.debug("Resolved Atlassian cloud ID from tenant_info: %s", cloud_id)
        return str(cloud_id)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            raise ValueError(
                "Authentication failed for Atlassian API token. "
                "Check ATLASSIAN_EMAIL, ATLASSIAN_API_TOKEN, and ATLASSIAN_SITE_URL."
            ) from e
        logger.error("HTTP error getting cloud ID from tenant_info: %s", e.response.status_code)
        raise ValueError(
            f"Failed to get Atlassian cloud ID from tenant_info: HTTP {e.response.status_code}"
        ) from e
    except httpx.RequestError as e:
        logger.error("Request error getting cloud ID from tenant_info: %s", e)
        raise ValueError("Failed to get Atlassian cloud ID: Network error") from e


async def _get_oauth_cloud_id(
    client: httpx.AsyncClient,
    service_type: AtlassianServiceType | None,
) -> str:
    url = f"{ATLASSIAN_API_BASE}{OAUTH_ACCESSIBLE_RESOURCES_PATH}"

    try:
        response = await client.get(url)
        response.raise_for_status()
        resources = response.json()

        if not resources:
            raise ValueError(
                "No accessible resources found. Ensure OAuth token has required scopes."
            )

        if service_type:
            resource = _find_resource_by_service(resources, service_type)
            if resource:
                cloud_id = resource["id"]
                logger.debug("Using %s cloud ID: %s", service_type, cloud_id)
                return cloud_id
            logger.warning(
                "No %s resource found, falling back to first available resource",
                service_type,
            )

        resource = _find_first_resource_with_id(resources)
        if resource:
            cloud_id = resource["id"]
            logger.debug("Using cloud ID (fallback): %s", cloud_id)
            return cloud_id

        raise ValueError("No cloud ID found in accessible resources")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            raise ValueError(
                "Authentication failed. Token may be expired. "
                "Complete OAuth flow again: GET /oauth/atlassian/authorize"
            ) from e
        logger.error("HTTP error getting cloud ID: %s", e.response.status_code)
        raise ValueError(f"Failed to get Atlassian cloud ID: HTTP {e.response.status_code}") from e
    except httpx.RequestError as e:
        logger.error("Request error getting cloud ID: %s", e)
        raise ValueError("Failed to get Atlassian cloud ID: Network error") from e


async def get_atlassian_cloud_id(
    client: httpx.AsyncClient,
    service_type: AtlassianServiceType | None = None,
    *,
    auth: AtlassianAuth | None = None,
) -> str:
    """
    Get the cloud ID for the authenticated Atlassian instance.

    OAuth Bearer: ``https://api.atlassian.com/oauth/token/accessible-resources``

    API token Basic: ``{site}/_edge/tenant_info``

    According to Atlassian documentation, API calls use:
    ``https://api.atlassian.com/ex/{service}/{cloudId}/...``

    Args:
        client: HTTP client with authentication headers configured
        service_type: Optional service type to filter OAuth resources (e.g. ``jira``)
        auth: Optional resolved auth; when mode is API token Basic, uses ``tenant_info``

    Returns
    -------
        Cloud ID string for the Atlassian instance

    Raises
    ------
        ValueError: If cloud ID cannot be retrieved
    """
    if auth and auth.cloud_id:
        return auth.cloud_id

    if auth and auth.mode == AtlassianAuthMode.API_TOKEN_BASIC:
        if not auth.site_url:
            raise ValueError("Atlassian site URL is required for API token cloud ID lookup")
        return await get_atlassian_cloud_id_from_site(client, auth.site_url)

    return await _get_oauth_cloud_id(client, service_type)
