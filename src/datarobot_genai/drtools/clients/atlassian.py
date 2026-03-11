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

"""Atlassian API client utilities for OAuth and cloud ID management."""

import logging
from typing import Any
from typing import Literal

import httpx
from datarobot.auth.datarobot.exceptions import OAuthServiceClientErr
from fastmcp.exceptions import ToolError

from datarobot_genai.drmcp.core.auth import get_access_token

logger = logging.getLogger(__name__)

# Atlassian Cloud API base URL
ATLASSIAN_API_BASE = "https://api.atlassian.com"

# API endpoint paths
OAUTH_ACCESSIBLE_RESOURCES_PATH = "/oauth/token/accessible-resources"

# Supported Atlassian service types
AtlassianServiceType = Literal["jira", "confluence"]


async def get_atlassian_access_token() -> str | ToolError:
    """
    Get Atlassian OAuth access token with error handling.

    Returns
    -------
        Access token string on success, ToolError on failure

    Example:
        ```python
        token = await get_atlassian_access_token()
        if isinstance(token, ToolError):
            # Handle error
            return token
        # Use token
        ```
    """
    try:
        access_token = await get_access_token("atlassian")
        if not access_token:
            logger.warning("Empty access token received")
            return ToolError("Received empty access token. Please complete the OAuth flow.")
        return access_token
    except OAuthServiceClientErr as e:
        logger.error(f"OAuth client error: {e}", exc_info=True)
        return ToolError(
            "Could not obtain access token for Atlassian. Make sure the OAuth "
            "permission was granted for the application to act on your behalf."
        )
    except Exception as e:
        logger.error(f"Unexpected error obtaining access token: {e}", exc_info=True)
        return ToolError("An unexpected error occurred while obtaining access token for Atlassian.")


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


def _find_first_resource_with_id(resources: list[dict[str, Any]]) -> dict[str, Any] | None:
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


async def get_atlassian_cloud_id(
    client: httpx.AsyncClient,
    service_type: AtlassianServiceType | None = None,
) -> str:
    """
    Get the cloud ID for the authenticated Atlassian instance.

    According to Atlassian OAuth 2.0 documentation, API calls should use:
    https://api.atlassian.com/ex/{service}/{cloudId}/rest/api/3/...

    Args:
        client: HTTP client with authentication headers configured
        service_type: Optional service type to filter by (e.g., "jira", "confluence").
            If provided, will prioritize resources matching this service type.

    Returns
    -------
        Cloud ID string for the Atlassian instance

    Raises
    ------
        ValueError: If cloud ID cannot be retrieved due to:
            - No accessible resources found
            - No cloud ID found in resources
            - Authentication failure (401)
            - HTTP request failure

    Example:
        ```python
        client = httpx.AsyncClient(headers={"Authorization": f"Bearer {token}"})
        cloud_id = await get_atlassian_cloud_id(client, service_type="jira")
        ```
    """
    url = f"{ATLASSIAN_API_BASE}{OAUTH_ACCESSIBLE_RESOURCES_PATH}"

    try:
        response = await client.get(url)
        response.raise_for_status()
        resources = response.json()

        if not resources:
            raise ValueError(
                "No accessible resources found. Ensure OAuth token has required scopes."
            )

        # If service_type is specified, try to find matching resource
        if service_type:
            resource = _find_resource_by_service(resources, service_type)
            if resource:
                cloud_id = resource["id"]
                logger.debug(f"Using {service_type} cloud ID: {cloud_id}")
                return cloud_id
            logger.warning(
                f"No {service_type} resource found, falling back to first available resource"
            )

        # Fallback: use the first resource with an ID
        resource = _find_first_resource_with_id(resources)
        if resource:
            cloud_id = resource["id"]
            logger.debug(f"Using cloud ID (fallback): {cloud_id}")
            return cloud_id

        raise ValueError("No cloud ID found in accessible resources")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            raise ValueError(
                "Authentication failed. Token may be expired. "
                "Complete OAuth flow again: GET /oauth/atlassian/authorize"
            ) from e
        logger.error(f"HTTP error getting cloud ID: {e.response.status_code}")
        raise ValueError(f"Failed to get Atlassian cloud ID: HTTP {e.response.status_code}") from e
    except httpx.RequestError as e:
        logger.error(f"Request error getting cloud ID: {e}")
        raise ValueError("Failed to get Atlassian cloud ID: Network error") from e
