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

import logging
from typing import Any

import httpx

from .atlassian import ATLASSIAN_API_BASE
from .atlassian import get_atlassian_cloud_id

logger = logging.getLogger(__name__)


class JiraClient:
    """Client for interacting with Jira API using OAuth access token."""

    def __init__(self, access_token: str):
        """
        Initialize Jira client with access token.

        Args:
            access_token: OAuth access token for Atlassian API
        """
        self.access_token = access_token
        self._client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )
        self._cloud_id: str | None = None

    async def _get_cloud_id(self) -> str:
        """
        Get the cloud ID for the authenticated Atlassian Jira instance.

        According to Atlassian OAuth 2.0 documentation, API calls should use:
        https://api.atlassian.com/ex/jira/{cloudId}/rest/api/3/...

        Returns
        -------
            Cloud ID string

        Raises
        ------
            ValueError: If cloud ID cannot be retrieved
        """
        if self._cloud_id:
            return self._cloud_id

        self._cloud_id = await get_atlassian_cloud_id(self._client, service_type="jira")
        return self._cloud_id

    async def get_jira_issue(self, issue_key: str) -> dict[str, Any]:
        """
        Get a Jira issue by its key.

        Args:
            issue_key: The key (ID) of the Jira issue, e.g., 'PROJ-123'

        Returns
        -------
            Dictionary containing the issue data

        Raises
        ------
            httpx.HTTPStatusError: If the API request fails
        """
        cloud_id = await self._get_cloud_id()
        url = f"{ATLASSIAN_API_BASE}/ex/jira/{cloud_id}/rest/api/3/issue/{issue_key}"

        response = await self._client.get(url)
        response.raise_for_status()
        return response.json()

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
