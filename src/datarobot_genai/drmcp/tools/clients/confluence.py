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

"""Async client for interacting with Confluence Cloud REST API.

At the moment of creating this client, official Confluence SDK is not supporting async.
"""

import logging
from http import HTTPStatus
from typing import Any

import httpx
from pydantic import BaseModel
from pydantic import Field

from .atlassian import ATLASSIAN_API_BASE
from .atlassian import get_atlassian_cloud_id

logger = logging.getLogger(__name__)


class ConfluencePage(BaseModel):
    """Pydantic model for Confluence page."""

    page_id: str = Field(..., description="The unique page ID")
    title: str = Field(..., description="Page title")
    space_id: str = Field(..., description="Space ID where the page resides")
    space_key: str | None = Field(None, description="Space key (if available)")
    body: str = Field(..., description="Page content in storage format (HTML-like)")

    def as_flat_dict(self) -> dict[str, Any]:
        """Return a flat dictionary representation of the page."""
        return {
            "page_id": self.page_id,
            "title": self.title,
            "space_id": self.space_id,
            "space_key": self.space_key,
            "body": self.body,
        }


class ConfluenceClient:
    """
    Client for interacting with Confluence API using OAuth access token.

    At the moment of creating this client, official Confluence SDK is not supporting async.
    """

    EXPAND_FIELDS = "body.storage,space"

    def __init__(self, access_token: str) -> None:
        """
        Initialize Confluence client with access token.

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
        Get the cloud ID for the authenticated Atlassian Confluence instance.

        According to Atlassian OAuth 2.0 documentation, API calls should use:
        https://api.atlassian.com/ex/confluence/{cloudId}/wiki/rest/api/...

        Returns
        -------
            Cloud ID string

        Raises
        ------
            ValueError: If cloud ID cannot be retrieved
        """
        if self._cloud_id:
            return self._cloud_id

        self._cloud_id = await get_atlassian_cloud_id(self._client, service_type="confluence")
        return self._cloud_id

    def _parse_response(self, data: dict) -> ConfluencePage:
        """Parse API response into ConfluencePage."""
        body_content = ""
        body = data.get("body", {})
        if isinstance(body, dict):
            storage = body.get("storage", {})
            if isinstance(storage, dict):
                body_content = storage.get("value", "")

        space = data.get("space", {})
        space_key = space.get("key") if isinstance(space, dict) else None
        space_id = space.get("id", "") if isinstance(space, dict) else data.get("spaceId", "")

        return ConfluencePage(
            page_id=str(data.get("id", "")),
            title=data.get("title", ""),
            space_id=str(space_id),
            space_key=space_key,
            body=body_content,
        )

    async def get_page_by_id(self, page_id: str) -> ConfluencePage:
        """
        Get a Confluence page by its ID.

        Args:
            page_id: The numeric page ID

        Returns
        -------
            ConfluencePage with page data

        Raises
        ------
            ValueError: If page is not found
            httpx.HTTPStatusError: If the API request fails
        """
        cloud_id = await self._get_cloud_id()
        url = f"{ATLASSIAN_API_BASE}/ex/confluence/{cloud_id}/wiki/rest/api/content/{page_id}"

        response = await self._client.get(url, params={"expand": self.EXPAND_FIELDS})

        if response.status_code == HTTPStatus.NOT_FOUND:
            raise ValueError(f"Page with ID '{page_id}' not found")

        response.raise_for_status()
        return self._parse_response(response.json())

    async def get_page_by_title(self, title: str, space_key: str) -> ConfluencePage:
        """
        Get a Confluence page by its title within a specific space.

        Args:
            title: The exact page title
            space_key: The space key where the page resides

        Returns
        -------
            ConfluencePage with page data

        Raises
        ------
            ValueError: If the page is not found
            httpx.HTTPStatusError: If the API request fails
        """
        cloud_id = await self._get_cloud_id()
        url = f"{ATLASSIAN_API_BASE}/ex/confluence/{cloud_id}/wiki/rest/api/content"

        response = await self._client.get(
            url,
            params={
                "title": title,
                "spaceKey": space_key,
                "expand": self.EXPAND_FIELDS,
            },
        )
        response.raise_for_status()

        data = response.json()
        results = data.get("results", [])

        if not results:
            raise ValueError(f"Page with title '{title}' not found in space '{space_key}'")

        return self._parse_response(results[0])

    async def __aenter__(self) -> "ConfluenceClient":
        """Async context manager entry."""
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any
    ) -> None:
        """Async context manager exit."""
        await self._client.aclose()
