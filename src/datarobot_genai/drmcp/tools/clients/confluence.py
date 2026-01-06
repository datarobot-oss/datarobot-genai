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


class ConfluenceError(Exception):
    """Exception for Confluence API errors."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


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


class ConfluenceComment(BaseModel):
    """Pydantic model for Confluence comment."""

    comment_id: str = Field(..., description="The unique comment ID")
    page_id: str = Field(..., description="The page ID where the comment was added")
    body: str = Field(..., description="Comment content in storage format")

    def as_flat_dict(self) -> dict[str, Any]:
        """Return a flat dictionary representation of the comment."""
        return {
            "comment_id": self.comment_id,
            "page_id": self.page_id,
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
            ConfluenceError: If page is not found
            httpx.HTTPStatusError: If the API request fails
        """
        cloud_id = await self._get_cloud_id()
        url = f"{ATLASSIAN_API_BASE}/ex/confluence/{cloud_id}/wiki/rest/api/content/{page_id}"

        response = await self._client.get(url, params={"expand": self.EXPAND_FIELDS})

        if response.status_code == HTTPStatus.NOT_FOUND:
            raise ConfluenceError(f"Page with ID '{page_id}' not found", status_code=404)

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
            ConfluenceError: If the page is not found
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
            raise ConfluenceError(
                f"Page with title '{title}' not found in space '{space_key}'", status_code=404
            )

        return self._parse_response(results[0])

    def _extract_error_message(self, response: httpx.Response) -> str:
        """Extract error message from Confluence API error response."""
        try:
            error_data = response.json()
            # Confluence API returns errors in different formats
            if "message" in error_data:
                return error_data["message"]
            if "errorMessages" in error_data and error_data["errorMessages"]:
                return "; ".join(error_data["errorMessages"])
            if "errors" in error_data:
                errors = error_data["errors"]
                if isinstance(errors, list):
                    return "; ".join(str(e) for e in errors)
                if isinstance(errors, dict):
                    return "; ".join(f"{k}: {v}" for k, v in errors.items())
        except Exception:
            pass
        return response.text or "Unknown error"

    async def create_page(
        self,
        space_key: str,
        title: str,
        body_content: str,
        parent_id: int | None = None,
    ) -> ConfluencePage:
        """
        Create a new Confluence page in a specified space.

        Args:
            space_key: The key of the Confluence space where the page should live
            title: The title of the new page
            body_content: The content in Confluence Storage Format (XML) or raw text
            parent_id: Optional ID of the parent page for creating a child page

        Returns
        -------
            ConfluencePage with the created page data

        Raises
        ------
            ConfluenceError: If space not found, parent page not found, duplicate title,
                            permission denied, or invalid content
            httpx.HTTPStatusError: If the API request fails with unexpected status
        """
        cloud_id = await self._get_cloud_id()
        url = f"{ATLASSIAN_API_BASE}/ex/confluence/{cloud_id}/wiki/rest/api/content"

        payload: dict[str, Any] = {
            "type": "page",
            "title": title,
            "space": {"key": space_key},
            "body": {
                "storage": {
                    "value": body_content,
                    "representation": "storage",
                }
            },
        }

        if parent_id is not None:
            payload["ancestors"] = [{"id": parent_id}]

        response = await self._client.post(url, json=payload)

        if response.status_code == HTTPStatus.NOT_FOUND:
            error_msg = self._extract_error_message(response)
            if parent_id is not None and "ancestor" in error_msg.lower():
                raise ConfluenceError(
                    f"Parent page with ID '{parent_id}' not found", status_code=404
                )
            raise ConfluenceError(
                f"Space '{space_key}' not found or resource unavailable: {error_msg}",
                status_code=404,
            )

        if response.status_code == HTTPStatus.CONFLICT:
            raise ConfluenceError(
                f"A page with title '{title}' already exists in space '{space_key}'",
                status_code=409,
            )

        if response.status_code == HTTPStatus.FORBIDDEN:
            raise ConfluenceError(
                f"Permission denied: you don't have access to create pages in space '{space_key}'",
                status_code=403,
            )

        if response.status_code == HTTPStatus.BAD_REQUEST:
            error_msg = self._extract_error_message(response)
            raise ConfluenceError(f"Invalid request: {error_msg}", status_code=400)

        if response.status_code == HTTPStatus.TOO_MANY_REQUESTS:
            raise ConfluenceError("Rate limit exceeded. Please try again later.", status_code=429)

        response.raise_for_status()

        return self._parse_response(response.json())

    def _parse_comment_response(self, data: dict, page_id: str) -> ConfluenceComment:
        """Parse API response into ConfluenceComment."""
        body_content = ""
        body = data.get("body", {})
        if isinstance(body, dict):
            storage = body.get("storage", {})
            if isinstance(storage, dict):
                body_content = storage.get("value", "")

        return ConfluenceComment(
            comment_id=str(data.get("id", "")),
            page_id=page_id,
            body=body_content,
        )

    async def add_comment(self, page_id: str, comment_body: str) -> ConfluenceComment:
        """
        Add a comment to a Confluence page.

        Args:
            page_id: The numeric page ID where the comment will be added
            comment_body: The text content of the comment

        Returns
        -------
            ConfluenceComment with the created comment data

        Raises
        ------
            ConfluenceError: If page not found, permission denied, or invalid content
            httpx.HTTPStatusError: If the API request fails with unexpected status
        """
        cloud_id = await self._get_cloud_id()
        url = f"{ATLASSIAN_API_BASE}/ex/confluence/{cloud_id}/wiki/rest/api/content"

        payload: dict[str, Any] = {
            "type": "comment",
            "container": {"id": page_id, "type": "page"},
            "body": {
                "storage": {
                    "value": comment_body,
                    "representation": "storage",
                }
            },
        }

        response = await self._client.post(url, json=payload)

        if response.status_code == HTTPStatus.NOT_FOUND:
            error_msg = self._extract_error_message(response)
            raise ConfluenceError(
                f"Page with ID '{page_id}' not found: {error_msg}",
                status_code=404,
            )

        if response.status_code == HTTPStatus.FORBIDDEN:
            raise ConfluenceError(
                f"Permission denied: you don't have access to add comments to page '{page_id}'",
                status_code=403,
            )

        if response.status_code == HTTPStatus.BAD_REQUEST:
            error_msg = self._extract_error_message(response)
            raise ConfluenceError(f"Invalid request: {error_msg}", status_code=400)

        if response.status_code == HTTPStatus.TOO_MANY_REQUESTS:
            raise ConfluenceError("Rate limit exceeded. Please try again later.", status_code=429)

        response.raise_for_status()

        return self._parse_comment_response(response.json(), page_id)

    async def __aenter__(self) -> "ConfluenceClient":
        """Async context manager entry."""
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any
    ) -> None:
        """Async context manager exit."""
        await self._client.aclose()
