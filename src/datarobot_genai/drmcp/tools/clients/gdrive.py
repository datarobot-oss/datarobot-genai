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

"""Google Drive API Client and utilities for OAuth."""

import logging
from typing import Annotated
from typing import Any

import httpx
from datarobot.auth.datarobot.exceptions import OAuthServiceClientErr
from fastmcp.exceptions import ToolError
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from datarobot_genai.drmcp.core.auth import get_access_token

logger = logging.getLogger(__name__)

DEFAULT_FIELDS = "nextPageToken,files(id,name,size,mimeType,webViewLink,createdTime,modifiedTime)"
DEFAULT_ORDER = "modifiedTime desc"
MAX_PAGE_SIZE = 100
LIMIT = 500


async def get_gdrive_access_token() -> str | ToolError:
    """
    Get Google Drive OAuth access token with error handling.

    Returns
    -------
        Access token string on success, ToolError on failure

    Example:
        ```python
        token = await get_gdrive_access_token()
        if isinstance(token, ToolError):
            # Handle error
            return token
        # Use token
        ```
    """
    try:
        access_token = await get_access_token("google")
        if not access_token:
            logger.warning("Empty access token received")
            return ToolError("Received empty access token. Please complete the OAuth flow.")
        return access_token
    except OAuthServiceClientErr as e:
        logger.error(f"OAuth client error: {e}", exc_info=True)
        return ToolError(
            "Could not obtain access token for Google. Make sure the OAuth "
            "permission was granted for the application to act on your behalf."
        )
    except Exception as e:
        logger.error(f"Unexpected error obtaining access token: {e}", exc_info=True)
        return ToolError("An unexpected error occurred while obtaining access token for Google.")


class GoogleDriveError(Exception):
    """Exception for Google Drive API errors."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


PrimitiveData = str | int | float | bool | None


class GoogleDriveFile(BaseModel):
    """Represents a file from Google Drive."""

    id: str
    name: str
    mime_type: Annotated[str, Field(alias="mimeType")]
    size: int | None = None
    web_view_link: Annotated[str | None, Field(alias="webViewLink")] = None
    created_time: Annotated[str | None, Field(alias="createdTime")] = None
    modified_time: Annotated[str | None, Field(alias="modifiedTime")] = None

    model_config = ConfigDict(populate_by_name=True)

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> "GoogleDriveFile":
        """Create a GoogleDriveFile from API response data."""
        return cls(
            id=data.get("id", "Unknown"),
            name=data.get("name", "Unknown"),
            mime_type=data.get("mimeType", "Unknown"),
            size=int(data["size"]) if data.get("size") else None,
            web_view_link=data.get("webViewLink"),
            created_time=data.get("createdTime"),
            modified_time=data.get("modifiedTime"),
        )


class PaginatedResult(BaseModel):
    """Result of a paginated API call."""

    files: list[GoogleDriveFile]
    next_page_token: str | None = None


class GoogleDriveClient:
    """Client for interacting with Google Drive API."""

    def __init__(self, access_token: str) -> None:
        self._client = httpx.AsyncClient(
            base_url="https://www.googleapis.com/drive/v3/files",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=30.0,
        )

    async def list_files(
        self,
        page_size: int,
        limit: int,
        page_token: str | None = None,
        query: str | None = None,
    ) -> PaginatedResult:
        """
        List files from Google Drive.

        It's public API for GoogleDriveClient.

        Args:
            page_size: Number of files to return per 1 gdrive api request.
            limit: Maximum number of files to return.
            page_token: Optional token (specific for gdrive api) allowing to query next page.
            query: Optional query to filter results.
                If not provided it'll list all authorized user files.
                If the query doesn't contain operators (contains, =, etc.), it will be treated as
                a name search: "name contains '{query}'".

        Returns
        -------
            List of Google Drive files.
        """
        if page_size <= 0:
            raise GoogleDriveError("Error: page size must be positive.")
        if limit <= 0:
            raise GoogleDriveError("Error: limit must be positive.")
        if limit < page_size:
            raise GoogleDriveError("Error: limit must be bigger than or equal to page size.")
        if limit % page_size != 0:
            raise GoogleDriveError("Error: limit must be multiplication of page size.")

        page_size = min(page_size, MAX_PAGE_SIZE)
        limit = min(limit, LIMIT)
        fetched = 0

        formatted_query = self._get_formatted_query(query)

        files: list[GoogleDriveFile] = []

        while fetched < limit:
            data = await self._list_files(
                page_size=page_size,
                page_token=page_token,
                query=formatted_query,
            )
            files.extend(data.files)
            fetched += len(data.files)
            page_token = data.next_page_token

            if not page_token:
                break

        return PaginatedResult(files=files, next_page_token=page_token)

    async def _list_files(
        self,
        page_size: int,
        page_token: str | None = None,
        query: str | None = None,
    ) -> PaginatedResult:
        """Fetch a page of files from Google Drive."""
        params: dict[str, PrimitiveData] = {
            "pageSize": page_size,
            "fields": DEFAULT_FIELDS,
            "orderBy": DEFAULT_ORDER,
        }
        if page_token:
            params["pageToken"] = page_token
        if query:
            params["q"] = query

        response = await self._client.get(url="/", params=params)
        response.raise_for_status()
        data = response.json()

        files = [
            GoogleDriveFile.from_api_response(file_data) for file_data in data.get("files", [])
        ]
        next_page_token = data.get("nextPageToken")
        return PaginatedResult(files=files, next_page_token=next_page_token)

    @staticmethod
    def _get_formatted_query(query: str | None) -> str | None:
        """Get formatted Google Drive API query.

        Args:
            query: Optional search query string (e.g., "name contains 'report'"").
                If the query doesn't contain operators (contains, =, etc.), it will be treated as
                a name search: "name contains '{query}'".

        Returns
        -------
            Correctly formatted query (if provided)
        """
        if not query:
            return None

        # If query doesn't look like a formatted query (no operators), format it as a name search
        # Check if query already has Google Drive API operators
        has_operator = any(
            op in query for op in [" contains ", "=", "!=", " in ", " and ", " or ", " not "]
        )
        formatted_query = query
        if not has_operator and query.strip():
            # Simple text search - format as name contains query
            # Escape backslashes first, then single quotes for Google Drive API
            escaped_query = query.replace("\\", "\\\\").replace("'", "\\'")
            formatted_query = f"name contains '{escaped_query}'"
            logger.debug(f"Auto-formatted query '{query}' to '{formatted_query}'")
        return formatted_query

    async def __aenter__(self) -> "GoogleDriveClient":
        """Async context manager entry."""
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any
    ) -> None:
        """Async context manager exit."""
        await self._client.aclose()
