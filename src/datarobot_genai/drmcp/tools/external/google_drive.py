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

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import aiohttp

from datarobot_genai.drmcp import dr_mcp_tool
from datarobot_genai.drmcp.core.auth import get_access_token

logger = logging.getLogger(__name__)

GOOGLE_DRIVE_API_BASE = "https://www.googleapis.com/drive/v3"
DEFAULT_PAGE_SIZE = 30
MAX_PAGE_SIZE = 1000
API_TIMEOUT = 30.0


@dataclass
class GoogleDriveFile:
    """Represents a Google Drive file with essential metadata."""

    id: str
    name: str
    mime_type: str
    modified_time: str | None = None
    size: int | None = None

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> GoogleDriveFile:
        """Create a GoogleDriveFile instance from API response data.

        Parameters
        ----------
        data : dict[str, Any]
            Raw file data from Google Drive API response.

        Returns
        -------
        GoogleDriveFile
            A parsed GoogleDriveFile instance.
        """
        return cls(
            id=data["id"],
            name=data["name"],
            mime_type=data.get("mimeType", "unknown"),
            modified_time=data.get("modifiedTime"),
            size=data.get("size"),
        )

    def format_summary(self) -> str:
        """Format file information as a single-line summary.

        Returns
        -------
        str
            Formatted string: "file_id: file_name (mime_type)".
        """
        return f"{self.id}: {self.name} ({self.mime_type})"


class GoogleDriveClient:
    """Client for interacting with Google Drive API v3."""

    def __init__(self, access_token: str) -> None:
        """Initialize the Google Drive client.

        Parameters
        ----------
        access_token : str
            OAuth access token for authenticating API requests.
        """
        self.access_token = access_token
        self.base_url = GOOGLE_DRIVE_API_BASE

    def _get_headers(self) -> dict[str, str]:
        """Get HTTP headers for API requests.

        Returns
        -------
        dict[str, str]
            Headers including OAuth authorization.
        """
        return {"Authorization": f"Bearer {self.access_token}"}

    async def list_files(self, page_size: int = DEFAULT_PAGE_SIZE) -> list[GoogleDriveFile]:
        """List files from Google Drive.

        Parameters
        ----------
        page_size : int, optional
            Maximum number of files to return (default: 10, max: 1000).

        Returns
        -------
        list[GoogleDriveFile]
            List of GoogleDriveFile objects.

        Raises
        ------
        aiohttp.ClientResponseError
            If the API returns an error status code.
        aiohttp.ClientError
            If the request fails due to network issues.
        """
        page_size = min(max(1, page_size), MAX_PAGE_SIZE)

        params = {
            "pageSize": page_size,
            "fields": "files(id, name, mimeType, modifiedTime, size)",
        }

        timeout = aiohttp.ClientTimeout(total=API_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(
                f"{self.base_url}/files",
                headers=self._get_headers(),
                params=params,
            ) as response:
                response.raise_for_status()
                data = await response.json()

        files_data = data.get("files", [])
        return [GoogleDriveFile.from_api_response(file) for file in files_data]


@dr_mcp_tool(tags={"data", "google", "list"})
async def list_google_drive_items(page_size: int = DEFAULT_PAGE_SIZE) -> str:
    """List Google Drive items using OAuth token granted by the user.

    Parameters
    ----------
    page_size : int, optional
        Maximum number of items to return (default: 30, max: 1000).

    Returns
    -------
    str
        A string summary of Google Drive items with their IDs and names.
    """
    try:
        oauth_access_token = await get_access_token(provider="google")
        client = GoogleDriveClient(access_token=oauth_access_token)
        files = await client.list_files(page_size=page_size)

        if not files:
            logger.info("No Google Drive items found")
            return "No Google Drive items found."

        result = "\n".join(file.format_summary() for file in files)
        logger.info(f"Found {len(files)} Google Drive items")
        return result

    except aiohttp.ClientResponseError as e:
        error_msg = f"Google Drive API error: {e.status} - {e.message}"
        logger.error(error_msg)
        return f"Error accessing Google Drive: {e.status}"
    except aiohttp.ClientError as e:
        logger.error(f"Request error when accessing Google Drive: {e}")
        return f"Error connecting to Google Drive: {str(e)}"
