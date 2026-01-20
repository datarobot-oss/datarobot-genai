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

import io
import json
import logging
import uuid
from typing import Annotated
from typing import Any
from typing import Literal

import httpx
from datarobot.auth.datarobot.exceptions import OAuthServiceClientErr
from fastmcp.exceptions import ToolError
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pypdf import PdfReader

from datarobot_genai.drmcp.core.auth import get_access_token

logger = logging.getLogger(__name__)

SUPPORTED_FIELDS = {
    "id",
    "name",
    "size",
    "mimeType",
    "webViewLink",
    "createdTime",
    "modifiedTime",
    "starred",
    "trashed",
}
SUPPORTED_FIELDS_STR = ",".join(SUPPORTED_FIELDS)
DEFAULT_FIELDS = f"nextPageToken,files({SUPPORTED_FIELDS_STR})"
GOOGLE_DRIVE_FOLDER_MIME = "application/vnd.google-apps.folder"
DEFAULT_ORDER = "modifiedTime desc"
MAX_PAGE_SIZE = 100
LIMIT = 500

GOOGLE_WORKSPACE_EXPORT_MIMES: dict[str, str] = {
    "application/vnd.google-apps.document": "text/markdown",
    "application/vnd.google-apps.spreadsheet": "text/csv",
    "application/vnd.google-apps.presentation": "text/plain",
}

# MIME type mappings for content conversion during upload to Google Workspace formats
UPLOAD_CONTENT_TYPES: dict[str, str] = {
    "application/vnd.google-apps.document": "text/plain",
    "application/vnd.google-apps.spreadsheet": "text/csv",
}

BINARY_MIME_PREFIXES = (
    "image/",
    "audio/",
    "video/",
    "application/zip",
    "application/octet-stream",
    "application/vnd.google-apps.drawing",
)

PDF_MIME_TYPE = "application/pdf"


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
    starred: bool | None = None
    trashed: bool | None = None

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
            starred=data.get("starred"),
            trashed=data.get("trashed"),
        )

    def as_flat_dict(self) -> dict[str, Any]:
        """Return a flat dictionary representation of the file."""
        result: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "mimeType": self.mime_type,
        }
        if self.size is not None:
            result["size"] = self.size
        if self.web_view_link is not None:
            result["webViewLink"] = self.web_view_link
        if self.created_time is not None:
            result["createdTime"] = self.created_time
        if self.modified_time is not None:
            result["modifiedTime"] = self.modified_time
        if self.starred is not None:
            result["starred"] = self.starred
        if self.trashed is not None:
            result["trashed"] = self.trashed
        return result


class PaginatedResult(BaseModel):
    """Result of a paginated API call."""

    files: list[GoogleDriveFile]
    next_page_token: str | None = None


class GoogleDriveFileContent(BaseModel):
    """Content retrieved from a Google Drive file."""

    id: str
    name: str
    mime_type: str
    content: str
    original_mime_type: str
    was_exported: bool = False
    size: int | None = None
    web_view_link: str | None = None

    def as_flat_dict(self) -> dict[str, Any]:
        """Return a flat dictionary representation of the file content."""
        result: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "mimeType": self.mime_type,
            "content": self.content,
            "originalMimeType": self.original_mime_type,
            "wasExported": self.was_exported,
        }
        if self.size is not None:
            result["size"] = self.size
        if self.web_view_link is not None:
            result["webViewLink"] = self.web_view_link
        return result


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
        folder_id: str | None = None,
        recursive: bool = False,
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
            folder_id: The ID of a specific folder to list or search within.
                If omitted, searches the entire Drive.
            recursive: If True, searches all subfolders.
                If False and folder_id is provided, only lists immediate children.

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
        formatted_query = self._build_query(query, folder_id)

        if not recursive or not folder_id:
            files, next_token = await self._fetch_paginated(
                page_size=page_size,
                limit=limit,
                page_token=page_token,
                query=formatted_query,
            )
            return PaginatedResult(files=files, next_page_token=next_token)

        files = await self._fetch_recursive(
            root_folder_id=folder_id,
            base_query=query,
            page_size=page_size,
            limit=limit,
        )

        return PaginatedResult(files=files, next_page_token=page_token)

    async def _fetch_paginated(
        self,
        page_size: int,
        limit: int,
        page_token: str | None,
        query: str | None,
    ) -> tuple[list[GoogleDriveFile], str | None]:
        fetched = 0
        files: list[GoogleDriveFile] = []
        next_page_token = page_token

        while fetched < limit:
            data = await self._list_files(
                page_size=page_size,
                page_token=next_page_token,
                query=query,
            )

            files.extend(data.files)
            fetched += len(data.files)
            next_page_token = data.next_page_token

            if not next_page_token:
                break

        return files, next_page_token

    async def _fetch_recursive(
        self,
        root_folder_id: str,
        base_query: str | None,
        page_size: int,
        limit: int,
    ) -> list[GoogleDriveFile]:
        collected: list[GoogleDriveFile] = []
        folders_to_visit: list[str] = [root_folder_id]

        while folders_to_visit and len(collected) < limit:
            current_folder = folders_to_visit.pop(0)

            query = self._build_query(base_query, current_folder)

            files, _ = await self._fetch_paginated(
                page_size=page_size,
                limit=limit - len(collected),
                page_token=None,
                query=query,
            )

            for file in files:
                collected.append(file)

                if file.mime_type == GOOGLE_DRIVE_FOLDER_MIME:
                    folders_to_visit.append(file.id)

                if len(collected) >= limit:
                    break

        return collected

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

    def _build_query(self, query: str | None, folder_id: str | None) -> str | None:
        """Build Google Drive API query.

        Args:
            query: Optional search query string (e.g., "name contains 'report'"").
                If the query doesn't contain operators (contains, =, etc.), it will be treated as
                a name search: "name contains '{query}'".
            folder_id: Optional folder id.
                If provided it'll narrow query to search/list only in given folder.

        Returns
        -------
            Correctly builded query (if provided)
        """
        base_query = self._get_formatted_query(query)

        if base_query:
            # Case #1 -- Some query provided and contains in parents (gdrive "folder id")
            if "in parents" in base_query and folder_id:
                logger.debug(
                    "In-parents (parent folder) already used in query. "
                    "Omiting folder_id argument. "
                    f"Query: {base_query} | FolderId: {folder_id}"
                )
                return base_query
            # Case #2 -- Some query provided without "in parents" and folder id provided.
            elif folder_id:
                return f"{base_query} and '{folder_id}' in parents"
            # Case #3 -- Query provided without "in parents" and no folder id.
            else:
                return base_query

        # Case #4 -- Base query is null but folder id provided
        if folder_id:
            return f"'{folder_id}' in parents"

        # Case #5 -- Neither query not folder provided
        return None

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

    @staticmethod
    def _is_binary_mime_type(mime_type: str) -> bool:
        """Check if MIME type indicates binary content that's not useful for LLM consumption.

        Args:
            mime_type: The MIME type to check.

        Returns
        -------
            True if the MIME type is considered binary, False otherwise.
        """
        return any(mime_type.startswith(prefix) for prefix in BINARY_MIME_PREFIXES)

    async def get_file_metadata(self, file_id: str) -> GoogleDriveFile:
        """Get file metadata from Google Drive.

        Args:
            file_id: The ID of the file to get metadata for.

        Returns
        -------
            GoogleDriveFile with file metadata.

        Raises
        ------
            GoogleDriveError: If the file is not found or access is denied.
        """
        params = {"fields": SUPPORTED_FIELDS_STR}
        response = await self._client.get(f"/{file_id}", params=params)

        if response.status_code == 404:
            raise GoogleDriveError(f"File with ID '{file_id}' not found.")
        if response.status_code == 403:
            raise GoogleDriveError(f"Permission denied: you don't have access to file '{file_id}'.")
        if response.status_code == 429:
            raise GoogleDriveError("Rate limit exceeded. Please try again later.")

        response.raise_for_status()
        return GoogleDriveFile.from_api_response(response.json())

    async def update_file_metadata(
        self,
        file_id: str,
        new_name: str | None = None,
        starred: bool | None = None,
        trashed: bool | None = None,
    ) -> GoogleDriveFile:
        """Update file metadata in Google Drive.

        Args:
            file_id: The ID of the file to update.
            new_name: A new name to rename the file. Must not be empty or whitespace.
            starred: Set to True to star the file or False to unstar it.
            trashed: Set to True to trash the file or False to restore it.

        Returns
        -------
            GoogleDriveFile with updated metadata.

        Raises
        ------
            GoogleDriveError: If no update fields are provided, file is not found,
                             access is denied, or the request is invalid.
        """
        if new_name is None and starred is None and trashed is None:
            raise GoogleDriveError(
                "At least one of new_name, starred, or trashed must be provided."
            )

        if new_name is not None and not new_name.strip():
            raise GoogleDriveError("new_name cannot be empty or whitespace.")

        body: dict[str, Any] = {}
        if new_name is not None:
            body["name"] = new_name
        if starred is not None:
            body["starred"] = starred
        if trashed is not None:
            body["trashed"] = trashed

        response = await self._client.patch(
            f"/{file_id}",
            json=body,
            params={"fields": SUPPORTED_FIELDS_STR, "supportsAllDrives": "true"},
        )

        if response.status_code == 404:
            raise GoogleDriveError(f"File with ID '{file_id}' not found.")
        if response.status_code == 403:
            raise GoogleDriveError(
                f"Permission denied: you don't have permission to update file '{file_id}'."
            )
        if response.status_code == 400:
            raise GoogleDriveError("Bad request: invalid parameters for file update.")
        if response.status_code == 429:
            raise GoogleDriveError("Rate limit exceeded. Please try again later.")

        response.raise_for_status()
        return GoogleDriveFile.from_api_response(response.json())

    async def _export_workspace_file(self, file_id: str, export_mime_type: str) -> str:
        """Export a Google Workspace file to the specified format.

        Args:
            file_id: The ID of the Google Workspace file.
            export_mime_type: The MIME type to export to (e.g., 'text/markdown').

        Returns
        -------
            The exported content as a string.

        Raises
        ------
            GoogleDriveError: If export fails.
        """
        response = await self._client.get(
            f"/{file_id}/export",
            params={"mimeType": export_mime_type},
        )

        if response.status_code == 404:
            raise GoogleDriveError(f"File with ID '{file_id}' not found.")
        if response.status_code == 403:
            raise GoogleDriveError(
                f"Permission denied: you don't have access to export file '{file_id}'."
            )
        if response.status_code == 400:
            raise GoogleDriveError(
                f"Cannot export file '{file_id}' to format '{export_mime_type}'. "
                "The file may not support this export format."
            )
        if response.status_code == 429:
            raise GoogleDriveError("Rate limit exceeded. Please try again later.")

        response.raise_for_status()
        return response.text

    async def _download_file(self, file_id: str) -> str:
        """Download a regular file's content from Google Drive as text."""
        content = await self._download_file_bytes(file_id)
        return content.decode("utf-8")

    async def _download_file_bytes(self, file_id: str) -> bytes:
        """Download a file's content as bytes from Google Drive.

        Args:
            file_id: The ID of the file to download.

        Returns
        -------
            The file content as bytes.

        Raises
        ------
            GoogleDriveError: If download fails.
        """
        response = await self._client.get(
            f"/{file_id}",
            params={"alt": "media"},
        )

        if response.status_code == 404:
            raise GoogleDriveError(f"File with ID '{file_id}' not found.")
        if response.status_code == 403:
            raise GoogleDriveError(
                f"Permission denied: you don't have access to download file '{file_id}'."
            )
        if response.status_code == 429:
            raise GoogleDriveError("Rate limit exceeded. Please try again later.")

        response.raise_for_status()
        return response.content

    def _extract_text_from_pdf(self, pdf_bytes: bytes) -> str:
        """Extract text from PDF bytes using pypdf.

        Args:
            pdf_bytes: The PDF file content as bytes.

        Returns
        -------
            Extracted text from the PDF.

        Raises
        ------
            GoogleDriveError: If PDF text extraction fails.
        """
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            text_parts = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            return "\n\n".join(text_parts)
        except Exception as e:
            raise GoogleDriveError(f"Failed to extract text from PDF: {e}")

    async def read_file_content(
        self, file_id: str, target_format: str | None = None
    ) -> GoogleDriveFileContent:
        """Read the content of a file from Google Drive.

        Google Workspace files (Docs, Sheets, Slides) are automatically exported to
        LLM-readable formats:
        - Google Docs -> Markdown (text/markdown)
        - Google Sheets -> CSV (text/csv)
        - Google Slides -> Plain text (text/plain)
        - PDF files -> Extracted text (text/plain)

        Regular text files are downloaded directly.
        Binary files (images, videos, etc.) will raise an error.

        Args:
            file_id: The ID of the file to read.
            target_format: Optional MIME type to export Google Workspace files to.
                If not specified, uses sensible defaults. Has no effect on non-Workspace files.

        Returns
        -------
            GoogleDriveFileContent with the file content and metadata.

        Raises
        ------
            GoogleDriveError: If the file cannot be read (not found, permission denied,
                             binary file, etc.).
        """
        file_metadata = await self.get_file_metadata(file_id)
        original_mime_type = file_metadata.mime_type

        if self._is_binary_mime_type(original_mime_type):
            raise GoogleDriveError(
                f"Binary files are not supported for reading. "
                f"File '{file_metadata.name}' has MIME type '{original_mime_type}'."
            )

        if original_mime_type == GOOGLE_DRIVE_FOLDER_MIME:
            raise GoogleDriveError(
                f"Cannot read content of a folder. '{file_metadata.name}' is a folder, not a file."
            )

        was_exported = False
        if original_mime_type in GOOGLE_WORKSPACE_EXPORT_MIMES:
            export_mime = target_format or GOOGLE_WORKSPACE_EXPORT_MIMES[original_mime_type]
            content = await self._export_workspace_file(file_id, export_mime)
            result_mime_type = export_mime
            was_exported = True
        elif original_mime_type == PDF_MIME_TYPE:
            pdf_bytes = await self._download_file_bytes(file_id)
            content = self._extract_text_from_pdf(pdf_bytes)
            result_mime_type = "text/plain"
            was_exported = True
        else:
            content = await self._download_file(file_id)
            result_mime_type = original_mime_type

        return GoogleDriveFileContent(
            id=file_metadata.id,
            name=file_metadata.name,
            mime_type=result_mime_type,
            content=content,
            original_mime_type=original_mime_type,
            was_exported=was_exported,
            size=file_metadata.size,
            web_view_link=file_metadata.web_view_link,
        )

    async def create_file(
        self,
        name: str,
        mime_type: str,
        parent_id: str | None = None,
        initial_content: str | None = None,
    ) -> GoogleDriveFile:
        """Create a new file or folder in Google Drive.

        Creates a new file with the specified name and MIME type. Optionally places
        it in a specific folder and populates it with initial content.

        For Google Workspace files (Docs, Sheets), the Drive API automatically
        converts plain text content to the appropriate format.

        Args:
            name: The name for the new file or folder.
            mime_type: The MIME type of the file (e.g., 'text/plain',
                'application/vnd.google-apps.document',
                'application/vnd.google-apps.folder').
            parent_id: Optional ID of the parent folder. If not specified,
                the file is created in the root of the user's Drive.
            initial_content: Optional text content to populate the file.
                Ignored for folders.

        Returns
        -------
            GoogleDriveFile with the created file's metadata.

        Raises
        ------
            GoogleDriveError: If file creation fails (permission denied,
                parent not found, rate limited, etc.).
        """
        metadata: dict[str, Any] = {
            "name": name,
            "mimeType": mime_type,
        }
        if parent_id:
            metadata["parents"] = [parent_id]

        if mime_type == GOOGLE_DRIVE_FOLDER_MIME or not initial_content:
            response = await self._client.post(
                "/",
                json=metadata,
                params={"fields": SUPPORTED_FIELDS_STR, "supportsAllDrives": "true"},
            )
        else:
            response = await self._create_file_with_content(
                metadata=metadata,
                content=initial_content,
                target_mime_type=mime_type,
            )

        if response.status_code == 404:
            raise GoogleDriveError(
                f"Parent folder with ID '{parent_id}' not found."
                if parent_id
                else "Resource not found."
            )
        if response.status_code == 403:
            raise GoogleDriveError(
                "Permission denied: you don't have permission to create files in this location."
            )
        if response.status_code == 400:
            raise GoogleDriveError(
                f"Bad request: invalid parameters for file creation. "
                f"Check that the MIME type '{mime_type}' is valid."
            )
        if response.status_code == 429:
            raise GoogleDriveError("Rate limit exceeded. Please try again later.")

        response.raise_for_status()
        return GoogleDriveFile.from_api_response(response.json())

    async def _create_file_with_content(
        self,
        metadata: dict[str, Any],
        content: str,
        target_mime_type: str,
    ) -> httpx.Response:
        """Create a file with content using multipart upload.

        Args:
            metadata: File metadata dictionary.
            content: Text content for the file.
            target_mime_type: The target MIME type for the file.

        Returns
        -------
            The HTTP response from the upload.
        """
        content_type = UPLOAD_CONTENT_TYPES.get(target_mime_type, "text/plain")
        boundary = f"===gdrive_boundary_{uuid.uuid4().hex}==="
        body_parts = [
            f"--{boundary}",
            "Content-Type: application/json; charset=UTF-8",
            "",
            json.dumps(metadata),
            f"--{boundary}",
            f"Content-Type: {content_type}",
            "",
            content,
            f"--{boundary}--",
        ]
        body = "\r\n".join(body_parts)

        upload_url = "https://www.googleapis.com/upload/drive/v3/files"
        return await self._client.post(
            upload_url,
            content=body.encode("utf-8"),
            params={
                "uploadType": "multipart",
                "fields": SUPPORTED_FIELDS_STR,
                "supportsAllDrives": "true",
            },
            headers={"Content-Type": f"multipart/related; boundary={boundary}"},
        )

    async def manage_access(
        self,
        *,
        file_id: str,
        action: Literal["add", "update", "remove"],
        role: Literal["reader", "commenter", "writer", "fileOrganizer", "organizer", "owner"]
        | None = None,
        email_address: str | None = None,
        permission_id: str | None = None,
        transfer_ownership: bool = False,
    ) -> str:
        """Manage access permissions for a Google Drive file or folder.

        Adds, updates, or removes sharing permissions on an existing Google Drive
        file or folder using the Google Drive Permissions API.

        This method supports granting access to users or groups, changing access
        roles, and revoking permissions. Ownership transfer is supported for files
        in "My Drive" when explicitly requested.

        Args:
            file_id: The ID of the Google Drive file or folder whose permissions
                are being managed.
            action: The permission operation to perform.
            role: The access role to assign or update. Valid values include
                Required for "add" and "update" actions.
            email_address: The email address of the user or group to grant access to.
                Required for the "add" action.
            permission_id: The ID of the permission to update or remove.
                Required for "update" and "remove" actions.
            transfer_ownership: Whether to transfer ownership of the file.
                Only applicable when action="update" and role="owner".

        Returns
        -------
            Permission id.
            For "add" its newly added permission.
            For "update"/"remove" its previous permission.

        Raises
        ------
            GoogleDriveError: If the permission operation fails (invalid arguments,
                insufficient permissions, resource not found, ownership transfer
                not allowed, rate limited, etc.).
        """
        if not file_id.strip():
            raise GoogleDriveError("Argument validation error: 'file_id' cannot be empty.")

        if action == "add" and not email_address:
            raise GoogleDriveError("'email_address' is required for action 'add'.")

        if action in ("update", "remove") and not permission_id:
            raise GoogleDriveError("'permission_id' is required for action 'update' or 'remove'.")

        if action != "remove" and not role:
            raise GoogleDriveError("'role' is required for action 'add' or 'update'.")

        if action == "add":
            response = await self._client.post(
                url=f"/{file_id}/permissions",
                json={
                    "type": "user",
                    "role": role,
                    "emailAddress": email_address,
                },
                params={"sendNotificationEmail": False, "supportsAllDrives": True},
            )

        elif action == "update":
            response = await self._client.patch(
                url=f"/{file_id}/permissions/{permission_id}",
                json={"role": role},
                params={"transferOwnership": transfer_ownership, "supportsAllDrives": True},
            )

        elif action == "remove":
            response = await self._client.delete(url=f"/{file_id}/permissions/{permission_id}")

        else:
            raise GoogleDriveError(f"Invalid action '{action}'")

        if response.status_code not in (200, 201, 204):
            raise GoogleDriveError(f"Drive API error {response.status_code}: {response.text}")

        if action == "add":
            return response.json()["id"]

        # Cannot be null here because of above validators
        return permission_id  # type: ignore

    async def __aenter__(self) -> "GoogleDriveClient":
        """Async context manager entry."""
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any
    ) -> None:
        """Async context manager exit."""
        await self._client.aclose()
