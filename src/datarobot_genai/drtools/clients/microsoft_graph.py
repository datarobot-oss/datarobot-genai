# Copyright 2026 DataRobot, Inc.
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

"""Microsoft Graph API Client for searching SharePoint and OneDrive content."""

import logging
from typing import Any
from typing import Literal
from urllib.parse import quote

import httpx
from datarobot.auth.datarobot.exceptions import OAuthServiceClientErr
from fastmcp.exceptions import ToolError
from pydantic import BaseModel
from pydantic import Field

from datarobot_genai.drmcp.core.auth import get_access_token

logger = logging.getLogger(__name__)

GRAPH_API_BASE = "https://graph.microsoft.com/v1.0"
MAX_SEARCH_RESULTS = 250


async def get_microsoft_graph_access_token() -> str | ToolError:
    """
    Get Microsoft Graph OAuth access token with error handling.

    Returns
    -------
        Access token string on success, ToolError on failure

    Example:
        ```python
        token = await get_microsoft_graph_access_token()
        if isinstance(token, ToolError):
            # Handle error
            return token
        # Use token
        ```
    """
    try:
        access_token = await get_access_token("microsoft")
        if not access_token:
            logger.warning("Empty access token received")
            return ToolError("Received empty access token. Please complete the OAuth flow.")
        return access_token
    except OAuthServiceClientErr as e:
        logger.error(f"OAuth client error: {e}", exc_info=True)
        return ToolError(
            "Could not obtain access token for Microsoft. Make sure the OAuth "
            "permission was granted for the application to act on your behalf."
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Unexpected error obtaining access token: {error_msg}", exc_info=True)
        return ToolError("An unexpected error occurred while obtaining access token for Microsoft.")


class MicrosoftGraphError(Exception):
    """Exception for Microsoft Graph API errors."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class MicrosoftGraphItem(BaseModel):
    """Represents an item (file or folder) from Microsoft Graph (SharePoint/OneDrive)."""

    id: str
    name: str
    web_url: str | None = Field(None, alias="webUrl")
    size: int | None = None
    created_datetime: str | None = Field(None, alias="createdDateTime")
    last_modified_datetime: str | None = Field(None, alias="lastModifiedDateTime")
    is_folder: bool = False
    mime_type: str | None = Field(None, alias="mimeType")
    drive_id: str | None = Field(None, alias="driveId")
    parent_folder_id: str | None = Field(None, alias="parentFolderId")

    model_config = {"populate_by_name": True}

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> "MicrosoftGraphItem":
        """Create a MicrosoftGraphItem from Microsoft Graph API response data."""
        parent_ref = data.get("parentReference", {})
        return cls(
            id=data.get("id", ""),
            name=data.get("name", "Unknown"),
            web_url=data.get("webUrl"),
            size=data.get("size"),
            created_datetime=data.get("createdDateTime"),
            last_modified_datetime=data.get("lastModifiedDateTime"),
            is_folder="folder" in data,
            mime_type=data.get("file", {}).get("mimeType") if "file" in data else None,
            drive_id=parent_ref.get("driveId"),
            parent_folder_id=parent_ref.get("id"),
        )


class MicrosoftGraphClient:
    """Client for interacting with Microsoft Graph API to search SharePoint and OneDrive content."""

    def __init__(self, access_token: str, site_url: str | None = None):
        """
        Initialize Microsoft Graph client with access token.

        Args:
            access_token: OAuth access token for Microsoft Graph API
            site_url: Optional SharePoint site URL (e.g., https://tenant.sharepoint.com/sites/sitename)
                     If not provided, searches across all accessible sites and OneDrive
        """
        self.access_token = access_token
        self.site_url = site_url
        self._client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )
        self._site_id: str | None = None

    async def _get_site_id(self) -> str:
        """Get the SharePoint site ID from the site URL or return root site ID."""
        if self._site_id:
            return self._site_id

        # If no site_url provided, use root site
        if not self.site_url:
            # Get root site ID
            graph_url = f"{GRAPH_API_BASE}/sites/root"
            try:
                response = await self._client.get(graph_url)
                response.raise_for_status()
                data = response.json()
                self._site_id = data.get("id", "")
                return self._site_id
            except httpx.HTTPStatusError as e:
                raise self._handle_http_error(e, "Failed to get root site ID") from e

        # Extract site path from URL
        # Format: https://{tenant}.sharepoint.com/sites/{site-name}
        # or: https://{tenant}.sharepoint.com/sites/{site-name}/...
        url_parts = self.site_url.replace("https://", "").split("/")
        if len(url_parts) < 3:
            raise MicrosoftGraphError(f"Invalid SharePoint site URL: {self.site_url}")

        hostname = url_parts[0]  # tenant.sharepoint.com
        site_path = "/".join(url_parts[1:])  # sites/site-name/...

        # Use Microsoft Graph API to get site ID
        graph_url = f"{GRAPH_API_BASE}/sites/{hostname}:/{site_path}"
        try:
            response = await self._client.get(graph_url)
            response.raise_for_status()
            data = response.json()
            self._site_id = data.get("id", "")
            return self._site_id
        except httpx.HTTPStatusError as e:
            raise self._handle_http_error(
                e, f"Failed to get site ID from URL: {self.site_url}"
            ) from e

    def _handle_http_error(
        self, error: httpx.HTTPStatusError, base_message: str
    ) -> MicrosoftGraphError:
        """Handle HTTP errors and return appropriate MicrosoftGraphError with user-friendly messages."""  # noqa: E501
        error_msg = base_message

        if error.response.status_code == 403:
            error_msg += (
                ": Insufficient permissions. Requires Sites.Read.All or Sites.Search.All "
                "permission."
            )
        elif error.response.status_code == 400:
            try:
                error_data = error.response.json()
                api_message = error_data.get("error", {}).get("message", "Invalid request")
                error_msg += f": {api_message}"
            except Exception:
                error_msg += ": Invalid request parameters."
        else:
            error_msg += f": HTTP {error.response.status_code}"

        return MicrosoftGraphError(error_msg)

    async def search_content(
        self,
        search_query: str,
        site_id: str | None = None,
        from_offset: int = 0,
        size: int = 250,
        entity_types: list[str] | None = None,
        filters: list[str] | None = None,
        include_hidden_content: bool = False,
        region: str | None = None,
    ) -> list[MicrosoftGraphItem]:
        """
        Search for content using Microsoft Graph API search.

        This tool utilizes Microsoft Graph's search engine to locate items across
        SharePoint sites, OneDrive, and other Microsoft 365 services. When a site
        is specified, it searches within that site. Otherwise, it searches across
        all accessible SharePoint sites and OneDrive.

        Args:
            search_query: The search string to find files, folders, or list items
            site_id: Optional site ID to scope the search. If not provided and site_url
                    is set, will use that site. If neither is provided, searches across
                    all accessible sites.
            from_offset: The zero-based index of the first result to return (default: 0).
                        Use this for pagination - increment by the size value to get the next page.
            size: Maximum number of results to return in this request (default: 250, max: 250).
                  The LLM should control pagination by making multiple calls with different
                  'from' values (e.g., from=0 size=250, then from=250 size=250, etc.).
            entity_types: Optional list of entity types to search. Valid values:
                         "driveItem", "listItem", "site", "list", "drive".
                         Default: ["driveItem", "listItem"]
            filters: Optional list of filter expressions (KQL syntax) to refine search results
            include_hidden_content: Whether to include hidden content in search results.
                                  Only works with delegated permissions, not application
                                  permissions.
            region: Optional region code for application permissions (e.g., "NAM", "EUR", "APC")

        Returns
        -------
            List of MicrosoftGraphItem objects matching the search query

        Raises
        ------
            MicrosoftGraphError: If the search fails
            httpx.HTTPStatusError: If the API request fails
        """
        if not search_query:
            raise MicrosoftGraphError("Search query cannot be empty")

        # Validate and limit size parameter
        size = min(max(1, size), MAX_SEARCH_RESULTS)  # Between 1 and 250
        from_offset = max(0, from_offset)  # Must be non-negative

        # Determine which site to search
        # If site_id is provided, use it directly; otherwise resolve from site_url if set
        if site_id:
            target_site_id = site_id
        elif self.site_url:
            target_site_id = await self._get_site_id()
        else:
            target_site_id = None

        # Use unified Microsoft Search API for both site-specific and organization-wide search
        # Reference: https://learn.microsoft.com/en-us/graph/api/search-query
        graph_url = f"{GRAPH_API_BASE}/search/query"

        # Default entity types: driveItem and listItem
        if entity_types is None:
            entity_types = ["driveItem", "listItem"]

        # Validate entity types
        valid_entity_types = ["driveItem", "listItem", "site", "list", "drive"]
        entity_types = [et for et in entity_types if et in valid_entity_types]
        if not entity_types:
            entity_types = ["driveItem", "listItem"]  # Fallback to default

        # Build search request payload
        # Reference: https://learn.microsoft.com/en-us/graph/search-concept-files
        query_parts = []

        # If searching within a specific site, add scoping using KQL syntax first
        if target_site_id:
            # Get site details to construct proper scoping query
            try:
                site_info_url = f"{GRAPH_API_BASE}/sites/{target_site_id}"
                site_response = await self._client.get(site_info_url)
                site_response.raise_for_status()
                site_data = site_response.json()
                site_web_url = site_data.get("webUrl", "")

                # Use KQL to scope search to the specific site
                # Format: path:"{site-url}"
                if site_web_url:
                    query_parts.append(f'path:"{site_web_url}"')
            except httpx.HTTPStatusError as e:
                raise self._handle_http_error(e, "Failed to get site details for scoping") from e
            except Exception as e:
                logger.warning(
                    f"Could not get site details for scoping, using un-scoped search: {e}"
                )
                # Fall back to un-scoped search if site details can't be retrieved

        # Add the main search query
        query_parts.append(search_query)

        # Add filters if provided (using AND operator for proper KQL syntax)
        if filters:
            # Join filters with AND operator for proper KQL syntax
            filter_string = " AND ".join(filters)
            query_parts.append(filter_string)

        # Combine all query parts with spaces
        query_string = " ".join(query_parts)

        # Build request payload with from and size parameters
        request_payload = {
            "entityTypes": entity_types,
            "query": {
                "queryString": query_string,
            },
            "from": from_offset,
            "size": size,
        }

        # Add includeHiddenContent (only works with delegated permissions)
        if include_hidden_content:
            request_payload["includeHiddenContent"] = True

        # Add region for application permissions
        if region:
            request_payload["region"] = region

        payload = {"requests": [request_payload]}

        try:
            response = await self._client.post(graph_url, json=payload)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as e:
            raise self._handle_http_error(e, "Failed to search SharePoint content") from e

        # Parse the Microsoft Search API response format
        # Reference: https://learn.microsoft.com/en-us/graph/search-concept-files
        results = []
        for request_result in data.get("value", []):
            hits_containers = request_result.get("hitsContainers", [])
            for container in hits_containers:
                hits = container.get("hits", [])
                for hit in hits:
                    resource = hit.get("resource", {})
                    if not resource:
                        continue

                    odata_type = resource.get("@odata.type", "")
                    transformed_resource = self._transform_search_resource(resource, odata_type)
                    # transformed_resource always returns a dict, so we can process it directly
                    results.append(MicrosoftGraphItem.from_api_response(transformed_resource))

        return results

    def _transform_search_resource(
        self, resource: dict[str, Any], odata_type: str
    ) -> dict[str, Any]:
        """Transform a search API resource to MicrosoftGraphItem-compatible format."""
        # Preserve original values from resource if they exist, otherwise use defaults
        # This ensures we don't lose data that might be present in the original response
        base_resource = {
            "id": resource.get("id", ""),
            "webUrl": resource.get("webUrl"),
            "createdDateTime": resource.get("createdDateTime"),
            "lastModifiedDateTime": resource.get("lastModifiedDateTime"),
            "size": resource.get("size"),
            "folder": resource.get("folder", {}),
            "file": resource.get("file", {}),
        }

        parent_ref = resource.get("parentReference", {})

        if odata_type == "#microsoft.graph.listItem":
            fields = resource.get("fields", {})
            base_resource.update(
                {
                    "name": fields.get("Title") or resource.get("name", "Unknown"),
                    "parentReference": {
                        "driveId": parent_ref.get("driveId"),
                        "id": parent_ref.get("id"),
                    },
                }
            )
        elif odata_type == "#microsoft.graph.site":
            base_resource.update(
                {
                    "name": resource.get("displayName") or resource.get("name", "Unknown"),
                    "parentReference": {},
                }
            )
        elif odata_type == "#microsoft.graph.list":
            base_resource.update(
                {
                    "name": resource.get("displayName") or resource.get("name", "Unknown"),
                    "parentReference": {
                        "siteId": parent_ref.get("siteId"),
                    },
                }
            )
        elif odata_type == "#microsoft.graph.drive":
            base_resource.update(
                {
                    "name": resource.get("name", "Unknown"),
                    "parentReference": {
                        "siteId": parent_ref.get("siteId"),
                    },
                }
            )
        else:
            # Standard driveItem - use resource as-is
            return resource

        return base_resource

    async def get_personal_drive_id(self) -> str:
        """Get the current user's personal OneDrive drive ID.

        Returns
        -------
            The drive ID string for the user's personal OneDrive.

        Raises
        ------
            MicrosoftGraphError: If the drive cannot be retrieved.
        """
        try:
            response = await self._client.get(f"{GRAPH_API_BASE}/me/drive")
            response.raise_for_status()
            data = response.json()
            return data["id"]
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            if status_code == 401:
                raise MicrosoftGraphError(
                    "Authentication failed. Access token may be expired or invalid."
                ) from e
            if status_code == 403:
                raise MicrosoftGraphError(
                    "Permission denied: cannot access personal OneDrive. "
                    "Requires Files.Read or Files.ReadWrite permission."
                ) from e
            raise MicrosoftGraphError(f"Failed to get personal OneDrive: HTTP {status_code}") from e

    async def create_file(
        self,
        drive_id: str,
        file_name: str,
        content: str,
        parent_folder_id: str = "root",
        conflict_behavior: str = "rename",
    ) -> MicrosoftGraphItem:
        """Create a text file in a drive (SharePoint document library or OneDrive).

        Uses Microsoft Graph's simple upload endpoint for files < 4MB.
        Files are created as text/plain content.

        Args:
            drive_id: The ID of the drive (document library) where the file will be created.
            file_name: The name of the file to create (e.g., 'report.txt').
            content: The text content to store in the file.
            parent_folder_id: ID of the parent folder. Defaults to "root" (drive root folder).
            conflict_behavior: How to handle name conflicts. Options:
                - "rename" (default): Auto-renames to 'filename (1).txt', etc.
                - "fail": Returns 409 Conflict error
                - "replace": Overwrites existing file

        Returns
        -------
            MicrosoftGraphItem representing the created file.

        Raises
        ------
            MicrosoftGraphError: If file creation fails.
        """
        if not drive_id or not drive_id.strip():
            raise MicrosoftGraphError("drive_id cannot be empty")
        if not file_name or not file_name.strip():
            raise MicrosoftGraphError("file_name cannot be empty")

        # URL encode the filename for path-based addressing
        encoded_name = quote(file_name, safe="")

        # Simple upload endpoint for files < 4MB
        # Reference: https://learn.microsoft.com/en-us/graph/api/driveitem-put-content
        upload_url = (
            f"{GRAPH_API_BASE}/drives/{drive_id}/items/{parent_folder_id}:/{encoded_name}:/content"
        )

        try:
            response = await self._client.put(
                upload_url,
                content=content.encode("utf-8"),
                headers={"Content-Type": "text/plain"},
                params={"@microsoft.graph.conflictBehavior": conflict_behavior},
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise self._handle_create_file_error(e, drive_id, file_name, parent_folder_id) from e

        return MicrosoftGraphItem.from_api_response(response.json())

    def _handle_create_file_error(
        self,
        error: httpx.HTTPStatusError,
        drive_id: str,
        file_name: str,
        parent_folder_id: str,
    ) -> MicrosoftGraphError:
        """Handle HTTP errors for file creation and return appropriate MicrosoftGraphError."""
        status_code = error.response.status_code
        error_msg = f"Failed to create file: HTTP {status_code}"

        if status_code == 400:
            try:
                error_data = error.response.json()
                api_message = error_data.get("error", {}).get("message", "Invalid request")
                error_msg = f"Bad request creating file: {api_message}"
            except Exception:
                error_msg = "Bad request: invalid parameters for file creation."
        elif status_code == 401:
            error_msg = "Authentication failed. Access token may be expired or invalid."
        elif status_code == 403:
            error_msg = (
                f"Permission denied: you don't have permission to create files in drive "
                f"'{drive_id}'. Requires Files.ReadWrite.All permission."
            )
        elif status_code == 404:
            error_msg = (
                f"Parent folder '{parent_folder_id}' not found in drive '{drive_id}'."
                if parent_folder_id != "root"
                else f"Drive '{drive_id}' not found."
            )
        elif status_code == 409:
            error_msg = f"File '{file_name}' already exists and conflict behavior is set to 'fail'."
        elif status_code == 429:
            error_msg = "Rate limit exceeded. Please try again later."

        return MicrosoftGraphError(error_msg)

    async def share_item(
        self,
        file_id: str,
        document_library_id: str,
        recipient_emails: list[str],
        role: Literal["read", "write"],
        send_invitation: bool,
    ) -> None:
        """
        Share sharepoint / ondrive item using Microsoft Graph API.
        Under the hood all resources in sharepoint/onedrive
            in MS Graph API are treated as 'driveItem'.

        Args:
            file_id: The ID of the file or folder to share.
            document_library_id: The ID of the document library containing the item.
            recipient_emails: A list of email addresses to invite.
            role: The role to assign.
            send_invitation: Flag determining if recipients should be notified

        Returns
        -------
            None

        Raises
        ------
            MicrosoftGraphError: If sharing fails
        """
        graph_url = f"{GRAPH_API_BASE}/drives/{document_library_id}/items/{file_id}/invite"

        payload = {
            "recipients": [{"email": email} for email in recipient_emails],
            "requireSignIn": True,
            "sendInvitation": send_invitation,
            "roles": [role],
        }

        response = await self._client.post(url=graph_url, json=payload)

        if response.status_code not in (200, 201):
            raise MicrosoftGraphError(
                f"Microsoft Graph API error {response.status_code}: {response.text}"
            )

    async def update_item_metadata(
        self,
        item_id: str,
        fields_to_update: dict[str, Any],
        site_id: str | None = None,
        list_id: str | None = None,
        drive_id: str | None = None,
    ) -> dict[str, Any]:
        """Update metadata on a SharePoint list item or OneDrive/SharePoint drive item.

        For SharePoint list items: Updates custom column values via the fields endpoint.
        For drive items: Updates properties like name and description.

        Args:
            item_id: The ID of the item to update.
            fields_to_update: Key-value pairs of metadata fields to modify.
            site_id: For SharePoint list items - the site ID.
            list_id: For SharePoint list items - the list ID.
            drive_id: For OneDrive/drive items - the drive ID.

        Returns
        -------
            The API response containing updated item data.

        Raises
        ------
            MicrosoftGraphError: If validation fails or the API request fails.
        """
        if not item_id or not item_id.strip():
            raise MicrosoftGraphError("item_id cannot be empty")
        if not fields_to_update:
            raise MicrosoftGraphError("fields_to_update cannot be empty")

        # Determine the endpoint based on provided parameters
        has_sharepoint_context = site_id is not None and list_id is not None
        has_drive_context = drive_id is not None

        if has_sharepoint_context and has_drive_context:
            raise MicrosoftGraphError(
                "Cannot specify both SharePoint (site_id + list_id) and OneDrive "
                "(document_library_id) context. Choose one."
            )

        if not has_sharepoint_context and not has_drive_context:
            raise MicrosoftGraphError(
                "Must specify either SharePoint context (site_id + list_id) or "
                "OneDrive context (document_library_id)."
            )

        try:
            if has_sharepoint_context:
                # PATCH /sites/{site-id}/lists/{list-id}/items/{item-id}/fields
                url = f"{GRAPH_API_BASE}/sites/{site_id}/lists/{list_id}/items/{item_id}/fields"
                response = await self._client.patch(url, json=fields_to_update)
            else:
                # Drive item: PATCH /drives/{drive-id}/items/{item-id}
                url = f"{GRAPH_API_BASE}/drives/{drive_id}/items/{item_id}"
                response = await self._client.patch(url, json=fields_to_update)

            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            raise self._handle_update_metadata_error(e, item_id) from e

    def _handle_update_metadata_error(
        self,
        error: httpx.HTTPStatusError,
        item_id: str,
    ) -> MicrosoftGraphError:
        """Handle HTTP errors for metadata updates and return appropriate MicrosoftGraphError."""
        status_code = error.response.status_code
        error_msg = f"Failed to update metadata: HTTP {status_code}"

        if status_code == 400:
            try:
                error_data = error.response.json()
                api_message = error_data.get("error", {}).get("message", "Invalid request")
                error_msg = f"Bad request updating metadata: {api_message}"
            except Exception:
                error_msg = "Bad request: invalid field names or values."
        elif status_code == 401:
            error_msg = "Authentication failed. Access token may be expired or invalid."
        elif status_code == 403:
            error_msg = (
                f"Permission denied: you don't have permission to update item '{item_id}'. "
                "Requires Sites.ReadWrite.All or Files.ReadWrite.All permission."
            )
        elif status_code == 404:
            error_msg = f"Item '{item_id}' not found."
        elif status_code == 409:
            error_msg = f"Conflict: item '{item_id}' may have been modified concurrently."
        elif status_code == 429:
            error_msg = "Rate limit exceeded. Please try again later."

        return MicrosoftGraphError(error_msg)

    async def __aenter__(self) -> "MicrosoftGraphClient":
        """Async context manager entry."""
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any
    ) -> None:
        """Async context manager exit."""
        await self._client.aclose()


def validate_site_url(site_url: str) -> str | None:
    """Validate SharePoint site URL and return user-friendly error message if invalid.

    Args:
        site_url: The SharePoint site URL to validate

    Returns
    -------
        None if valid, or a user-friendly error message if invalid
    """
    if not site_url:
        return (
            "SharePoint site URL is required. "
            "Please provide a valid SharePoint site URL (e.g., https://yourtenant.sharepoint.com/sites/yoursite)."
        )

    site_url = site_url.strip()

    if not site_url.startswith("https://"):
        return (
            f"Invalid SharePoint site URL: '{site_url}'. "
            "The URL must start with 'https://'. "
            "Example: https://yourtenant.sharepoint.com/sites/yoursite"
        )

    if "sharepoint.com" not in site_url.lower():
        return (
            f"Invalid SharePoint site URL: '{site_url}'. "
            "The URL must be a SharePoint site URL containing 'sharepoint.com'. "
            "Example: https://yourtenant.sharepoint.com/sites/yoursite"
        )

    # Check basic URL structure
    url_parts = site_url.replace("https://", "").split("/")
    if len(url_parts) < 1 or not url_parts[0]:
        return (
            f"Invalid SharePoint site URL format: '{site_url}'. "
            "The URL must include a domain name. "
            "Example: https://yourtenant.sharepoint.com/sites/yoursite"
        )

    # Check if it looks like a valid SharePoint site URL
    domain = url_parts[0]
    if not domain.endswith("sharepoint.com"):
        return (
            f"Invalid SharePoint site URL: '{site_url}'. "
            "The domain must end with 'sharepoint.com'. "
            "Example: https://yourtenant.sharepoint.com/sites/yoursite"
        )

    return None
