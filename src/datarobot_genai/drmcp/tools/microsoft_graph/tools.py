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

"""Microsoft Graph MCP tools for searching SharePoint and OneDrive content."""

import logging
from typing import Annotated
from typing import Any
from typing import Literal

from fastmcp.exceptions import ToolError
from fastmcp.tools.tool import ToolResult

from datarobot_genai.drmcp.core.mcp_instance import dr_mcp_tool
from datarobot_genai.drmcp.tools.clients.microsoft_graph import MicrosoftGraphClient
from datarobot_genai.drmcp.tools.clients.microsoft_graph import MicrosoftGraphError
from datarobot_genai.drmcp.tools.clients.microsoft_graph import get_microsoft_graph_access_token
from datarobot_genai.drmcp.tools.clients.microsoft_graph import validate_site_url

logger = logging.getLogger(__name__)


@dr_mcp_tool(
    tags={
        "microsoft",
        "graph api",
        "sharepoint",
        "drive",
        "list",
        "search",
        "files",
        "find",
        "contents",
    }
)
async def microsoft_graph_search_content(
    *,
    search_query: Annotated[str, "The search string to find files, folders, or list items."],
    site_url: Annotated[
        str | None,
        "Optional SharePoint site URL to scope the search "
        "(e.g., https://tenant.sharepoint.com/sites/sitename). "
        "If not provided, searches across all accessible sites.",
    ] = None,
    site_id: Annotated[
        str | None,
        "Optional ID of the site to scope the search. If provided, takes precedence over site_url.",
    ] = None,
    from_offset: Annotated[
        int,
        "The zero-based index of the first result to return. Use this for pagination. "
        "Default: 0 (start from the beginning). To get the next page, increment by the size "
        "value (e.g., first page: from=0 size=250, second page: from=250 size=250, "
        "third page: from=500 size=250).",
    ] = 0,
    size: Annotated[
        int,
        "Maximum number of results to return in this request. Default is 250, max is 250. "
        "The LLM should control pagination by making multiple calls with different 'from' values.",
    ] = 250,
    entity_types: Annotated[
        list[str] | None,
        "Optional list of entity types to search. Valid values: 'driveItem', 'listItem', "
        "'site', 'list', 'drive'. Default: ['driveItem', 'listItem']. "
        "Multiple types can be specified.",
    ] = None,
    filters: Annotated[
        list[str] | None,
        "Optional list of KQL filter expressions to refine search results "
        "(e.g., ['fileType:docx', 'size>1000']).",
    ] = None,
    include_hidden_content: Annotated[
        bool,
        "Whether to include hidden content in search results. Only works with delegated "
        "permissions, not application permissions. Default: False.",
    ] = False,
    region: Annotated[
        str | None,
        "Optional region code for application permissions (e.g., 'NAM', 'EUR', 'APC'). "
        "Required when using application permissions to search SharePoint content in "
        "specific regions.",
    ] = None,
) -> ToolResult | ToolError:
    """
    Search for SharePoint and OneDrive content using Microsoft Graph Search API.

    Search Scope:
    - When site_url or site_id is provided: searches within the specified SharePoint site
    - When neither is provided: searches across all accessible SharePoint sites and OneDrive

    Supported Entity Types:
    - driveItem: Files and folders in document libraries and OneDrive
    - listItem: Items in SharePoint lists
    - site: SharePoint sites
    - list: SharePoint lists
    - drive: Document libraries/drives

    Filtering:
    - Filters use KQL (Keyword Query Language) syntax
    - Multiple filters are combined with AND operators
    - Examples: ['fileType:docx', 'size>1000', 'lastModifiedTime>2024-01-01']
    - Filters are applied in addition to the search query

    Pagination:
    - Controlled via from_offset (zero-based index) and size parameters
    - Maximum size per request: 250 results
    - To paginate: increment from_offset by size value for each subsequent page
    - Example pagination sequence:
      * Page 1: from_offset=0, size=250 (returns results 0-249)
      * Page 2: from_offset=250, size=250 (returns results 250-499)
      * Page 3: from_offset=500, size=250 (returns results 500-749)

    API Reference:
    - Endpoint: POST /search/query
    - Documentation: https://learn.microsoft.com/en-us/graph/api/search-query
    - Search concepts: https://learn.microsoft.com/en-us/graph/search-concept-files

    Permissions:
    - Requires Sites.Read.All or Sites.Search.All permission
    - include_hidden_content only works with delegated permissions
    - region parameter is required for application permissions in multi-region environments
    """
    if not search_query:
        raise ToolError("Argument validation error: 'search_query' cannot be empty.")

    # Validate site_url if provided
    if site_url:
        validation_error = validate_site_url(site_url)
        if validation_error:
            raise ToolError(validation_error)

    access_token = await get_microsoft_graph_access_token()
    if isinstance(access_token, ToolError):
        raise access_token

    async with MicrosoftGraphClient(access_token=access_token, site_url=site_url) as client:
        items = await client.search_content(
            search_query=search_query,
            site_id=site_id,
            from_offset=from_offset,
            size=size,
            entity_types=entity_types,
            filters=filters,
            include_hidden_content=include_hidden_content,
            region=region,
        )

    results = []
    for item in items:
        result_dict = {
            "id": item.id,  # Unique ID of the file, folder, or list item
            "name": item.name,
            "webUrl": item.web_url,
            "size": item.size,
            "createdDateTime": item.created_datetime,
            "lastModifiedDateTime": item.last_modified_datetime,
            "isFolder": item.is_folder,
            "mimeType": item.mime_type,
            # Document library/drive ID (driveId in Microsoft Graph API)
            "documentLibraryId": item.drive_id,
            "parentFolderId": item.parent_folder_id,  # Parent folder ID
        }
        results.append(result_dict)

    n = len(results)
    return ToolResult(
        content=(
            f"Successfully searched Microsoft Graph and retrieved {n} result(s) for "
            f"'{search_query}' (from={from_offset}, size={size})."
        ),
        structured_content={
            "query": search_query,
            "siteUrl": site_url,
            "siteId": site_id,
            "from": from_offset,
            "size": size,
            "results": results,
            "count": n,
        },
    )


@dr_mcp_tool(tags={"microsoft", "graph api", "sharepoint", "onedrive", "share"}, enabled=False)
async def microsoft_graph_share_item(
    *,
    file_id: Annotated[str, "The ID of the file or folder to share."],
    document_library_id: Annotated[str, "The ID of the document library containing the item."],
    recipient_emails: Annotated[list[str], "A list of email addresses to invite."],
    role: Annotated[Literal["read", "write"], "The role to assign: 'read' or 'write'."] = "read",
    send_invitation: Annotated[
        bool, "Flag determining if recipients should be notified. Default False"
    ] = False,
) -> ToolResult | ToolError:
    """
    Share a SharePoint or Onedrive file or folder with one or more users.
    It works with internal users or existing guest users in the
    tenant. It does NOT create new guest accounts and does NOT use the tenant-level
    /invitations endpoint.

    Microsoft Graph API is treating OneDrive and SharePoint resources as driveItem.

    API Reference:
    - DriveItem Resource Type: https://learn.microsoft.com/en-us/graph/api/resources/driveitem
    - API Documentation: https://learn.microsoft.com/en-us/graph/api/driveitem-invite
    """
    if not file_id or not file_id.strip():
        raise ToolError("Argument validation error: 'file_id' cannot be empty.")

    if not document_library_id or not document_library_id.strip():
        raise ToolError("Argument validation error: 'document_library_id' cannot be empty.")

    if not recipient_emails:
        raise ToolError("Argument validation error: you must provide at least one 'recipient'.")

    access_token = await get_microsoft_graph_access_token()
    if isinstance(access_token, ToolError):
        raise access_token

    async with MicrosoftGraphClient(access_token=access_token) as client:
        await client.share_item(
            file_id=file_id,
            document_library_id=document_library_id,
            recipient_emails=recipient_emails,
            role=role,
            send_invitation=send_invitation,
        )

    n = len(recipient_emails)
    return ToolResult(
        content=(
            f"Successfully shared file {file_id} "
            f"from document library {document_library_id} "
            f"with {n} recipients with '{role}' role."
        ),
        structured_content={
            "fileId": file_id,
            "documentLibraryId": document_library_id,
            "recipientEmails": recipient_emails,
            "n": n,
            "role": role,
        },
    )


@dr_mcp_tool(
    tags={
        "microsoft",
        "graph api",
        "sharepoint",
        "onedrive",
        "document library",
        "create",
        "file",
        "write",
    },
    enabled=False,
)
async def microsoft_create_file(
    *,
    file_name: Annotated[str, "The name of the file to create (e.g., 'report.txt')."],
    content_text: Annotated[str, "The raw text content to be stored in the file."],
    document_library_id: Annotated[
        str | None,
        "The ID of the document library (Drive). If not provided, saves to personal OneDrive.",
    ] = None,
    parent_folder_id: Annotated[
        str | None,
        "ID of the parent folder. Defaults to 'root' (root of the drive).",
    ] = "root",
) -> ToolResult | ToolError:
    """
    Create a new text file in SharePoint or OneDrive.

    **Personal OneDrive:** Just provide file_name and content_text.
    The file saves to your personal OneDrive root folder.

    **SharePoint:** Provide document_library_id to save to a specific
    SharePoint site. Get the ID from microsoft_graph_search_content
    results ('documentLibraryId' field).

    **Conflict Resolution:** If a file with the same name exists,
    it will be automatically renamed (e.g., 'report (1).txt').
    """
    if not file_name or not file_name.strip():
        raise ToolError("Error: file_name is required.")
    if not content_text:
        raise ToolError("Error: content_text is required.")

    access_token = await get_microsoft_graph_access_token()
    if isinstance(access_token, ToolError):
        raise access_token

    folder_id = parent_folder_id if parent_folder_id else "root"

    async with MicrosoftGraphClient(access_token=access_token) as client:
        # Auto-fetch personal OneDrive if no library specified
        if document_library_id is None:
            drive_id = await client.get_personal_drive_id()
            is_personal_onedrive = True
        else:
            drive_id = document_library_id
            is_personal_onedrive = False

        created_file = await client.create_file(
            drive_id=drive_id,
            file_name=file_name.strip(),
            content=content_text,
            parent_folder_id=folder_id,
            conflict_behavior="rename",
        )

    return ToolResult(
        content=f"File '{created_file.name}' created successfully.",
        structured_content={
            "file_name": created_file.name,
            "destination": "onedrive" if is_personal_onedrive else "sharepoint",
            "driveId": drive_id,
            "id": created_file.id,
            "webUrl": created_file.web_url,
            "parentFolderId": created_file.parent_folder_id,
        },
    )


@dr_mcp_tool(
    tags={
        "microsoft",
        "graph api",
        "sharepoint",
        "onedrive",
        "metadata",
        "update",
        "fields",
        "compliance",
    },
    enabled=False,
)
async def microsoft_update_metadata(
    *,
    item_id: Annotated[str, "The ID of the file or list item to update."],
    fields_to_update: Annotated[
        dict[str, Any],
        "Key-value pairs of metadata fields to modify. "
        "For SharePoint list items: any custom column values. "
        "For drive items: 'name' and/or 'description'.",
    ],
    site_id: Annotated[
        str | None,
        "The site ID (required for SharePoint list items, along with list_id).",
    ] = None,
    list_id: Annotated[
        str | None,
        "The list ID (required for SharePoint list items, along with site_id).",
    ] = None,
    document_library_id: Annotated[
        str | None,
        "The drive ID (required for OneDrive/drive item updates). "
        "Cannot be used together with site_id and list_id.",
    ] = None,
) -> ToolResult | ToolError:
    """
    Update metadata on a SharePoint list item or OneDrive/SharePoint drive item.

    **SharePoint List Items:** Provide site_id and list_id to update custom
    column values on a list item. All custom columns can be updated.

    **OneDrive/Drive Items:** Provide document_library_id to update drive item
    properties. Only 'name' and 'description' fields can be updated.

    **Context Requirements:**
    - For SharePoint list items: Both site_id AND list_id are required
    - For OneDrive/drive items: document_library_id is required
    - Cannot specify both contexts simultaneously

    **Examples:**
    - SharePoint list item: Update a 'Status' column to 'Approved'
    - Drive item: Rename a file or update its description

    **Permissions:**
    - Requires Sites.ReadWrite.All or Files.ReadWrite.All permission
    """
    if not item_id or not item_id.strip():
        raise ToolError("Error: item_id is required.")
    if not fields_to_update:
        raise ToolError("Error: fields_to_update is required and cannot be empty.")

    # Validate context parameters
    has_sharepoint_context = site_id is not None and list_id is not None
    has_partial_sharepoint_context = (site_id is not None) != (list_id is not None)
    has_drive_context = document_library_id is not None

    if has_partial_sharepoint_context:
        raise ToolError(
            "Error: For SharePoint list items, both site_id and list_id must be provided."
        )

    if has_sharepoint_context and has_drive_context:
        raise ToolError(
            "Error: Cannot specify both SharePoint (site_id + list_id) and OneDrive "
            "(document_library_id) context. Choose one."
        )

    if not has_sharepoint_context and not has_drive_context:
        raise ToolError(
            "Error: Must specify either SharePoint context (site_id + list_id) or "
            "OneDrive context (document_library_id)."
        )

    access_token = await get_microsoft_graph_access_token()
    if isinstance(access_token, ToolError):
        raise access_token

    try:
        async with MicrosoftGraphClient(access_token=access_token) as client:
            result = await client.update_item_metadata(
                item_id=item_id.strip(),
                fields_to_update=fields_to_update,
                site_id=site_id,
                list_id=list_id,
                drive_id=document_library_id,
            )
    except MicrosoftGraphError as e:
        logger.error(f"Microsoft Graph error updating metadata: {e}")
        raise ToolError(str(e))
    except Exception as e:
        logger.error(f"Unexpected error updating metadata: {e}", exc_info=True)
        raise ToolError(f"An unexpected error occurred while updating metadata: {str(e)}")

    context_type = "sharepoint_list_item" if has_sharepoint_context else "drive_item"

    # Build structured content similar to microsoft_create_file pattern
    structured: dict[str, Any] = {
        "item_id": item_id,
        "context_type": context_type,
        "fields_updated": list(fields_to_update.keys()),
    }

    # Add context-specific IDs for traceability
    if has_sharepoint_context:
        structured["site_id"] = site_id
        structured["list_id"] = list_id
    else:
        structured["document_library_id"] = document_library_id

    # Include relevant response data
    if isinstance(result, dict):
        # For drive items, include key properties if present
        if has_drive_context:
            if "id" in result:
                structured["id"] = result["id"]
            if "name" in result:
                structured["name"] = result["name"]
            if "webUrl" in result:
                structured["webUrl"] = result["webUrl"]
            if "description" in result:
                structured["description"] = result.get("description")
        # For list items, the response is the fields object itself
        else:
            structured["updated_fields"] = result

    return ToolResult(
        content=f"Metadata updated successfully for item '{item_id}'.",
        structured_content=structured,
    )
