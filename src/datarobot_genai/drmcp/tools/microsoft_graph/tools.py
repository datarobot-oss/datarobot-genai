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

    try:
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
    except MicrosoftGraphError as e:
        logger.error(f"Microsoft Graph error searching content: {e}")
        raise ToolError(str(e))
    except Exception as e:
        logger.error(f"Unexpected error searching Microsoft Graph content: {e}", exc_info=True)
        raise ToolError(
            f"An unexpected error occurred while searching Microsoft Graph content: {str(e)}"
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
