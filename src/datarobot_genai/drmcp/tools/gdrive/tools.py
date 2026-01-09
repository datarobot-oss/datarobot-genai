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

"""Google Drive MCP tools for interacting with Google Drive API."""

import logging
from typing import Annotated

from fastmcp.exceptions import ToolError
from fastmcp.tools.tool import ToolResult

from datarobot_genai.drmcp.core.mcp_instance import dr_mcp_tool
from datarobot_genai.drmcp.tools.clients.gdrive import LIMIT
from datarobot_genai.drmcp.tools.clients.gdrive import MAX_PAGE_SIZE
from datarobot_genai.drmcp.tools.clients.gdrive import SUPPORTED_FIELDS
from datarobot_genai.drmcp.tools.clients.gdrive import SUPPORTED_FIELDS_STR
from datarobot_genai.drmcp.tools.clients.gdrive import GoogleDriveClient
from datarobot_genai.drmcp.tools.clients.gdrive import GoogleDriveError
from datarobot_genai.drmcp.tools.clients.gdrive import get_gdrive_access_token

logger = logging.getLogger(__name__)


@dr_mcp_tool(tags={"google", "gdrive", "list", "search", "files", "find", "contents"})
async def gdrive_find_contents(
    *,
    page_size: Annotated[
        int, f"Maximum number of files to return per page (max {MAX_PAGE_SIZE})."
    ] = 10,
    limit: Annotated[int, f"Total maximum number of files to return (max {LIMIT})."] = 50,
    page_token: Annotated[
        str | None, "The token for the next page of results, retrieved from a previous call."
    ] = None,
    query: Annotated[
        str | None, "Optional filter to narrow results (e.g., 'trashed = false')."
    ] = None,
    folder_id: Annotated[
        str | None,
        "The ID of a specific folder to list or search within. "
        "If omitted, searches the entire Drive.",
    ] = None,
    recursive: Annotated[
        bool,
        "If True, searches all subfolders. "
        "If False and folder_id is provided, only lists immediate children.",
    ] = False,
    fields: Annotated[
        list[str] | None,
        "Optional list of metadata fields to include. Ex. id, name, mimeType. "
        f"Default = {SUPPORTED_FIELDS_STR}",
    ] = None,
) -> ToolResult | ToolError:
    """
    Search or list files in the user's Google Drive with pagination and filtering support.
    Use this tool to discover file names and IDs for use with other tools.

    Limit must be bigger than or equal to page size and it must be multiplication of page size.
    Ex.
        page size = 10 limit = 50
        page size = 3 limit = 3
        page size = 12 limit = 36
    """
    access_token = await get_gdrive_access_token()
    if isinstance(access_token, ToolError):
        raise access_token

    try:
        async with GoogleDriveClient(access_token) as client:
            data = await client.list_files(
                page_size=page_size,
                page_token=page_token,
                query=query,
                limit=limit,
                folder_id=folder_id,
                recursive=recursive,
            )
    except GoogleDriveError as e:
        logger.error(f"Google Drive error listing files: {e}")
        raise ToolError(str(e))
    except Exception as e:
        logger.error(f"Unexpected error listing Google Drive files: {e}")
        raise ToolError(f"An unexpected error occurred while listing Google Drive files: {str(e)}")

    filtered_fields = set(fields).intersection(SUPPORTED_FIELDS) if fields else SUPPORTED_FIELDS
    number_of_files = len(data.files)
    next_page_info = (
        f"Next page token needed to fetch more data: {data.next_page_token}"
        if data.next_page_token
        else "There're no more pages."
    )
    return ToolResult(
        content=f"Successfully listed {number_of_files} files. {next_page_info}",
        structured_content={
            "files": [
                file.model_dump(by_alias=True, include=filtered_fields) for file in data.files
            ],
            "count": number_of_files,
            "nextPageToken": data.next_page_token,
        },
    )
