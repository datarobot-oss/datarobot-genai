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

"""Confluence MCP tools for interacting with Confluence Cloud."""

import logging
from typing import Annotated

from fastmcp.exceptions import ToolError
from fastmcp.tools.tool import ToolResult

from datarobot_genai.drmcp.core.mcp_instance import dr_mcp_tool
from datarobot_genai.drmcp.tools.clients.atlassian import get_atlassian_access_token
from datarobot_genai.drmcp.tools.clients.confluence import ConfluenceClient

logger = logging.getLogger(__name__)


@dr_mcp_tool(tags={"confluence", "read", "get", "page"})
async def confluence_get_page(
    *,
    page_id_or_title: Annotated[str, "The ID or the exact title of the Confluence page."],
    space_key: Annotated[
        str | None,
        "Required if identifying the page by title. The space key (e.g., 'PROJ').",
    ] = None,
) -> ToolResult | ToolError:
    """Retrieve the content of a specific Confluence page.

    Use this tool to fetch Confluence pages by their numeric ID or by title.
    Returns page content in HTML storage format.

    Usage:
        - By ID: page_id_or_title="856391684"
        - By title: page_id_or_title="Meeting Notes", space_key="TEAM"

    When using a page title, the space_key parameter is required.
    """
    if not page_id_or_title:
        raise ToolError("Argument validation error: 'page_id_or_title' cannot be empty.")

    access_token = await get_atlassian_access_token()
    if isinstance(access_token, ToolError):
        raise access_token

    try:
        async with ConfluenceClient(access_token) as client:
            if page_id_or_title.isdigit():
                page_response = await client.get_page_by_id(page_id_or_title)
            else:
                if not space_key:
                    raise ToolError(
                        "Argument validation error: "
                        "'space_key' is required when identifying a page by title."
                    )
                page_response = await client.get_page_by_title(page_id_or_title, space_key)
    except ValueError as e:
        logger.error(f"Value error getting Confluence page: {e}")
        raise ToolError(str(e))
    except Exception as e:
        logger.error(f"Unexpected error getting Confluence page: {e}")
        raise ToolError(
            f"An unexpected error occurred while getting Confluence page "
            f"'{page_id_or_title}': {str(e)}"
        )

    return ToolResult(
        content=f"Successfully retrieved page '{page_response.title}'.",
        structured_content=page_response.as_flat_dict(),
    )
