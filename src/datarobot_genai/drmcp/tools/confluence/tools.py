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
from datarobot_genai.drmcp.tools.clients.confluence import ConfluenceError

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

    return ToolResult(
        content=f"Successfully retrieved page '{page_response.title}'.",
        structured_content=page_response.as_flat_dict(),
    )


@dr_mcp_tool(tags={"confluence", "write", "create", "page"})
async def confluence_create_page(
    *,
    space_key: Annotated[str, "The key of the Confluence space where the new page should live."],
    title: Annotated[str, "The title of the new page."],
    body_content: Annotated[
        str,
        "The content of the page, typically in Confluence Storage Format (XML) or raw text.",
    ],
    parent_id: Annotated[
        int | None,
        "The ID of the parent page, used to create a child page.",
    ] = None,
) -> ToolResult:
    """Create a new documentation page in a specified Confluence space.

    Use this tool to create new Confluence pages with content in storage format.
    The page will be created at the root level of the space unless a parent_id
    is provided, in which case it will be created as a child page.

    Usage:
        - Root page: space_key="PROJ", title="New Page", body_content="<p>Content</p>"
        - Child page: space_key="PROJ", title="Sub Page", body_content="<p>Content</p>",
                      parent_id=123456
    """
    if not all([space_key, title, body_content]):
        raise ToolError(
            "Argument validation error: space_key, title, and body_content are required fields."
        )

    access_token = await get_atlassian_access_token()
    if isinstance(access_token, ToolError):
        raise access_token

    async with ConfluenceClient(access_token) as client:
        page_response = await client.create_page(
            space_key=space_key,
            title=title,
            body_content=body_content,
            parent_id=parent_id,
        )

    return ToolResult(
        content=f"New page '{title}' created successfully in space '{space_key}'.",
        structured_content={"new_page_id": page_response.page_id, "title": page_response.title},
    )


@dr_mcp_tool(tags={"confluence", "write", "add", "comment"})
async def confluence_add_comment(
    *,
    page_id: Annotated[str, "The numeric ID of the page where the comment will be added."],
    comment_body: Annotated[str, "The text content of the comment."],
) -> ToolResult:
    """Add a new comment to a specified Confluence page for collaboration.

    Use this tool to add comments to Confluence pages to facilitate collaboration
    and discussion. Comments are added at the page level.

    Usage:
        - Add comment: page_id="856391684", comment_body="Great work on this documentation!"
    """
    if not page_id:
        raise ToolError("Argument validation error: 'page_id' cannot be empty.")

    if not comment_body:
        raise ToolError("Argument validation error: 'comment_body' cannot be empty.")

    access_token = await get_atlassian_access_token()
    if isinstance(access_token, ToolError):
        raise access_token

    async with ConfluenceClient(access_token) as client:
        comment_response = await client.add_comment(
            page_id=page_id,
            comment_body=comment_body,
        )

    return ToolResult(
        content=f"Comment added successfully to page ID {page_id}.",
        structured_content={
            "comment_id": comment_response.comment_id,
            "page_id": page_id,
        },
    )


@dr_mcp_tool(tags={"confluence", "search", "content"})
async def confluence_search(
    *,
    cql_query: Annotated[
        str,
        "The CQL (Confluence Query Language) string used to filter content, "
        "e.g., 'type=page and space=DOC'.",
    ],
    max_results: Annotated[int, "Maximum number of content items to return. Default is 10."] = 10,
    include_body: Annotated[
        bool,
        "If True, fetch full page body content for each result (slower, "
        "makes additional API calls). Default is False, which returns only excerpts.",
    ] = False,
) -> ToolResult:
    """
    Search Confluence pages and content efficiently using a CQL query string.
    This pushes the search logic to the Confluence API (Push-Down).

    Refer to Confluence documentation for advanced searching using CQL:
    https://developer.atlassian.com/cloud/confluence/advanced-searching-using-cql/
    """
    if not cql_query:
        raise ToolError("Argument validation error: 'cql_query' cannot be empty.")

    if max_results < 1 or max_results > 100:
        raise ToolError("Argument validation error: 'max_results' must be between 1 and 100.")

    access_token = await get_atlassian_access_token()
    if isinstance(access_token, ToolError):
        raise access_token

    async with ConfluenceClient(access_token) as client:
        results = await client.search_confluence_content(
            cql_query=cql_query, max_results=max_results
        )

        # If include_body is True, fetch full content for each page
        if include_body and results:
            data = []
            for result in results:
                flat = result.as_flat_dict()
                try:
                    page = await client.get_page_by_id(result.id)
                    flat["body"] = page.body
                except ConfluenceError:
                    flat["body"] = None  # Keep excerpt if page fetch fails
                data.append(flat)
        else:
            data = [result.as_flat_dict() for result in results]

    n = len(results)
    return ToolResult(
        content=f"Successfully executed CQL query and retrieved {n} result(s).",
        structured_content={"data": data, "count": n},
    )


@dr_mcp_tool(tags={"confluence", "write", "update", "page"})
async def confluence_update_page(
    *,
    page_id: Annotated[str, "The ID of the Confluence page to update."],
    new_body_content: Annotated[
        str,
        "The full updated content of the page in Confluence Storage Format (XML) or raw text.",
    ],
    version_number: Annotated[
        int,
        "The current version number of the page, required to prevent update conflicts. "
        "Get this from the confluence_get_page tool.",
    ],
) -> ToolResult:
    """Update the content of an existing Confluence page.

    Requires the current version number to ensure atomic updates.
    Use this tool to update the body content of an existing Confluence page.
    The version_number is required for optimistic locking - it prevents overwriting
    changes made by others since you last fetched the page.

    Usage:
        page_id="856391684", new_body_content="<p>New content</p>", version_number=5

    Important: Always fetch the page first using confluence_get_page to get the
    current version number before updating.
    """
    if not page_id:
        raise ToolError("Argument validation error: 'page_id' cannot be empty.")

    if not new_body_content:
        raise ToolError("Argument validation error: 'new_body_content' cannot be empty.")

    if version_number < 1:
        raise ToolError(
            "Argument validation error: 'version_number' must be a positive integer (>= 1)."
        )

    access_token = await get_atlassian_access_token()
    if isinstance(access_token, ToolError):
        raise access_token

    async with ConfluenceClient(access_token) as client:
        page_response = await client.update_page(
            page_id=page_id,
            new_body_content=new_body_content,
            version_number=version_number,
        )

    return ToolResult(
        content=f"Page ID {page_id} updated successfully to version {page_response.version}.",
        structured_content={
            "updated_page_id": page_response.page_id,
            "new_version": page_response.version,
        },
    )
