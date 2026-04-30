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
from typing import Any

from datarobot_genai.drtools.core import tool_metadata
from datarobot_genai.drtools.core.clients.atlassian import get_atlassian_access_token
from datarobot_genai.drtools.core.clients.confluence import ConfluenceClient
from datarobot_genai.drtools.core.clients.confluence import ConfluenceError
from datarobot_genai.drtools.core.exceptions import ToolError
from datarobot_genai.drtools.core.exceptions import ToolErrorKind

logger = logging.getLogger(__name__)

_CQL_DOCS = "https://developer.atlassian.com/cloud/confluence/advanced-searching-using-cql/"
_PAGE_REST_DOCS = "https://developer.atlassian.com/cloud/confluence/rest/v2/api-group-page/"
_PAGE_BODY_DOCS = (
    "https://developer.atlassian.com/cloud/confluence/rest/v2/api-group-page/#api-pages-post"
)
_FOOTER_COMMENTS_DOCS = (
    "https://developer.atlassian.com/cloud/confluence/rest/v2/api-group-footer-comments/"
)


@tool_metadata(
    tags={"confluence", "read", "get", "page"},
    description=(
        "[Confluence—get page] Use when you have a numeric page id, or an exact page title plus "
        "space_key, and need the page body (storage HTML). Not CQL multi-page search "
        "(confluence_search), not Jira (jira_get_issue).\n\n"
        'Examples: By ID: page_id_or_title="856391684". By title: '
        'page_id_or_title="Meeting Notes", space_key="TEAM" (space_key is required when '
        "using a title).\n\n"
        f"Reference: {_PAGE_REST_DOCS}"
    ),
)
async def confluence_get_page(
    *,
    page_id_or_title: Annotated[str, "The ID or the exact title of the Confluence page."],
    space_key: Annotated[
        str | None,
        "Required if identifying the page by title. The space key (e.g., 'PROJ').",
    ] = None,
) -> dict[str, Any]:
    if not page_id_or_title:
        raise ToolError(
            "Argument validation error: 'page_id_or_title' cannot be empty.",
            kind=ToolErrorKind.VALIDATION,
        )

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
                    "'space_key' is required when identifying a page by title.",
                    kind=ToolErrorKind.VALIDATION,
                )
            page_response = await client.get_page_by_title(page_id_or_title, space_key)

    return page_response.as_flat_dict()


@tool_metadata(
    tags={"confluence", "write", "create", "page"},
    description=(
        "[Confluence—create page] Use when publishing a new page in a space (space_key, title, "
        "body storage format), optional parent page id. Not updating an existing page "
        "(confluence_update_page), not comments (confluence_add_comment).\n\n"
        'Examples: Root page: space_key="PROJ", title="New Page", '
        'body_content="<p>Content</p>". Child page: same fields plus parent_id=123456.\n\n'
        f"References (body representation / storage): {_PAGE_BODY_DOCS} {_PAGE_REST_DOCS}"
    ),
)
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
) -> dict[str, Any]:
    if not all([space_key, title, body_content]):
        raise ToolError(
            "Argument validation error: space_key, title, and body_content are required fields.",
            kind=ToolErrorKind.VALIDATION,
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

    return {"new_page_id": page_response.page_id, "title": page_response.title}


@tool_metadata(
    tags={"confluence", "write", "add", "comment"},
    description=(
        "[Confluence—comment] Use when appending a page-level comment on an existing page by "
        "numeric page_id. Not page body edits (confluence_update_page), not new pages "
        "(confluence_create_page).\n\n"
        'Example: page_id="856391684", comment_body="Great work on this documentation!"\n\n'
        f"Reference: {_FOOTER_COMMENTS_DOCS}"
    ),
)
async def confluence_add_comment(
    *,
    page_id: Annotated[str, "The numeric ID of the page where the comment will be added."],
    comment_body: Annotated[str, "The text content of the comment."],
) -> dict[str, Any]:
    if not page_id:
        raise ToolError(
            "Argument validation error: 'page_id' cannot be empty.", kind=ToolErrorKind.VALIDATION
        )

    if not comment_body:
        raise ToolError(
            "Argument validation error: 'comment_body' cannot be empty.",
            kind=ToolErrorKind.VALIDATION,
        )

    access_token = await get_atlassian_access_token()
    if isinstance(access_token, ToolError):
        raise access_token

    async with ConfluenceClient(access_token) as client:
        comment_response = await client.add_comment(
            page_id=page_id,
            comment_body=comment_body,
        )

    return {
        "comment_id": comment_response.comment_id,
        "page_id": page_id,
    }


@tool_metadata(
    tags={"confluence", "search", "content"},
    description=(
        "[Confluence—CQL search] Use when finding pages or content with Confluence Query Language "
        "(type, space, text filters). Optional full body per hit. "
        "Not Jira JQL (jira_search_issues), "
        "not single-page fetch by id alone (confluence_get_page when key already known).\n\n"
        'Example: cql_query="type=page and space=DOC", max_results=10, include_body=false.\n\n'
        f"Reference (CQL): {_CQL_DOCS}"
    ),
)
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
) -> dict[str, Any]:
    if not cql_query:
        raise ToolError(
            "Argument validation error: 'cql_query' cannot be empty.", kind=ToolErrorKind.VALIDATION
        )

    if max_results < 1 or max_results > 100:
        raise ToolError(
            "Argument validation error: 'max_results' must be between 1 and 100.",
            kind=ToolErrorKind.VALIDATION,
        )

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
    return {"data": data, "count": n}


@tool_metadata(
    tags={"confluence", "write", "update", "page"},
    description=(
        "[Confluence—update page] Use when replacing page body content for an existing page_id; "
        "version_number must match current version from confluence_get_page (optimistic lock). "
        "Not new pages (confluence_create_page), not comments (confluence_add_comment).\n\n"
        'Example: page_id="856391684", new_body_content="<p>New content</p>", '
        "version_number=5. Always call confluence_get_page first to read the current "
        f"version_number.\n\nReferences (body representation / storage): {_PAGE_BODY_DOCS} "
        f"{_PAGE_REST_DOCS}"
    ),
)
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
) -> dict[str, Any]:
    if not page_id:
        raise ToolError(
            "Argument validation error: 'page_id' cannot be empty.", kind=ToolErrorKind.VALIDATION
        )

    if not new_body_content:
        raise ToolError(
            "Argument validation error: 'new_body_content' cannot be empty.",
            kind=ToolErrorKind.VALIDATION,
        )

    if version_number < 1:
        raise ToolError(
            "Argument validation error: 'version_number' must be a positive integer (>= 1).",
            kind=ToolErrorKind.VALIDATION,
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

    return {
        "updated_page_id": page_response.page_id,
        "new_version": page_response.version,
    }
