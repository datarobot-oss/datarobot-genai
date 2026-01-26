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

"""GitHub MCP tools for interacting with GitHub repositories, issues, and pull requests.

These tools proxy calls to GitHub's official MCP server at https://api.githubcopilot.com/mcp/.
Tool discovery does not require authentication, but tool execution requires a GitHub OAuth token.

Reference: https://github.com/github/github-mcp-server
"""

import json
import logging
from typing import Annotated
from typing import Any

from fastmcp.exceptions import ToolError
from fastmcp.tools.tool import ToolResult

from datarobot_genai.drmcp.core.mcp_instance import dr_mcp_tool
from datarobot_genai.drmcp.tools.clients.github import GitHubMCPClient
from datarobot_genai.drmcp.tools.clients.github import get_github_access_token

logger = logging.getLogger(__name__)


def _extract_tool_content(result: dict[str, Any]) -> Any:
    """Extract content from MCP tool result.

    The GitHub MCP server returns results in the format:
    {
        "content": [
            {"type": "text", "text": "..."}
        ]
    }
    """
    content = result.get("content", [])
    if not content:
        return result

    # If there's a single text content, return just the text
    if len(content) == 1 and content[0].get("type") == "text":
        text = content[0].get("text", "")
        # Try to parse as JSON if it looks like JSON
        if text.startswith("{") or text.startswith("["):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text
        return text

    # Otherwise return all content items
    return content


@dr_mcp_tool(tags={"github", "issues", "create"})
async def github_create_issue(
    *,
    owner: Annotated[str, "Repository owner (username or organization)"],
    repo: Annotated[str, "Repository name"],
    title: Annotated[str, "Issue title"],
    body: Annotated[str | None, "Issue body content (Markdown supported)"] = None,
    labels: Annotated[list[str] | None, "List of label names to apply"] = None,
    assignees: Annotated[list[str] | None, "List of usernames to assign"] = None,
    milestone: Annotated[int | None, "Milestone number to associate"] = None,
) -> ToolResult:
    """
    Create a new issue in a GitHub repository.

    This tool creates a new issue with the specified title and optional body,
    labels, assignees, and milestone. Markdown is supported in the body.

    Required GitHub token scopes: repo (for private repos) or public_repo (for public repos)
    """
    if not owner or not owner.strip():
        raise ToolError("Argument validation error: 'owner' cannot be empty.")
    if not repo or not repo.strip():
        raise ToolError("Argument validation error: 'repo' cannot be empty.")
    if not title or not title.strip():
        raise ToolError("Argument validation error: 'title' cannot be empty.")

    access_token = await get_github_access_token()
    if isinstance(access_token, ToolError):
        raise access_token

    arguments: dict[str, Any] = {
        "owner": owner.strip(),
        "repo": repo.strip(),
        "title": title.strip(),
    }
    if body:
        arguments["body"] = body
    if labels:
        arguments["labels"] = labels
    if assignees:
        arguments["assignees"] = assignees
    if milestone is not None:
        arguments["milestone"] = milestone

    async with GitHubMCPClient(access_token) as client:
        result = await client.call_tool("create_issue", arguments)

    content = _extract_tool_content(result)

    return ToolResult(
        content=f"Successfully created issue '{title}' in {owner}/{repo}.",
        structured_content=content if isinstance(content, dict) else {"result": content},
    )


@dr_mcp_tool(tags={"github", "issues", "search"})
async def github_search_issues(
    *,
    query: Annotated[
        str,
        "Search query using GitHub search syntax "
        "(e.g., 'is:open is:issue label:bug repo:owner/repo')",
    ],
    sort: Annotated[
        str | None,
        "Sort field: 'comments', 'reactions', 'created', 'updated'. Default: best match",
    ] = None,
    order: Annotated[str | None, "Sort order: 'asc' or 'desc'. Default: 'desc'"] = None,
    per_page: Annotated[int, "Results per page (max 100)"] = 30,
    page: Annotated[int, "Page number for pagination"] = 1,
) -> ToolResult:
    """
    Search for issues and pull requests across GitHub repositories.

    Uses GitHub's powerful search syntax. Examples:
    - 'is:open is:issue repo:owner/repo' - Open issues in a specific repo
    - 'is:pr is:merged author:username' - Merged PRs by a user
    - 'label:bug label:help-wanted' - Issues with specific labels
    - 'in:title error' - Issues with 'error' in the title

    Reference: https://docs.github.com/en/search-github/searching-on-github/searching-issues-and-pull-requests
    """
    if not query or not query.strip():
        raise ToolError("Argument validation error: 'query' cannot be empty.")

    access_token = await get_github_access_token()
    if isinstance(access_token, ToolError):
        raise access_token

    arguments: dict[str, Any] = {
        "q": query.strip(),
        "per_page": min(per_page, 100),
        "page": max(page, 1),
    }
    if sort:
        arguments["sort"] = sort
    if order:
        arguments["order"] = order

    async with GitHubMCPClient(access_token) as client:
        result = await client.call_tool("search_issues", arguments)

    content = _extract_tool_content(result)

    # Try to get count from result
    count = 0
    if isinstance(content, dict):
        count = content.get("total_count", len(content.get("items", [])))
    elif isinstance(content, list):
        count = len(content)

    return ToolResult(
        content=f"Searched GitHub issues. Found {count} result(s) for query: '{query}'.",
        structured_content=content if isinstance(content, dict) else {"results": content},
    )


@dr_mcp_tool(tags={"github", "pull_requests", "create"})
async def github_create_pull_request(
    *,
    owner: Annotated[str, "Repository owner (username or organization)"],
    repo: Annotated[str, "Repository name"],
    title: Annotated[str, "Pull request title"],
    head: Annotated[str, "The name of the branch where your changes are implemented"],
    base: Annotated[str, "The name of the branch you want the changes pulled into"],
    body: Annotated[str | None, "Pull request body content (Markdown supported)"] = None,
    draft: Annotated[bool, "Create as a draft pull request"] = False,
    maintainer_can_modify: Annotated[bool, "Allow maintainers to modify the pull request"] = True,
) -> ToolResult:
    """
    Create a new pull request in a GitHub repository.

    Creates a pull request to merge changes from the head branch into the base branch.
    The head branch must exist and contain commits not in the base branch.

    Required GitHub token scopes: repo
    """
    if not owner or not owner.strip():
        raise ToolError("Argument validation error: 'owner' cannot be empty.")
    if not repo or not repo.strip():
        raise ToolError("Argument validation error: 'repo' cannot be empty.")
    if not title or not title.strip():
        raise ToolError("Argument validation error: 'title' cannot be empty.")
    if not head or not head.strip():
        raise ToolError("Argument validation error: 'head' cannot be empty.")
    if not base or not base.strip():
        raise ToolError("Argument validation error: 'base' cannot be empty.")

    access_token = await get_github_access_token()
    if isinstance(access_token, ToolError):
        raise access_token

    arguments: dict[str, Any] = {
        "owner": owner.strip(),
        "repo": repo.strip(),
        "title": title.strip(),
        "head": head.strip(),
        "base": base.strip(),
        "draft": draft,
        "maintainer_can_modify": maintainer_can_modify,
    }
    if body:
        arguments["body"] = body

    async with GitHubMCPClient(access_token) as client:
        result = await client.call_tool("create_pull_request", arguments)

    content = _extract_tool_content(result)

    return ToolResult(
        content=f"Created pull request '{title}' in {owner}/{repo} ({head} -> {base}).",
        structured_content=content if isinstance(content, dict) else {"result": content},
    )


@dr_mcp_tool(tags={"github", "repository", "files", "read"})
async def github_get_file_contents(
    *,
    owner: Annotated[str, "Repository owner (username or organization)"],
    repo: Annotated[str, "Repository name"],
    path: Annotated[str, "Path to the file or directory in the repository"],
    ref: Annotated[
        str | None, "Git reference (branch, tag, or commit SHA). Default: default branch"
    ] = None,
) -> ToolResult:
    """
    Get the contents of a file or directory from a GitHub repository.

    For files, returns the content (decoded if text). For directories, returns
    a list of entries. Supports files up to 1MB in size via this API.

    Required GitHub token scopes: repo (for private repos) or none (for public repos)
    """
    if not owner or not owner.strip():
        raise ToolError("Argument validation error: 'owner' cannot be empty.")
    if not repo or not repo.strip():
        raise ToolError("Argument validation error: 'repo' cannot be empty.")
    if not path:
        raise ToolError("Argument validation error: 'path' cannot be empty.")

    access_token = await get_github_access_token()
    if isinstance(access_token, ToolError):
        raise access_token

    arguments: dict[str, Any] = {
        "owner": owner.strip(),
        "repo": repo.strip(),
        "path": path.strip("/"),
    }
    if ref:
        arguments["ref"] = ref.strip()

    async with GitHubMCPClient(access_token) as client:
        result = await client.call_tool("get_file_contents", arguments)

    content = _extract_tool_content(result)

    return ToolResult(
        content=f"Successfully retrieved contents of '{path}' from {owner}/{repo}.",
        structured_content=content if isinstance(content, dict) else {"content": content},
    )


@dr_mcp_tool(tags={"github", "code", "search"})
async def github_search_code(
    *,
    query: Annotated[
        str,
        "Search query using GitHub code search syntax "
        "(e.g., 'function repo:owner/repo language:python')",
    ],
    sort: Annotated[str | None, "Sort field: 'indexed'. Default: best match"] = None,
    order: Annotated[str | None, "Sort order: 'asc' or 'desc'. Default: 'desc'"] = None,
    per_page: Annotated[int, "Results per page (max 100)"] = 30,
    page: Annotated[int, "Page number for pagination"] = 1,
) -> ToolResult:
    """
    Search for code across GitHub repositories.

    Uses GitHub's code search syntax. Examples:
    - 'function repo:owner/repo' - Search in a specific repo
    - 'import pandas language:python' - Python files with pandas import
    - 'filename:config.yml path:/.github' - Config files in .github directory
    - 'extension:js "use strict"' - JavaScript files with "use strict"

    Note: Code search requires authentication and has rate limits.
    Reference: https://docs.github.com/en/search-github/searching-on-github/searching-code
    """
    if not query or not query.strip():
        raise ToolError("Argument validation error: 'query' cannot be empty.")

    access_token = await get_github_access_token()
    if isinstance(access_token, ToolError):
        raise access_token

    arguments: dict[str, Any] = {
        "q": query.strip(),
        "per_page": min(per_page, 100),
        "page": max(page, 1),
    }
    if sort:
        arguments["sort"] = sort
    if order:
        arguments["order"] = order

    async with GitHubMCPClient(access_token) as client:
        result = await client.call_tool("search_code", arguments)

    content = _extract_tool_content(result)

    count = 0
    if isinstance(content, dict):
        count = content.get("total_count", len(content.get("items", [])))
    elif isinstance(content, list):
        count = len(content)

    return ToolResult(
        content=f"Successfully searched GitHub code. Found {count} result(s) for query: '{query}'.",
        structured_content=content if isinstance(content, dict) else {"results": content},
    )


@dr_mcp_tool(tags={"github", "issues", "comments", "create"})
async def github_add_issue_comment(
    *,
    owner: Annotated[str, "Repository owner (username or organization)"],
    repo: Annotated[str, "Repository name"],
    issue_number: Annotated[int, "Issue or pull request number"],
    body: Annotated[str, "Comment body content (Markdown supported)"],
) -> ToolResult:
    """
    Add a comment to an issue or pull request.

    Comments support Markdown formatting including code blocks, images,
    and @mentions. This works for both issues and pull requests.

    Required GitHub token scopes: repo (for private repos) or public_repo (for public repos)
    """
    if not owner or not owner.strip():
        raise ToolError("Argument validation error: 'owner' cannot be empty.")
    if not repo or not repo.strip():
        raise ToolError("Argument validation error: 'repo' cannot be empty.")
    if issue_number <= 0:
        raise ToolError("Argument validation error: 'issue_number' must be a positive integer.")
    if not body or not body.strip():
        raise ToolError("Argument validation error: 'body' cannot be empty.")

    access_token = await get_github_access_token()
    if isinstance(access_token, ToolError):
        raise access_token

    arguments: dict[str, Any] = {
        "owner": owner.strip(),
        "repo": repo.strip(),
        "issue_number": issue_number,
        "body": body,
    }

    async with GitHubMCPClient(access_token) as client:
        result = await client.call_tool("add_issue_comment", arguments)

    content = _extract_tool_content(result)

    return ToolResult(
        content=f"Successfully added comment to issue #{issue_number} in {owner}/{repo}.",
        structured_content=content if isinstance(content, dict) else {"result": content},
    )


@dr_mcp_tool(tags={"github", "repository", "commits", "list"})
async def github_list_commits(
    *,
    owner: Annotated[str, "Repository owner (username or organization)"],
    repo: Annotated[str, "Repository name"],
    sha: Annotated[
        str | None, "SHA or branch name to start listing commits from. Default: default branch"
    ] = None,
    path: Annotated[str | None, "Only commits containing this file path"] = None,
    author: Annotated[str | None, "GitHub username or email to filter by author"] = None,
    since: Annotated[str | None, "Only commits after this date (ISO 8601 format)"] = None,
    until: Annotated[str | None, "Only commits before this date (ISO 8601 format)"] = None,
    per_page: Annotated[int, "Results per page (max 100)"] = 30,
    page: Annotated[int, "Page number for pagination"] = 1,
) -> ToolResult:
    """
    List commits in a GitHub repository.

    Returns commit history with optional filtering by branch, path, author, and date range.
    Results include commit SHA, message, author, and date information.

    Required GitHub token scopes: repo (for private repos) or none (for public repos)
    """
    if not owner or not owner.strip():
        raise ToolError("Argument validation error: 'owner' cannot be empty.")
    if not repo or not repo.strip():
        raise ToolError("Argument validation error: 'repo' cannot be empty.")

    access_token = await get_github_access_token()
    if isinstance(access_token, ToolError):
        raise access_token

    arguments: dict[str, Any] = {
        "owner": owner.strip(),
        "repo": repo.strip(),
        "per_page": min(per_page, 100),
        "page": max(page, 1),
    }
    if sha:
        arguments["sha"] = sha.strip()
    if path:
        arguments["path"] = path.strip()
    if author:
        arguments["author"] = author.strip()
    if since:
        arguments["since"] = since
    if until:
        arguments["until"] = until

    async with GitHubMCPClient(access_token) as client:
        result = await client.call_tool("list_commits", arguments)

    content = _extract_tool_content(result)

    count = len(content) if isinstance(content, list) else 0

    return ToolResult(
        content=f"Successfully retrieved {count} commit(s) from {owner}/{repo}.",
        structured_content={"commits": content} if isinstance(content, list) else content,
    )


@dr_mcp_tool(tags={"github", "issues", "read", "get"})
async def github_get_issue(
    *,
    owner: Annotated[str, "Repository owner (username or organization)"],
    repo: Annotated[str, "Repository name"],
    issue_number: Annotated[int, "Issue number"],
) -> ToolResult:
    """
    Get details of a specific issue from a GitHub repository.

    Returns full issue details including title, body, state, labels,
    assignees, milestone, and comments count.

    Required GitHub token scopes: repo (for private repos) or none (for public repos)
    """
    if not owner or not owner.strip():
        raise ToolError("Argument validation error: 'owner' cannot be empty.")
    if not repo or not repo.strip():
        raise ToolError("Argument validation error: 'repo' cannot be empty.")
    if issue_number <= 0:
        raise ToolError("Argument validation error: 'issue_number' must be a positive integer.")

    access_token = await get_github_access_token()
    if isinstance(access_token, ToolError):
        raise access_token

    arguments: dict[str, Any] = {
        "owner": owner.strip(),
        "repo": repo.strip(),
        "issue_number": issue_number,
    }

    async with GitHubMCPClient(access_token) as client:
        result = await client.call_tool("get_issue", arguments)

    content = _extract_tool_content(result)

    return ToolResult(
        content=f"Successfully retrieved issue #{issue_number} from {owner}/{repo}.",
        structured_content=content if isinstance(content, dict) else {"issue": content},
    )


@dr_mcp_tool(tags={"github", "pull_requests", "read", "get"})
async def github_get_pull_request(
    *,
    owner: Annotated[str, "Repository owner (username or organization)"],
    repo: Annotated[str, "Repository name"],
    pull_number: Annotated[int, "Pull request number"],
) -> ToolResult:
    """
    Get details of a specific pull request from a GitHub repository.

    Returns full PR details including title, body, state, head/base branches,
    merge status, reviewers, and more.

    Required GitHub token scopes: repo (for private repos) or none (for public repos)
    """
    if not owner or not owner.strip():
        raise ToolError("Argument validation error: 'owner' cannot be empty.")
    if not repo or not repo.strip():
        raise ToolError("Argument validation error: 'repo' cannot be empty.")
    if pull_number <= 0:
        raise ToolError("Argument validation error: 'pull_number' must be a positive integer.")

    access_token = await get_github_access_token()
    if isinstance(access_token, ToolError):
        raise access_token

    arguments: dict[str, Any] = {
        "owner": owner.strip(),
        "repo": repo.strip(),
        "pull_number": pull_number,
    }

    async with GitHubMCPClient(access_token) as client:
        result = await client.call_tool("get_pull_request", arguments)

    content = _extract_tool_content(result)

    return ToolResult(
        content=f"Successfully retrieved pull request #{pull_number} from {owner}/{repo}.",
        structured_content=content if isinstance(content, dict) else {"pull_request": content},
    )
