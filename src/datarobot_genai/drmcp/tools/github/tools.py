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

"""GitHub MCP tools helper functions.

This module provides helper functions for GitHub MCP tool execution.
Tools are dynamically registered from the remote MCP server or manifest
via register.py. This module provides the shared execution logic.

Reference: https://github.com/github/github-mcp-server
"""

import json
import logging
from typing import Any

from fastmcp.exceptions import ToolError
from fastmcp.tools.tool import ToolResult

from datarobot_genai.drmcp.tools.clients.github import GitHubMCPClient
from datarobot_genai.drmcp.tools.clients.github import GitHubMCPError
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

    Args:
        result: Raw result from the GitHub MCP server

    Returns
    -------
        Extracted and parsed content
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


async def call_github_tool(tool_name: str, arguments: dict[str, Any]) -> ToolResult:
    """Call a GitHub MCP tool and return the result.

    This is the shared execution function used by all dynamically registered
    GitHub tools.

    Args:
        tool_name: The name of the GitHub MCP tool to call
        arguments: Arguments to pass to the tool

    Returns
    -------
        ToolResult with the execution result

    Raises
    ------
        ToolError: If authentication fails
        GitHubMCPError: If the tool call fails
    """
    access_token = await get_github_access_token()
    if isinstance(access_token, ToolError):
        raise access_token

    try:
        async with GitHubMCPClient(access_token) as client:
            result = await client.call_tool(tool_name, arguments)

        content = _extract_tool_content(result)

        return ToolResult(
            content=f"Successfully executed GitHub tool '{tool_name}'.",
            structured_content=content if isinstance(content, dict) else {"result": content},
        )
    except GitHubMCPError as e:
        logger.error(f"GitHub MCP tool '{tool_name}' failed: {e}")
        raise ToolError(str(e)) from e
