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

"""GitHub MCP client for proxying tool calls to GitHub's remote MCP server."""

import json
import logging
from typing import Any

import httpx
from datarobot.auth.datarobot.exceptions import OAuthServiceClientErr
from fastmcp.exceptions import ToolError

from datarobot_genai.drmcp.core.auth import get_access_token

logger = logging.getLogger(__name__)

# GitHub MCP server URL
GITHUB_MCP_SERVER_URL = "https://api.githubcopilot.com/mcp/"


async def get_github_access_token() -> str | ToolError:
    """
    Get GitHub OAuth access token with error handling.

    Returns
    -------
        Access token string on success, ToolError on failure

    Example:
        ```python
        token = await get_github_access_token()
        if isinstance(token, ToolError):
            # Handle error
            return token
        # Use token
        ```
    """
    try:
        access_token = await get_access_token("github")
        if not access_token:
            logger.warning("Empty access token received")
            return ToolError("Received empty access token. Please complete the OAuth flow.")
        return access_token
    except OAuthServiceClientErr as e:
        logger.error(f"OAuth client error: {e}", exc_info=True)
        return ToolError(
            "Could not obtain access token for GitHub. Make sure the OAuth "
            "permission was granted for the application to act on your behalf."
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Unexpected error obtaining access token: {error_msg}", exc_info=True)
        return ToolError("An unexpected error occurred while obtaining access token for GitHub.")


class GitHubMCPError(Exception):
    """Exception for GitHub MCP API errors."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class GitHubMCPClient:
    """Client for proxying tool calls to GitHub's remote MCP server.

    The GitHub MCP server provides 51+ tools for interacting with GitHub resources
    including repositories, issues, pull requests, code search, and more.

    API Reference:
    - GitHub MCP Server: https://github.com/github/github-mcp-server
    - Remote URL: https://api.githubcopilot.com/mcp/
    """

    def __init__(self, access_token: str, toolsets: list[str] | None = None):
        """
        Initialize GitHub MCP client with access token.

        Args:
            access_token: OAuth access token for GitHub API
            toolsets: Optional list of toolsets to request (e.g., ["all"] or ["repos", "issues"]).
                     These are sent via the X-MCP-Toolsets header.
        """
        self.access_token = access_token
        self._toolsets = toolsets

        headers: dict[str, str] = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }
        if toolsets:
            headers["X-MCP-Toolsets"] = ",".join(toolsets)

        self._client = httpx.AsyncClient(
            headers=headers,
            timeout=60.0,  # Longer timeout for MCP operations
        )
        self._request_id = 0

    def _next_request_id(self) -> str:
        """Generate the next request ID for JSON-RPC calls."""
        self._request_id += 1
        return str(self._request_id)

    @staticmethod
    def _parse_response(response: httpx.Response) -> dict[str, Any]:
        """Parse response, handling both JSON and SSE (Server-Sent Events) formats.

        The GitHub MCP server may return responses as either standard JSON or
        as SSE (text/event-stream) format. This method handles both cases.

        Args:
            response: The HTTP response to parse

        Returns
        -------
            Parsed JSON response as a dictionary

        Raises
        ------
            GitHubMCPError: If SSE response contains no data or parsing fails
        """
        content_type = response.headers.get("content-type", "")

        if "text/event-stream" in content_type:
            # Parse SSE format: look for lines starting with "data: "
            for line in response.text.splitlines():
                if line.startswith("data: "):
                    try:
                        return json.loads(line[6:])
                    except json.JSONDecodeError as e:
                        raise GitHubMCPError(f"Failed to parse SSE data: {e}") from e
            raise GitHubMCPError("SSE response contained no data")

        # Standard JSON response
        return response.json()

    async def list_tools(self) -> list[dict[str, Any]]:
        """
        List all available tools from the GitHub MCP server.

        Returns
        -------
            List of tool definitions with name, description, and inputSchema

        Raises
        ------
            GitHubMCPError: If the request fails
        """
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_request_id(),
            "method": "tools/list",
            "params": {},
        }

        try:
            response = await self._client.post(GITHUB_MCP_SERVER_URL, json=payload)
            response.raise_for_status()
            result = self._parse_response(response)

            # Check for JSON-RPC error
            if "error" in result:
                error = result["error"]
                error_message = error.get("message", "Unknown error")
                error_code = error.get("code", "")
                raise GitHubMCPError(f"GitHub MCP error ({error_code}): {error_message}")

            return result.get("result", {}).get("tools", [])

        except httpx.HTTPStatusError as e:
            raise self._handle_http_error(e, "Failed to list tools") from e
        except httpx.RequestError as e:
            logger.error(f"Request error listing GitHub MCP tools: {e}")
            raise GitHubMCPError(f"Network error listing GitHub MCP tools: {e}") from e

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """
        Call a tool on the GitHub MCP server.

        Args:
            tool_name: The name of the tool to call
            arguments: The arguments to pass to the tool

        Returns
        -------
            The result from the tool execution

        Raises
        ------
            GitHubMCPError: If the tool call fails
        """
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_request_id(),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
            },
        }

        try:
            response = await self._client.post(GITHUB_MCP_SERVER_URL, json=payload)
            response.raise_for_status()
            result = self._parse_response(response)

            # Check for JSON-RPC error
            if "error" in result:
                error = result["error"]
                error_message = error.get("message", "Unknown error")
                error_code = error.get("code", "")
                raise GitHubMCPError(f"GitHub MCP error ({error_code}): {error_message}")

            return result.get("result", {})

        except httpx.HTTPStatusError as e:
            raise self._handle_http_error(e, f"Failed to call tool '{tool_name}'") from e
        except httpx.RequestError as e:
            logger.error(f"Request error calling GitHub MCP: {e}")
            raise GitHubMCPError(f"Network error calling GitHub MCP: {e}") from e

    def _handle_http_error(self, error: httpx.HTTPStatusError, base_message: str) -> GitHubMCPError:
        """Handle HTTP errors and return appropriate GitHubMCPError with user-friendly messages."""
        status_code = error.response.status_code
        error_msg = base_message

        if status_code == 401:
            error_msg += (
                ": Authentication failed. Your GitHub token may be expired or invalid. "
                "Please re-authenticate with GitHub."
            )
        elif status_code == 403:
            error_msg += (
                ": Permission denied. Your GitHub token may not have the required scopes "
                "for this operation."
            )
        elif status_code == 404:
            error_msg += ": The requested resource was not found on GitHub."
        elif status_code == 422:
            try:
                error_data = error.response.json()
                api_message = error_data.get("error", {}).get("message", "Validation failed")
                error_msg += f": {api_message}"
            except Exception:
                error_msg += ": Invalid request parameters."
        elif status_code == 429:
            error_msg += ": Rate limit exceeded. Please wait before making more requests."
        else:
            error_msg += f": HTTP {status_code}"

        return GitHubMCPError(error_msg)

    async def __aenter__(self) -> "GitHubMCPClient":
        """Async context manager entry."""
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any
    ) -> None:
        """Async context manager exit."""
        await self._client.aclose()
