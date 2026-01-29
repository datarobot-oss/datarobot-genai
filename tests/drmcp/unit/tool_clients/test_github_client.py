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

"""Tests for GitHub MCP client."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import httpx
import pytest
from datarobot.auth.datarobot.exceptions import OAuthServiceClientErr
from fastmcp.exceptions import ToolError

from datarobot_genai.drmcp.tools.clients.github import GitHubMCPClient
from datarobot_genai.drmcp.tools.clients.github import GitHubMCPError
from datarobot_genai.drmcp.tools.clients.github import get_github_access_token


class TestGetGitHubAccessToken:
    """Test get_github_access_token function."""

    @pytest.mark.asyncio
    async def test_get_access_token_success(self) -> None:
        """Test successful access token retrieval via OAuth."""
        with patch(
            "datarobot_genai.drmcp.tools.clients.github.get_access_token",
            new_callable=AsyncMock,
            return_value="oauth-token",
        ):
            result = await get_github_access_token()
            assert result == "oauth-token"

    @pytest.mark.asyncio
    async def test_returns_tool_error_on_empty_token(self) -> None:
        """Test that ToolError is returned when OAuth returns empty token."""
        with patch(
            "datarobot_genai.drmcp.tools.clients.github.get_access_token",
            new_callable=AsyncMock,
            return_value="",
        ):
            result = await get_github_access_token()
            assert isinstance(result, ToolError)
            assert "empty access token" in str(result).lower()

    @pytest.mark.asyncio
    async def test_returns_tool_error_on_oauth_error(self) -> None:
        """Test that ToolError is returned on OAuth error."""
        oauth_error = OAuthServiceClientErr("OAuth error")
        with patch(
            "datarobot_genai.drmcp.tools.clients.github.get_access_token",
            new_callable=AsyncMock,
            side_effect=oauth_error,
        ):
            result = await get_github_access_token()
            assert isinstance(result, ToolError)
            assert "Could not obtain access token" in str(result)

    @pytest.mark.asyncio
    async def test_returns_tool_error_on_unexpected_error(self) -> None:
        """Test that ToolError is returned on unexpected error."""
        with patch(
            "datarobot_genai.drmcp.tools.clients.github.get_access_token",
            new_callable=AsyncMock,
            side_effect=ValueError("Unexpected"),
        ):
            result = await get_github_access_token()
            assert isinstance(result, ToolError)
            assert "unexpected error" in str(result).lower()


class TestGitHubMCPClientInit:
    """Test GitHubMCPClient initialization."""

    def test_init_with_token_only(self) -> None:
        """Test initialization with just a token."""
        client = GitHubMCPClient("test-token")
        assert client.access_token == "test-token"
        assert client._toolsets is None

    def test_init_with_toolsets(self) -> None:
        """Test initialization with toolsets."""
        client = GitHubMCPClient("test-token", toolsets=["all"])
        assert client._toolsets == ["all"]


class TestGitHubMCPClientParseResponse:
    """Test GitHubMCPClient._parse_response method."""

    def test_parse_json_response(self) -> None:
        """Test parsing standard JSON response."""
        mock_response = MagicMock()
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"result": {"tools": []}}

        result = GitHubMCPClient._parse_response(mock_response)
        assert result == {"result": {"tools": []}}

    def test_parse_sse_response(self) -> None:
        """Test parsing SSE response."""
        mock_response = MagicMock()
        mock_response.headers = {"content-type": "text/event-stream"}
        mock_response.text = 'data: {"result": {"tools": []}}\n'

        result = GitHubMCPClient._parse_response(mock_response)
        assert result == {"result": {"tools": []}}

    def test_parse_sse_response_multiple_lines(self) -> None:
        """Test parsing SSE response with multiple lines."""
        mock_response = MagicMock()
        mock_response.headers = {"content-type": "text/event-stream"}
        mock_response.text = 'event: message\ndata: {"result": "success"}\n\n'

        result = GitHubMCPClient._parse_response(mock_response)
        assert result == {"result": "success"}

    def test_parse_sse_response_no_data(self) -> None:
        """Test parsing SSE response with no data raises error."""
        mock_response = MagicMock()
        mock_response.headers = {"content-type": "text/event-stream"}
        mock_response.text = "event: ping\n\n"

        with pytest.raises(GitHubMCPError, match="no data"):
            GitHubMCPClient._parse_response(mock_response)


class TestGitHubMCPClientListTools:
    """Test GitHubMCPClient.list_tools method."""

    @pytest.mark.asyncio
    async def test_list_tools_success(self) -> None:
        """Test successful tool listing."""
        mock_tools = [
            {"name": "get_me", "description": "Get user info", "inputSchema": {}},
            {"name": "create_issue", "description": "Create issue", "inputSchema": {}},
        ]
        mock_response = MagicMock()
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"result": {"tools": mock_tools}}
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=mock_response
        ):
            async with GitHubMCPClient("test-token") as client:
                tools = await client.list_tools()

        assert len(tools) == 2
        assert tools[0]["name"] == "get_me"
        assert tools[1]["name"] == "create_issue"

    @pytest.mark.asyncio
    async def test_list_tools_with_jsonrpc_error(self) -> None:
        """Test handling of JSON-RPC error in response."""
        mock_response = MagicMock()
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"error": {"code": -32600, "message": "Invalid request"}}
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=mock_response
        ):
            async with GitHubMCPClient("test-token") as client:
                with pytest.raises(GitHubMCPError, match="Invalid request"):
                    await client.list_tools()


class TestGitHubMCPClientCallTool:
    """Test GitHubMCPClient.call_tool method."""

    @pytest.mark.asyncio
    async def test_call_tool_success(self) -> None:
        """Test successful tool call."""
        mock_result = {"content": [{"type": "text", "text": '{"login": "testuser"}'}]}
        mock_response = MagicMock()
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"result": mock_result}
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=mock_response
        ):
            async with GitHubMCPClient("test-token") as client:
                result = await client.call_tool("get_me", {})

        assert result == mock_result

    @pytest.mark.asyncio
    async def test_call_tool_http_401_error(self) -> None:
        """Test handling of 401 authentication error."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Unauthorized", request=MagicMock(), response=mock_response
        )

        with patch.object(
            httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=mock_response
        ):
            async with GitHubMCPClient("test-token") as client:
                with pytest.raises(GitHubMCPError, match="Authentication failed"):
                    await client.call_tool("get_me", {})

    @pytest.mark.asyncio
    async def test_call_tool_http_403_error(self) -> None:
        """Test handling of 403 permission error."""
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Forbidden", request=MagicMock(), response=mock_response
        )

        with patch.object(
            httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=mock_response
        ):
            async with GitHubMCPClient("test-token") as client:
                with pytest.raises(GitHubMCPError, match="Permission denied"):
                    await client.call_tool("create_issue", {"owner": "test", "repo": "test"})

    @pytest.mark.asyncio
    async def test_call_tool_network_error(self) -> None:
        """Test handling of network errors."""
        with patch.object(
            httpx.AsyncClient,
            "post",
            new_callable=AsyncMock,
            side_effect=httpx.RequestError("Connection failed", request=MagicMock()),
        ):
            async with GitHubMCPClient("test-token") as client:
                with pytest.raises(GitHubMCPError, match="Network error"):
                    await client.call_tool("get_me", {})
