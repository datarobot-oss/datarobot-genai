# Copyright 2025 DataRobot, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from datarobot_genai.llama_index.mcp import load_mcp_tools


class TestLoadMCPTools:
    """Test async MCP tools loading."""

    @pytest.mark.asyncio
    async def test_load_mcp_tools_no_configuration(self):
        """Test loading tools when no MCP server is configured."""
        with patch.dict(os.environ, {}, clear=True):
            tools = await load_mcp_tools()
            assert isinstance(tools, list)
            assert len(tools) == 0

    @pytest.mark.asyncio
    @patch("datarobot_genai.llama_index.mcp.aget_tools_from_mcp_url", new_callable=AsyncMock)
    async def test_load_mcp_tools_with_external_url(self, mock_aget):
        """Test loading tools with external MCP URL."""
        mock_tools = [MagicMock(), MagicMock()]
        mock_aget.return_value = mock_tools

        test_url = "https://mcp-server.example.com/mcp"
        with patch.dict(os.environ, {"EXTERNAL_MCP_URL": test_url}, clear=True):
            tools = await load_mcp_tools()
            assert tools == mock_tools
            mock_aget.assert_awaited_once()
            # Check that the function was called with correct parameters
            call_args = mock_aget.await_args
            assert call_args[1]["command_or_url"] == test_url.rstrip("/")
            assert call_args[1]["client"].headers == {}

    @pytest.mark.asyncio
    @patch("datarobot_genai.llama_index.mcp.aget_tools_from_mcp_url", new_callable=AsyncMock)
    async def test_load_mcp_tools_with_datarobot_deployment(self, mock_aget):
        """Test loading tools with DataRobot deployment ID."""
        mock_tools = [MagicMock()]
        mock_aget.return_value = mock_tools

        deployment_id = "abc123def456789012345678"
        api_base = "https://app.datarobot.com/api/v2"
        api_key = "test-api-key"

        with patch.dict(
            os.environ,
            {
                "MCP_DEPLOYMENT_ID": deployment_id,
                "DATAROBOT_ENDPOINT": api_base,
                "DATAROBOT_API_TOKEN": api_key,
            },
            clear=True,
        ):
            tools = await load_mcp_tools()
            assert tools == mock_tools
            mock_aget.assert_awaited_once()
            # Check that the function was called with correct parameters
            call_args = mock_aget.await_args
            expected_url = f"{api_base}/deployments/{deployment_id}/directAccess/mcp"
            assert call_args[1]["command_or_url"] == expected_url
            assert call_args[1]["client"].headers["Authorization"] == f"Bearer {api_key}"

    @pytest.mark.asyncio
    @patch("datarobot_genai.llama_index.mcp.aget_tools_from_mcp_url", new_callable=AsyncMock)
    async def test_load_mcp_tools_connection_error(self, mock_aget):
        """Test loading tools handles connection errors gracefully."""
        mock_aget.side_effect = Exception("Connection failed")

        test_url = "https://mcp-server.example.com/mcp"
        with patch.dict(os.environ, {"EXTERNAL_MCP_URL": test_url}, clear=True):
            tools = await load_mcp_tools()
            assert isinstance(tools, list)
            assert len(tools) == 0

    @pytest.mark.asyncio
    @patch("datarobot_genai.llama_index.mcp.aget_tools_from_mcp_url", new_callable=AsyncMock)
    async def test_load_mcp_tools_returns_none(self, mock_aget):
        """Test loading tools when aget_tools_from_mcp_url returns None."""
        mock_aget.return_value = None

        test_url = "https://mcp-server.example.com/mcp"
        with patch.dict(os.environ, {"EXTERNAL_MCP_URL": test_url}, clear=True):
            tools = await load_mcp_tools()
            assert isinstance(tools, list)
            assert len(tools) == 0

    @pytest.mark.asyncio
    @patch("datarobot_genai.llama_index.mcp.aget_tools_from_mcp_url", new_callable=AsyncMock)
    async def test_load_mcp_tools_with_parameters(self, mock_aget):
        """Test loading tools with explicit api_base and api_key parameters."""
        mock_tools = [MagicMock()]
        mock_aget.return_value = mock_tools

        deployment_id = "abc123def456789012345678"
        custom_api_base = "https://custom.datarobot.com/api/v2"
        custom_api_key = "custom-key"

        with patch.dict(
            os.environ,
            {
                "MCP_DEPLOYMENT_ID": deployment_id,
                "DATAROBOT_API_TOKEN": custom_api_key,
                "DATAROBOT_ENDPOINT": custom_api_base,
            },
            clear=True,
        ):
            tools = await load_mcp_tools()
            assert tools == mock_tools
            mock_aget.assert_awaited_once()
            # Check that the function was called with custom parameters
            call_args = mock_aget.await_args
            expected_url = f"{custom_api_base}/deployments/{deployment_id}/directAccess/mcp"
            assert call_args[1]["command_or_url"] == expected_url
            assert call_args[1]["client"].headers["Authorization"] == f"Bearer {custom_api_key}"

    @pytest.mark.asyncio
    @patch("datarobot_genai.llama_index.mcp.aget_tools_from_mcp_url", new_callable=AsyncMock)
    async def test_load_mcp_tools_with_forwarded_headers(self, mock_aget):
        """Test loading tools with forwarded headers including scoped token."""
        mock_tools = [MagicMock()]
        mock_aget.return_value = mock_tools

        deployment_id = "abc123def456789012345678"
        api_base = "https://app.datarobot.com/api/v2"
        api_key = "test-api-key"
        forwarded_headers = {
            "x-datarobot-api-key": "scoped-token-123",
        }

        with patch.dict(
            os.environ,
            {
                "MCP_DEPLOYMENT_ID": deployment_id,
                "DATAROBOT_ENDPOINT": api_base,
                "DATAROBOT_API_TOKEN": api_key,
            },
            clear=True,
        ):
            tools = await load_mcp_tools(forwarded_headers=forwarded_headers)
            assert tools == mock_tools
            mock_aget.assert_awaited_once()
            # Check that forwarded headers are included in client headers
            call_args = mock_aget.await_args
            client_headers = call_args[1]["client"].headers
            assert client_headers["x-datarobot-api-key"] == "scoped-token-123"
            assert client_headers["Authorization"] == f"Bearer {api_key}"
