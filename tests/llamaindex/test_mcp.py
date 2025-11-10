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

from datarobot_genai.core.mcp.common import MCPConfig
from datarobot_genai.llama_index.mcp import load_mcp_tools


class TestMCPConfig:
    """Test MCP configuration management."""

    def test_mcp_config_without_configuration(self):
        """Test MCP config when no environment variables are set."""
        with patch.dict(os.environ, {}, clear=True):
            config = MCPConfig()
            assert config.external_mcp_url is None
            assert config.mcp_deployment_id is None
            assert config.server_config is None

    def test_mcp_config_with_external_url(self):
        """Test MCP config with external URL."""
        test_url = "https://mcp-server.example.com/mcp"
        with patch.dict(os.environ, {"EXTERNAL_MCP_URL": test_url}, clear=True):
            config = MCPConfig()
            assert config.external_mcp_url == test_url
            assert config.server_config is not None
            assert config.server_config["url"] == test_url.rstrip("/")
            assert config.server_config["transport"] == "streamable-http"
            assert config.server_config["headers"] == {}

    def test_mcp_config_with_datarobot_deployment_id(self):
        """Test MCP config with DataRobot deployment ID."""
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
            config = MCPConfig()
            assert config.mcp_deployment_id == deployment_id
            assert config.server_config is not None
            expected_url = f"{api_base}/deployments/{deployment_id}/directAccess/mcp"
            assert config.server_config["url"] == expected_url
            assert config.server_config["transport"] == "streamable-http"
            assert config.server_config["headers"]["Authorization"] == f"Bearer {api_key}"

    def test_mcp_config_with_datarobot_deployment_id_and_bearer_token(self):
        """Test MCP config with DataRobot deployment ID and Bearer token already formatted."""
        deployment_id = "abc123def456789012345678"
        api_base = "https://app.datarobot.com/api/v2"
        api_key = "Bearer test-api-key"

        with patch.dict(
            os.environ,
            {
                "MCP_DEPLOYMENT_ID": deployment_id,
                "DATAROBOT_ENDPOINT": api_base,
                "DATAROBOT_API_TOKEN": api_key,
            },
            clear=True,
        ):
            config = MCPConfig()
            assert config.server_config["headers"]["Authorization"] == api_key

    def test_mcp_config_with_datarobot_deployment_id_no_api_key(self):
        """Test MCP config with DataRobot deployment ID but no API key."""
        deployment_id = "abc123def456789012345678"

        with patch.dict(os.environ, {"MCP_DEPLOYMENT_ID": deployment_id}, clear=True):
            config = MCPConfig()
            assert config.server_config is None

    def test_mcp_config_with_datarobot_deployment_id_no_deployment_id(self):
        """Test MCP config with API key but no deployment ID."""
        api_key = "test-api-key"

        with patch.dict(os.environ, {"DATAROBOT_API_TOKEN": api_key}, clear=True):
            config = MCPConfig()
            assert config.server_config is None

    def test_mcp_config_url_construction_with_trailing_slash(self):
        """Test URL construction when api_base has trailing slash."""
        deployment_id = "abc123def456789012345678"
        api_base = "https://app.datarobot.com/api/v2/"
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
            config = MCPConfig()
            expected_url = "https://app.datarobot.com/api/v2/deployments/abc123def456789012345678/directAccess/mcp"
            assert config.server_config["url"] == expected_url

    def test_mcp_config_priority_external_over_deployment(self):
        """Test that EXTERNAL_MCP_URL takes priority over MCP_DEPLOYMENT_ID."""
        external_url = "https://external-mcp.com/mcp"
        deployment_id = "abc123def456789012345678"
        api_key = "test-api-key"

        with patch.dict(
            os.environ,
            {
                "EXTERNAL_MCP_URL": external_url,
                "MCP_DEPLOYMENT_ID": deployment_id,
                "DATAROBOT_API_TOKEN": api_key,
            },
            clear=True,
        ):
            config = MCPConfig()
            assert config.server_config["url"] == external_url.rstrip("/")
            assert config.server_config["headers"] == {}

    def test_mcp_config_with_explicit_parameters(self):
        """Test MCP config with explicit api_base and api_key parameters."""
        deployment_id = "abc123def456789012345678"
        custom_api_base = "https://custom.datarobot.com/api/v2"
        custom_api_key = "custom-key"

        with patch.dict(os.environ, {}, clear=True):
            config = MCPConfig(api_base=custom_api_base, api_key=custom_api_key)
            # Without MCP_DEPLOYMENT_ID, should return None
            assert config.server_config is None

        with patch.dict(os.environ, {"MCP_DEPLOYMENT_ID": deployment_id}, clear=True):
            config = MCPConfig(api_base=custom_api_base, api_key=custom_api_key)
            assert config.server_config is not None
            expected_url = f"{custom_api_base}/deployments/{deployment_id}/directAccess/mcp"
            assert config.server_config["url"] == expected_url
            assert config.server_config["headers"]["Authorization"] == f"Bearer {custom_api_key}"


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
            assert call_args[1]["url"] == test_url.rstrip("/")
            assert call_args[1]["transport"] == "streamable-http"
            # headers should always be present, empty dict when no custom headers
            assert call_args[1]["headers"] == {}

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
            assert call_args[1]["url"] == expected_url
            assert call_args[1]["transport"] == "streamable-http"
            assert call_args[1]["headers"]["Authorization"] == f"Bearer {api_key}"

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

        with patch.dict(os.environ, {"MCP_DEPLOYMENT_ID": deployment_id}, clear=True):
            tools = await load_mcp_tools(api_base=custom_api_base, api_key=custom_api_key)
            assert tools == mock_tools
            mock_aget.assert_awaited_once()
            # Check that the function was called with custom parameters
            call_args = mock_aget.await_args
            expected_url = f"{custom_api_base}/deployments/{deployment_id}/directAccess/mcp"
            assert call_args[1]["url"] == expected_url
            assert call_args[1]["headers"]["Authorization"] == f"Bearer {custom_api_key}"
