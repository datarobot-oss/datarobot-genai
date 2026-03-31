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

from datarobot_genai.core.mcp import MCPConfig
from datarobot_genai.llama_index.mcp import mcp_tools_context


@pytest.fixture(autouse=True)
def empty_agent_auth_context():
    with patch.dict(os.environ, {}, clear=True):
        yield


@pytest.fixture
def mock_aget():
    with patch(
        "datarobot_genai.llama_index.mcp.aget_tools_from_mcp_url", new_callable=AsyncMock
    ) as mock:
        yield mock


class TestLoadMCPTools:
    """Test async MCP tools loading."""

    async def test_mcp_tools_context_no_configuration(self):
        """Test loading tools when no MCP server is configured."""
        with patch.dict(os.environ, {}, clear=True):
            mcp_config = MCPConfig()
            async with mcp_tools_context(mcp_config) as tools:
                assert isinstance(tools, list)
                assert len(tools) == 0

    async def test_mcp_tools_context_with_datarobot_deployment(self, mock_aget):
        """Test loading tools with DataRobot deployment ID."""
        mock_tools = [MagicMock()]
        mock_aget.return_value = mock_tools

        deployment_id = "abc123def456789012345678"
        api_base = "https://app.datarobot.com/api/v2"
        api_key = "test-api-key"

        mcp_config = MCPConfig(
            mcp_deployment_id=deployment_id,
            datarobot_endpoint=api_base,
            datarobot_api_token=api_key,
        )
        async with mcp_tools_context(mcp_config) as tools:
            assert tools == mock_tools
            mock_aget.assert_awaited_once()
            # Check that the function was called with correct parameters
            call_args = mock_aget.await_args
            expected_url = f"{api_base}/deployments/{deployment_id}/directAccess/mcp"
            assert call_args[1]["command_or_url"] == expected_url
            assert call_args[1]["client"].headers["Authorization"] == f"Bearer {api_key}"

    async def test_mcp_tools_context_connection_error(self, mock_aget):
        """Test loading tools handles connection errors gracefully."""
        mock_aget.side_effect = Exception("Connection failed")

        test_url = "https://mcp-server.example.com/mcp"
        mcp_config = MCPConfig(external_mcp_url=test_url)
        async with mcp_tools_context(mcp_config) as tools:
            assert isinstance(tools, list)
            assert len(tools) == 0

    async def test_load_mcp_tools_returns_none(self, mock_aget):
        """Test loading tools when aget_tools_from_mcp_url returns None."""
        mock_aget.return_value = None

        test_url = "https://mcp-server.example.com/mcp"
        mcp_config = MCPConfig(external_mcp_url=test_url)
        async with mcp_tools_context(mcp_config) as tools:
            assert isinstance(tools, list)
            assert len(tools) == 0

    async def test_mcp_tools_context_with_parameters(self, mock_aget):
        """Test loading tools with explicit api_base and api_key parameters."""
        mock_tools = [MagicMock()]
        mock_aget.return_value = mock_tools

        deployment_id = "abc123def456789012345678"
        custom_api_base = "https://custom.datarobot.com/api/v2"
        custom_api_key = "custom-key"

        mcp_config = MCPConfig(
            mcp_deployment_id=deployment_id,
            datarobot_endpoint=custom_api_base,
            datarobot_api_token=custom_api_key,
        )
        async with mcp_tools_context(mcp_config) as tools:
            assert tools == mock_tools
            mock_aget.assert_awaited_once()
            # Check that the function was called with custom parameters
            call_args = mock_aget.await_args
            expected_url = f"{custom_api_base}/deployments/{deployment_id}/directAccess/mcp"
            assert call_args[1]["command_or_url"] == expected_url
            assert call_args[1]["client"].headers["Authorization"] == f"Bearer {custom_api_key}"

    async def test_mcp_tools_context_with_forwarded_headers(self, mock_aget):
        """Test loading tools with forwarded headers including scoped token."""
        mock_tools = [MagicMock()]
        mock_aget.return_value = mock_tools

        deployment_id = "abc123def456789012345678"
        api_base = "https://app.datarobot.com/api/v2"
        api_key = "test-api-key"
        forwarded_headers = {
            "x-datarobot-api-key": "scoped-token-123",
        }

        mcp_config = MCPConfig(
            mcp_deployment_id=deployment_id,
            datarobot_endpoint=api_base,
            datarobot_api_token=api_key,
            forwarded_headers=forwarded_headers,
        )
        async with mcp_tools_context(mcp_config) as tools:
            assert tools == mock_tools
            mock_aget.assert_awaited_once()
            # Check that forwarded headers are included in client headers
            call_args = mock_aget.await_args
            client_headers = call_args[1]["client"].headers
            assert client_headers["x-datarobot-api-key"] == "scoped-token-123"
            assert client_headers["Authorization"] == f"Bearer {api_key}"

    @pytest.mark.usefixtures("mock_aget")
    async def test_mcp_tools_context_exception_is_propagated(self):
        """Test that exceptions are propagated from aget_tools_from_mcp_url."""
        test_url = "https://mcp-server.example.com/mcp"
        mcp_config = MCPConfig(external_mcp_url=test_url)
        with pytest.raises(RuntimeError):
            async with mcp_tools_context(mcp_config):
                raise RuntimeError("Connection failed")
