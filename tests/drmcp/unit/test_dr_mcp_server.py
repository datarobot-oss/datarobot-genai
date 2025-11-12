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
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from fastmcp import FastMCP

from datarobot_genai.drmcp.core.dr_mcp_server import DataRobotMCPServer
from datarobot_genai.drmcp.core.dynamic_tools.deployment.adapters.default import Metadata
from datarobot_genai.drmcp.core.mcp_instance import mcp


@pytest.fixture
def mock_mcp() -> MagicMock:
    """Create a mock FastMCP instance."""
    mock = MagicMock(spec=FastMCP)
    mock._mcp_list_tools = AsyncMock(
        return_value=[MagicMock(name="tool1"), MagicMock(name="tool2")]
    )
    return mock


@pytest.fixture
def mock_config() -> MagicMock:
    """Create a mock configuration."""
    mock = MagicMock()
    mock.has.return_value = True
    mock.mcp_server_register_dynamic_tools_on_startup = False
    mock.app_log_level = "INFO"
    return mock


@pytest.fixture
def metadata_factory():
    def _factory(name, description):
        return Metadata(
            {
                "name": name,
                "description": description,
                "method": "POST",
                "endpoint": "/predict",
                "headers": {},
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "json": {
                            "type": "object",
                            "properties": {"foo": {"type": "string"}},
                            "required": ["foo"],
                        }
                    },
                    "required": ["json"],
                },
                "tags": {"deployment", "datarobot"},
            }
        )

    return _factory


class TestDataRobotMCPServer:
    """Test suite for DataRobotMCPServer class."""

    def test_initialization(self, mock_mcp: MagicMock) -> None:
        """Test server initialization with default transport."""
        server = DataRobotMCPServer(mock_mcp)
        assert server._mcp == mock_mcp
        assert server._mcp_transport == "streamable-http"

    def test_initialization_stdio_transport(self, mock_mcp: MagicMock) -> None:
        """Test server initialization with stdio transport."""
        server = DataRobotMCPServer(mock_mcp, transport="stdio")
        assert server._mcp == mock_mcp
        assert server._mcp_transport == "stdio"

    @patch("datarobot_genai.drmcp.core.dr_mcp_server.get_config")
    @patch(
        "datarobot_genai.drmcp.core.dynamic_tools.deployment.register.get_datarobot_tool_deployments"
    )
    @patch("datarobot_genai.drmcp.core.dynamic_tools.deployment.register.dr.Deployment.get")
    @patch("datarobot_genai.drmcp.core.dynamic_tools.deployment.config.get_api_client")
    @patch("datarobot_genai.drmcp.core.dynamic_tools.deployment.config.get_mcp_tool_metadata")
    def test_run_with_dynamic_tool_registration(
        self,
        mock_get_mcp_tool_metadata,
        mock_get_api_client,
        mock_deployment_get,
        mock_get_mcp_tool_deployments,
        mock_get_config,
        mock_config,
        mock_mcp,
        metadata_factory,
    ):
        # setup config mock
        mock_config.mcp_server_register_dynamic_tools_on_startup = True
        mock_config.mcp_server_register_dynamic_prompts_on_startup = False
        mock_get_config.return_value = mock_config
        mock_get_api_client.return_value = MagicMock(
            endpoint="https://test.datarobot.com/api/v2/",
            token="fake-test-token",
        )

        # setup mock deployment objects
        mock_deployment1 = MagicMock(
            id="id1",
            label="tool3",
            description="foo - tool",
            prediction_environment={"platform": "datarobotServerless"},
        )
        mock_deployment2 = MagicMock(
            id="id2",
            label="tool4",
            description="foo bar baz - tool",
            default_prediction_server={
                "url": "https://test.datarobot.com/predApi/v1/",
                "datarobot-key": "fake-test-datarobot-key",
            },
        )

        mock_get_mcp_tool_deployments.return_value = ["id1", "id2"]
        mock_deployment_get.side_effect = [mock_deployment1, mock_deployment2]
        mock_get_mcp_tool_metadata.side_effect = [
            metadata_factory("tool3", "foo - tool"),
            metadata_factory("tool4", "foo bar baz - tool"),
        ]

        server = DataRobotMCPServer(mock_mcp, transport="stdio")
        server.run()

        registered_tools = list(mcp._tool_manager._tools)

        assert "tool3" in registered_tools, f"tool3 missing in tools: {registered_tools}"
        assert "tool4" in registered_tools, f"tool4 missing in tools: {registered_tools}"
        # Assert mocks called as expected
        mock_deployment_get.assert_any_call("id1")
        mock_deployment_get.assert_any_call("id2")
        assert mock_get_mcp_tool_metadata.call_count == 2

    @patch("datarobot_genai.drmcp.core.dr_mcp_server.get_credentials")
    def test_run_missing_config(self, mock_get_credentials: MagicMock, mock_mcp: MagicMock) -> None:
        """Test server run with missing configuration."""
        mock_creds = MagicMock()
        mock_creds.has_datarobot_credentials.return_value = False
        mock_get_credentials.return_value = mock_creds

        server = DataRobotMCPServer(mock_mcp)
        with pytest.raises(ValueError, match="Missing required DataRobot credentials"):
            server.run()

    @patch("datarobot_genai.drmcp.core.dr_mcp_server.get_config")
    @patch("datarobot_genai.drmcp.core.dr_mcp_server.asyncio")
    @patch("datarobot_genai.drmcp.core.dr_mcp_server.get_credentials")
    def test_run_success(
        self,
        mock_get_credentials: MagicMock,
        mock_asyncio: MagicMock,
        mock_get_config: MagicMock,
        mock_mcp: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        """Test successful server run."""
        mock_get_config.return_value = mock_config
        mock_creds = MagicMock()
        mock_creds.has_datarobot_credentials.return_value = True
        mock_get_credentials.return_value = mock_creds

        # Mock asyncio methods
        mock_loop = MagicMock()
        mock_asyncio.new_event_loop.return_value = mock_loop
        mock_mcp.run_streamable_http_async = AsyncMock()

        server = DataRobotMCPServer(mock_mcp, transport="stdio")
        server.run()

        # Verify event loop was created and used
        mock_asyncio.new_event_loop.assert_called_once()
        mock_asyncio.set_event_loop.assert_called_once_with(mock_loop)
        # Should be called twice: once for run_server, once for pre_server_shutdown
        assert mock_loop.run_until_complete.call_count == 2

        # Verify tools were listed
        mock_mcp._mcp_list_tools.assert_called_once()

    @patch("datarobot_genai.drmcp.core.dr_mcp_server.get_config")
    @patch("datarobot_genai.drmcp.core.dr_mcp_server.asyncio")
    @patch("datarobot_genai.drmcp.core.dr_mcp_server.get_credentials")
    def test_run_server_error(
        self,
        mock_get_credentials: MagicMock,
        mock_asyncio: MagicMock,
        mock_get_config: MagicMock,
        mock_mcp: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        """Test server run with MCP error."""
        mock_get_config.return_value = mock_config
        mock_creds = MagicMock()
        mock_creds.has_datarobot_credentials.return_value = True
        mock_get_credentials.return_value = mock_creds

        # Mock asyncio methods
        mock_loop = MagicMock()
        mock_asyncio.new_event_loop.return_value = mock_loop

        # Set up the loop.run_until_complete to raise the exception
        mock_loop.run_until_complete.side_effect = Exception("Server failed to start")

        server = DataRobotMCPServer(mock_mcp)
        with pytest.raises(Exception, match="Server failed to start"):
            server.run()

    @patch("datarobot_genai.drmcp.core.dr_mcp_server.get_config")
    @patch("datarobot_genai.drmcp.core.dr_mcp_server.asyncio")
    @patch("datarobot_genai.drmcp.core.dr_mcp_server.get_credentials")
    def test_run_lists_tools(
        self,
        mock_get_credentials: MagicMock,
        mock_asyncio: MagicMock,
        mock_get_config: MagicMock,
        mock_mcp: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        """Test that tools are listed before server start."""
        mock_get_config.return_value = mock_config
        mock_creds = MagicMock()
        mock_creds.has_datarobot_credentials.return_value = True
        mock_get_credentials.return_value = mock_creds

        # Mock asyncio methods
        mock_loop = MagicMock()
        mock_asyncio.new_event_loop.return_value = mock_loop
        mock_mcp.run_streamable_http_async = AsyncMock()

        mock_tools = [MagicMock(name="tool1"), MagicMock(name="tool2")]
        mock_mcp._mcp_list_tools = AsyncMock(return_value=mock_tools)

        server = DataRobotMCPServer(mock_mcp)
        server.run()

        # Verify tools were listed before server start
        mock_mcp._mcp_list_tools.assert_called_once()
        # Should be called twice: once for run_server, once for pre_server_shutdown
        assert mock_loop.run_until_complete.call_count == 2
