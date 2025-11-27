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
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from datarobot_genai.crewai.mcp import mcp_tools_context


@pytest.fixture
def mock_adapter():
    """Fixture for mocking MCPServerAdapter."""
    with patch("datarobot_genai.crewai.mcp.MCPServerAdapter") as mock:
        yield mock


class TestMCPToolsContext:
    """Test MCP tools context manager."""

    def test_mcp_tools_context_no_configuration(self):
        """Test context manager when no MCP server is configured."""
        with patch.dict(os.environ, {}, clear=True):
            with mcp_tools_context() as tools:
                assert tools == []

    def test_mcp_tools_context_with_external_url(self, mock_adapter):
        """Test context manager with external MCP URL."""
        mock_tools = [MagicMock(), MagicMock()]
        mock_adapter_instance = MagicMock()
        mock_adapter_instance.__enter__.return_value = mock_tools
        mock_adapter_instance.__exit__.return_value = None
        mock_adapter.return_value = mock_adapter_instance

        test_url = "https://mcp-server.example.com/mcp"
        with patch.dict(os.environ, {"EXTERNAL_MCP_URL": test_url}, clear=True):
            with mcp_tools_context() as tools:
                assert tools == mock_tools
                mock_adapter.assert_called_once()
                # Check that the server config was passed correctly
                call_args = mock_adapter.call_args[0][0]
                assert call_args["url"] == test_url
                assert call_args["transport"] == "streamable-http"

    def test_mcp_tools_context_with_datarobot_deployment(
        self, mock_adapter, agent_auth_context_data
    ):
        """Test context manager with DataRobot deployment ID."""
        mock_tools = [MagicMock()]
        mock_adapter_instance = MagicMock()
        mock_adapter_instance.__enter__.return_value = mock_tools
        mock_adapter_instance.__exit__.return_value = None
        mock_adapter.return_value = mock_adapter_instance

        deployment_id = "abc123def456789012345678"
        api_base = "https://app.datarobot.com/api/v2"
        api_key = "test-api-key"
        secret_key = "my-secret-key"

        with patch.dict(
            os.environ,
            {
                "MCP_DEPLOYMENT_ID": deployment_id,
                "DATAROBOT_ENDPOINT": api_base,
                "DATAROBOT_API_TOKEN": api_key,
                "SESSION_SECRET_KEY": secret_key,
            },
            clear=True,
        ):
            with mcp_tools_context(authorization_context=agent_auth_context_data) as tools:
                assert tools == mock_tools
                mock_adapter.assert_called_once()
                # Check that the server config was passed correctly
                call_args = mock_adapter.call_args[0][0]
                expected_url = f"{api_base}/deployments/{deployment_id}/directAccess/mcp"
                assert call_args["url"] == expected_url
                assert call_args["transport"] == "streamable-http"
                assert call_args["headers"]["Authorization"] == f"Bearer {api_key}"
                assert call_args["headers"]["X-DataRobot-Authorization-Context"] is not None

    def test_mcp_tools_context_with_forwarded_headers(self, mock_adapter, agent_auth_context_data):
        """Test context manager with forwarded headers including scoped token."""
        mock_tools = [MagicMock()]
        mock_adapter_instance = MagicMock()
        mock_adapter_instance.__enter__.return_value = mock_tools
        mock_adapter_instance.__exit__.return_value = None
        mock_adapter.return_value = mock_adapter_instance

        deployment_id = "abc123def456789012345678"
        api_base = "https://app.datarobot.com/api/v2"
        api_key = "test-api-key"
        secret_key = "my-secret-key"
        forwarded_headers = {
            "x-datarobot-api-key": "scoped-token-123",
            "x-custom-header": "custom-value",
        }

        with patch.dict(
            os.environ,
            {
                "MCP_DEPLOYMENT_ID": deployment_id,
                "DATAROBOT_ENDPOINT": api_base,
                "DATAROBOT_API_TOKEN": api_key,
                "SESSION_SECRET_KEY": secret_key,
            },
            clear=True,
        ):
            with mcp_tools_context(
                authorization_context=agent_auth_context_data, forwarded_headers=forwarded_headers
            ) as tools:
                assert tools == mock_tools
                mock_adapter.assert_called_once()
                # Check that forwarded headers are included in the server config
                call_args = mock_adapter.call_args[0][0]
                assert call_args["headers"]["x-datarobot-api-key"] == "scoped-token-123"
                assert call_args["headers"]["x-custom-header"] == "custom-value"
                assert call_args["headers"]["Authorization"] == f"Bearer {api_key}"
