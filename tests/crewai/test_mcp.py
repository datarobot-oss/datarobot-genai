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
from pydantic import BaseModel

from datarobot_genai.core.mcp import MCPConfig
from datarobot_genai.crewai.mcp import _EmptyArgsSchema
from datarobot_genai.crewai.mcp import _sanitize_tool_schemas
from datarobot_genai.crewai.mcp import mcp_tools_context


@pytest.fixture
def mock_tools():
    return [MagicMock(), MagicMock()]


@pytest.fixture
def mock_adapter(mock_tools):
    """Fixture for mocking MCPServerAdapter."""
    with patch("datarobot_genai.crewai.mcp.MCPServerAdapter") as mock:
        mock_adapter_instance = MagicMock()
        mock_adapter_instance.__enter__.return_value = mock_tools
        mock_adapter_instance.__exit__.return_value = None
        mock.return_value = mock_adapter_instance
        yield mock


@pytest.fixture(autouse=True)
def clear_environment_variables():
    with patch.dict(os.environ, {}, clear=True):
        yield


class TestSanitizeToolSchemas:
    def test_none_args_schema_replaced_with_empty_schema(self):
        tool = MagicMock()
        tool.args_schema = None
        result = _sanitize_tool_schemas([tool])
        assert result[0].args_schema is _EmptyArgsSchema

    def test_existing_args_schema_untouched(self):
        class MySchema(BaseModel):
            value: int

        tool = MagicMock()
        tool.args_schema = MySchema
        result = _sanitize_tool_schemas([tool])
        assert result[0].args_schema is MySchema

    def test_mixed_tools_sanitized_correctly(self):
        class MySchema(BaseModel):
            value: int

        tool_with_schema = MagicMock()
        tool_with_schema.args_schema = MySchema
        tool_without_schema = MagicMock()
        tool_without_schema.args_schema = None
        result = _sanitize_tool_schemas([tool_with_schema, tool_without_schema])
        assert result[0].args_schema is MySchema
        assert result[1].args_schema is _EmptyArgsSchema

    def test_empty_args_schema_produces_valid_openai_parameters(self):
        # Azure OpenAI requires parameters to be type "object"; verify the
        # fallback schema serializes correctly.
        schema = _EmptyArgsSchema.model_json_schema()
        assert schema.get("type") == "object"


class TestMCPToolsContext:
    """Test MCP tools context manager."""

    async def test_mcp_tools_context_no_configuration(self):
        """Test context manager when no MCP server is configured."""
        with patch.dict(os.environ, {}, clear=True):
            mcp_config = MCPConfig()
            async with mcp_tools_context(mcp_config) as tools:
                assert tools == []

    async def test_mcp_tools_context_with_external_url(self, mock_adapter, mock_tools):
        """Test context manager with external MCP URL."""
        test_url = "https://mcp-server.example.com/mcp"
        mcp_config = MCPConfig(external_mcp_url=test_url)
        async with mcp_tools_context(mcp_config) as tools:
            assert tools == mock_tools
            mock_adapter.assert_called_once()
            # Check that the server config was passed correctly
            call_args = mock_adapter.call_args[0][0]
            assert call_args["url"] == test_url
            assert call_args["transport"] == "streamable-http"

    async def test_mcp_tools_context_with_datarobot_deployment(
        self, mock_adapter, agent_auth_context_data, mock_tools
    ):
        """Test context manager with DataRobot deployment ID."""
        deployment_id = "abc123def456789012345678"
        api_base = "https://app.datarobot.com/api/v2"
        api_key = "test-api-key"

        mcp_config = MCPConfig(
            mcp_deployment_id=deployment_id,
            datarobot_endpoint=api_base,
            datarobot_api_token=api_key,
            authorization_context=agent_auth_context_data,
        )
        async with mcp_tools_context(mcp_config) as tools:
            assert tools == mock_tools
            mock_adapter.assert_called_once()
            # Check that the server config was passed correctly
            call_args = mock_adapter.call_args[0][0]
            expected_url = f"{api_base}/deployments/{deployment_id}/directAccess/mcp"
            assert call_args["url"] == expected_url
            assert call_args["transport"] == "streamable-http"
            assert call_args["headers"]["Authorization"] == f"Bearer {api_key}"
            assert call_args["headers"]["X-DataRobot-Authorization-Context"] is not None

    async def test_mcp_tools_context_with_forwarded_headers(
        self, mock_adapter, agent_auth_context_data, mock_tools
    ):
        """Test context manager with forwarded headers including scoped token."""
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
            authorization_context=agent_auth_context_data,
            forwarded_headers=forwarded_headers,
        )
        async with mcp_tools_context(mcp_config) as tools:
            assert tools == mock_tools
            mock_adapter.assert_called_once()
            # Check that forwarded headers are included in the server config
            call_args = mock_adapter.call_args[0][0]
            assert call_args["headers"]["x-datarobot-api-key"] == "scoped-token-123"
            assert call_args["headers"]["Authorization"] == f"Bearer {api_key}"

    @pytest.mark.usefixtures("mock_adapter")
    async def test_mcp_tools_context_propagates_exceptions(self):
        """Test context manager propagates exceptions."""
        test_url = "https://mcp-server.example.com/mcp"
        mcp_config = MCPConfig(external_mcp_url=test_url)
        with pytest.raises(RuntimeError):
            async with mcp_tools_context(mcp_config):
                raise RuntimeError("Connection failed")

    async def test_mcp_tools_context_sanitizes_none_args_schema(self, mock_adapter):
        """Tools with None args_schema get a valid empty schema so 
        Azure OpenAI doesn't reject them."""
        null_schema_tool = MagicMock()
        null_schema_tool.args_schema = None
        with patch("datarobot_genai.crewai.mcp.MCPServerAdapter") as mock:
            instance = MagicMock()
            instance.__enter__.return_value = [null_schema_tool]
            instance.__exit__.return_value = None
            mock.return_value = instance
            test_url = "https://mcp-server.example.com/mcp"
            mcp_config = MCPConfig(external_mcp_url=test_url)
            async with mcp_tools_context(mcp_config) as tools:
                assert len(tools) == 1
                assert tools[0].args_schema is _EmptyArgsSchema

    async def test_mcp_tools_context_preserves_existing_args_schema(self, mock_adapter):
        """Tools that already have a valid args_schema are left untouched."""

        class MySchema(BaseModel):
            query: str

        tool_with_schema = MagicMock()
        tool_with_schema.args_schema = MySchema
        with patch("datarobot_genai.crewai.mcp.MCPServerAdapter") as mock:
            instance = MagicMock()
            instance.__enter__.return_value = [tool_with_schema]
            instance.__exit__.return_value = None
            mock.return_value = instance
            test_url = "https://mcp-server.example.com/mcp"
            mcp_config = MCPConfig(external_mcp_url=test_url)
            async with mcp_tools_context(mcp_config) as tools:
                assert tools[0].args_schema is MySchema

    async def test_mcp_tools_context_connection_error_yields_empty(self):
        """Test graceful fallback when MCP server connection fails."""
        test_url = "https://mcp-server.example.com/mcp"
        mcp_config = MCPConfig(external_mcp_url=test_url)
        with patch(
            "datarobot_genai.crewai.mcp.MCPServerAdapter", side_effect=ConnectionError("refused")
        ):
            async with mcp_tools_context(mcp_config) as tools:
                assert tools == []
