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

# Sequence import removed as it's not used
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from datarobot_genai.drmcp.core.dr_mcp_server import DataRobotMCPServer
from datarobot_genai.drmcp.core.mcp_instance import DataRobotMCP
from datarobot_genai.drmcp.core.mcp_instance import mcp
from datarobot_genai.drmcp.core.telemetry import _set_otel_attributes
from datarobot_genai.drmcp.core.telemetry import get_trace_id
from datarobot_genai.drmcp.core.telemetry import initialize_telemetry


@pytest.mark.asyncio
async def test_tagged_tool_decorator() -> None:
    """Test that the mcp.tool decorator properly applies tags."""
    mcp = DataRobotMCP(name="test")

    @mcp.tool(tags={"test", "example"})
    def test_function() -> str:
        return "test"

    # Get the tool as exposed via MCP (includes meta)
    tools = await mcp._list_tools_mcp()
    assert len(tools) == 1

    tool = list(tools)[0]
    assert hasattr(tool, "meta")
    assert tool.meta is not None
    assert tool.meta.get("_fastmcp", {}).get("tags") == ["example", "test"]


@pytest.mark.asyncio
async def test_tagged_tool_with_additional_annotations() -> None:
    """Test that mcp.tool can handle additional annotations."""
    mcp = DataRobotMCP(name="test")

    @mcp.tool(tags={"test"}, annotations={"title": "Test Tool", "readOnlyHint": True})
    def test_function() -> str:
        return "test"

    tools = await mcp._tool_manager.get_tools()
    assert len(tools) == 1

    tool = list(tools.values())[0]
    assert tool.annotations is not None
    assert hasattr(tool.annotations, "title")
    assert tool.annotations.title == "Test Tool"
    assert hasattr(tool.annotations, "readOnlyHint")
    assert tool.annotations.readOnlyHint is True
    assert tool.tags == {"test"}


@pytest.mark.asyncio
async def test_tool_without_tags() -> None:
    """Test that tools work without tags."""
    mcp = DataRobotMCP(name="test")

    @mcp.tool()
    def test_function() -> str:
        return "test"

    tools = await mcp._tool_manager.get_tools()
    assert len(tools) == 1

    tool = list(tools.values())[0]
    # Should work fine without annotations
    assert tool.name == "test_function"


def test_main_module() -> None:
    """Test the main module execution."""
    # Test that we can import and instantiate the server
    server = DataRobotMCPServer(mcp)
    assert server is not None
    assert hasattr(server, "run")


def test_dr_mcp_server_error_handling() -> None:
    """Test DataRobotMCPServer error handling."""
    # Test server initialization with streamable-http transport
    server = DataRobotMCPServer(mcp, transport="streamable-http")
    assert server is not None

    # Test server initialization with different transport
    server = DataRobotMCPServer(mcp, transport="stdio")
    assert server is not None


def test_dr_mcp_server_run_without_credentials() -> None:
    """Test DataRobotMCPServer run method without credentials."""
    # Mock credentials to return False for has_datarobot_credentials
    mock_credentials = Mock()
    mock_credentials.has_datarobot_credentials.return_value = False

    with patch(
        "datarobot_genai.drmcp.core.dr_mcp_server.get_credentials", return_value=mock_credentials
    ):
        server = DataRobotMCPServer(mcp)

        # Should raise ValueError when credentials are missing
        with pytest.raises(ValueError, match="Missing required DataRobot credentials"):
            server.run()


def test_telemetry_functions() -> None:
    """Test telemetry functions."""
    # Mock the current span and context
    mock_span = Mock()
    mock_context = Mock()
    mock_context.is_valid = True
    mock_context.trace_id = 123456  # Example trace ID
    mock_span.get_span_context.return_value = mock_context

    expected_trace_id = "0000000000000000000000000001e240"  # Hex format of 123456

    with (
        patch("opentelemetry.trace.get_current_span", return_value=mock_span),
    ):
        # Test get_trace_id when span exists
        trace_id = get_trace_id()
        assert trace_id is not None
        assert trace_id == expected_trace_id

    # Test set_otel_attributes
    mock_span = Mock()
    attributes = {
        "simple": "value",
        "nested": {"key": "value"},
        "number": 42,
        "boolean": True,
    }

    _set_otel_attributes(mock_span, attributes)

    # Check that set_attribute was called for flattened attributes
    expected_calls = {
        "simple": "value",
        "nested.key": "value",
        "number": 42,
        "boolean": True,
    }
    mock_span.set_attributes.assert_called_with(expected_calls)


def test_telemetry_initialization() -> None:
    """Test telemetry initialization."""
    mcp_mock = Mock()
    # Test with telemetry disabled
    with patch("datarobot_genai.drmcp.core.telemetry.get_config") as mock_config:
        mock_config.return_value.otel_enabled = False
        result = initialize_telemetry(mcp_mock)
        assert result is None

    # Test with telemetry enabled
    with patch("datarobot_genai.drmcp.core.telemetry.get_config") as mock_config:
        mock_config.return_value.otel_enabled = True
        mock_config.return_value.mcp_server_name = "test-app"

        with patch("datarobot_genai.drmcp.core.telemetry._setup_otel_env_variables"):
            with patch("datarobot_genai.drmcp.core.telemetry._setup_otel_exporter"):
                with patch("datarobot_genai.drmcp.core.telemetry._setup_http_instrumentors"):
                    with patch(
                        "datarobot_genai.drmcp.core.telemetry.trace.get_tracer"
                    ) as mock_tracer:
                        mock_span = Mock()
                        mock_tracer.return_value.start_span.return_value = mock_span

                        initialize_telemetry(mcp_mock)
                        assert mcp_mock.add_middleware.called
