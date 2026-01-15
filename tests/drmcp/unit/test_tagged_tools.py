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
from datarobot_genai.drmcp.core.mcp_instance import TaggedFastMCP
from datarobot_genai.drmcp.core.mcp_instance import mcp
from datarobot_genai.drmcp.core.telemetry import _set_otel_attributes
from datarobot_genai.drmcp.core.telemetry import get_trace_id
from datarobot_genai.drmcp.core.telemetry import initialize_telemetry
from datarobot_genai.drmcp.core.tool_filter import filter_tools_by_tags
from datarobot_genai.drmcp.core.tool_filter import get_tool_tags
from datarobot_genai.drmcp.core.tool_filter import get_tools_by_tag
from datarobot_genai.drmcp.core.tool_filter import list_all_tags


@pytest.mark.asyncio
async def test_tagged_tool_decorator() -> None:
    """Test that the mcp.tool decorator properly applies tags."""
    mcp = TaggedFastMCP(name="test")

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
    mcp = TaggedFastMCP(name="test")

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
    mcp = TaggedFastMCP(name="test")

    @mcp.tool()
    def test_function() -> str:
        return "test"

    tools = await mcp._tool_manager.get_tools()
    assert len(tools) == 1

    tool = list(tools.values())[0]
    # Should work fine without annotations
    assert tool.name == "test_function"


def test_filter_tools_by_tags() -> None:
    """Test filtering tools by tags."""
    # Create mock tools with annotations
    tool1 = Mock()
    tool1.annotations = Mock()
    tool1.annotations.tags = ["deployment", "management"]

    tool2 = Mock()
    tool2.annotations = Mock()
    tool2.annotations.tags = ["model", "info"]

    tool3 = Mock()
    tool3.annotations = Mock()
    tool3.annotations.tags = ["deployment", "model"]

    tools = [tool1, tool2, tool3]

    # Test filtering by single tag
    deployment_tools = filter_tools_by_tags(list(tools), ["deployment"])
    assert len(deployment_tools) == 2
    assert tool1 in deployment_tools
    assert tool3 in deployment_tools

    # Test filtering by multiple tags (any match)
    model_tools = filter_tools_by_tags(list(tools), ["model", "management"])
    assert len(model_tools) == 3  # All tools match at least one tag

    # Test filtering by multiple tags (all match)
    deployment_model_tools = filter_tools_by_tags(
        list(tools), ["deployment", "model"], match_all=True
    )
    assert len(deployment_model_tools) == 1
    assert tool3 in deployment_model_tools


def test_get_tool_tags() -> None:
    """Test getting tags from a tool."""
    tool = Mock()
    tool.annotations = Mock()
    tool.annotations.tags = ["deployment", "management"]

    tags = get_tool_tags(tool)
    assert tags == ["deployment", "management"]


def test_get_tool_tags_no_annotations() -> None:
    """Test getting tags from a tool without annotations."""
    tool = Mock()
    tool.annotations = None

    tags = get_tool_tags(tool)
    assert tags == []


def test_list_all_tags() -> None:
    """Test listing all unique tags from tools."""
    tool1 = Mock()
    tool1.annotations = Mock()
    tool1.annotations.tags = ["deployment", "management"]

    tool2 = Mock()
    tool2.annotations = Mock()
    tool2.annotations.tags = ["model", "info"]

    tool3 = Mock()
    tool3.annotations = Mock()
    tool3.annotations.tags = ["deployment", "model"]

    tools = [tool1, tool2, tool3]

    all_tags = list_all_tags(list(tools))
    expected_tags = ["deployment", "info", "management", "model"]
    assert all_tags == expected_tags


def test_get_tools_by_tag() -> None:
    """Test getting tools by a specific tag."""
    tool1 = Mock()
    tool1.annotations = Mock()
    tool1.annotations.tags = ["deployment", "management"]

    tool2 = Mock()
    tool2.annotations = Mock()
    tool2.annotations.tags = ["model", "info"]

    tool3 = Mock()
    tool3.annotations = Mock()
    tool3.annotations.tags = ["deployment", "model"]

    tools = [tool1, tool2, tool3]

    deployment_tools = get_tools_by_tag(list(tools), "deployment")
    assert len(deployment_tools) == 2
    assert tool1 in deployment_tools
    assert tool3 in deployment_tools


@pytest.mark.asyncio
async def test_list_tools_filtering() -> None:
    """Test the enhanced list_tools method with tag filtering."""
    mcp = TaggedFastMCP(name="test")

    @mcp.tool(tags={"data", "read"})
    async def read_file(file_path: str) -> str:
        return f"Reading file: {file_path}"

    @mcp.tool(tags={"data", "write"})
    async def write_file(file_path: str, content: str) -> str:
        return f"Writing to file: {file_path}"

    @mcp.tool(tags={"model", "train"})
    async def train_model(dataset: str) -> str:
        return f"Training model on dataset: {dataset}"

    @mcp.tool(tags={"model", "deployment"})
    async def deploy_model(model_id: str) -> str:
        return f"Deploying model {model_id}"

    # Test getting all tools
    all_tools = await mcp.list_tools()
    assert len(all_tools) == 4

    # Test filtering by single tag
    data_tools = await mcp.list_tools(tags=["data"])
    assert len(data_tools) == 2
    tool_names = [tool.name for tool in data_tools]
    assert "read_file" in tool_names
    assert "write_file" in tool_names

    # Test filtering by multiple tags (OR logic)
    model_tools = await mcp.list_tools(tags=["model"])
    assert len(model_tools) == 2
    tool_names = [tool.name for tool in model_tools]
    assert "train_model" in tool_names
    assert "deploy_model" in tool_names

    # Test filtering by multiple tags (AND logic)
    model_deployment_tools = await mcp.list_tools(tags=["model", "deployment"], match_all=True)
    assert len(model_deployment_tools) == 1
    assert model_deployment_tools[0].name == "deploy_model"

    # Test filtering with no matches
    no_tools = await mcp.list_tools(tags=["nonexistent"])
    assert len(no_tools) == 0


@pytest.mark.asyncio
async def test_get_all_tags() -> None:
    """Test the get_all_tags method."""
    mcp = TaggedFastMCP(name="test")

    @mcp.tool(tags={"data", "read"})
    async def read_file(file_path: str) -> str:
        return f"Reading file: {file_path}"

    @mcp.tool(tags={"data", "write"})
    async def write_file(file_path: str, content: str) -> str:
        return f"Writing to file: {file_path}"

    @mcp.tool(tags={"model", "train"})
    async def train_model(dataset: str) -> str:
        return f"Training model on dataset: {dataset}"

    # Test getting all tags
    all_tags = await mcp.get_all_tags()
    expected_tags = ["data", "model", "read", "train", "write"]
    assert all_tags == expected_tags


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
