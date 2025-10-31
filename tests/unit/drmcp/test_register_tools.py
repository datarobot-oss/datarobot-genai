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

import base64
from collections.abc import Generator
from typing import NamedTuple
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from mcp.server.fastmcp import Context
from mcp.shared.context import RequestContext
from mcp.types import Tool

# Add import for the function we're testing
from datarobot_genai.drmcp.core.mcp_instance import register_tools
from datarobot_genai.drmcp.core.utils import format_response_as_tool_result


class FormatResponseTestCase(NamedTuple):
    """Test case for _format_response_as_tool_result function."""

    data: bytes
    content_type: str
    charset: str
    expected_type: str
    expected_data_key: str


class CharsetTestCase(NamedTuple):
    """Test case for charset encoding in _format_response_as_tool_result."""

    data: bytes
    charset: str
    expected_decoded: str


@pytest.fixture
def mock_context() -> Context[MagicMock, MagicMock, MagicMock]:
    """Create a mock Context with headers."""
    mock_request = MagicMock()
    mock_request.headers = {"x-agent-id": "test-agent-123"}
    ctx = Context(
        request_context=RequestContext(
            request=mock_request,
            request_id="req-123",
            meta=None,
            session=MagicMock(),
            lifespan_context=MagicMock(),
        )
    )
    return ctx


@pytest.fixture
def mock_memory_manager() -> MagicMock:
    """Create a mock MemoryManager."""
    mock = MagicMock()
    mock.get_active_storage_id_for_agent = AsyncMock(return_value="test-storage-456")
    return mock


@pytest.fixture
def mock_mcp() -> Generator[MagicMock, None, None]:
    """Set up mock MCP to handle tool registration verification.

    Args:
        mock_mcp: The MCP mock to setup
        tool_name: The name of the tool that will be registered
    """
    registered_tools: list[Tool] = []

    with patch("datarobot_genai.drmcp.core.mcp_instance.mcp") as mock_mcp:

        def add_tool(tool: Tool) -> Tool:
            """Mock add_tool that tracks the registered tool."""
            registered_tools.append(tool)
            return tool

        async def list_tools() -> list[Tool]:
            """Mock list_tools that returns registered tools."""
            return registered_tools

        mock_mcp.add_tool = MagicMock(side_effect=add_tool)
        mock_mcp.list_tools = AsyncMock(side_effect=list_tools)
        yield mock_mcp


@pytest.mark.asyncio
@patch("datarobot_genai.drmcp.core.mcp_instance.MemoryManager")
@patch("datarobot_genai.drmcp.core.mcp_instance.get_memory_manager")
async def test_register_tools_basic(
    mock_get_memory_manager: MagicMock,
    mock_memory_manager_cls: MagicMock,
    mock_mcp: MagicMock,
) -> None:
    """Test basic tool registration without tags or memory management."""

    # Setup
    async def dummy_tool() -> None:
        pass

    # Execute
    await register_tools(dummy_tool, name="test_tool", description="Test tool description")

    # Verify
    mock_mcp.add_tool.assert_called_once()
    registered_tool = mock_mcp.add_tool.call_args[0][0]
    assert registered_tool.name == "test_tool"
    assert registered_tool.description == "Test tool description"


@pytest.mark.asyncio
async def test_register_tools_with_tags(mock_mcp: MagicMock) -> None:
    """Test tool registration with tags."""

    # Setup
    async def dummy_tool() -> None:
        pass

    test_tags = {"tag1", "tag2"}

    # Execute
    await register_tools(dummy_tool, name="test_tool", tags=test_tags)

    # Verify
    mock_mcp.add_tool.assert_called_once()
    registered_tool = mock_mcp.add_tool.call_args[0][0]
    assert registered_tool.name == "test_tool"
    assert registered_tool.tags == test_tags
    assert registered_tool.annotations.tags == test_tags


@pytest.mark.asyncio
@patch("datarobot_genai.drmcp.core.mcp_instance.MemoryManager")
@patch("datarobot_genai.drmcp.core.mcp_instance.get_memory_manager")
async def test_register_tools_with_memory_management(
    mock_get_memory_manager: MagicMock,
    mock_memory_manager_cls: MagicMock,
    mock_mcp: MagicMock,
    mock_context: MagicMock,
    mock_memory_manager: MagicMock,
) -> None:
    """Test tool registration with memory management."""
    # Setup
    mock_memory_manager_cls.is_initialized = MagicMock(return_value=True)
    mock_get_memory_manager.return_value = mock_memory_manager

    # Create a tool that expects memory management args
    async def dummy_tool(
        ctx: Context[MagicMock, MagicMock, MagicMock],
        agent_id: str | None = None,
        storage_id: str | None = None,
    ) -> dict[str, str | None]:
        return {"agent_id": agent_id, "storage_id": storage_id}

    # Execute
    await register_tools(dummy_tool, name="test_tool")

    # Get the wrapped function from add_tool call
    registered_tool = mock_mcp.add_tool.call_args[0][0]

    # Call the wrapped function with a context
    result = await registered_tool.fn(ctx=mock_context)

    # Verify memory IDs were properly injected
    assert result == {"agent_id": "test-agent-123", "storage_id": "test-storage-456"}


class TestFormatResponseAsToolResult:
    @pytest.mark.parametrize(
        "test_case",
        [
            pytest.param(
                FormatResponseTestCase(
                    data=b"Hello, world!",
                    content_type="text/plain",
                    charset="utf-8",
                    expected_type="text",
                    expected_data_key="data",
                ),
                id="text/plain",
            ),
            pytest.param(
                FormatResponseTestCase(
                    data=b'{"key": "value", "number": 42}',
                    content_type="application/json",
                    charset="utf-8",
                    expected_type="text",
                    expected_data_key="data",
                ),
                id="application/json",
            ),
            pytest.param(
                FormatResponseTestCase(
                    data=b"<html><body>Test</body></html>",
                    content_type="text/html",
                    charset="utf-8",
                    expected_type="text",
                    expected_data_key="data",
                ),
                id="text/html",
            ),
            pytest.param(
                FormatResponseTestCase(
                    data=b"name,age,city\nAlice,30,NYC\nBob,25,LA\nCharlie,35,SF",
                    content_type="text/csv",
                    charset="utf-8",
                    expected_type="text",
                    expected_data_key="data",
                ),
                id="text/csv-dataframe",
            ),
            pytest.param(
                FormatResponseTestCase(
                    data=b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR",
                    content_type="image/png",
                    charset="utf-8",
                    expected_type="image",
                    expected_data_key="data_base64",
                ),
                id="image/png",
            ),
            pytest.param(
                FormatResponseTestCase(
                    data=b"\xff\xd8\xff\xe0\x00\x10JFIF",
                    content_type="image/jpeg",
                    charset="utf-8",
                    expected_type="image",
                    expected_data_key="data_base64",
                ),
                id="image/jpeg",
            ),
            pytest.param(
                FormatResponseTestCase(
                    data=b"%PDF-1.7\n%\xc2\xa5\xc2\xb1\xc3\xab\n\n1 0 obj\n<< /Type /Catalog",
                    content_type="application/pdf",
                    charset="utf-8",
                    expected_type="binary",
                    expected_data_key="data_base64",
                ),
                id="application/pdf-report",
            ),
            pytest.param(
                FormatResponseTestCase(
                    data=b"\x00\x01\x02\x03\x04\x05",
                    content_type="application/octet-stream",
                    charset="utf-8",
                    expected_type="binary",
                    expected_data_key="data_base64",
                ),
                id="application/octet-stream",
            ),
        ],
    )
    def test_format_response_by_content_type(self, test_case: FormatResponseTestCase) -> None:
        """Test formatting responses for various content types."""
        result = format_response_as_tool_result(
            test_case.data, test_case.content_type, test_case.charset
        )
        assert result.structured_content["type"] == test_case.expected_type
        assert result.structured_content["mime_type"] == test_case.content_type

        if test_case.expected_data_key == "data":
            assert result.structured_content["data"] == test_case.data.decode(test_case.charset)
            assert "data_base64" not in result.structured_content
        else:
            assert result.structured_content["data_base64"] == base64.b64encode(
                test_case.data
            ).decode(test_case.charset)
            assert "data" not in result.structured_content

    def test_format_response_missing_content_type(self) -> None:
        """Test formatting response when Content-Type header is missing."""
        data = b"Some data"
        content_type = ""
        charset = "utf-8"
        result = format_response_as_tool_result(data, content_type, charset)
        assert result.structured_content["type"] == "binary"
        assert result.structured_content["mime_type"] == ""
        assert result.structured_content["data_base64"] == base64.b64encode(data).decode(charset)

    @pytest.mark.parametrize(
        "test_case",
        [
            pytest.param(
                CharsetTestCase(
                    data="Héllo Wörld".encode("latin-1"),
                    charset="latin-1",
                    expected_decoded="Héllo Wörld",
                ),
                id="latin-1-encoding",
            ),
            pytest.param(
                CharsetTestCase(
                    data="Hello 世界".encode(),
                    charset="utf-8",
                    expected_decoded="Hello 世界",
                ),
                id="utf-8-encoding",
            ),
        ],
    )
    def test_format_response_with_charset(self, test_case: CharsetTestCase) -> None:
        """Test formatting response with different character encodings."""
        result = format_response_as_tool_result(test_case.data, "text/plain", test_case.charset)
        assert result.structured_content["type"] == "text"
        assert result.structured_content["data"] == test_case.expected_decoded

    def test_format_response_default_charset(self) -> None:
        """Test that default charset is utf-8 when not specified."""
        data = "Hello 世界".encode()
        content_type = "text/plain"
        charset = "utf-8"
        result = format_response_as_tool_result(data, content_type, charset)
        assert result.structured_content["type"] == "text"
        assert result.structured_content["data"] == "Hello 世界"

    def test_format_response_case_insensitive_content_type(self) -> None:
        """Test that Content-Type matching is case-insensitive."""
        data = b"Test data"
        content_type = "TEXT/PLAIN"
        charset = "utf-8"
        result = format_response_as_tool_result(data, content_type, charset)
        assert result.structured_content["type"] == "text"
        assert result.structured_content["mime_type"] == "text/plain"

    def test_format_empty_response(self) -> None:
        """Test formatting empty response."""
        data = b""
        content_type = "text/plain"
        charset = "utf-8"
        result = format_response_as_tool_result(data, content_type, charset)
        assert result.structured_content["type"] == "text"
        assert result.structured_content["data"] == ""
