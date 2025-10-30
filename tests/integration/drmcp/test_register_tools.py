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
import json
from typing import Any
from typing import cast

import pytest
from fastmcp.tools import FunctionTool
from mcp.server.fastmcp import Context
from mcp.server.session import ServerSession
from mcp.types import TextContent

from datarobot_genai.drmcp.core.mcp_instance import mcp
from datarobot_genai.drmcp.core.mcp_instance import register_tools


@pytest.mark.asyncio
class TestMCPRegisterToolsIntegration:
    """Integration tests for tool registration.
    These tests are semi-integration because they use the global MCP instance instead of a session.
    The session is not initialized in this test because its a client session that uses
    stdio protocol.
    The register_tools is a server side function that uses the global MCP instance.
    """

    async def test_register_tools_basic(self) -> None:
        """Test basic tool registration with name and description."""
        tool_output = {"foo_arg": 1, "bar_arg": 2}

        # Define a simple tool
        async def test_tool() -> dict[str, int]:
            return tool_output

        initial_tools = await mcp.list_tools()
        assert all(tool.name != "test_tool" for tool in initial_tools)

        # Register the tool
        await register_tools(
            test_tool,
            name="test_tool",
            description="A test tool",
        )

        # Verify tool is registered
        tool = await mcp.get_tool("test_tool")
        assert isinstance(tool, FunctionTool)
        tool_result = await tool.run({})
        # Assert the tool returns structured output
        assert tool_result.structured_content == tool_output
        # Assert the tool returns unstructured output
        assert json.loads(cast(TextContent, tool_result.content[0]).text) == tool_output

    async def test_register_tools_with_tags(self) -> None:
        """Test tool registration with tags and annotations."""

        # Define a tool with tags
        async def tagged_tool(ctx: Context[ServerSession, dict[str, Any]]) -> str:
            return "tagged_response"

        initial_tools = await mcp.list_tools()
        assert all(tool.name != "tagged_tool" for tool in initial_tools)

        test_tags = {"test", "integration"}
        # Register the tool
        await register_tools(
            tagged_tool,
            name="tagged_tool",
            tags=test_tags,
        )

        # Verify tool is registered with correct tags
        tools = await mcp.list_tools()
        assert any(tool.name == "tagged_tool" for tool in tools)

        # Verify tool annotations
        registered_tool = next(tool for tool in tools if tool.name == "tagged_tool")
        assert registered_tool.annotations is not None
        annotations_dict = registered_tool.annotations.model_dump()
        assert annotations_dict.get("tags") == test_tags
