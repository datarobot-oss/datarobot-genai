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

from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import patch

from nat.builder.workflow_builder import WorkflowBuilder
from nat.plugins.mcp.client.client_base import MCPBaseClient
from nat.plugins.mcp.client.client_impl import MCPFunctionGroup
from pydantic import BaseModel

from datarobot_genai.nat.datarobot_mcp_client import DataRobotMCPClientConfig
from datarobot_genai.nat.datarobot_mcp_client import DataRobotMCPServerConfig


class _InputSchema(BaseModel):
    """Input schema for fake tools used in testing."""

    param: str


class _FakeTool:
    """Fake tool class for testing MCP tool functionality."""

    def __init__(self, name: str, description: str = "desc") -> None:
        self.name = name
        self.description = description
        self.input_schema = _InputSchema

    async def acall(self, args: dict[str, Any]) -> str:
        """Simulate tool execution by returning a formatted response."""
        return f"ok {args['param']}"

    def set_description(self, description: str) -> None:
        """Allow description to be updated for testing purposes."""
        if description is not None:
            self.description = description


class _FakeMCPClient(MCPBaseClient):
    """Fake MCP client for testing client-server interactions."""

    def __init__(
        self,
        *,
        tools: dict[str, _FakeTool],
        url: str | None = None,
    ) -> None:
        super().__init__("streamable-http")
        self._tools = tools
        self.url = url

    async def get_tool(self, name: str) -> _FakeTool:
        """Retrieve a tool by name."""
        return self._tools[name]

    async def get_tools(self) -> dict[str, _FakeTool]:
        """Retrieve all tools."""
        return self._tools

    @asynccontextmanager
    async def connect_to_server(self):
        """Support async context manager for testing."""
        yield self


async def test_datarobot_mcp_client():
    with patch(
        "datarobot_genai.nat.datarobot_mcp_client.DataRobotMCPStreamableHTTPClient"
    ) as mock_client:
        fake_tools = {"a": _FakeTool("a", "da"), "b": _FakeTool("b", "db")}

        def make_fake_client(url, *args, **kwargs):
            return _FakeMCPClient(tools=fake_tools, url=url)

        mock_client.side_effect = make_fake_client
        server_config = DataRobotMCPServerConfig()
        config = DataRobotMCPClientConfig(server=server_config)
        async with WorkflowBuilder() as builder:
            await builder.add_function_group("datarobot_mcp_tools", config)
            function_group = await builder.get_function_group("datarobot_mcp_tools")
            assert isinstance(function_group, MCPFunctionGroup)
            # Verify the happy path: fake client was used and tools were registered
            all_functions = await function_group.get_all_functions()
            # Function names are prefixed with the group name (e.g. datarobot_mcp_tools__a)
            assert "datarobot_mcp_tools__a" in all_functions
            assert "datarobot_mcp_tools__b" in all_functions
