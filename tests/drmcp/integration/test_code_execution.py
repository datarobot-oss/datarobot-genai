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

"""Integration tests for the execute_code MCP tool."""

import json

import pytest
from mcp.types import TextContent

from datarobot_genai.drmcp.test_utils.mcp_utils_integration import integration_test_mcp_session
from datarobot_genai.drmcp.test_utils.mcp_utils_integration import (
    integration_test_server_params_with_env,
)


def _code_execution_server_params():
    """Return server params with code_execution tools enabled."""
    return integration_test_server_params_with_env({"ENABLE_CODE_EXECUTION_TOOLS": "true"})


@pytest.mark.asyncio
class TestMCPCodeExecutionToolsIntegration:
    """Integration tests for the execute_code tool (uses stub DR client)."""

    async def test_execute_code_tool_registered(self) -> None:
        """Verify execute_code is registered in the MCP session."""
        server_params = _code_execution_server_params()
        async with integration_test_mcp_session(server_params=server_params) as session:
            result = await session.list_tools()
            tool_names = [t.name for t in result.tools]
            assert "execute_code" in tool_names

    async def test_execute_code_noop_sandbox_returns_error(self) -> None:
        """NoopSandbox returns an error and empty stdout/stderr/result."""
        server_params = _code_execution_server_params()
        async with integration_test_mcp_session(server_params=server_params) as session:
            result = await session.call_tool(
                "execute_code",
                {"code": "print('hello')", "session_id": "test-session"},
            )
            assert len(result.content) > 0
            content = result.content[0]
            assert isinstance(content, TextContent)
            data = json.loads(content.text)
            assert "error" in data
            assert data["error"] is not None
            assert "not available" in data["error"].lower() or "planned" in data["error"].lower()
            assert data["stdout"] == ""
            assert data["stderr"] == ""
            assert data["result"] is None

    async def test_execute_code_missing_code_raises_error(self) -> None:
        """Calling execute_code without code should raise a ToolError."""
        server_params = _code_execution_server_params()
        async with integration_test_mcp_session(server_params=server_params) as session:
            result = await session.call_tool(
                "execute_code",
                {},
            )
            assert result.isError
            content = result.content[0]
            assert isinstance(content, TextContent)
            assert "Code must be provided" in content.text

    async def test_execute_code_tool_has_expected_tags(self) -> None:
        """Verify execute_code has the expected tags (code_execution, python, sandbox, daria)."""
        server_params = _code_execution_server_params()
        async with integration_test_mcp_session(server_params=server_params) as session:
            result = await session.list_tools()
            execute_code_tool = next((t for t in result.tools if t.name == "execute_code"), None)
            assert execute_code_tool is not None
            if execute_code_tool.meta:
                fastmcp_meta = execute_code_tool.meta.get("_fastmcp", {})
                tags = set(fastmcp_meta.get("tags", []))
                assert "code_execution" in tags
                assert "daria" in tags
