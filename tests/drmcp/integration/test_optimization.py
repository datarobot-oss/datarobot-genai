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

"""Integration tests for the cuopt_solve MCP tool."""

import json

import pytest
from mcp.client.stdio import StdioServerParameters
from mcp.types import TextContent

from datarobot_genai.drmcp.test_utils.mcp_utils_integration import (
    integration_test_mcp_server_params,
    integration_test_mcp_session,
)

# A minimal LP problem definition for testing
_SIMPLE_LP_PROBLEM = {
    "type": "lp",
    "objective": {"minimize": ["x", "y"], "coefficients": [1.0, 2.0]},
    "constraints": [{"lhs": ["x", "y"], "coefficients": [1.0, 1.0], "rhs": 1.0, "sense": ">="}],
    "bounds": {"x": [0, None], "y": [0, None]},
}

# Deployment ID used in stubs (any non-empty string works with stub_post)
_STUB_CUOPT_DEPLOYMENT_ID = "stub_cuopt_deployment_id"


def _optimization_server_params(cuopt_deployment_id: str | None = None) -> StdioServerParameters:
    """Return server params with optimization tools enabled."""
    params = integration_test_mcp_server_params(use_stub=True)
    env = dict(params.env or {})
    env["ENABLE_OPTIMIZATION_TOOLS"] = "true"
    if cuopt_deployment_id:
        env["CUOPT_DEPLOYMENT_ID"] = cuopt_deployment_id
    return StdioServerParameters(
        command=params.command,
        args=params.args,
        env=env,
    )


@pytest.mark.asyncio
class TestMCPOptimizationToolsIntegration:
    """Integration tests for the cuopt_solve tool (uses stub DR client)."""

    async def test_cuopt_solve_tool_registered(self) -> None:
        """Verify cuopt_solve is registered in the MCP session."""
        server_params = _optimization_server_params()
        async with integration_test_mcp_session(server_params=server_params) as session:
            result = await session.list_tools()
            tool_names = [t.name for t in result.tools]
            assert "cuopt_solve" in tool_names

    async def test_cuopt_solve_missing_deployment_id_raises_error(self) -> None:
        """cuopt_solve without CUOPT_DEPLOYMENT_ID raises a ToolError."""
        # Don't set CUOPT_DEPLOYMENT_ID in env
        server_params = _optimization_server_params(cuopt_deployment_id=None)
        async with integration_test_mcp_session(server_params=server_params) as session:
            result = await session.call_tool(
                "cuopt_solve",
                {"problem_definition": _SIMPLE_LP_PROBLEM},
            )
            assert result.isError
            content = result.content[0]
            assert isinstance(content, TextContent)
            assert "CUOPT_DEPLOYMENT_ID" in content.text

    async def test_cuopt_solve_missing_problem_definition_raises_error(self) -> None:
        """cuopt_solve without problem_definition raises a ToolError."""
        server_params = _optimization_server_params(
            cuopt_deployment_id=_STUB_CUOPT_DEPLOYMENT_ID
        )
        async with integration_test_mcp_session(server_params=server_params) as session:
            result = await session.call_tool(
                "cuopt_solve",
                {},
            )
            assert result.isError
            content = result.content[0]
            assert isinstance(content, TextContent)
            assert "Problem definition must be provided" in content.text

    async def test_cuopt_solve_returns_solution(self) -> None:
        """cuopt_solve with valid inputs returns a solution from the stub."""
        server_params = _optimization_server_params(
            cuopt_deployment_id=_STUB_CUOPT_DEPLOYMENT_ID
        )
        async with integration_test_mcp_session(server_params=server_params) as session:
            result = await session.call_tool(
                "cuopt_solve",
                {"problem_definition": _SIMPLE_LP_PROBLEM},
            )
            assert not result.isError
            assert len(result.content) > 0
            content = result.content[0]
            assert isinstance(content, TextContent)
            data = json.loads(content.text)
            assert data["preview"] is False
            assert data["status"] == "optimal"
            assert data["objective_value"] == 42.0
            assert data["solution"] is not None

    async def test_cuopt_solve_preview_mode(self) -> None:
        """cuopt_solve with preview=True returns validation result."""
        server_params = _optimization_server_params(
            cuopt_deployment_id=_STUB_CUOPT_DEPLOYMENT_ID
        )
        async with integration_test_mcp_session(server_params=server_params) as session:
            result = await session.call_tool(
                "cuopt_solve",
                {"problem_definition": _SIMPLE_LP_PROBLEM, "preview": True},
            )
            assert not result.isError
            assert len(result.content) > 0
            content = result.content[0]
            assert isinstance(content, TextContent)
            data = json.loads(content.text)
            assert data["preview"] is True
            assert "validation" in data

    async def test_cuopt_solve_tool_has_expected_tags(self) -> None:
        """Verify cuopt_solve has the expected tags (optimization, cuopt, solver, daria)."""
        server_params = _optimization_server_params()
        async with integration_test_mcp_session(server_params=server_params) as session:
            result = await session.list_tools()
            cuopt_tool = next(
                (t for t in result.tools if t.name == "cuopt_solve"), None
            )
            assert cuopt_tool is not None
            if cuopt_tool.meta:
                fastmcp_meta = cuopt_tool.meta.get("_fastmcp", {})
                tags = set(fastmcp_meta.get("tags", []))
                assert "optimization" in tags
                assert "daria" in tags
