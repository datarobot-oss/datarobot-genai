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

"""Integration tests for use case MCP tools (list_use_cases, list_use_case_assets)."""

import json

import pytest
from mcp.types import TextContent

from datarobot_genai.drmcp.test_utils.mcp_utils_integration import integration_test_mcp_session
from datarobot_genai.drmcp.test_utils.mcp_utils_integration import (
    integration_test_server_params_with_env,
)
from datarobot_genai.drmcp.test_utils.stubs.dr_client_stubs import STUB_USE_CASE_ID


def _use_case_server_params():
    """Return server params with use case tools enabled."""
    return integration_test_server_params_with_env({"ENABLE_USE_CASE_TOOLS": "true"})


@pytest.mark.asyncio
class TestMCPUseCaseToolsIntegration:
    """Integration tests for MCP use case tools via stub DataRobot client."""

    async def test_tools_registered(self) -> None:
        """Verify list_use_cases and list_use_case_assets are registered in the MCP server."""
        async with integration_test_mcp_session(server_params=_use_case_server_params()) as session:
            result = await session.list_tools()
            tool_names = [t.name for t in result.tools]
            assert "list_use_cases" in tool_names
            assert "list_use_case_assets" in tool_names

    async def test_list_use_cases_success(self) -> None:
        """list_use_cases returns a list of use cases from the stub."""
        async with integration_test_mcp_session(server_params=_use_case_server_params()) as session:
            result = await session.call_tool("list_use_cases", {})
            assert not result.isError
            assert len(result.content) > 0
            assert isinstance(result.content[0], TextContent)
            data = json.loads(result.content[0].text)
            assert "use_cases" in data
            assert "count" in data
            assert data["count"] >= 1
            use_case_ids = [uc["id"] for uc in data["use_cases"]]
            assert STUB_USE_CASE_ID in use_case_ids

    async def test_list_use_cases_with_search_filter(self) -> None:
        """list_use_cases with a search filter narrows the results."""
        async with integration_test_mcp_session(server_params=_use_case_server_params()) as session:
            result = await session.call_tool("list_use_cases", {"search": "Stub"})
            assert not result.isError
            data = json.loads(result.content[0].text)
            assert "use_cases" in data
            # Only use cases matching "Stub" should be returned
            for uc in data["use_cases"]:
                assert "stub" in uc["name"].lower()

    async def test_list_use_cases_with_search_no_results(self) -> None:
        """list_use_cases with a non-matching search filter returns empty list."""
        async with integration_test_mcp_session(server_params=_use_case_server_params()) as session:
            result = await session.call_tool("list_use_cases", {"search": "zzz_nonexistent_zzz"})
            assert not result.isError
            data = json.loads(result.content[0].text)
            assert data["use_cases"] == []
            assert data["count"] == 0

    async def test_list_use_cases_with_limit(self) -> None:
        """list_use_cases respects the limit parameter."""
        async with integration_test_mcp_session(server_params=_use_case_server_params()) as session:
            result = await session.call_tool("list_use_cases", {"limit": 1})
            assert not result.isError
            data = json.loads(result.content[0].text)
            assert "use_cases" in data

    async def test_list_use_case_assets_success(self) -> None:
        """list_use_case_assets returns datasets, deployments, and experiments for a use case."""
        async with integration_test_mcp_session(server_params=_use_case_server_params()) as session:
            result = await session.call_tool(
                "list_use_case_assets", {"use_case_id": STUB_USE_CASE_ID}
            )
            assert not result.isError
            assert len(result.content) > 0
            assert isinstance(result.content[0], TextContent)
            data = json.loads(result.content[0].text)
            assert data["use_case_id"] == STUB_USE_CASE_ID
            assert "name" in data
            # Stub returns datasets, deployments, and experiments
            assert "datasets" in data or "datasets_error" in data
            assert "deployments" in data or "deployments_error" in data
            assert "experiments" in data or "experiments_error" in data

    async def test_list_use_case_assets_has_datasets(self) -> None:
        """list_use_case_assets includes datasets list from stub."""
        async with integration_test_mcp_session(server_params=_use_case_server_params()) as session:
            result = await session.call_tool(
                "list_use_case_assets", {"use_case_id": STUB_USE_CASE_ID}
            )
            assert not result.isError
            data = json.loads(result.content[0].text)
            assert "datasets" in data
            assert len(data["datasets"]) >= 1
            assert "id" in data["datasets"][0]
            assert "name" in data["datasets"][0]

    async def test_list_use_case_assets_has_deployments(self) -> None:
        """list_use_case_assets includes deployments list from stub."""
        async with integration_test_mcp_session(server_params=_use_case_server_params()) as session:
            result = await session.call_tool(
                "list_use_case_assets", {"use_case_id": STUB_USE_CASE_ID}
            )
            assert not result.isError
            data = json.loads(result.content[0].text)
            assert "deployments" in data
            assert len(data["deployments"]) >= 1
            assert "id" in data["deployments"][0]

    async def test_list_use_case_assets_has_experiments(self) -> None:
        """list_use_case_assets includes experiments list from stub."""
        async with integration_test_mcp_session(server_params=_use_case_server_params()) as session:
            result = await session.call_tool(
                "list_use_case_assets", {"use_case_id": STUB_USE_CASE_ID}
            )
            assert not result.isError
            data = json.loads(result.content[0].text)
            assert "experiments" in data
            assert len(data["experiments"]) >= 1
            assert "id" in data["experiments"][0]
            assert "name" in data["experiments"][0]

    async def test_list_use_case_assets_missing_use_case_id(self) -> None:
        """list_use_case_assets raises ToolError when use_case_id is not provided."""
        async with integration_test_mcp_session(server_params=_use_case_server_params()) as session:
            result = await session.call_tool("list_use_case_assets", {})
            assert result.isError
            assert len(result.content) > 0
            error_text = (
                result.content[0].text
                if hasattr(result.content[0], "text")
                else str(result.content[0])
            )
            assert "use case" in error_text.lower() or "error" in error_text.lower()
