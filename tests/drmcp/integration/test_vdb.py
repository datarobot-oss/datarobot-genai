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

"""Integration tests for VDB (Vector Database) MCP tools."""

import json

import pytest
from mcp.types import TextContent

from datarobot_genai.drmcp.test_utils.mcp_utils_integration import integration_test_mcp_session
from datarobot_genai.drmcp.test_utils.mcp_utils_integration import (
    integration_test_server_params_with_env,
)
from datarobot_genai.drmcp.test_utils.stubs.dr_client_stubs import STUB_VDB_DEPLOYMENT_ID


def _vdb_server_params():
    """Return server params with VDB tools enabled."""
    return integration_test_server_params_with_env({"ENABLE_VDB_TOOLS": "true"})


@pytest.mark.asyncio
class TestMCPVDBToolsIntegration:
    """Integration tests for MCP VDB tools (vdb_list, vdb_query)."""

    async def test_tools_registered(self) -> None:
        """Verify that VDB tools are registered and visible in the MCP session."""
        async with integration_test_mcp_session(server_params=_vdb_server_params()) as session:
            result = await session.list_tools()
            tool_names = [t.name for t in result.tools]
            assert "vdb_list" in tool_names
            assert "vdb_query" in tool_names

    async def test_vdb_list_returns_vdbs(self) -> None:
        """vdb_list should return only VDB-capable deployments."""
        async with integration_test_mcp_session(server_params=_vdb_server_params()) as session:
            result = await session.call_tool("vdb_list", {})

            assert not result.isError, (
                f"vdb_list failed: {result.content[0].text if result.content else 'no content'}"  # type: ignore[union-attr]
            )
            assert len(result.content) > 0
            assert isinstance(result.content[0], TextContent)

            data = json.loads(result.content[0].text)
            assert "vector_databases" in data
            assert "count" in data
            # Stub returns 2 deployments but only 1 is a VDB
            assert data["count"] == 1
            assert len(data["vector_databases"]) == 1

            vdb = data["vector_databases"][0]
            assert vdb["deployment_id"] == STUB_VDB_DEPLOYMENT_ID
            assert vdb["label"] == "Stub VDB Deployment"
            assert vdb["status"] == "active"

    async def test_vdb_list_structure(self) -> None:
        """vdb_list result must have the expected keys per VDB entry."""
        async with integration_test_mcp_session(server_params=_vdb_server_params()) as session:
            result = await session.call_tool("vdb_list", {})

            assert not result.isError
            data = json.loads(result.content[0].text)  # type: ignore[union-attr]

            for vdb in data["vector_databases"]:
                assert "deployment_id" in vdb
                assert "label" in vdb
                assert "status" in vdb

    async def test_vdb_query_returns_documents(self) -> None:
        """vdb_query should return a list of matching documents."""
        async with integration_test_mcp_session(server_params=_vdb_server_params()) as session:
            result = await session.call_tool(
                "vdb_query",
                {
                    "deployment_id": STUB_VDB_DEPLOYMENT_ID,
                    "query": "What is DataRobot?",
                    "num_results": 2,
                },
            )

            assert not result.isError, (
                f"vdb_query failed: {result.content[0].text if result.content else 'no content'}"  # type: ignore[union-attr]
            )
            assert len(result.content) > 0
            assert isinstance(result.content[0], TextContent)

            data = json.loads(result.content[0].text)
            assert "deployment_id" in data
            assert data["deployment_id"] == STUB_VDB_DEPLOYMENT_ID
            assert "documents" in data
            assert "count" in data
            assert data["count"] == 2
            assert len(data["documents"]) == 2

    async def test_vdb_query_document_structure(self) -> None:
        """Returned documents should include page_content and metadata fields."""
        async with integration_test_mcp_session(server_params=_vdb_server_params()) as session:
            result = await session.call_tool(
                "vdb_query",
                {
                    "deployment_id": STUB_VDB_DEPLOYMENT_ID,
                    "query": "AutoML",
                },
            )

            assert not result.isError
            data = json.loads(result.content[0].text)  # type: ignore[union-attr]

            assert len(data["documents"]) > 0
            doc = data["documents"][0]
            assert "page_content" in doc
            assert "metadata" in doc

    async def test_vdb_query_missing_deployment_id(self) -> None:
        """vdb_query without deployment_id must return an error."""
        async with integration_test_mcp_session(server_params=_vdb_server_params()) as session:
            result = await session.call_tool(
                "vdb_query",
                {"query": "What is DataRobot?"},
            )

            assert result.isError
            assert len(result.content) > 0
            error_text = result.content[0].text  # type: ignore[union-attr]
            assert "Deployment ID" in error_text or "deployment_id" in error_text.lower()

    async def test_vdb_query_missing_query(self) -> None:
        """vdb_query without a query string must return an error."""
        async with integration_test_mcp_session(server_params=_vdb_server_params()) as session:
            result = await session.call_tool(
                "vdb_query",
                {"deployment_id": STUB_VDB_DEPLOYMENT_ID},
            )

            assert result.isError
            assert len(result.content) > 0
            error_text = result.content[0].text  # type: ignore[union-attr]
            assert "Query" in error_text or "query" in error_text.lower()

    async def test_vdb_query_with_retrieval_mode(self) -> None:
        """vdb_query should accept retrieval_mode parameter."""
        async with integration_test_mcp_session(server_params=_vdb_server_params()) as session:
            result = await session.call_tool(
                "vdb_query",
                {
                    "deployment_id": STUB_VDB_DEPLOYMENT_ID,
                    "query": "machine learning",
                    "num_results": 1,
                    "retrieval_mode": "maximal_marginal_relevance",
                },
            )

            assert not result.isError
            data = json.loads(result.content[0].text)  # type: ignore[union-attr]
            assert "documents" in data
            assert "count" in data
