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

import pytest
from mcp.types import TextContent

from datarobot_genai.drmcp.test_utils.mcp_utils_integration import integration_test_mcp_session


@pytest.mark.asyncio
class TestMCPDataToolsIntegration:
    """Integration tests for MCP data tools (uses DR client stubs)."""

    async def test_data_tools_registered(self) -> None:
        """Verify new data tools are registered and callable via MCP session."""
        async with integration_test_mcp_session() as session:
            result = await session.list_tools()
            tool_names = [t.name for t in result.tools]
            assert "get_dataset_details" in tool_names
            assert "list_datastores" in tool_names
            assert "browse_datastore" in tool_names
            assert "query_datastore" in tool_names

    async def test_get_dataset_details_success(self) -> None:
        """Test get_dataset_details returns metadata and sample rows."""
        async with integration_test_mcp_session() as session:
            result = await session.call_tool(
                "get_dataset_details",
                {"dataset_id": "stub_dataset_id"},
            )
            assert not result.isError
            assert len(result.content) > 0
            assert isinstance(result.content[0], TextContent)
            data = json.loads(result.content[0].text)
            assert data["id"] == "stub_dataset_id"
            assert data["name"] == "stub_dataset.csv"
            assert "columns" in data
            assert "sample" in data
            assert len(data["sample"]) > 0

    async def test_get_dataset_details_not_found(self) -> None:
        """Test get_dataset_details with nonexistent dataset."""
        async with integration_test_mcp_session() as session:
            result = await session.call_tool(
                "get_dataset_details",
                {"dataset_id": "nonexistent_dataset"},
            )
            assert result.isError
            result_text = result.content[0].text  # type: ignore[union-attr]
            assert "not found" in result_text.lower() or "error" in result_text.lower()

    async def test_get_dataset_details_missing_id(self) -> None:
        """Test get_dataset_details with no dataset_id."""
        async with integration_test_mcp_session() as session:
            result = await session.call_tool(
                "get_dataset_details",
                {},
            )
            assert result.isError
            result_text = result.content[0].text  # type: ignore[union-attr]
            assert "Dataset ID must be provided" in result_text

    async def test_list_datastores_success(self) -> None:
        """Test list_datastores returns available data connections."""
        async with integration_test_mcp_session() as session:
            result = await session.call_tool("list_datastores", {})
            assert not result.isError
            assert isinstance(result.content[0], TextContent)
            data = json.loads(result.content[0].text)
            assert "datastores" in data
            assert "count" in data
            assert data["count"] >= 1
            assert data["datastores"][0]["id"] == "stub_datastore_id"
            assert data["datastores"][0]["canonical_name"] == "Test PostgreSQL"

    async def test_browse_datastore_success(self) -> None:
        """Test browse_datastore returns table listings."""
        async with integration_test_mcp_session() as session:
            result = await session.call_tool(
                "browse_datastore",
                {"datastore_id": "stub_datastore_id"},
            )
            assert not result.isError
            assert isinstance(result.content[0], TextContent)
            data = json.loads(result.content[0].text)
            assert data["datastore_id"] == "stub_datastore_id"
            assert "items" in data
            assert data["count"] >= 1

    async def test_browse_datastore_missing_id(self) -> None:
        """Test browse_datastore with no datastore_id."""
        async with integration_test_mcp_session() as session:
            result = await session.call_tool("browse_datastore", {})
            assert result.isError
            result_text = result.content[0].text  # type: ignore[union-attr]
            assert "Datastore ID must be provided" in result_text

    async def test_query_datastore_success(self) -> None:
        """Test query_datastore executes SQL and returns results."""
        async with integration_test_mcp_session() as session:
            result = await session.call_tool(
                "query_datastore",
                {
                    "datastore_id": "stub_datastore_id",
                    "sql": "SELECT * FROM users LIMIT 10",
                },
            )
            assert not result.isError
            assert isinstance(result.content[0], TextContent)
            data = json.loads(result.content[0].text)
            assert "rows" in data
            assert "columns" in data
            assert data["row_count"] >= 1

    async def test_query_datastore_missing_params(self) -> None:
        """Test query_datastore with missing required parameters."""
        async with integration_test_mcp_session() as session:
            # Missing sql
            result = await session.call_tool(
                "query_datastore",
                {"datastore_id": "stub_datastore_id"},
            )
            assert result.isError
            result_text = result.content[0].text  # type: ignore[union-attr]
            assert "SQL query must be provided" in result_text

            # Missing datastore_id
            result = await session.call_tool(
                "query_datastore",
                {"sql": "SELECT 1"},
            )
            assert result.isError
            result_text = result.content[0].text  # type: ignore[union-attr]
            assert "Datastore ID must be provided" in result_text
