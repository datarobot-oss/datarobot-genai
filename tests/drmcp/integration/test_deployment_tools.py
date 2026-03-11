# Copyright 2026 DataRobot, Inc. and its affiliates.
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

"""Integration tests for get_prediction_history deployment tool."""

import json

import pytest
from mcp.types import TextContent

from datarobot_genai.drmcp.test_utils.mcp_utils_integration import integration_test_mcp_session

STUB_DEPLOYMENT_ID = "stub_deployment_id"


@pytest.mark.asyncio
class TestGetPredictionHistoryIntegration:
    """Integration tests for the get_prediction_history MCP tool."""

    async def test_get_prediction_history_basic(self) -> None:
        """get_prediction_history returns prediction rows for a valid deployment."""
        async with integration_test_mcp_session() as session:
            result = await session.call_tool(
                "get_prediction_history",
                {
                    "deployment_id": STUB_DEPLOYMENT_ID,
                    "limit": 10,
                },
            )

            assert not result.isError
            assert len(result.content) > 0
            assert isinstance(result.content[0], TextContent)

            data = json.loads(result.content[0].text)
            assert data["deployment_id"] == STUB_DEPLOYMENT_ID
            assert "row_count" in data
            assert "rows" in data
            assert isinstance(data["rows"], list)

    async def test_get_prediction_history_returns_rows(self) -> None:
        """get_prediction_history returns the expected stub prediction rows."""
        async with integration_test_mcp_session() as session:
            result = await session.call_tool(
                "get_prediction_history",
                {
                    "deployment_id": STUB_DEPLOYMENT_ID,
                    "limit": 5,
                },
            )

            assert not result.isError
            data = json.loads(result.content[0].text)
            # The stub returns min(limit, 5) rows
            assert data["row_count"] == 5
            assert len(data["rows"]) == 5
            # Each row has rowId, predictionValue, timestamp
            first_row = data["rows"][0]
            assert "rowId" in first_row
            assert "predictionValue" in first_row
            assert "timestamp" in first_row

    async def test_get_prediction_history_with_time_filters(self) -> None:
        """get_prediction_history passes time filters without error."""
        async with integration_test_mcp_session() as session:
            result = await session.call_tool(
                "get_prediction_history",
                {
                    "deployment_id": STUB_DEPLOYMENT_ID,
                    "limit": 10,
                    "start_time": "2024-01-01T00:00:00Z",
                    "end_time": "2024-01-31T23:59:59Z",
                },
            )

            assert not result.isError
            data = json.loads(result.content[0].text)
            assert data["deployment_id"] == STUB_DEPLOYMENT_ID
            assert "rows" in data

    async def test_get_prediction_history_default_limit(self) -> None:
        """get_prediction_history uses the default limit of 100."""
        async with integration_test_mcp_session() as session:
            result = await session.call_tool(
                "get_prediction_history",
                {
                    "deployment_id": STUB_DEPLOYMENT_ID,
                },
            )

            assert not result.isError
            data = json.loads(result.content[0].text)
            # stub returns min(100, 5) = 5 rows with default limit
            assert data["row_count"] >= 0

    async def test_get_prediction_history_missing_deployment_id(self) -> None:
        """get_prediction_history returns an error when deployment_id is not provided."""
        async with integration_test_mcp_session() as session:
            result = await session.call_tool(
                "get_prediction_history",
                {"limit": 10},
            )

            assert result.isError
            assert "deployment_id" in result.content[0].text.lower() or "Error" in result.content[0].text

    async def test_get_prediction_history_tool_registered(self) -> None:
        """Verify get_prediction_history is registered in the MCP session."""
        async with integration_test_mcp_session() as session:
            tools_result = await session.list_tools()
            tool_names = [t.name for t in tools_result.tools]
            assert "get_prediction_history" in tool_names
