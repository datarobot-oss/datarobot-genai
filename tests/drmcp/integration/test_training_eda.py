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

"""Integration tests for catalog EDA tools (stub DataRobot client via MCP stdio server)."""

import json

import pytest
from mcp.types import TextContent

from datarobot_genai.drmcp.test_utils.mcp_utils_integration import integration_test_mcp_session
from datarobot_genai.drmcp.test_utils.stubs.dr_client_stubs import STUB_DATASET_ID


@pytest.mark.asyncio
class TestGetExploratoryInsightsIntegration:
    """MCP integration tests for ``catalog_get_eda_insights`` including catalog API profile."""

    async def test_catalog_get_eda_insights_registered(self) -> None:
        """``catalog_get_eda_insights`` is registered when predictive tools load."""
        async with integration_test_mcp_session() as session:
            listed = await session.list_tools()
            tool_names = [t.name for t in listed.tools]
            assert "catalog_get_eda_insights" in tool_names

    async def test_catalog_get_eda_insights_with_catalog_feature_profile(self) -> None:
        """``feature_col`` returns DataRobot catalog-style stats and optional histogram."""
        async with integration_test_mcp_session() as session:
            result = await session.call_tool(
                "catalog_get_eda_insights",
                {
                    "dataset_id": STUB_DATASET_ID,
                    "feature_col": "sales",
                    "include_feature_histogram": True,
                },
            )

            assert not result.isError
            assert isinstance(result.content[0], TextContent)
            data = json.loads(result.content[0].text)

            assert "dataset_summary" in data
            profile = data["catalog_feature_profile"]
            assert profile["source"] == "datarobot_catalog_all_features_details"
            assert profile["statistics_basis"] == "eda_sample_as_computed_by_datarobot"
            assert profile["data_persisted"] is True
            feat = profile["feature"]
            assert feat["name"] == "sales"
            assert feat["mean"] == 550.0
            assert "histogram" in profile
            assert profile["histogram"]["plot"][0]["label"] == "stub-bin"

    async def test_catalog_get_eda_insights_invalid_feature_col(self) -> None:
        """Unknown ``feature_col`` yields a tool error."""
        async with integration_test_mcp_session() as session:
            result = await session.call_tool(
                "catalog_get_eda_insights",
                {
                    "dataset_id": STUB_DATASET_ID,
                    "feature_col": "not_a_column",
                },
            )

            assert result.isError
            text = result.content[0].text  # type: ignore[union-attr]
            assert "feature_col" in text or "not_a_column" in text
