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

"""Integration tests for modeling_get_modeldetails and catalog_check_timeseries_eligibility."""

import json

import pytest
from mcp.types import TextContent

from datarobot_genai.drmcp.test_utils.mcp_utils_integration import integration_test_mcp_session
from datarobot_genai.drmcp.test_utils.stubs.dr_client_stubs import STUB_DATASET_ID
from datarobot_genai.drmcp.test_utils.stubs.dr_client_stubs import STUB_PROJECT_ID


@pytest.mark.asyncio
class TestGetModelDetailsIntegration:
    """Integration tests for the modeling_get_modeldetails MCP tool."""

    async def test_modeling_get_modeldetails_basic(self) -> None:
        """modeling_get_modeldetails returns model info without optional fields."""
        async with integration_test_mcp_session() as session:
            result = await session.call_tool(
                "modeling_get_modeldetails",
                {
                    "project_id": STUB_PROJECT_ID,
                    "model_id": "model_1",
                    "include_feature_impact": False,
                    "include_roc_curve": False,
                },
            )

            assert not result.isError
            assert len(result.content) > 0
            assert isinstance(result.content[0], TextContent)

            data = json.loads(result.content[0].text)
            assert data["project_id"] == STUB_PROJECT_ID
            assert data["model_id"] == "model_1"
            assert "Keras" in data["model_type"]
            assert data["target"] == "sentiment"
            assert data["metric"] == "AUC"
            assert "AUC" in data["metrics"]
            assert data["featurelist_name"] == "Informative Features"
            assert data["sample_pct"] == 64.0
            assert "feature_impact" not in data
            assert "roc_curve" not in data

    async def test_modeling_get_modeldetails_with_feature_impact(self) -> None:
        """modeling_get_modeldetails includes feature impact when requested."""
        async with integration_test_mcp_session() as session:
            result = await session.call_tool(
                "modeling_get_modeldetails",
                {
                    "project_id": STUB_PROJECT_ID,
                    "model_id": "model_1",
                    "include_feature_impact": True,
                    "include_roc_curve": False,
                },
            )

            assert not result.isError
            data = json.loads(result.content[0].text)
            assert "feature_impact" in data
            fi = data["feature_impact"]
            assert isinstance(fi, list)
            assert len(fi) > 0
            feature_names = [f["featureName"] for f in fi]
            assert "text_review" in feature_names

    async def test_modeling_get_modeldetails_with_roc_curve(self) -> None:
        """modeling_get_modeldetails includes ROC curve when requested."""
        async with integration_test_mcp_session() as session:
            result = await session.call_tool(
                "modeling_get_modeldetails",
                {
                    "project_id": STUB_PROJECT_ID,
                    "model_id": "model_1",
                    "include_feature_impact": False,
                    "include_roc_curve": True,
                },
            )

            assert not result.isError
            data = json.loads(result.content[0].text)
            assert "roc_curve" in data
            roc = data["roc_curve"]
            assert roc["source"] == "validation"
            assert isinstance(roc["roc_points"], list)
            assert len(roc["roc_points"]) > 0

    async def test_modeling_get_modeldetails_with_all_optional_fields(self) -> None:
        """modeling_get_modeldetails returns all fields when both optional flags are True."""
        async with integration_test_mcp_session() as session:
            result = await session.call_tool(
                "modeling_get_modeldetails",
                {
                    "project_id": STUB_PROJECT_ID,
                    "model_id": "model_2",
                    "include_feature_impact": True,
                    "include_roc_curve": True,
                },
            )

            assert not result.isError
            data = json.loads(result.content[0].text)
            assert data["model_id"] == "model_2"
            assert "Random Forest" in data["model_type"]
            assert "feature_impact" in data
            assert "roc_curve" in data

    async def test_modeling_get_modeldetails_missing_project_id(self) -> None:
        """modeling_get_modeldetails returns an error when project_id is not provided."""
        async with integration_test_mcp_session() as session:
            result = await session.call_tool(
                "modeling_get_modeldetails",
                {"model_id": "model_1"},
            )

            assert result.isError
            text = result.content[0].text.lower()
            assert "project_id" in text or "error" in text

    async def test_modeling_get_modeldetails_missing_model_id(self) -> None:
        """modeling_get_modeldetails returns an error when model_id is not provided."""
        async with integration_test_mcp_session() as session:
            result = await session.call_tool(
                "modeling_get_modeldetails",
                {"project_id": STUB_PROJECT_ID},
            )

            assert result.isError
            assert "model_id" in result.content[0].text.lower() or "Error" in result.content[0].text

    async def test_modeling_get_modeldetails_tool_registered(self) -> None:
        """Verify modeling_get_modeldetails is registered in the MCP session."""
        async with integration_test_mcp_session() as session:
            tools_result = await session.list_tools()
            tool_names = [t.name for t in tools_result.tools]
            assert "modeling_get_modeldetails" in tool_names


@pytest.mark.asyncio
class TestIsEligibleForTimeseriesTrainingIntegration:
    """Integration tests for the catalog_check_timeseries_eligibility MCP tool."""

    async def test_eligible_dataset(self) -> None:
        """catalog_check_timeseries_eligibility returns ELIGIBLE for a valid dataset."""
        async with integration_test_mcp_session() as session:
            result = await session.call_tool(
                "catalog_check_timeseries_eligibility",
                {
                    "dataset_id": STUB_DATASET_ID,
                    "datetime_column": "date",
                    "target_column": "sales",
                },
            )

            assert not result.isError
            data = json.loads(result.content[0].text)
            assert data["status"] == "ELIGIBLE"
            assert data["errors"] == []
            assert any("Row count" in info for info in data["info"])

    async def test_eligible_dataset_with_series_id(self) -> None:
        """catalog_check_timeseries_eligibility returns ELIGIBLE with optional series_id_column."""
        async with integration_test_mcp_session() as session:
            result = await session.call_tool(
                "catalog_check_timeseries_eligibility",
                {
                    "dataset_id": STUB_DATASET_ID,
                    "datetime_column": "date",
                    "target_column": "sales",
                    "series_id_column": "store_id",
                },
            )

            assert not result.isError
            data = json.loads(result.content[0].text)
            assert data["status"] == "ELIGIBLE"
            assert data["errors"] == []

    async def test_not_eligible_missing_datetime_column(self) -> None:
        """catalog_check_timeseries_eligibility is NOT_ELIGIBLE without datetime column."""
        async with integration_test_mcp_session() as session:
            result = await session.call_tool(
                "catalog_check_timeseries_eligibility",
                {
                    "dataset_id": STUB_DATASET_ID,
                    "datetime_column": "nonexistent_date_col",
                    "target_column": "sales",
                },
            )

            assert not result.isError
            data = json.loads(result.content[0].text)
            assert data["status"] == "NOT_ELIGIBLE"
            assert any("nonexistent_date_col" in err for err in data["errors"])

    async def test_not_eligible_missing_target_column(self) -> None:
        """catalog_check_timeseries_eligibility returns NOT_ELIGIBLE when target column missing."""
        async with integration_test_mcp_session() as session:
            result = await session.call_tool(
                "catalog_check_timeseries_eligibility",
                {
                    "dataset_id": STUB_DATASET_ID,
                    "datetime_column": "date",
                    "target_column": "nonexistent_target",
                },
            )

            assert not result.isError
            data = json.loads(result.content[0].text)
            assert data["status"] == "NOT_ELIGIBLE"
            assert any("nonexistent_target" in err for err in data["errors"])

    async def test_missing_dataset_id(self) -> None:
        """catalog_check_timeseries_eligibility returns an error when dataset_id is missing."""
        async with integration_test_mcp_session() as session:
            result = await session.call_tool(
                "catalog_check_timeseries_eligibility",
                {
                    "datetime_column": "date",
                    "target_column": "sales",
                },
            )

            assert result.isError

    async def test_missing_datetime_column(self) -> None:
        """catalog_check_timeseries_eligibility returns an error when datetime_column is missing."""
        async with integration_test_mcp_session() as session:
            result = await session.call_tool(
                "catalog_check_timeseries_eligibility",
                {
                    "dataset_id": STUB_DATASET_ID,
                    "target_column": "sales",
                },
            )

            assert result.isError

    async def test_missing_target_column(self) -> None:
        """catalog_check_timeseries_eligibility returns an error when target_column is missing."""
        async with integration_test_mcp_session() as session:
            result = await session.call_tool(
                "catalog_check_timeseries_eligibility",
                {
                    "dataset_id": STUB_DATASET_ID,
                    "datetime_column": "date",
                },
            )

            assert result.isError

    async def test_tool_registered(self) -> None:
        """Verify catalog_check_timeseries_eligibility is registered in the MCP session."""
        async with integration_test_mcp_session() as session:
            tools_result = await session.list_tools()
            tool_names = [t.name for t in tools_result.tools]
            assert "catalog_check_timeseries_eligibility" in tool_names
