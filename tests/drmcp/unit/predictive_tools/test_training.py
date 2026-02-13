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

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pandas as pd
import pytest
from fastmcp.exceptions import ToolError
from fastmcp.tools.tool import ToolResult

from datarobot_genai.drmcp.tools.predictive import training


@pytest.mark.asyncio
async def test_analyze_dataset() -> None:
    mock_dataset = MagicMock()
    mock_df = pd.DataFrame(
        {
            "num_col": [1, 2, 3],
            "cat_col": ["a", "b", "c"],
            "date_col": pd.date_range("2021-01-01", periods=3),
            "text_col": ["long text " * 10] * 3,
            "target": [0, 1, 0],
        }
    )
    mock_dataset.get_as_dataframe.return_value = mock_df

    with (
        patch(
            "datarobot_genai.drmcp.tools.predictive.training.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drmcp.tools.predictive.training.DataRobotClient") as mock_drc,
    ):
        mock_client = MagicMock()
        mock_client.Dataset.get.return_value = mock_dataset
        mock_drc.return_value.get_client.return_value = mock_client
        result = await training.analyze_dataset(dataset_id="test_dataset_id")
        assert hasattr(result, "structured_content")
        insights = result.structured_content

        assert insights["total_columns"] == 5
        assert insights["total_rows"] == 3
        assert "num_col" in insights["numerical_columns"]
        assert "cat_col" in insights["categorical_columns"]
        assert "date_col" in insights["datetime_columns"]
        assert "text_col" in insights["text_columns"]
        assert "target" in insights["potential_targets"]


@pytest.mark.asyncio
async def test_suggest_use_cases() -> None:
    mock_dataset = MagicMock()
    mock_df = pd.DataFrame(
        {
            "features": [1, 2, 3],
            "binary_target": [0, 1, 0],
            "multi_target": ["a", "b", "c"],
            "regression_target": [10.5, 20.1, 30.8],
        }
    )
    mock_dataset.get_as_dataframe.return_value = mock_df

    # Mock analyze_dataset to return ToolResult
    mock_insights_dict = {
        "total_columns": 4,
        "total_rows": 3,
        "numerical_columns": ["features"],
        "categorical_columns": ["multi_target"],
        "datetime_columns": [],
        "text_columns": [],
        "potential_targets": ["binary_target", "multi_target", "regression_target"],
        "missing_data_summary": {},
    }
    mock_insights = ToolResult(
        structured_content=mock_insights_dict,
    )

    with (
        patch(
            "datarobot_genai.drmcp.tools.predictive.training.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drmcp.tools.predictive.training.DataRobotClient") as mock_drc,
        patch(
            "datarobot_genai.drmcp.tools.predictive.training.analyze_dataset",
            return_value=mock_insights,
        ),
    ):
        mock_client = MagicMock()
        mock_client.Dataset.get.return_value = mock_dataset
        mock_drc.return_value.get_client.return_value = mock_client
        result = await training.suggest_use_cases(dataset_id="test_dataset_id")
        assert hasattr(result, "structured_content")
        suggestions = result.structured_content["use_case_suggestions"]

        assert len(suggestions) > 0
        assert any(s["problem_type"] == "Binary Classification" for s in suggestions)
        assert any(s["problem_type"] == "Multiclass Classification" for s in suggestions)
        assert any(s["problem_type"] == "Regression" for s in suggestions)


@pytest.mark.asyncio
async def test_get_exploratory_insights() -> None:
    mock_dataset = MagicMock()
    mock_df = pd.DataFrame({"features": [1, 2, 3], "target": [0, 1, 0]})
    mock_dataset.get_as_dataframe.return_value = mock_df

    # Mock analyze_dataset to return ToolResult
    mock_insights_dict = {
        "total_columns": 2,
        "total_rows": 3,
        "numerical_columns": ["features", "target"],
        "categorical_columns": [],
        "datetime_columns": [],
        "text_columns": [],
        "potential_targets": ["target"],
        "missing_data_summary": {},
    }
    mock_insights = ToolResult(
        structured_content=mock_insights_dict,
    )

    with (
        patch(
            "datarobot_genai.drmcp.tools.predictive.training.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drmcp.tools.predictive.training.DataRobotClient") as mock_drc,
        patch(
            "datarobot_genai.drmcp.tools.predictive.training.analyze_dataset",
            return_value=mock_insights,
        ),
    ):
        mock_client = MagicMock()
        mock_client.Dataset.get.return_value = mock_dataset
        mock_drc.return_value.get_client.return_value = mock_client
        result = await training.get_exploratory_insights(
            dataset_id="test_dataset_id", target_col="target"
        )
        assert hasattr(result, "structured_content")
        insights = result.structured_content

        assert "dataset_summary" in insights
        assert "target_analysis" in insights
        assert "feature_correlations" in insights
        assert "missing_data" in insights
        assert "data_types" in insights


@pytest.mark.asyncio
async def test_start_autopilot_new_project() -> None:
    mock_dataset = MagicMock()
    mock_dataset.id = "test_dataset_id"
    mock_project = MagicMock()
    mock_project.id = "test_project_id"
    mock_project.get_status.return_value = "running"
    mock_project.use_case_id = None

    with (
        patch(
            "datarobot_genai.drmcp.tools.predictive.training.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drmcp.tools.predictive.training.DataRobotClient") as mock_drc,
    ):
        mock_client = MagicMock()
        mock_client.Dataset.create_from_url.return_value = mock_dataset
        mock_client.Project.create_from_dataset.return_value = mock_project
        mock_drc.return_value.get_client.return_value = mock_client
        result = await training.start_autopilot(
            target="target",
            dataset_url="http://test.com/data.csv",
            project_name="Test Project",
        )
        assert hasattr(result, "structured_content")
        response = result.structured_content

        assert response["project_id"] == "test_project_id"
        assert response["target"] == "target"
        assert response["mode"] == "quick"
        assert response["status"] == "running"


@pytest.mark.asyncio
async def test_start_autopilot_existing_project() -> None:
    mock_project = MagicMock()
    mock_project.id = "test_project_id"
    mock_project.get_status.return_value = "running"
    mock_project.use_case_id = None

    with (
        patch(
            "datarobot_genai.drmcp.tools.predictive.training.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drmcp.tools.predictive.training.DataRobotClient") as mock_drc,
    ):
        mock_client = MagicMock()
        mock_client.Project.get.return_value = mock_project
        mock_drc.return_value.get_client.return_value = mock_client
        result = await training.start_autopilot(
            target="target", project_id="test_project_id", mode="comprehensive"
        )
        assert hasattr(result, "structured_content")
        response = result.structured_content

        assert response["project_id"] == "test_project_id"
        assert response["target"] == "target"
        assert response["mode"] == "comprehensive"
        assert response["status"] == "running"


@pytest.mark.asyncio
async def test_get_model_roc_curve() -> None:
    mock_project = MagicMock()
    mock_model = MagicMock()
    mock_roc_curve = MagicMock()
    mock_roc_curve.roc_points = [
        {
            "accuracy": 0.8,
            "f1_score": 0.75,
            "threshold": 0.5,
            "true_positive_rate": 0.8,
            "false_positive_rate": 0.2,
        }
    ]
    mock_roc_curve.negative_class_predictions = [0.1, 0.2]
    mock_roc_curve.positive_class_predictions = [0.8, 0.9]
    mock_model.get_roc_curve.return_value = mock_roc_curve

    with (
        patch(
            "datarobot_genai.drmcp.tools.predictive.training.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drmcp.tools.predictive.training.DataRobotClient") as mock_drc,
    ):
        mock_client = MagicMock()
        mock_client.Project.get.return_value = mock_project
        mock_client.Model.get.return_value = mock_model
        mock_drc.return_value.get_client.return_value = mock_client
        result = await training.get_model_roc_curve(
            project_id="test_project_id", model_id="test_model_id"
        )
        assert hasattr(result, "structured_content")
        response = result.structured_content

        assert "data" in response
        assert "roc_points" in response["data"]
        assert len(response["data"]["roc_points"]) == 1


@pytest.mark.asyncio
async def test_get_model_feature_impact() -> None:
    mock_project = MagicMock()
    mock_model = MagicMock()
    mock_feature_impact = [
        {"feature": "feature1", "impact": 0.8},
        {"feature": "feature2", "impact": 0.5},
    ]
    mock_model.get_or_request_feature_impact.return_value = mock_feature_impact

    with (
        patch(
            "datarobot_genai.drmcp.tools.predictive.training.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drmcp.tools.predictive.training.DataRobotClient") as mock_drc,
    ):
        mock_client = MagicMock()
        mock_client.Project.get.return_value = mock_project
        mock_client.Model.get.return_value = mock_model
        mock_drc.return_value.get_client.return_value = mock_client
        result = await training.get_model_feature_impact(
            project_id="test_project_id", model_id="test_model_id"
        )
        assert hasattr(result, "structured_content")
        response = result.structured_content

        assert "data" in response
        assert len(response["data"]) == 2


@pytest.mark.asyncio
async def test_get_model_lift_chart() -> None:
    mock_project = MagicMock()
    mock_model = MagicMock()
    mock_lift_chart = MagicMock()
    mock_lift_chart.bins = [
        {"actual": 0.8, "predicted": 0.75, "bin_weight": 0.2},
        {"actual": 0.6, "predicted": 0.55, "bin_weight": 0.3},
    ]
    mock_lift_chart.source_model_id = "test_model_id"
    mock_lift_chart.target_class = "class1"
    mock_model.get_lift_chart.return_value = mock_lift_chart

    with (
        patch(
            "datarobot_genai.drmcp.tools.predictive.training.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drmcp.tools.predictive.training.DataRobotClient") as mock_drc,
    ):
        mock_client = MagicMock()
        mock_client.Project.get.return_value = mock_project
        mock_client.Model.get.return_value = mock_model
        mock_drc.return_value.get_client.return_value = mock_client
        result = await training.get_model_lift_chart(
            project_id="test_project_id", model_id="test_model_id"
        )
        assert hasattr(result, "structured_content")
        response = result.structured_content

        assert "data" in response
        assert "bins" in response["data"]
        assert len(response["data"]["bins"]) == 2


@pytest.mark.asyncio
async def test_start_autopilot_validation() -> None:
    """Test validation of input parameters for start_autopilot."""
    # Test missing dataset info for new project
    with (
        patch(
            "datarobot_genai.drmcp.tools.predictive.training.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drmcp.tools.predictive.training.DataRobotClient") as mock_drc,
    ):
        mock_drc.return_value.get_client.return_value = MagicMock()
        with pytest.raises(
            ToolError,
            match=(
                "Error in start_autopilot: ToolError: "
                "Either dataset_url or dataset_id must be provided"
            ),
        ):
            await training.start_autopilot(target="target")

        # Test conflicting dataset inputs
        with pytest.raises(
            ToolError,
            match=(
                "Error in start_autopilot: ToolError: "
                "Please provide either dataset_url or dataset_id, not both"
            ),
        ):
            await training.start_autopilot(
                target="target",
                dataset_url="http://test.com/data.csv",
                dataset_id="test_dataset_id",
            )

        # Test missing target
        with pytest.raises(
            ToolError,
            match="Error in start_autopilot: ToolError: Target variable must be specified",
        ):
            await training.start_autopilot(target="", dataset_url="http://test.com/data.csv")


@pytest.mark.asyncio
async def test_suggest_use_cases_dataset_not_found() -> None:
    """Test that suggest_use_cases provides a clear error message when dataset is not found."""

    # Create a mock ClientError that matches the DataRobot SDK error format
    class MockClientError(Exception):
        def __init__(self, message: str, status_code: int = 404):
            self.message = message
            self.status_code = status_code
            super().__init__(f"{status_code} client error: {message}")

    with (
        patch(
            "datarobot_genai.drmcp.tools.predictive.training.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drmcp.tools.predictive.training.DataRobotClient") as mock_drc,
    ):
        mock_client = MagicMock()
        mock_client.Dataset.get.side_effect = MockClientError(
            "{'message': 'Not Found'}", status_code=404
        )
        mock_drc.return_value.get_client.return_value = mock_client
        with pytest.raises(
            ToolError,
            match=r"Dataset 'nonexistent_dataset_id' not found",
        ):
            await training.suggest_use_cases(dataset_id="nonexistent_dataset_id")
