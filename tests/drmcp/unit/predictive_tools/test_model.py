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
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import polars as pl
import pytest
from fastmcp.exceptions import ToolError

from datarobot_genai.drtools.predictive import model
from datarobot_genai.drtools.predictive.model import ModelEncoder
from datarobot_genai.drtools.predictive.model import model_to_dict


def _patch_model_client(mock_client: MagicMock):
    """Return (patch get_datarobot_access_token, patch DataRobotClient).
    Use: with p1, p2 as mock_drc.
    """
    return (
        patch(
            "datarobot_genai.drtools.predictive.model.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drtools.predictive.model.DataRobotClient"),
    )


@pytest.mark.asyncio
async def test_get_best_model_success() -> None:
    mock_client = MagicMock()
    mock_project = MagicMock()
    mock_model1 = MagicMock(id="m1", model_type="XGBoost", metrics={"AUC": {"validation": 0.9}})
    mock_model2 = MagicMock(
        id="m2", model_type="Random Forest", metrics={"AUC": {"validation": 0.8}}
    )
    mock_project.get_models.return_value = [mock_model1, mock_model2]
    mock_client.Project.get.return_value = mock_project
    p1, p2 = _patch_model_client(mock_client)
    with p1, p2 as mock_drc:
        mock_drc.return_value.get_client.return_value = mock_client
        result = await model.get_best_model(project_id="pid", metric="AUC")
    assert isinstance(result, dict)
    assert result["project_id"] == "pid"
    assert result["best_model"]["model_type"] == "XGBoost"
    expected_auc = 0.9
    assert result["best_model"]["metrics"]["AUC"]["validation"] == expected_auc


@pytest.mark.asyncio
async def test_get_best_model_no_models() -> None:
    mock_client = MagicMock()
    mock_project = MagicMock()
    mock_project.get_models.return_value = []
    mock_client.Project.get.return_value = mock_project
    p1, p2 = _patch_model_client(mock_client)
    with p1, p2 as mock_drc:
        mock_drc.return_value.get_client.return_value = mock_client
        with pytest.raises(
            ToolError, match="Error in get_best_model: ToolError: No models found for this project."
        ):
            await model.get_best_model(project_id="pid", metric="AUC")


@pytest.mark.asyncio
async def test_get_best_model_project_not_found() -> None:
    mock_client = MagicMock()
    mock_client.Project.get.return_value = None
    p1, p2 = _patch_model_client(mock_client)
    with p1, p2 as mock_drc:
        mock_drc.return_value.get_client.return_value = mock_client
        with pytest.raises(
            ToolError, match="Error in get_best_model: ToolError: Project with ID pid not found."
        ):
            await model.get_best_model(project_id="pid", metric="AUC")


@pytest.mark.asyncio
async def test_get_best_model_error() -> None:
    with patch(
        "datarobot_genai.drtools.predictive.model.get_datarobot_access_token",
        new_callable=AsyncMock,
        side_effect=Exception("fail"),
    ):
        with pytest.raises(Exception) as exc_info:
            await model.get_best_model(project_id="pid", metric="AUC")
        assert "Error in get_best_model: Exception: fail" == str(exc_info.value)


@pytest.mark.asyncio
async def test_score_dataset_with_model_success() -> None:
    mock_client = MagicMock()
    mock_project = MagicMock()
    mock_model = MagicMock()
    mock_job = MagicMock(id="jobid")
    mock_model.score.return_value = mock_job
    mock_model.model_type = "type1"
    mock_model.metrics = {"AUC": 0.9}
    mock_client.Model.get.return_value = mock_model
    mock_client.Project.get.return_value = mock_project
    p1, p2 = _patch_model_client(mock_client)
    with p1, p2 as mock_drc:
        mock_drc.return_value.get_client.return_value = mock_client
        result = await model.score_dataset_with_model(
            project_id="pid", model_id="mid", dataset_url="url"
        )
    mock_client.Project.get.assert_called_once_with("pid")
    mock_client.Model.get.assert_called_once_with(mock_project, "mid")
    mock_model.score.assert_called_once_with("url")
    assert isinstance(result, dict)
    assert result["scoring_job_id"] == "jobid"


@pytest.mark.asyncio
async def test_score_dataset_with_model_project_not_found() -> None:
    project_id = "pid"
    mock_client = MagicMock()
    exception_message = (
        "404 client error: {'message': 'Project with ID " + project_id + " does not exist'}"
    )
    mock_client.Project.get.side_effect = Exception(exception_message)
    p1, p2 = _patch_model_client(mock_client)
    with p1, p2 as mock_drc:
        mock_drc.return_value.get_client.return_value = mock_client
        with pytest.raises(Exception) as exc_info:
            await model.score_dataset_with_model(
                project_id=project_id, model_id="mid", dataset_url="url"
            )
    assert "Error in score_dataset_with_model: Exception: " + exception_message == str(
        exc_info.value
    )


@pytest.mark.asyncio
async def test_score_dataset_with_model_model_not_found() -> None:
    mock_client = MagicMock()
    mock_project = MagicMock()
    mock_project.get_models.return_value = []
    mock_client.Project.get.return_value = mock_project
    exception_message = "404 client error: {'message': 'Leaderboard Item Not Found'}"
    mock_client.Model.get.side_effect = Exception(exception_message)
    p1, p2 = _patch_model_client(mock_client)
    with p1, p2 as mock_drc:
        mock_drc.return_value.get_client.return_value = mock_client
        with pytest.raises(Exception) as exc_info:
            await model.score_dataset_with_model(
                project_id="pid", model_id="mid", dataset_url="url"
            )
    assert "Error in score_dataset_with_model: Exception: " + exception_message == str(
        exc_info.value
    )


@pytest.mark.asyncio
async def test_score_dataset_with_model_error() -> None:
    with patch(
        "datarobot_genai.drtools.predictive.model.get_datarobot_access_token",
        new_callable=AsyncMock,
        side_effect=Exception("fail"),
    ):
        with pytest.raises(Exception) as exc_info:
            await model.score_dataset_with_model(
                project_id="pid", model_id="mid", dataset_url="url"
            )
        assert "Error in score_dataset_with_model: Exception: fail" == str(exc_info.value)


@pytest.mark.asyncio
async def test_get_model_details_success() -> None:
    mock_client = MagicMock()
    mock_project = MagicMock()
    mock_project.target = "target_col"
    mock_project.metric = "AUC"
    mock_model = MagicMock(
        id="mid",
        model_type="XGBoost",
        featurelist_name="Informative Features",
        metrics={"AUC": {"validation": 0.9}},
        sample_pct=64,
    )
    mock_model.get_or_request_feature_impact.return_value = [{"feature": "f1", "impact": 0.8}]
    mock_client.Project.get.return_value = mock_project
    mock_client.Model.get.return_value = mock_model
    p1, p2 = _patch_model_client(mock_client)
    with p1, p2 as mock_drc:
        mock_drc.return_value.get_client.return_value = mock_client
        result = await model.get_model_details(project_id="pid", model_id="mid")
    assert isinstance(result, dict)
    assert result["model_id"] == "mid"
    assert result["model_type"] == "XGBoost"
    assert result["target"] == "target_col"


@pytest.mark.asyncio
async def test_get_model_details_missing_project_id() -> None:
    """Required param project_id is enforced by signature (wrapped as ToolError)."""
    with pytest.raises(ToolError, match="project_id"):
        await model.get_model_details(model_id="mid")


@pytest.mark.asyncio
async def test_get_model_details_missing_model_id() -> None:
    """Required param model_id is enforced by signature (wrapped as ToolError)."""
    with pytest.raises(ToolError, match="model_id"):
        await model.get_model_details(project_id="pid")


@pytest.mark.asyncio
async def test_get_model_details_feature_impact_error() -> None:
    mock_client = MagicMock()
    mock_project = MagicMock()
    mock_project.target = "target_col"
    mock_project.metric = "AUC"
    mock_model = MagicMock(id="mid", model_type="XGBoost", metrics={}, sample_pct=64)
    mock_model.get_or_request_feature_impact.side_effect = Exception("not available")
    mock_client.Project.get.return_value = mock_project
    mock_client.Model.get.return_value = mock_model
    p1, p2 = _patch_model_client(mock_client)
    with p1, p2 as mock_drc:
        mock_drc.return_value.get_client.return_value = mock_client
        result = await model.get_model_details(
            project_id="pid", model_id="mid", include_feature_impact=True
        )
    assert "feature_impact_error" in result


@pytest.mark.asyncio
async def test_is_eligible_for_timeseries_training_success() -> None:
    mock_client = MagicMock()
    mock_dataset = MagicMock()
    # Build with polars, convert to pandas at boundary (SDK returns pandas)
    dates = pl.date_range(
        pl.date(2020, 1, 1), pl.date(2020, 1, 1) + pl.duration(days=199), eager=True
    )
    pandas_df = pl.DataFrame(
        {
            "date": dates,
            "target": range(200),
            "feature": range(200),
        }
    ).to_pandas()
    mock_dataset.get_as_dataframe.return_value = pandas_df
    mock_client.Dataset.get.return_value = mock_dataset
    with (
        patch(
            "datarobot_genai.drtools.predictive.model.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drtools.predictive.model.DataRobotClient") as mock_drc,
    ):
        mock_drc.return_value.get_client.return_value = mock_client
        result = await model.is_eligible_for_timeseries_training(
            dataset_id="ds1", datetime_column="date", target_column="target"
        )
    assert result["status"] == "ELIGIBLE"
    assert result["errors"] == []


@pytest.mark.asyncio
async def test_is_eligible_for_timeseries_training_too_few_rows() -> None:
    mock_client = MagicMock()
    mock_dataset = MagicMock()
    # Build with polars, convert to pandas at boundary (SDK returns pandas)
    dates = pl.date_range(
        pl.date(2020, 1, 1), pl.date(2020, 1, 1) + pl.duration(days=49), eager=True
    )
    pandas_df = pl.DataFrame(
        {
            "date": dates,
            "target": range(50),
        }
    ).to_pandas()
    mock_dataset.get_as_dataframe.return_value = pandas_df
    mock_client.Dataset.get.return_value = mock_dataset
    with (
        patch(
            "datarobot_genai.drtools.predictive.model.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drtools.predictive.model.DataRobotClient") as mock_drc,
    ):
        mock_drc.return_value.get_client.return_value = mock_client
        result = await model.is_eligible_for_timeseries_training(
            dataset_id="ds1", datetime_column="date", target_column="target"
        )
    assert result["status"] == "NOT_ELIGIBLE"
    assert any("Too few rows" in e for e in result["errors"])


@pytest.mark.asyncio
async def test_is_eligible_for_timeseries_training_missing_params() -> None:
    with pytest.raises(ToolError, match="Error in is_eligible_for_timeseries_training"):
        await model.is_eligible_for_timeseries_training(
            datetime_column="date", target_column="target"
        )


class TestModelToDict:
    """Test cases for model_to_dict function."""

    def test_model_to_dict_success(self):
        """Test successful conversion of model to dict."""
        model = Mock()
        model.id = "model123"
        model.model_type = "regression"
        model.metrics = {"AUC": 0.85}

        result = model_to_dict(model)

        assert result == {"id": "model123", "model_type": "regression", "metrics": {"AUC": 0.85}}

    def test_model_to_dict_attribute_error(self):
        """Test model_to_dict handles AttributeError gracefully."""

        # Create a class that raises AttributeError when accessing model_type
        class ModelWithError:
            def __init__(self):
                self.id = "model123"
                self.metrics = {"AUC": 0.85}

            @property
            def model_type(self):
                raise AttributeError("No model_type")

        model = ModelWithError()
        result = model_to_dict(model)

        assert result == {"id": "model123", "model_type": "unknown"}

    def test_model_to_dict_all_attributes_missing(self):
        """Test model_to_dict when all attributes are missing."""

        # Create a class that raises AttributeError when accessing any attribute
        class ModelWithAllErrors:
            @property
            def id(self):
                raise AttributeError("No id")

            @property
            def model_type(self):
                raise AttributeError("No model_type")

            @property
            def metrics(self):
                raise AttributeError("No metrics")

        model = ModelWithAllErrors()
        result = model_to_dict(model)

        assert result == {"id": "unknown", "model_type": "unknown"}

    def test_model_to_dict_partial_attributes_missing(self):
        """Test model_to_dict when some attributes are missing."""

        # Create a class that raises AttributeError when accessing model_type
        class ModelWithPartialError:
            def __init__(self):
                self.id = "model123"
                self.metrics = {"AUC": 0.85}

            @property
            def model_type(self):
                raise AttributeError("No model_type")

        model = ModelWithPartialError()
        result = model_to_dict(model)

        assert result == {"id": "model123", "model_type": "unknown"}


class TestModelEncoder:
    """Test cases for ModelEncoder class."""

    def test_model_encoder_with_model_object(self):
        """Test ModelEncoder with DataRobot Model object."""
        # Create a mock that behaves like a Model object
        model = Mock()
        model.id = "model123"
        model.model_type = "regression"
        model.metrics = {"AUC": 0.85}

        # Mock the isinstance check to return True for Model
        with patch("datarobot_genai.drtools.predictive.model.isinstance", return_value=True):
            encoder = ModelEncoder()
            result = encoder.default(model)

            assert result == {
                "id": "model123",
                "model_type": "regression",
                "metrics": {"AUC": 0.85},
            }

    def test_model_encoder_with_non_model_object(self):
        """Test ModelEncoder with non-Model object falls back to default."""
        encoder = ModelEncoder()

        # Test with a string - should raise TypeError
        with pytest.raises(TypeError, match="Object of type str is not JSON serializable"):
            encoder.default("test_string")

    def test_model_encoder_json_serialization(self):
        """Test ModelEncoder with JSON serialization."""
        # Create a mock that behaves like a Model object
        model = Mock()
        model.id = "model123"
        model.model_type = "regression"
        model.metrics = {"AUC": 0.85}

        # Mock the isinstance check to return True for Model
        with patch("datarobot_genai.drtools.predictive.model.isinstance", return_value=True):
            data = {"model": model, "other": "value"}
            result = json.dumps(data, cls=ModelEncoder)

            expected = (
                '{"model": {"id": "model123", "model_type": "regression", '
                '"metrics": {"AUC": 0.85}}, "other": "value"}'
            )
            assert result == expected
