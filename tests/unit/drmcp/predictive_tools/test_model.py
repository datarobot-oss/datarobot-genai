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

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from datarobot_genai.drmcp.tools.predictive import model


@pytest.mark.asyncio
async def test_get_best_model_success() -> None:
    with patch("datarobot_genai.drmcp.tools.predictive.model.get_sdk_client") as mock_get_client:
        mock_client = MagicMock()
        mock_project = MagicMock()
        mock_model1 = MagicMock(id="m1", model_type="XGBoost", metrics={"AUC": {"validation": 0.9}})
        mock_model2 = MagicMock(
            id="m2", model_type="Random Forest", metrics={"AUC": {"validation": 0.8}}
        )
        mock_project.get_models.return_value = [mock_model1, mock_model2]
        mock_client.Project.get.return_value = mock_project
        mock_get_client.return_value = mock_client

        result = await model.get_best_model("pid", "AUC")
        assert result == "Best model: XGBoost with AUC: 0.90"


@pytest.mark.asyncio
async def test_get_best_model_no_models() -> None:
    with patch("datarobot_genai.drmcp.tools.predictive.model.get_sdk_client") as mock_get_client:
        mock_client = MagicMock()
        mock_project = MagicMock()
        mock_project.get_models.return_value = []
        mock_client.Project.get.return_value = mock_project
        mock_get_client.return_value = mock_client
        with pytest.raises(Exception) as exc_info:
            await model.get_best_model("pid", "AUC")
        assert "Error in get_best_model: Exception: No models found for this project." == str(
            exc_info.value
        )


@pytest.mark.asyncio
async def test_get_best_model_project_not_found() -> None:
    with patch("datarobot_genai.drmcp.tools.predictive.model.get_sdk_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.Project.get.return_value = None
        mock_get_client.return_value = mock_client
        with pytest.raises(Exception) as exc_info:
            await model.get_best_model("pid", "AUC")
        assert "Error in get_best_model: Exception: Project with ID pid not found." == str(
            exc_info.value
        )


@pytest.mark.asyncio
async def test_get_best_model_error() -> None:
    with patch(
        "datarobot_genai.drmcp.tools.predictive.model.get_sdk_client", side_effect=Exception("fail")
    ):
        with pytest.raises(Exception) as exc_info:
            await model.get_best_model("pid", "AUC")
        assert "Error in get_best_model: Exception: fail" == str(exc_info.value)


@pytest.mark.asyncio
async def test_score_dataset_with_model_success() -> None:
    with patch("datarobot_genai.drmcp.tools.predictive.model.get_sdk_client") as mock_get_client:
        mock_client = MagicMock()
        mock_project = MagicMock()
        mock_model = MagicMock()
        mock_job = MagicMock(id="jobid")
        mock_model.score.return_value = mock_job
        mock_model.model_type = "type1"
        mock_model.metrics = {"AUC": 0.9}
        mock_client.Model.get.return_value = mock_model
        mock_client.Project.get.return_value = mock_project
        mock_get_client.return_value = mock_client
        result = await model.score_dataset_with_model("pid", "mid", "url")
        mock_client.Project.get.assert_called_once_with("pid")
        mock_client.Model.get.assert_called_once_with(mock_project, "mid")
        mock_model.score.assert_called_once_with("url")
        assert "Scoring job started: jobid" in result


@pytest.mark.asyncio
async def test_score_dataset_with_model_project_not_found() -> None:
    with patch("datarobot_genai.drmcp.tools.predictive.model.get_sdk_client") as mock_get_client:
        project_id = "pid"
        mock_client = MagicMock()
        exception_message = (
            "404 client error: {'message': 'Project with ID " + project_id + " does not exist'}"
        )
        mock_client.Project.get.side_effect = Exception(exception_message)
        mock_get_client.return_value = mock_client
        with pytest.raises(Exception) as exc_info:
            await model.score_dataset_with_model(project_id, "mid", "url")
        assert "Error in score_dataset_with_model: Exception: " + exception_message == str(
            exc_info.value
        )


@pytest.mark.asyncio
async def test_score_dataset_with_model_model_not_found() -> None:
    with patch("datarobot_genai.drmcp.tools.predictive.model.get_sdk_client") as mock_get_client:
        mock_client = MagicMock()
        mock_project = MagicMock()
        mock_project.get_models.return_value = []
        mock_client.Project.get.return_value = mock_project
        exception_message = "404 client error: {'message': 'Leaderboard Item Not Found'}"
        mock_client.Model.get.side_effect = Exception(exception_message)
        mock_get_client.return_value = mock_client
        with pytest.raises(Exception) as exc_info:
            await model.score_dataset_with_model("pid", "mid", "url")
        assert "Error in score_dataset_with_model: Exception: " + exception_message == str(
            exc_info.value
        )


@pytest.mark.asyncio
async def test_score_dataset_with_model_error() -> None:
    with patch(
        "datarobot_genai.drmcp.tools.predictive.model.get_sdk_client", side_effect=Exception("fail")
    ):
        with pytest.raises(Exception) as exc_info:
            await model.score_dataset_with_model("pid", "mid", "url")
        assert "Error in score_dataset_with_model: Exception: fail" == str(exc_info.value)
