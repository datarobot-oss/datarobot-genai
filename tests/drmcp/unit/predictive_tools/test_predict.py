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
from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch

import datarobot as dr
import pytest

from datarobot_genai.drmcp.core.shared import MCPError
from datarobot_genai.drmcp.tools.predictive import predict

FEATURE_VALUE = 0.5


@pytest.fixture()
def patch_predict_dependencies() -> Generator[dict[str, Any], None, None]:
    with (
        patch(
            "datarobot_genai.drmcp.tools.predictive.predict.get_or_create_s3_credential"
        ) as mock_cred,
        patch(
            "datarobot_genai.drmcp.tools.predictive.predict.make_output_settings",
            side_effect=lambda cred: {
                "url": "s3://bucket/key",
                "credential_id": "cid",
                "type": "s3",
            },
        ),
        patch(
            "datarobot_genai.drmcp.tools.predictive.predict.dr.BatchPredictionJob"
        ) as mock_batch_job,
        patch(
            "datarobot_genai.drmcp.tools.predictive.predict.generate_presigned_url",
            return_value="https://dummy-presigned-url",
        ),
    ):
        yield {
            "mock_cred": mock_cred,
            "mock_make_output_settings": None,  # not used directly
            "mock_batch_job": mock_batch_job,
            "mock_generate_presigned_url": None,  # not used directly
        }


@pytest.mark.asyncio
async def test_predict_by_file_path(
    patch_predict_dependencies: dict[str, Any],
) -> None:
    mock_job = MagicMock()
    mock_job.id = "jobid"
    mock_job.status = "COMPLETED"
    patch_predict_dependencies["mock_batch_job"].score.return_value = mock_job
    patch_predict_dependencies["mock_cred"].return_value = MagicMock(credential_id="cid")
    result = await predict.predict_by_file_path("dep", "file.csv", 5)
    patch_predict_dependencies["mock_batch_job"].score.assert_called_once()
    assert "Finished Batch Prediction job ID jobid" in result


@pytest.mark.asyncio
async def test_predict_by_ai_catalog(
    patch_predict_dependencies: dict[str, Any],
    monkeypatch: Any,
) -> None:
    with patch("datarobot_genai.drmcp.tools.predictive.predict.uuid.uuid4", return_value="uuid"):
        mock_job = MagicMock()
        mock_job.id = "jobid"
        mock_job.status = "COMPLETED"
        patch_predict_dependencies["mock_batch_job"].score.return_value = mock_job
        mock_dataset = MagicMock()
        mock_client = MagicMock()
        mock_client.Dataset.get.return_value = mock_dataset
        monkeypatch.setattr(predict, "get_sdk_client", lambda: mock_client)
        result = await predict.predict_by_ai_catalog("dep", "dsid", 5)
        patch_predict_dependencies["mock_batch_job"].score.assert_called_once()
        assert "Finished Batch Prediction job ID jobid" in result


@pytest.mark.asyncio
async def test_predict_from_project_data(
    patch_predict_dependencies: dict[str, Any],
) -> None:
    with patch("datarobot_genai.drmcp.tools.predictive.predict.uuid.uuid4", return_value="uuid"):
        mock_job = MagicMock()
        mock_job.id = "jobid"
        mock_job.status = "COMPLETED"
        patch_predict_dependencies["mock_batch_job"].score.return_value = mock_job
        result = await predict.predict_from_project_data("dep", "pid", "dsid", "holdout", 5)
        patch_predict_dependencies["mock_batch_job"].score.assert_called_once()
        assert "Finished Batch Prediction job ID jobid" in result


@pytest.mark.asyncio
async def test_predict_by_file_path_timeout(
    patch_predict_dependencies: dict[str, Any],
) -> None:
    mock_job = MagicMock()
    mock_job.id = "jobid"
    mock_job.status = "IN_PROGRESS"
    mock_job.wait_for_completion.side_effect = dr.errors.AsyncTimeoutError(
        "Job did not complete within the timeout period."
    )
    patch_predict_dependencies["mock_batch_job"].score.return_value = mock_job
    patch_predict_dependencies["mock_cred"].return_value = MagicMock(credential_id="cid")
    with pytest.raises(MCPError) as exc_info:
        await predict.predict_by_file_path("dep", "file.csv", 1)
    assert (
        "Error in predict_by_file_path: AsyncTimeoutError: Job did not complete within the "
        "timeout period." == str(exc_info.value)
    )


@pytest.mark.asyncio
async def test_predict_by_file_path_failure_error(
    patch_predict_dependencies: dict[str, Any],
) -> None:
    mock_job = MagicMock()
    mock_job.id = "jobid"
    mock_job.status = "FAILED"
    mock_job.wait_for_completion.side_effect = dr.errors.AsyncFailureError(
        "Job failed for some reason."
    )
    patch_predict_dependencies["mock_batch_job"].score.return_value = mock_job
    patch_predict_dependencies["mock_cred"].return_value = MagicMock(credential_id="cid")
    with pytest.raises(MCPError) as exc_info:
        await predict.predict_by_file_path("dep", "file.csv", 1)
    assert "Error in predict_by_file_path: AsyncFailureError: Job failed for some reason." == str(
        exc_info.value
    )


@pytest.mark.asyncio
async def test_predict_by_file_path_unsuccessful_error(
    patch_predict_dependencies: dict[str, Any],
) -> None:
    mock_job = MagicMock()
    mock_job.id = "jobid"
    mock_job.status = "FAILED"
    mock_job.wait_for_completion.side_effect = dr.errors.AsyncProcessUnsuccessfulError(
        "Job was unsuccessful."
    )
    patch_predict_dependencies["mock_batch_job"].score.return_value = mock_job
    patch_predict_dependencies["mock_cred"].return_value = MagicMock(credential_id="cid")
    with pytest.raises(MCPError) as exc_info:
        await predict.predict_by_file_path("dep", "file.csv", 1)
    assert (
        "Error in predict_by_file_path: AsyncProcessUnsuccessfulError: Job was unsuccessful."
        == str(exc_info.value)
    )


def test_get_or_create_s3_credential_create(monkeypatch: Any) -> None:
    mock_cred = MagicMock()
    monkeypatch.setattr(predict.dr.Credential, "list", lambda: [])
    monkeypatch.setattr(predict.dr.Credential, "create_s3", lambda **kwargs: mock_cred)
    # Mock get_credentials to return credentials with AWS creds
    mock_credentials = MagicMock()
    mock_credentials.has_aws_credentials.return_value = True
    mock_credentials.get_aws_credentials.return_value = (
        "test-key",
        "test-secret",
        None,
    )
    monkeypatch.setattr(predict, "get_credentials", lambda: mock_credentials)
    cred = predict.get_or_create_s3_credential()
    assert cred is mock_cred


def test_get_or_create_s3_credential_existing(monkeypatch: Any) -> None:
    mock_cred = MagicMock()
    mock_cred.name = "dr_mcp_server_temp_storage_s3_cred"
    monkeypatch.setattr(predict.dr.Credential, "list", lambda: [mock_cred])
    monkeypatch.setattr(
        predict.dr.Credential,
        "create_s3",
        lambda **kwargs: (_ for _ in ()).throw(Exception("Should not be called")),
    )
    cred = predict.get_or_create_s3_credential()
    assert cred is mock_cred


def test_make_output_settings() -> None:
    class Cred:
        credential_id = "cid"

    out, bucket, key = predict.make_output_settings(Cred())
    assert out["credential_id"] == "cid"
    assert out["type"] == "s3"
    assert bucket in out["url"]
    assert key in out["url"]


@pytest.mark.asyncio
async def test_get_prediction_explanations_basic(monkeypatch: Any) -> None:
    mock_model = MagicMock()
    mock_model.get_or_request_prediction_explanations = MagicMock(
        return_value=[
            {"row_id": 1, "feature1": 0.5, "feature2": -0.2},
            {"row_id": 2, "feature1": 0.1, "feature2": 0.3},
        ]
    )
    mock_project = MagicMock()
    mock_client = MagicMock()
    mock_client.Project.get.return_value = mock_project
    mock_client.Model.get.return_value = mock_model
    monkeypatch.setattr(predict, "get_sdk_client", lambda: mock_client)

    result_json = await predict.get_prediction_explanations(
        project_id="pid", model_id="mid", dataset_id="dsid", max_explanations=2
    )
    result = result_json if isinstance(result_json, dict) else json.loads(result_json)
    assert "explanations" in result
    assert isinstance(result["explanations"], list)
    assert result["ui_panel"] == ["prediction-distribution"]
    assert result["explanations"][0]["feature1"] == FEATURE_VALUE


@pytest.mark.asyncio
async def test_get_prediction_explanations_empty(monkeypatch: Any) -> None:
    mock_model = MagicMock()
    mock_model.get_or_request_prediction_explanations = MagicMock(return_value=[])
    mock_project = MagicMock()
    mock_client = MagicMock()
    mock_client.Project.get.return_value = mock_project
    mock_client.Model.get.return_value = mock_model
    monkeypatch.setattr(predict, "get_sdk_client", lambda: mock_client)

    result_json = await predict.get_prediction_explanations(
        project_id="pid", model_id="mid", dataset_id="dsid"
    )
    result = result_json if isinstance(result_json, dict) else json.loads(result_json)
    assert "explanations" in result
    assert result["explanations"] == []
    assert result["ui_panel"] == ["prediction-distribution"]


@pytest.mark.asyncio
async def test_get_prediction_explanations_sdk_error(monkeypatch: Any) -> None:
    mock_model = MagicMock()
    mock_model.get_or_request_prediction_explanations = MagicMock(
        side_effect=Exception("SDK error")
    )
    mock_project = MagicMock()
    mock_client = MagicMock()
    mock_client.Project.get.return_value = mock_project
    mock_client.Model.get.return_value = mock_model
    monkeypatch.setattr(predict, "get_sdk_client", lambda: mock_client)

    result_json = await predict.get_prediction_explanations(
        project_id="pid", model_id="mid", dataset_id="dsid"
    )
    result = result_json if isinstance(result_json, dict) else json.loads(result_json)
    assert "Error in get_prediction_explanations" in result.get("error", "")
