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

from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import datarobot as dr
import pytest

from datarobot_genai.drtools.core.constants import MAX_INLINE_SIZE
from datarobot_genai.drtools.core.exceptions import ToolError
from datarobot_genai.drtools.predictive import predict

DOWNLOAD_URL = "https://app.example/api/v2/batchPredictions/jobid/download/"


def _completed_job() -> MagicMock:
    mock_job = MagicMock()
    mock_job.id = "jobid"
    mock_job.status = "COMPLETED"
    mock_job.get_status.return_value = {"links": {"download": DOWNLOAD_URL}}
    return mock_job


@pytest.fixture()
def patch_predict_dependencies() -> Generator[dict[str, Any], None, None]:
    with patch(
        "datarobot_genai.drtools.predictive.predict.dr.BatchPredictionJob"
    ) as mock_batch_job:
        yield {"mock_batch_job": mock_batch_job}


@pytest.mark.asyncio
async def test_predict_by_ai_catalog(
    patch_predict_dependencies: dict[str, Any],
) -> None:
    with (
        patch(
            "datarobot_genai.drtools.predictive.predict.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drtools.predictive.predict.DataRobotClient") as mock_drc,
    ):
        mock_job = _completed_job()
        patch_predict_dependencies["mock_batch_job"].score.return_value = mock_job
        mock_dataset = MagicMock()
        mock_client = MagicMock()
        mock_client.Dataset.get.return_value = mock_dataset
        mock_drc.return_value.get_client.return_value = mock_client
        result = await predict.predict_by_ai_catalog(
            deployment_id="dep", dataset_id="dsid", timeout=5
        )
        patch_predict_dependencies["mock_batch_job"].score.assert_called_once_with(
            deployment="dep",
            intake_settings={"type": "dataset", "dataset": mock_dataset},
            output_settings=None,
        )
        assert isinstance(result, dict)
        assert result["job_id"] == "jobid"
        assert result["deployment_id"] == "dep"
        assert "Scoring dataset dsid" in result["input_desc"]
        assert result["url"] == DOWNLOAD_URL


@pytest.mark.asyncio
async def test_predict_from_project_data(
    patch_predict_dependencies: dict[str, Any],
) -> None:
    with (
        patch(
            "datarobot_genai.drtools.predictive.predict.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drtools.predictive.predict.DataRobotClient") as mock_drc,
    ):
        mock_drc.return_value.get_client.return_value = MagicMock()
        mock_job = _completed_job()
        patch_predict_dependencies["mock_batch_job"].score.return_value = mock_job
        result = await predict.predict_from_project_data(
            deployment_id="dep",
            project_id="pid",
            dataset_id="dsid",
            partition="holdout",
            timeout=5,
        )
        patch_predict_dependencies["mock_batch_job"].score.assert_called_once_with(
            deployment="dep",
            intake_settings={
                "type": "dss",
                "project_id": "pid",
                "partition": "holdout",
                "dataset_id": "dsid",
            },
            output_settings=None,
        )
        assert isinstance(result, dict)
        assert result["job_id"] == "jobid"
        assert result["deployment_id"] == "dep"
        assert "Scoring project pid" in result["input_desc"]
        assert result["url"] == DOWNLOAD_URL


@pytest.mark.asyncio
async def test_predict_by_ai_catalog_timeout(
    patch_predict_dependencies: dict[str, Any],
) -> None:
    mock_job = MagicMock()
    mock_job.id = "jobid"
    mock_job.status = "IN_PROGRESS"
    mock_job.wait_for_completion.side_effect = dr.errors.AsyncTimeoutError(
        "Job did not complete within the timeout period."
    )
    with (
        patch(
            "datarobot_genai.drtools.predictive.predict.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drtools.predictive.predict.DataRobotClient") as mock_drc,
    ):
        mock_dataset = MagicMock()
        mock_client = MagicMock()
        mock_client.Dataset.get.return_value = mock_dataset
        mock_drc.return_value.get_client.return_value = mock_client
        patch_predict_dependencies["mock_batch_job"].score.return_value = mock_job
        with pytest.raises(dr.errors.AsyncTimeoutError) as exc_info:
            await predict.predict_by_ai_catalog(deployment_id="dep", dataset_id="dsid", timeout=1)
    assert "Job did not complete within the timeout period." == str(exc_info.value)


@pytest.mark.asyncio
async def test_predict_by_ai_catalog_failure_error(
    patch_predict_dependencies: dict[str, Any],
) -> None:
    mock_job = MagicMock()
    mock_job.id = "jobid"
    mock_job.status = "FAILED"
    mock_job.wait_for_completion.side_effect = dr.errors.AsyncFailureError(
        "Job failed for some reason."
    )
    with (
        patch(
            "datarobot_genai.drtools.predictive.predict.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drtools.predictive.predict.DataRobotClient") as mock_drc,
    ):
        mock_dataset = MagicMock()
        mock_client = MagicMock()
        mock_client.Dataset.get.return_value = mock_dataset
        mock_drc.return_value.get_client.return_value = mock_client
        patch_predict_dependencies["mock_batch_job"].score.return_value = mock_job
        with pytest.raises(dr.errors.AsyncFailureError) as exc_info:
            await predict.predict_by_ai_catalog(deployment_id="dep", dataset_id="dsid", timeout=1)
    assert "Job failed for some reason." == str(exc_info.value)


@pytest.mark.asyncio
async def test_predict_by_ai_catalog_unsuccessful_error(
    patch_predict_dependencies: dict[str, Any],
) -> None:
    mock_job = MagicMock()
    mock_job.id = "jobid"
    mock_job.status = "FAILED"
    mock_job.wait_for_completion.side_effect = dr.errors.AsyncProcessUnsuccessfulError(
        "Job was unsuccessful."
    )
    with (
        patch(
            "datarobot_genai.drtools.predictive.predict.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drtools.predictive.predict.DataRobotClient") as mock_drc,
    ):
        mock_dataset = MagicMock()
        mock_client = MagicMock()
        mock_client.Dataset.get.return_value = mock_dataset
        mock_drc.return_value.get_client.return_value = mock_client
        patch_predict_dependencies["mock_batch_job"].score.return_value = mock_job
        with pytest.raises(dr.errors.AsyncProcessUnsuccessfulError) as exc_info:
            await predict.predict_by_ai_catalog(deployment_id="dep", dataset_id="dsid", timeout=1)
    assert "Job was unsuccessful." == str(exc_info.value)


@pytest.mark.asyncio
async def test_predict_by_ai_catalog_missing_download_url(
    patch_predict_dependencies: dict[str, Any],
) -> None:
    mock_job = MagicMock()
    mock_job.id = "jobid"
    mock_job.status = "COMPLETED"
    mock_job.get_status.return_value = {"links": {}}
    with (
        patch(
            "datarobot_genai.drtools.predictive.predict.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drtools.predictive.predict.DataRobotClient") as mock_drc,
        patch("datarobot_genai.drtools.predictive.predict.time.sleep"),
    ):
        mock_dataset = MagicMock()
        mock_client = MagicMock()
        mock_client.Dataset.get.return_value = mock_dataset
        mock_drc.return_value.get_client.return_value = mock_client
        patch_predict_dependencies["mock_batch_job"].score.return_value = mock_job
        with pytest.raises(ToolError, match="no download URL"):
            await predict.predict_by_ai_catalog(deployment_id="dep", dataset_id="dsid", timeout=600)


@pytest.mark.asyncio
async def test_get_batch_prediction_results_success() -> None:
    mock_job = MagicMock()
    mock_job.id = "abc123"

    def _download(buf: Any, timeout: int = 120, read_timeout: int = 660) -> None:
        buf.write(b"x,y\n1,2\n")

    mock_job.download.side_effect = _download
    with (
        patch(
            "datarobot_genai.drtools.predictive.predict.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drtools.predictive.predict.DataRobotClient") as mock_drc,
        patch(
            "datarobot_genai.drtools.predictive.predict.dr.BatchPredictionJob.get",
            return_value=mock_job,
        ),
    ):
        mock_drc.return_value.get_client.return_value = MagicMock()
        result = await predict.get_batch_prediction_results(job_id="abc123")
    assert result["job_id"] == "abc123"
    assert result["mime_type"] == "text/csv"
    assert result["data"] == "x,y\n1,2\n"
    assert result["size_bytes"] == 8
    mock_job.download.assert_called_once()


@pytest.mark.asyncio
async def test_get_batch_prediction_results_empty_job_id() -> None:
    with pytest.raises(ToolError, match="job_id"):
        await predict.get_batch_prediction_results(job_id="   ")


@pytest.mark.asyncio
async def test_get_batch_prediction_results_too_large() -> None:
    mock_job = MagicMock()
    mock_job.id = "jid"
    mock_job.get_status.return_value = {"links": {"download": "https://example.com/batch/dl"}}

    def _download(buf: Any, timeout: int = 120, read_timeout: int = 660) -> None:
        buf.write(b"a" * (MAX_INLINE_SIZE + 1))

    mock_job.download.side_effect = _download
    with (
        patch(
            "datarobot_genai.drtools.predictive.predict.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drtools.predictive.predict.DataRobotClient") as mock_drc,
        patch(
            "datarobot_genai.drtools.predictive.predict.dr.BatchPredictionJob.get",
            return_value=mock_job,
        ),
    ):
        mock_drc.return_value.get_client.return_value = MagicMock()
        with pytest.raises(ToolError, match="exceeding the inline limit") as exc:
            await predict.get_batch_prediction_results(job_id="jid")
        assert "https://example.com/batch/dl" in str(exc.value)


@pytest.mark.asyncio
async def test_get_batch_prediction_results_download_runtime_error() -> None:
    mock_job = MagicMock()
    mock_job.id = "jid"
    mock_job.get_status.return_value = {"links": {"download": "https://example.com/batch/dl2"}}
    mock_job.download.side_effect = RuntimeError("queue full")
    with (
        patch(
            "datarobot_genai.drtools.predictive.predict.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drtools.predictive.predict.DataRobotClient") as mock_drc,
        patch(
            "datarobot_genai.drtools.predictive.predict.dr.BatchPredictionJob.get",
            return_value=mock_job,
        ),
    ):
        mock_drc.return_value.get_client.return_value = MagicMock()
        with pytest.raises(ToolError, match="queue full") as exc:
            await predict.get_batch_prediction_results(job_id="jid")
        assert "https://example.com/batch/dl2" in str(exc.value)
