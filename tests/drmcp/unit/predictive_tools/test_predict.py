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

from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from datarobot_genai.drtools.core.constants import MAX_INLINE_SIZE
from datarobot_genai.drtools.core.exceptions import ToolError
from datarobot_genai.drtools.predictive import predict

DOWNLOAD_URL = "https://app.example/api/v2/batchPredictions/jobid/download/"


def _running_job() -> MagicMock:
    mock_job = MagicMock()
    mock_job.id = "jobid"
    mock_job.get_status.return_value = {
        "status": "RUNNING",
        "links": {},
        "job_spec": {"output_settings": {"type": "localFile"}},
    }
    return mock_job


@pytest.fixture()
def mock_batch_prediction_job() -> MagicMock:
    return MagicMock()


@pytest.mark.asyncio
async def test_predict_by_ai_catalog(mock_batch_prediction_job: MagicMock) -> None:
    with patch("datarobot_genai.drtools.predictive.predict.dr_client") as mock_dr_client:
        mock_job = _running_job()
        mock_batch_prediction_job.score.return_value = mock_job
        mock_dataset = MagicMock()
        mock_client = MagicMock()
        mock_client.Dataset.get.return_value = mock_dataset
        mock_client.BatchPredictionJob = mock_batch_prediction_job
        mock_dr_client.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_dr_client.return_value.__aexit__ = AsyncMock(return_value=False)
        result = await predict.predict_by_ai_catalog(deployment_id="dep", dataset_id="dsid")
        mock_batch_prediction_job.score.assert_called_once_with(
            deployment="dep",
            intake_settings={"type": "dataset", "dataset": mock_dataset},
            output_settings=None,
        )
        assert isinstance(result, dict)
        assert result["job_id"] == "jobid"
        assert result["deployment_id"] == "dep"
        assert "Scoring dataset dsid" in result["input_desc"]
        assert result["batch_job_status"] == "RUNNING"
        assert result["url"] is None
        assert "get_batch_prediction_job_status" in result["note"]
        mock_job.wait_for_completion.assert_not_called()


@pytest.mark.asyncio
async def test_predict_by_ai_catalog_submit_returns_url_when_ready(
    mock_batch_prediction_job: MagicMock,
) -> None:
    mock_job = MagicMock()
    mock_job.id = "jobid"
    mock_job.get_status.return_value = {
        "status": "COMPLETED",
        "links": {"download": DOWNLOAD_URL},
        "job_spec": {"output_settings": {"type": "localFile"}},
    }
    with patch("datarobot_genai.drtools.predictive.predict.dr_client") as mock_dr_client:
        mock_dataset = MagicMock()
        mock_client = MagicMock()
        mock_client.Dataset.get.return_value = mock_dataset
        mock_client.BatchPredictionJob = mock_batch_prediction_job
        mock_batch_prediction_job.score.return_value = mock_job
        mock_dr_client.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_dr_client.return_value.__aexit__ = AsyncMock(return_value=False)
        result = await predict.predict_by_ai_catalog(deployment_id="dep", dataset_id="dsid")
    assert result["url"] == DOWNLOAD_URL
    assert result["batch_job_status"] == "COMPLETED"


@pytest.mark.asyncio
async def test_predict_by_ai_catalog_immediate_failure(
    mock_batch_prediction_job: MagicMock,
) -> None:
    mock_job = MagicMock()
    mock_job.id = "jobid"
    mock_job.get_status.return_value = {
        "status": "FAILED",
        "status_details": "bad",
        "links": {},
    }
    with patch("datarobot_genai.drtools.predictive.predict.dr_client") as mock_dr_client:
        mock_dataset = MagicMock()
        mock_client = MagicMock()
        mock_client.Dataset.get.return_value = mock_dataset
        mock_client.BatchPredictionJob = mock_batch_prediction_job
        mock_batch_prediction_job.score.return_value = mock_job
        mock_dr_client.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_dr_client.return_value.__aexit__ = AsyncMock(return_value=False)
        with pytest.raises(ToolError, match="FAILED"):
            await predict.predict_by_ai_catalog(deployment_id="dep", dataset_id="dsid")


@pytest.mark.asyncio
async def test_predict_by_ai_catalog_non_local_file_output_raises(
    mock_batch_prediction_job: MagicMock,
) -> None:
    mock_job = MagicMock()
    mock_job.id = "jobid"
    mock_job.get_status.return_value = {
        "status": "RUNNING",
        "links": {},
        "job_spec": {"output_settings": {"type": "jdbc"}},
    }
    with patch("datarobot_genai.drtools.predictive.predict.dr_client") as mock_dr_client:
        mock_dataset = MagicMock()
        mock_client = MagicMock()
        mock_client.Dataset.get.return_value = mock_dataset
        mock_client.BatchPredictionJob = mock_batch_prediction_job
        mock_batch_prediction_job.score.return_value = mock_job
        mock_dr_client.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_dr_client.return_value.__aexit__ = AsyncMock(return_value=False)
        with pytest.raises(ToolError, match="not local file streaming"):
            await predict.predict_by_ai_catalog(deployment_id="dep", dataset_id="dsid")


@pytest.mark.asyncio
async def test_predict_from_project_data(mock_batch_prediction_job: MagicMock) -> None:
    with patch("datarobot_genai.drtools.predictive.predict.dr_client") as mock_dr_client:
        mock_client = MagicMock()
        mock_client.BatchPredictionJob = mock_batch_prediction_job
        mock_dr_client.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_dr_client.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_job = _running_job()
        mock_batch_prediction_job.score.return_value = mock_job
        result = await predict.predict_from_project_data(
            deployment_id="dep",
            project_id="pid",
            dataset_id="dsid",
            partition="holdout",
        )
        mock_batch_prediction_job.score.assert_called_once_with(
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
        mock_job.wait_for_completion.assert_not_called()


@pytest.mark.asyncio
async def test_get_batch_prediction_job_status(mock_batch_prediction_job: MagicMock) -> None:
    mock_job = MagicMock()
    mock_job.id = "jid"
    mock_job.get_status.return_value = {
        "status": "COMPLETED",
        "links": {"download": "https://example.com/dl"},
        "percentage_completed": 100.0,
        "elapsed_time_sec": 12,
        "status_details": "",
        "job_spec": {"output_settings": {"type": "localFile"}},
    }
    mock_batch_prediction_job.get.return_value = mock_job
    with patch("datarobot_genai.drtools.predictive.predict.dr_client") as mock_dr_client:
        mock_client = MagicMock()
        mock_client.BatchPredictionJob = mock_batch_prediction_job
        mock_dr_client.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_dr_client.return_value.__aexit__ = AsyncMock(return_value=False)
        result = await predict.get_batch_prediction_job_status(job_id="jid")
    assert result["job_id"] == "jid"
    assert result["batch_job_status"] == "COMPLETED"
    assert result["url"] == "https://example.com/dl"
    assert result["percentage_completed"] == 100.0


@pytest.mark.asyncio
async def test_get_batch_prediction_job_status_non_local_file_raises(
    mock_batch_prediction_job: MagicMock,
) -> None:
    mock_job = MagicMock()
    mock_job.id = "jid"
    mock_job.get_status.return_value = {
        "status": "RUNNING",
        "links": {},
        "job_spec": {"output_settings": {"type": "s3"}},
    }
    mock_batch_prediction_job.get.return_value = mock_job
    with patch("datarobot_genai.drtools.predictive.predict.dr_client") as mock_dr_client:
        mock_client = MagicMock()
        mock_client.BatchPredictionJob = mock_batch_prediction_job
        mock_dr_client.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_dr_client.return_value.__aexit__ = AsyncMock(return_value=False)
        with pytest.raises(ToolError, match="not local file streaming"):
            await predict.get_batch_prediction_job_status(job_id="jid")


@pytest.mark.asyncio
async def test_get_batch_prediction_job_status_terminal(
    mock_batch_prediction_job: MagicMock,
) -> None:
    mock_job = MagicMock()
    mock_job.id = "jid"
    mock_job.get_status.return_value = {
        "status": "FAILED",
        "status_details": "boom",
        "links": {},
    }
    mock_batch_prediction_job.get.return_value = mock_job
    with patch("datarobot_genai.drtools.predictive.predict.dr_client") as mock_dr_client:
        mock_client = MagicMock()
        mock_client.BatchPredictionJob = mock_batch_prediction_job
        mock_dr_client.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_dr_client.return_value.__aexit__ = AsyncMock(return_value=False)
        with pytest.raises(ToolError, match="FAILED"):
            await predict.get_batch_prediction_job_status(job_id="jid")


@pytest.mark.asyncio
async def test_get_batch_prediction_job_status_empty_job_id() -> None:
    with pytest.raises(ToolError, match="job_id"):
        await predict.get_batch_prediction_job_status(job_id="  ")


@pytest.mark.asyncio
async def test_get_batch_prediction_results_success(mock_batch_prediction_job: MagicMock) -> None:
    mock_job = MagicMock()
    mock_job.id = "abc123"

    def _download(buf: Any, timeout: int = 120, read_timeout: int = 660) -> None:
        buf.write(b"x,y\n1,2\n")

    mock_job.download.side_effect = _download
    mock_batch_prediction_job.get.return_value = mock_job
    with patch("datarobot_genai.drtools.predictive.predict.dr_client") as mock_dr_client:
        mock_client = MagicMock()
        mock_client.BatchPredictionJob = mock_batch_prediction_job
        mock_dr_client.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_dr_client.return_value.__aexit__ = AsyncMock(return_value=False)
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
async def test_get_batch_prediction_results_too_large(mock_batch_prediction_job: MagicMock) -> None:
    mock_job = MagicMock()
    mock_job.id = "jid"
    mock_job.get_status.return_value = {"links": {"download": "https://example.com/batch/dl"}}

    def _download(buf: Any, timeout: int = 120, read_timeout: int = 660) -> None:
        buf.write(b"a" * (MAX_INLINE_SIZE + 1))

    mock_job.download.side_effect = _download
    mock_batch_prediction_job.get.return_value = mock_job
    with patch("datarobot_genai.drtools.predictive.predict.dr_client") as mock_dr_client:
        mock_client = MagicMock()
        mock_client.BatchPredictionJob = mock_batch_prediction_job
        mock_dr_client.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_dr_client.return_value.__aexit__ = AsyncMock(return_value=False)
        with pytest.raises(ToolError, match="exceeding the inline limit") as exc:
            await predict.get_batch_prediction_results(job_id="jid")
        assert "https://example.com/batch/dl" in str(exc.value)


@pytest.mark.asyncio
async def test_get_batch_prediction_results_download_runtime_error(
    mock_batch_prediction_job: MagicMock,
) -> None:
    mock_job = MagicMock()
    mock_job.id = "jid"
    mock_job.get_status.return_value = {"links": {"download": "https://example.com/batch/dl2"}}
    mock_job.download.side_effect = RuntimeError("queue full")
    mock_batch_prediction_job.get.return_value = mock_job
    with patch("datarobot_genai.drtools.predictive.predict.dr_client") as mock_dr_client:
        mock_client = MagicMock()
        mock_client.BatchPredictionJob = mock_batch_prediction_job
        mock_dr_client.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_dr_client.return_value.__aexit__ = AsyncMock(return_value=False)
        with pytest.raises(ToolError, match="queue full") as exc:
            await predict.get_batch_prediction_results(job_id="jid")
        assert "https://example.com/batch/dl2" in str(exc.value)
