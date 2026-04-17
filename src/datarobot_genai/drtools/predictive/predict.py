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

import io
import logging
from typing import Annotated
from typing import Any

import datarobot as dr

from datarobot_genai.drtools.core import tool_metadata
from datarobot_genai.drtools.core.clients.datarobot import DataRobotClient
from datarobot_genai.drtools.core.clients.datarobot import get_datarobot_access_token
from datarobot_genai.drtools.core.constants import MAX_INLINE_SIZE
from datarobot_genai.drtools.core.exceptions import ToolError

logger = logging.getLogger(__name__)


def _batch_job_download_url(job: Any) -> str | None:
    """Best-effort download URL from batch job status (get_status links.download)."""
    try:
        return job.get_status().get("links", {}).get("download")
    except Exception:
        return None


def _tool_error_with_batch_url(message: str, url: str | None) -> ToolError:
    if url:
        return ToolError(f"{message} url: {url}")
    return ToolError(message)


def _handle_prediction_resource(job: Any, deployment_id: str, input_desc: str) -> dict[str, Any]:
    """Return batch job metadata and an authenticated download URL for scored CSV."""
    status = job.get_status()
    download_url = status.get("links", {}).get("download")
    if not download_url:
        raise ToolError(
            "Batch prediction finished but no download URL is available. "
            "Confirm the job used local file streaming output (default batch output)."
        )
    return {
        "job_id": job.id,
        "deployment_id": deployment_id,
        "input_desc": input_desc,
        "url": download_url,
        "name": f"Predictions for {deployment_id}",
        "mime_type": "text/csv",
    }


def wait_for_preds_and_cache_results(
    job: Any, deployment_id: str, input_desc: str, timeout: int
) -> dict[str, Any]:
    job.wait_for_completion(timeout)
    if job.status in ["ERROR", "FAILED", "ABORTED"]:
        logger.error(f"Job failed with status {job.status}")
        raise ToolError(f"Job failed with status {job.status}")
    return _handle_prediction_resource(job, deployment_id, input_desc)


@tool_metadata(tags={"predictive", "prediction", "read", "scoring", "batch"})
async def predict_by_ai_catalog(
    deployment_id: Annotated[str, "The ID of the DataRobot deployment to use for prediction"],
    dataset_id: Annotated[str, "The ID of the AI Catalog dataset to use for prediction"],
    timeout: Annotated[int, "Timeout in seconds for the batch prediction job"] | None = 600,
) -> dict[str, Any]:
    """
    Make predictions using a DataRobot deployment and an AI Catalog dataset using the DataRobot
    Python SDK.
    Use this tool when asked to score data stored in AI Catalog by dataset id.

    Returns prediction job details including job ID, deployment ID, and a DataRobot URL to download
    scored results (batch local file streaming). Use get_batch_prediction_results with that job_id
    to retrieve CSV using server credentials.
    """
    if not deployment_id or not deployment_id.strip():
        raise ToolError("Argument validation error: 'deployment_id' cannot be empty.")
    if not dataset_id or not dataset_id.strip():
        raise ToolError("Argument validation error: 'dataset_id' cannot be empty.")

    token = await get_datarobot_access_token()
    client = DataRobotClient(token).get_client()
    dataset = client.Dataset.get(dataset_id)
    job = dr.BatchPredictionJob.score(
        deployment=deployment_id,
        intake_settings={  # type: ignore[arg-type]
            "type": "dataset",
            "dataset": dataset,
        },
        output_settings=None,
    )
    return wait_for_preds_and_cache_results(
        job, deployment_id, f"Scoring dataset {dataset_id}.", timeout or 600
    )


@tool_metadata(tags={"predictive", "prediction", "read", "scoring", "batch"})
async def predict_from_project_data(
    deployment_id: Annotated[str, "The ID of the DataRobot deployment to use for prediction"],
    project_id: Annotated[str, "The ID of the DataRobot project to use for prediction"],
    dataset_id: Annotated[
        str, "The ID of the external dataset, usually stored in AI Catalog, to use for prediction"
    ]
    | None = None,
    partition: Annotated[
        str,
        "The partition of the DataRobot dataset to use ('holdout', 'validation', 'allBacktest')",
    ]
    | None = None,
    timeout: Annotated[int, "Timeout in seconds for the batch prediction job"] | None = 600,
) -> dict[str, Any]:
    """
    Make predictions using a DataRobot deployment using the training data associated with the
    project that created the deployment.
    Use this tool to score holdout, validation, or allBacktest partitions of the training data.
    Can request a specific partition of the data, or use an external dataset (with dataset_id)
    stored in AI Catalog.

    Returns prediction job details including job ID, deployment ID, and a DataRobot URL to download
    scored results (batch local file streaming). Use get_batch_prediction_results with that job_id
    to retrieve CSV using server credentials.
    """
    if not deployment_id or not deployment_id.strip():
        raise ToolError("Argument validation error: 'deployment_id' cannot be empty.")
    if not project_id or not project_id.strip():
        raise ToolError("Argument validation error: 'project_id' cannot be empty.")
    if dataset_id is not None and (not dataset_id or not dataset_id.strip()):
        raise ToolError("Argument validation error: 'dataset_id' cannot be empty.")
    if partition is not None and (not partition or not partition.strip()):
        raise ToolError("Argument validation error: 'partition' cannot be empty.")

    token = await get_datarobot_access_token()
    DataRobotClient(token).get_client()
    intake_settings: dict[str, Any] = {
        "type": "dss",
        "project_id": project_id,
    }
    if partition:
        intake_settings["partition"] = partition
    if dataset_id:
        intake_settings["dataset_id"] = dataset_id
    job = dr.BatchPredictionJob.score(
        deployment=deployment_id,
        intake_settings=intake_settings,  # type: ignore[arg-type]
        output_settings=None,
    )
    return wait_for_preds_and_cache_results(
        job, deployment_id, f"Scoring project {project_id}.", timeout or 600
    )


@tool_metadata(tags={"predictive", "prediction", "read", "scoring", "batch"})
async def get_batch_prediction_results(
    job_id: Annotated[
        str,
        (
            "Batch prediction job_id from predict_by_ai_catalog, "
            "predict_from_project_data, or related tools"
        ),
    ],
    download_timeout: Annotated[
        int, "Seconds to wait for the download URL to become available"
    ] = 120,
    download_read_timeout: Annotated[
        int, "HTTP read timeout in seconds for streaming the CSV"
    ] = 660,
) -> dict[str, Any]:
    """
    Download scored CSV for a completed batch prediction job using the request DataRobot API token.

    Use this after batch prediction tools return a job_id and url, when you want the CSV inline
    without configuring a separate HTTP client. Output is limited to 1 MiB; for larger results,
    download using the returned url with DataRobot API authentication.
    """
    if not job_id or not job_id.strip():
        raise ToolError("Argument validation error: 'job_id' cannot be empty.")

    token = await get_datarobot_access_token()
    DataRobotClient(token).get_client()
    job = dr.BatchPredictionJob.get(job_id.strip())
    download_url = _batch_job_download_url(job)
    buf = io.BytesIO()
    try:
        job.download(buf, timeout=download_timeout, read_timeout=download_read_timeout)
    except RuntimeError as e:
        raise _tool_error_with_batch_url(
            str(e), download_url or _batch_job_download_url(job)
        ) from e
    except Exception as e:
        raise _tool_error_with_batch_url(
            f"Failed to download batch prediction results: {e}",
            download_url or _batch_job_download_url(job),
        ) from e
    raw = buf.getvalue()
    if len(raw) > MAX_INLINE_SIZE:
        url = download_url or _batch_job_download_url(job)
        raise _tool_error_with_batch_url(
            f"Downloaded CSV is {len(raw)} bytes, exceeding the inline limit "
            f"of {MAX_INLINE_SIZE} bytes. "
            "Fetch with DataRobot API authentication or run batch scoring with smaller data.",
            url,
        )
    try:
        data = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise _tool_error_with_batch_url(
            "Batch prediction CSV is not valid UTF-8.",
            download_url or _batch_job_download_url(job),
        ) from e

    return {
        "job_id": job.id,
        "mime_type": "text/csv",
        "size_bytes": len(raw),
        "data": data,
    }
