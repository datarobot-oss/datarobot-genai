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

import asyncio
import io
import logging
import time
from typing import Annotated
from typing import Any

import datarobot as dr
from datarobot.errors import ClientError

from datarobot_genai.drtools.core import tool_metadata
from datarobot_genai.drtools.core.clients.datarobot import DataRobotClient
from datarobot_genai.drtools.core.clients.datarobot import get_datarobot_access_token
from datarobot_genai.drtools.core.constants import MAX_INLINE_SIZE
from datarobot_genai.drtools.core.exceptions import ToolError
from datarobot_genai.drtools.core.exceptions import ToolErrorKind
from datarobot_genai.drtools.predictive.client_exceptions import raise_tool_error_for_client_error

logger = logging.getLogger(__name__)

# After completion, links.download can lag (busy queues). SDK download() waits up to 120s for the
# URL; we only poll status until the link appears—~30s balances real deployments vs giving up.
_DOWNLOAD_LINK_MAX_ATTEMPTS = 60
_DOWNLOAD_LINK_POLL_SEC = 0.5

_TERMINAL_FAILURE_STATUSES = frozenset({"ERROR", "FAILED", "ABORTED"})


def _batch_job_download_url(job: Any) -> str | None:
    """Best-effort download URL from batch job status (get_status links.download)."""
    try:
        return job.get_status().get("links", {}).get("download")
    except Exception:
        return None


def _tool_error_with_batch_url(message: str, url: str | None) -> ToolError:
    if url:
        return ToolError(f"{message} url: {url}", kind=ToolErrorKind.UPSTREAM)
    return ToolError(message, kind=ToolErrorKind.UPSTREAM)


def _handle_prediction_resource(job: Any, deployment_id: str, input_desc: str) -> dict[str, Any]:
    """Return batch job metadata and an authenticated download URL for scored CSV."""
    download_url: str | None = None
    last_status: dict[str, Any] | None = None
    for attempt in range(_DOWNLOAD_LINK_MAX_ATTEMPTS):
        status = job.get_status()
        last_status = status
        st = status.get("status")
        if st in _TERMINAL_FAILURE_STATUSES:
            details = status.get("status_details", "")
            logger.error("Batch job %s terminal status=%s details=%s", job.id, st, details)
            raise ToolError(
                f"Batch prediction job failed: status={st!r} details={details!r}",
                kind=ToolErrorKind.UPSTREAM,
            )
        download_url = status.get("links", {}).get("download")
        if download_url:
            break
        output_settings = (status.get("job_spec") or {}).get("output_settings") or {}
        out_type = output_settings.get("type")
        if out_type and out_type != "localFile":
            raise ToolError(
                "Batch job output is not local file streaming; there is no download URL for this "
                f"output type ({out_type!r}). Use the configured sink (e.g. JDBC/S3) instead.",
                kind=ToolErrorKind.UPSTREAM,
            )
        if attempt < _DOWNLOAD_LINK_MAX_ATTEMPTS - 1:
            time.sleep(_DOWNLOAD_LINK_POLL_SEC)
    if not download_url:
        st = (last_status or {}).get("status")
        details = (last_status or {}).get("status_details", "")
        raise ToolError(
            "Batch prediction finished but no download URL appeared after polling. "
            f"Last status={st!r} details={details!r}. "
            "Confirm the job used local file (streaming) output, or retry later with the same "
            "job_id.",
            kind=ToolErrorKind.UPSTREAM,
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
    if job.status in _TERMINAL_FAILURE_STATUSES:
        logger.error("Job failed with status %s", job.status)
        raise ToolError(f"Job failed with status {job.status}", kind=ToolErrorKind.UPSTREAM)
    return _handle_prediction_resource(job, deployment_id, input_desc)


async def _wait_for_preds_and_cache_results_async(
    job: Any, deployment_id: str, input_desc: str, timeout: int
) -> dict[str, Any]:
    """Run blocking SDK wait in a thread pool so the MCP/async event loop is not stalled."""
    return await asyncio.to_thread(
        wait_for_preds_and_cache_results, job, deployment_id, input_desc, timeout
    )


@tool_metadata(
    tags={"predictive", "prediction", "read", "scoring", "batch"},
    description=(
        "[Predict—deployment + catalog dataset] Use when the user wants batch scoring of one AI "
        "Catalog tabular dataset through an MLOps deployment (deployment_id + dataset_id). Waits "
        "until the job finishes. Returns job_id and an authenticated download URL for scored CSV. "
        "Not the same as scoring with a leaderboard model inside a project "
        "(score_dataset_with_model), not scoring project holdout/validation partitions "
        "(predict_from_project_data), and not inline pasted rows (predict_realtime). Use "
        "get_batch_prediction_results for CSV text when small enough; for synchronous rows from "
        "catalog data, consider predict_by_ai_catalog_rt."
    ),
)
async def predict_by_ai_catalog(
    *,
    deployment_id: Annotated[str, "MLOps deployment id."],
    dataset_id: Annotated[str, "AI Catalog dataset id to score."],
    timeout: Annotated[int, "Max seconds to wait for batch job completion."] | None = 600,
) -> dict[str, Any]:
    if not deployment_id or not deployment_id.strip():
        raise ToolError(
            "Argument validation error: 'deployment_id' cannot be empty.",
            kind=ToolErrorKind.VALIDATION,
        )
    if not dataset_id or not dataset_id.strip():
        raise ToolError(
            "Argument validation error: 'dataset_id' cannot be empty.",
            kind=ToolErrorKind.VALIDATION,
        )

    token = await get_datarobot_access_token()
    client = DataRobotClient(token).get_client()
    try:
        dataset = client.Dataset.get(dataset_id)
        job = dr.BatchPredictionJob.score(
            deployment=deployment_id,
            intake_settings={  # type: ignore[arg-type]
                "type": "dataset",
                "dataset": dataset,
            },
            output_settings=None,
        )
    except ClientError as e:
        raise_tool_error_for_client_error(e)
    return await _wait_for_preds_and_cache_results_async(
        job, deployment_id, f"Scoring dataset {dataset_id}.", timeout or 600
    )


@tool_metadata(
    tags={"predictive", "prediction", "read", "scoring", "batch"},
    description=(
        "[Predict—deployment + project splits] Use when the user asks for batch predictions from a "
        "deployment using data that already lives in the modeling project: training, holdout, "
        "validation, or allBacktest partitions (pass partition, e.g. holdout). Optional catalog "
        "dataset_id if they linked another dataset to the project. This is the right tool for "
        "phrases like 'batch score the holdout' or 'predict on the training partition', not for "
        "only a catalog dataset_id without project context, and not for inline CSV/JSON in the "
        "message. Returns job_id and scored CSV download URL like predict_by_ai_catalog, not "
        "inline rows."
    ),
)
async def predict_from_project_data(
    *,
    deployment_id: Annotated[str, "MLOps deployment id."],
    project_id: Annotated[str, "DataRobot project id that supplies training/holdout data."],
    dataset_id: Annotated[
        str,
        "Optional catalog dataset id for project-linked scoring (alternative to partition-only).",
    ]
    | None = None,
    partition: Annotated[
        str,
        "Project data split: holdout, validation, or allBacktest.",
    ]
    | None = None,
    timeout: Annotated[int, "Max seconds to wait for batch job completion."] | None = 600,
) -> dict[str, Any]:
    if not deployment_id or not deployment_id.strip():
        raise ToolError(
            "Argument validation error: 'deployment_id' cannot be empty.",
            kind=ToolErrorKind.VALIDATION,
        )
    if not project_id or not project_id.strip():
        raise ToolError(
            "Argument validation error: 'project_id' cannot be empty.",
            kind=ToolErrorKind.VALIDATION,
        )
    if dataset_id is not None and (not dataset_id or not dataset_id.strip()):
        raise ToolError(
            "Argument validation error: 'dataset_id' cannot be empty.",
            kind=ToolErrorKind.VALIDATION,
        )
    if partition is not None and (not partition or not partition.strip()):
        raise ToolError(
            "Argument validation error: 'partition' cannot be empty.", kind=ToolErrorKind.VALIDATION
        )

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
    try:
        job = dr.BatchPredictionJob.score(
            deployment=deployment_id,
            intake_settings=intake_settings,  # type: ignore[arg-type]
            output_settings=None,
        )
    except ClientError as e:
        raise_tool_error_for_client_error(e)
    return await _wait_for_preds_and_cache_results_async(
        job, deployment_id, f"Scoring project {project_id}.", timeout or 600
    )


@tool_metadata(
    tags={"predictive", "prediction", "read", "scoring", "batch"},
    description=(
        "[Predict—fetch batch CSV] Use after a batch job completed: returns scored CSV text "
        "inline given job_id from predict_by_ai_catalog or predict_from_project_data. Only when "
        "the file is under the inline size cap; otherwise use the job download URL from the "
        "batch tool with authenticated fetch. Does not start scoring and does not apply to "
        "score_dataset_with_model jobs (different API surface)."
    ),
)
async def get_batch_prediction_results(
    *,
    job_id: Annotated[
        str,
        "job_id returned by predict_by_ai_catalog or predict_from_project_data.",
    ],
    download_timeout: Annotated[int, "Seconds to wait for the download to become ready."] = 120,
    download_read_timeout: Annotated[int, "Seconds allowed for streaming the CSV body."] = 660,
) -> dict[str, Any]:
    if not job_id or not job_id.strip():
        raise ToolError(
            "Argument validation error: 'job_id' cannot be empty.", kind=ToolErrorKind.VALIDATION
        )

    token = await get_datarobot_access_token()
    DataRobotClient(token).get_client()
    try:
        job = dr.BatchPredictionJob.get(job_id.strip())
    except ClientError as e:
        raise_tool_error_for_client_error(e)
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
