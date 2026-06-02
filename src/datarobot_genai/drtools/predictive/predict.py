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
from datarobot.errors import ClientError

from datarobot_genai.drtools.core import tool_metadata
from datarobot_genai.drtools.core.clients.datarobot import ThreadSafeDataRobotClient
from datarobot_genai.drtools.core.constants import MAX_INLINE_SIZE
from datarobot_genai.drtools.core.exceptions import ToolError
from datarobot_genai.drtools.core.exceptions import ToolErrorKind
from datarobot_genai.drtools.predictive.client_exceptions import raise_tool_error_for_client_error

logger = logging.getLogger(__name__)

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


def _batch_job_submitted_payload(job: Any, deployment_id: str, input_desc: str) -> dict[str, Any]:
    """Return right after score(): job id, current status, download URL if already present."""
    status = job.get_status()
    st = status.get("status")
    if st in _TERMINAL_FAILURE_STATUSES:
        details = status.get("status_details", "")
        logger.error("Batch job %s terminal status=%s details=%s", job.id, st, details)
        raise ToolError(
            f"Batch prediction job failed: status={st!r} details={details!r}",
            kind=ToolErrorKind.UPSTREAM,
        )
    download_url = status.get("links", {}).get("download")
    output_settings = (status.get("job_spec") or {}).get("output_settings") or {}
    out_type = output_settings.get("type")
    if out_type and out_type != "localFile":
        raise ToolError(
            "Batch job output is not local file streaming; there is no download URL for this "
            f"output type ({out_type!r}). Use the configured sink (e.g. JDBC/S3) instead.",
            kind=ToolErrorKind.UPSTREAM,
        )
    return {
        "job_id": job.id,
        "deployment_id": deployment_id,
        "input_desc": input_desc,
        "url": download_url,
        "batch_job_status": st,
        "name": f"Predictions for {deployment_id}",
        "mime_type": "text/csv",
        "note": (
            "This tool only submits the batch job. You MUST poll get_batch_prediction_job_status "
            "with job_id until batch_job_status is COMPLETED and url is present (retry every few "
            "seconds while RUNNING or INITIALIZING). Then call get_batch_prediction_results for "
            "inline CSV if small enough, or fetch url with DataRobot API authentication."
        ),
    }


@tool_metadata(
    tags={"predictive", "prediction", "read", "scoring", "batch"},
    description=(
        "[Predict—deployment + catalog dataset] Submits batch scoring of one AI Catalog tabular "
        "dataset through an MLOps deployment (deployment_id + dataset_id). Returns immediately "
        "with job_id and initial batch_job_status; does NOT wait for completion. Required "
        "follow-up: poll get_batch_prediction_job_status until COMPLETED and url is set, then "
        "get_batch_prediction_results or authenticated download of url. Not leaderboard scoring "
        "(score_dataset_with_model), not project splits (predict_from_project_data), not inline "
        "rows (predict_realtime). For synchronous small catalog scoring use "
        "predict_by_ai_catalog_rt."
    ),
)
async def predict_by_ai_catalog(
    *,
    deployment_id: Annotated[str, "MLOps deployment id."],
    dataset_id: Annotated[str, "AI Catalog dataset id to score."],
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

    with ThreadSafeDataRobotClient().request_user_client():
        try:
            dataset = dr.Dataset.get(dataset_id)
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
        return _batch_job_submitted_payload(job, deployment_id, f"Scoring dataset {dataset_id}.")


@tool_metadata(
    tags={"predictive", "prediction", "read", "scoring", "batch"},
    description=(
        "[Predict—deployment + project splits] Submits batch predictions from a deployment using "
        "project data: training, holdout, validation, or allBacktest (partition), optional "
        "catalog dataset_id. Returns immediately with job_id like predict_by_ai_catalog; does NOT "
        "wait. Required follow-up: poll get_batch_prediction_job_status until COMPLETED and url, "
        "then get_batch_prediction_results or download url. Not catalog-only scoring without "
        "project context, not inline CSV in chat."
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

    with ThreadSafeDataRobotClient().request_user_client():
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
        return _batch_job_submitted_payload(job, deployment_id, f"Scoring project {project_id}.")


@tool_metadata(
    tags={"predictive", "prediction", "read", "scoring", "batch"},
    description=(
        "[Predict—batch job status] REQUIRED polling step after predict_by_ai_catalog or "
        "predict_from_project_data. Pass job_id from the submit response. Returns "
        "batch_job_status, optional url when the scored file is ready, and progress fields. "
        "Poll every few seconds until batch_job_status is COMPLETED and url is non-null, then "
        "get_batch_prediction_results or download url. Lightweight (no CSV body). Raises on "
        "terminal failure. Not for score_dataset_with_model jobs."
    ),
)
async def get_batch_prediction_job_status(
    *,
    job_id: Annotated[
        str,
        "Batch prediction job id from predict_by_ai_catalog or predict_from_project_data.",
    ],
) -> dict[str, Any]:
    if not job_id or not job_id.strip():
        raise ToolError(
            "Argument validation error: 'job_id' cannot be empty.", kind=ToolErrorKind.VALIDATION
        )

    with ThreadSafeDataRobotClient().request_user_client():
        try:
            job = dr.BatchPredictionJob.get(job_id.strip())
        except ClientError as e:
            raise_tool_error_for_client_error(e)
        status = job.get_status()
        st = status.get("status")
        if st in _TERMINAL_FAILURE_STATUSES:
            details = status.get("status_details", "")
            raise ToolError(
                f"Batch prediction job failed: status={st!r} details={details!r}",
                kind=ToolErrorKind.UPSTREAM,
            )
        download_url = status.get("links", {}).get("download")
        output_settings = (status.get("job_spec") or {}).get("output_settings") or {}
        out_type = output_settings.get("type")
        if out_type and out_type != "localFile":
            raise ToolError(
                "Batch job output is not local file streaming; there is no download URL for this "
                f"output type ({out_type!r}). Use the configured sink (e.g. JDBC/S3) instead.",
                kind=ToolErrorKind.UPSTREAM,
            )
        return {
            "job_id": job.id,
            "batch_job_status": st,
            "url": download_url,
            "percentage_completed": status.get("percentage_completed"),
            "elapsed_time_sec": status.get("elapsed_time_sec"),
            "status_details": status.get("status_details"),
        }


@tool_metadata(
    tags={"predictive", "prediction", "read", "scoring", "batch"},
    description=(
        "[Predict—fetch batch CSV] After get_batch_prediction_job_status shows COMPLETED and "
        "url, returns scored CSV text inline for job_id when under the inline size cap. If too "
        "large, the error includes the download url—fetch with API authentication. Does not start "
        "scoring; not for score_dataset_with_model jobs."
    ),
)
async def get_batch_prediction_results(
    *,
    job_id: Annotated[
        str,
        "job_id from predict_by_ai_catalog or predict_from_project_data (same as status polling).",
    ],
    download_timeout: Annotated[int, "Seconds to wait for the download to become ready."] = 120,
    download_read_timeout: Annotated[int, "Seconds allowed for streaming the CSV body."] = 660,
) -> dict[str, Any]:
    if not job_id or not job_id.strip():
        raise ToolError(
            "Argument validation error: 'job_id' cannot be empty.", kind=ToolErrorKind.VALIDATION
        )

    with ThreadSafeDataRobotClient().request_user_client():
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
