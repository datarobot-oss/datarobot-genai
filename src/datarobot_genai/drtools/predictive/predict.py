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

import logging
import uuid
from typing import Annotated
from typing import Any

import datarobot as dr

from datarobot_genai.drtools.core import tool_metadata
from datarobot_genai.drtools.core.clients.datarobot import DataRobotClient
from datarobot_genai.drtools.core.clients.datarobot import get_datarobot_access_token
from datarobot_genai.drtools.core.clients.s3 import generate_presigned_url
from datarobot_genai.drtools.core.clients.s3 import get_s3_bucket_info
from datarobot_genai.drtools.core.credentials import get_credentials
from datarobot_genai.drtools.core.exceptions import ToolError

logger = logging.getLogger(__name__)


def _handle_prediction_resource(
    job: Any, bucket: str, key: str, deployment_id: str, input_desc: str
) -> dict[str, Any]:
    """Handle prediction results and return a structured response."""
    s3_url = generate_presigned_url(bucket, key)
    return {
        "job_id": job.id,
        "deployment_id": deployment_id,
        "input_desc": input_desc,
        "uri": s3_url,
        "url": s3_url,
        "name": f"Predictions for {deployment_id}",
        "mime_type": "text/csv",
    }


def get_or_create_s3_credential() -> Any:
    existing_creds = dr.Credential.list()
    for cred in existing_creds:
        if cred.name == "dr_mcp_server_temp_storage_s3_cred":
            return cred

    if get_credentials().has_aws_credentials():
        aws_access_key_id, aws_secret_access_key, aws_session_token = (
            get_credentials().get_aws_credentials()
        )
        cred = dr.Credential.create_s3(
            name="dr_mcp_server_temp_storage_s3_cred",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
        )
        return cred

    raise Exception("No AWS credentials found in your MCP deployment.")


def make_output_settings(cred: Any) -> tuple[dict[str, Any], str, str]:
    bucket_info = get_s3_bucket_info()
    s3_bucket = bucket_info["bucket"]
    s3_prefix = bucket_info["prefix"]
    s3_key = f"{s3_prefix}{uuid.uuid4()}.csv"
    s3_url = f"s3://{s3_bucket}/{s3_key}"

    return (
        {
            "type": "s3",
            "url": s3_url,
            "credential_id": cred.credential_id,
        },
        s3_bucket,
        s3_key,
    )


def wait_for_preds_and_cache_results(
    job: Any, bucket: str, key: str, deployment_id: str, input_desc: str, timeout: int
) -> dict[str, Any]:
    job.wait_for_completion(timeout)
    if job.status in ["ERROR", "FAILED", "ABORTED"]:
        logger.error(f"Job failed with status {job.status}")
        raise ToolError(f"Job failed with status {job.status}")
    return _handle_prediction_resource(job, bucket, key, deployment_id, input_desc)


@tool_metadata(tags={"predictive", "prediction", "read", "scoring", "batch"})
async def predict_by_file_path(
    deployment_id: Annotated[str, "The ID of the DataRobot deployment to use for prediction"],
    file_path: Annotated[str, "Path to a CSV file to use as input data."],
    timeout: Annotated[int, "Timeout in seconds for the batch prediction job"] | None = 600,
) -> dict[str, Any]:
    """
    Make predictions using a DataRobot deployment and a local CSV file using the DataRobot Python
    SDK. Use this tool to score large amounts of data, for small amounts of data use the
    predict_realtime tool.

    Returns prediction job details including job ID, deployment ID, and S3 URL of results.
    """
    if not deployment_id or not deployment_id.strip():
        raise ToolError("Argument validation error: 'deployment_id' cannot be empty.")
    if not file_path or not file_path.strip():
        raise ToolError("Argument validation error: 'file_path' cannot be empty.")

    output_settings, bucket, key = make_output_settings(get_or_create_s3_credential())
    job = dr.BatchPredictionJob.score(
        deployment=deployment_id,
        intake_settings={  # type: ignore[arg-type]
            "type": "localFile",
            "file": file_path,
        },
        output_settings=output_settings,  # type: ignore[arg-type]
    )
    return wait_for_preds_and_cache_results(
        job, bucket, key, deployment_id, f"Scoring file {file_path}.", timeout or 600
    )


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

    Returns prediction job details including job ID, deployment ID, and S3 URL of results.
    """
    if not deployment_id or not deployment_id.strip():
        raise ToolError("Argument validation error: 'deployment_id' cannot be empty.")
    if not dataset_id or not dataset_id.strip():
        raise ToolError("Argument validation error: 'dataset_id' cannot be empty.")

    token = await get_datarobot_access_token()
    client = DataRobotClient(token).get_client()
    output_settings, bucket, key = make_output_settings(get_or_create_s3_credential())
    dataset = client.Dataset.get(dataset_id)
    job = dr.BatchPredictionJob.score(
        deployment=deployment_id,
        intake_settings={  # type: ignore[arg-type]
            "type": "dataset",
            "dataset": dataset,
        },
        output_settings=output_settings,  # type: ignore[arg-type]
    )
    return wait_for_preds_and_cache_results(
        job, bucket, key, deployment_id, f"Scoring dataset {dataset_id}.", timeout or 600
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

    Returns prediction job details including job ID, deployment ID, and S3 URL of results.
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
    output_settings, bucket, key = make_output_settings(get_or_create_s3_credential())
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
        output_settings=output_settings,  # type: ignore[arg-type]
    )
    return wait_for_preds_and_cache_results(
        job, bucket, key, deployment_id, f"Scoring project {project_id}.", timeout or 600
    )
