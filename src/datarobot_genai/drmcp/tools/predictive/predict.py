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
import logging
import uuid
from typing import Annotated
from typing import Any

import datarobot as dr
from fastmcp.exceptions import ToolError
from fastmcp.resources import HttpResource
from fastmcp.resources import ResourceManager
from fastmcp.tools.tool import ToolResult

from datarobot_genai.drmcp import dr_mcp_tool
from datarobot_genai.drmcp.core.clients import get_credentials
from datarobot_genai.drmcp.core.utils import generate_presigned_url
from datarobot_genai.drmcp.tools.clients.datarobot import DataRobotClient
from datarobot_genai.drmcp.tools.clients.datarobot import get_datarobot_access_token
from datarobot_genai.drmcp.tools.clients.s3 import get_s3_bucket_info

logger = logging.getLogger(__name__)


def _handle_prediction_resource(
    job: Any, bucket: str, key: str, deployment_id: str, input_desc: str
) -> ToolResult:
    s3_url = generate_presigned_url(bucket, key)
    resource_manager = ResourceManager()
    resource = HttpResource(
        uri=s3_url,  # type: ignore[arg-type]
        url=s3_url,
        name=f"Predictions for {deployment_id}",
        mime_type="text/csv",
    )
    resource_manager.add_resource(resource)
    return ToolResult(
        structured_content={
            "job_id": job.id,
            "deployment_id": deployment_id,
            "input_desc": input_desc,
            "s3_url": s3_url,
        },
    )


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
) -> ToolError | ToolResult:
    job.wait_for_completion(timeout)
    if job.status in ["ERROR", "FAILED", "ABORTED"]:
        logger.error(f"Job failed with status {job.status}")
        raise ToolError(f"Job failed with status {job.status}")
    return _handle_prediction_resource(job, bucket, key, deployment_id, input_desc)


@dr_mcp_tool(tags={"predictive", "prediction", "read", "scoring", "batch"})
async def predict_by_file_path(
    deployment_id: Annotated[str, "The ID of the DataRobot deployment to use for prediction"]
    | None = None,
    file_path: Annotated[str, "Path to a CSV file to use as input data."] | None = None,
    timeout: Annotated[int, "Timeout in seconds for the batch prediction job"] | None = 600,
) -> ToolError | ToolResult:
    """
    Make predictions using a DataRobot deployment and a local CSV file using the DataRobot Python
    SDK. Use this tool to score large amounts of data, for small amounts of data use the
    predict_realtime tool.
    """
    if not deployment_id:
        raise ToolError("Deployment ID must be provided")
    if not file_path:
        raise ToolError("File path must be provided")

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


@dr_mcp_tool(tags={"predictive", "prediction", "read", "scoring", "batch"})
async def predict_by_ai_catalog(
    deployment_id: Annotated[str, "The ID of the DataRobot deployment to use for prediction"]
    | None = None,
    dataset_id: Annotated[str, "The ID of the AI Catalog dataset to use for prediction"]
    | None = None,
    timeout: Annotated[int, "Timeout in seconds for the batch prediction job"] | None = 600,
) -> ToolError | ToolResult:
    """
    Make predictions using a DataRobot deployment and an AI Catalog dataset using the DataRobot
    Python SDK.
    Use this tool when asked to score data stored in AI Catalog by dataset id.
    """
    if not deployment_id:
        raise ToolError("Deployment ID must be provided")
    if not dataset_id:
        raise ToolError("Dataset ID must be provided")

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


@dr_mcp_tool(tags={"predictive", "prediction", "read", "scoring", "batch"})
async def predict_from_project_data(
    deployment_id: Annotated[str, "The ID of the DataRobot deployment to use for prediction"]
    | None = None,
    project_id: Annotated[str, "The ID of the DataRobot project to use for prediction"]
    | None = None,
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
) -> ToolError | ToolResult:
    """
    Make predictions using a DataRobot deployment using the training data associated with the
    project that created the deployment.
    Use this tool to score holdout, validation, or allBacktest partitions of the training data.
    Can request a specific partition of the data, or use an external dataset (with dataset_id)
    stored in AI Catalog.
    """
    if not deployment_id:
        raise ToolError("Deployment ID must be provided")
    if not project_id:
        raise ToolError("Project ID must be provided")

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


# FIXME
# @dr_mcp_tool(tags={"prediction", "explanations", "shap"})
async def get_prediction_explanations(
    project_id: str,
    model_id: str,
    dataset_id: str,
    max_explanations: int = 100,
) -> str:
    """
    Calculate prediction explanations (SHAP values) for a given model and dataset.

    Args:
        project_id: The ID of the DataRobot project.
        model_id: The ID of the model to use for explanations.
        dataset_id: The ID of the dataset to explain predictions for.
        max_explanations: Maximum number of explanations per row (default 100).

    Returns
    -------
        JSON string containing the prediction explanations for each row.
    """
    token = await get_datarobot_access_token()
    client = DataRobotClient(token).get_client()
    project = client.Project.get(project_id)
    model = client.Model.get(project=project, model_id=model_id)
    try:
        explanations = model.get_or_request_prediction_explanations(
            dataset_id=dataset_id, max_explanations=max_explanations
        )
        return json.dumps(
            {"explanations": explanations, "ui_panel": ["prediction-distribution"]},
            indent=2,
        ).replace("'", "'")
    except Exception as e:
        logger.error(f"Error in get_prediction_explanations: {type(e).__name__}: {e}")
        return json.dumps(
            {"error": f"Error in get_prediction_explanations: {type(e).__name__}: {e}"}
        )
