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
from typing import Any

import datarobot as dr
from fastmcp.resources import HttpResource
from fastmcp.resources import ResourceManager

from datarobot_genai.drmcp.core.clients import get_credentials
from datarobot_genai.drmcp.core.clients import get_sdk_client
from datarobot_genai.drmcp.core.mcp_instance import dr_mcp_tool
from datarobot_genai.drmcp.core.utils import generate_presigned_url
from datarobot_genai.drmcp.tools.clients.s3 import get_s3_bucket_info

logger = logging.getLogger(__name__)


def _handle_prediction_resource(
    job: Any, bucket: str, key: str, deployment_id: str, input_desc: str
) -> str:
    s3_url = generate_presigned_url(bucket, key)
    resource_manager = ResourceManager()
    resource = HttpResource(
        uri=s3_url,  # type: ignore[arg-type]
        url=s3_url,
        name=f"Predictions for {deployment_id}",
        mime_type="text/csv",
    )
    resource_manager.add_resource(resource)
    return (
        f"Finished Batch Prediction job ID {job.id} for deployment ID {deployment_id}. "
        f"{input_desc} Results uploaded to {s3_url}. "
        f"Job status: {job.status} and you can find the job on the DataRobot UI at "
        f"/deployments/batch-jobs. "
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
) -> str:
    job.wait_for_completion(timeout)
    if job.status in ["ERROR", "FAILED", "ABORTED"]:
        logger.error(f"Job failed with status {job.status}")
        return f"Job failed with status {job.status}"
    return _handle_prediction_resource(job, bucket, key, deployment_id, input_desc)


@dr_mcp_tool(tags={"prediction", "scoring", "batch"})
async def predict_by_file_path(
    deployment_id: str,
    file_path: str,
    timeout: int = 600,
) -> str:
    """
    Make predictions using a DataRobot deployment and a local CSV file using the DataRobot Python
    SDK. Use this tool to score large amounts of data, for small amounts of data use the
    predict_realtime tool.
    Args:
        deployment_id: The ID of the DataRobot deployment to use for prediction.
        file_path: Path to a CSV file to use as input data.
        timeout: Timeout in seconds for the batch prediction job (default 300).

    Returns
    -------
        A string summary of the batch prediction job and download link if available.
    """
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
        job, bucket, key, deployment_id, f"Scoring file {file_path}.", timeout
    )


@dr_mcp_tool(tags={"prediction", "scoring", "batch"})
async def predict_by_ai_catalog(
    deployment_id: str,
    dataset_id: str,
    timeout: int = 600,
) -> str:
    """
    Make predictions using a DataRobot deployment and an AI Catalog dataset using the DataRobot
    Python SDK.

    Use this tool when asked to score data stored in AI Catalog by dataset id.
    Args:
        deployment_id: The ID of the DataRobot deployment to use for prediction.
        dataset_id: ID of an AI Catalog item to use as input data.
        timeout: Timeout in seconds for the batch prediction job (default 300).

    Returns
    -------
        A string summary of the batch prediction job and download link if available.
    """
    output_settings, bucket, key = make_output_settings(get_or_create_s3_credential())
    client = get_sdk_client()
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
        job, bucket, key, deployment_id, f"Scoring dataset {dataset_id}.", timeout
    )


@dr_mcp_tool(tags={"prediction", "scoring", "batch"})
async def predict_from_project_data(
    deployment_id: str,
    project_id: str,
    dataset_id: str | None = None,
    partition: str | None = None,
    timeout: int = 600,
) -> str:
    """
    Make predictions using a DataRobot deployment using the training data associated with the
    project that created the deployment.
    Use this tool to score holdout, validation, or allBacktest partitions of the training data.
    Can request a specific partition of the data, or use an external dataset (with dataset_id)
    stored in AI Catalog.
    Args:
        deployment_id: (Required)The ID of the DataRobot deployment to use for prediction.
        project_id: (Required) The ID of the DataRobot project to use for prediction. Can be found
         by using the get_model_info_from_deployment tool.
        dataset_id: (Optional) The ID of the external dataset, ususally stored in AI Catalog, to
            use for prediction.
        partition: (Optional)The partition of the DataRobot dataset to use for prediction, could be
            'holdout', 'validation', or 'allBacktest'.
        timeout: (Optional) Timeout in seconds for the batch prediction job (default 600).

    Returns
    -------
        A string summary of the batch prediction job and download link if available.
    """
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
        job, bucket, key, deployment_id, f"Scoring project {project_id}.", timeout
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
    client = get_sdk_client()
    project = client.Project.get(project_id)
    model = client.Model.get(project=project, model_id=model_id)
    try:
        explanations = model.get_or_request_prediction_explanations(
            dataset_id=dataset_id, max_explanations=max_explanations
        )
        return json.dumps(
            {"explanations": explanations, "ui_panel": ["prediction-distribution"]},
            indent=2,
        )
    except Exception as e:
        logger.error(f"Error in get_prediction_explanations: {type(e).__name__}: {e}")
        return json.dumps(
            {"error": f"Error in get_prediction_explanations: {type(e).__name__}: {e}"}
        )
