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
import os
from typing import Annotated
from typing import Any

from datarobot_genai.drtools.core import tool_metadata
from datarobot_genai.drtools.core.clients.datarobot import MODEL_EXTENSIONS
from datarobot_genai.drtools.core.clients.datarobot import REQUIRED_FILES
from datarobot_genai.drtools.core.clients.datarobot import DataRobotClient
from datarobot_genai.drtools.core.clients.datarobot import deploy_custom_model_impl
from datarobot_genai.drtools.core.clients.datarobot import find_model_file_in_folder
from datarobot_genai.drtools.core.clients.datarobot import get_datarobot_access_token
from datarobot_genai.drtools.core.exceptions import ToolError

logger = logging.getLogger(__name__)


@tool_metadata(tags={"predictive", "deployment", "read", "management", "list", "daria"})
async def list_deployments() -> dict[str, Any]:
    """List all DataRobot deployments for the authenticated user."""
    token = await get_datarobot_access_token()
    client = DataRobotClient(token).get_client()
    deployments = client.Deployment.list()
    if not deployments:
        return {"deployments": []}
    deployments_dict = {d.id: d.label for d in deployments}
    return {"deployments": deployments_dict}


@tool_metadata(tags={"predictive", "deployment", "read", "model", "info", "daria"})
async def get_model_info_from_deployment(
    *,
    deployment_id: Annotated[str, "The ID of the DataRobot deployment"],
) -> dict[str, Any]:
    """Retrieve model info associated with a given deployment ID."""
    if not deployment_id:
        raise ToolError("Deployment ID must be provided")
    token = await get_datarobot_access_token()
    client = DataRobotClient(token).get_client()
    try:
        deployment = client.Deployment.get(deployment_id)
    except Exception as e:
        error_str = str(e)
        if "404" in error_str or "Not Found" in error_str:
            raise ToolError(
                f"Deployment '{deployment_id}' not found. Please verify the deployment ID exists "
                "and you have access to it."
            )
        raise ToolError(f"Failed to retrieve deployment '{deployment_id}': {error_str}")
    return deployment.model


@tool_metadata(tags={"predictive", "deployment", "write", "model", "create", "daria"})
async def deploy_model(
    *,
    model_id: Annotated[str, "The ID of the DataRobot model to deploy"],
    label: Annotated[str, "The label/name for the deployment"],
    description: Annotated[str, "Optional description for the deployment"] | None = None,
) -> dict[str, Any]:
    """Deploy a model by creating a new DataRobot deployment."""
    if not model_id:
        raise ToolError("Model ID must be provided")
    if not label:
        raise ToolError("Model label must be provided")

    token = await get_datarobot_access_token()
    client = DataRobotClient(token).get_client()
    prediction_servers = client.PredictionServer.list()
    if not prediction_servers:
        raise ToolError("No prediction servers available for deployment.")
    deployment = client.Deployment.create_from_learning_model(
        model_id=model_id,
        label=label,
        description=description,
        default_prediction_server_id=prediction_servers[0].id,
    )
    return {
        "deployment_id": deployment.id,
        "label": label,
    }


@tool_metadata(tags={"predictive", "deployment", "write", "custom", "create"})
async def deploy_custom_model(
    *,
    model_folder: Annotated[
        str, "Path to directory with custom.py, requirements.txt, and optionally a model file"
    ],
    name: Annotated[str, "Single name used for both custom model and deployment"],
    target_type: Annotated[str, "Target type: binary, regression, or multiclass"],
    target_name: Annotated[str, "Target column name"],
    model_file_path: Annotated[
        str,
        "Optional path to model file. If not set and folder contains none, ToolError is raised.",
    ]
    | None = None,
    positive_class_label: Annotated[str, "For binary: positive class label"] | None = None,
    negative_class_label: Annotated[str, "For binary: negative class label"] | None = None,
    class_labels: Annotated[list[str], "For multiclass: list of class labels"] | None = None,
    deployment_label: Annotated[str, "Deployment label; defaults to name"] | None = None,
    execution_environment_id: Annotated[str, "Optional execution environment ID"] | None = None,
    description: Annotated[str, "Optional description"] | None = None,
) -> dict[str, Any]:
    """Deploy a custom inference model (e.g., .pkl) to DataRobot MLOps.

    Requires a model file in the folder or model_file_path.
    """
    if not model_folder:
        raise ToolError("model_folder must be provided")
    if not name:
        raise ToolError("name must be provided")
    if not target_type:
        raise ToolError("target_type must be provided")
    if not target_name:
        raise ToolError("target_name must be provided")

    model_folder = os.path.abspath(model_folder)
    if not os.path.isdir(model_folder):
        raise ToolError(f"model_folder is not a directory: {model_folder}")
    for f in REQUIRED_FILES:
        if not os.path.isfile(os.path.join(model_folder, f)):
            raise ToolError(f"model_folder must contain {f}")
    resolved_path: str | None = None
    if model_file_path:
        p = (
            model_file_path
            if os.path.isabs(model_file_path)
            else os.path.join(model_folder, model_file_path)
        )
        if os.path.isfile(p):
            resolved_path = p
        else:
            raise ToolError(f"model_file_path does not exist: {p}")
    if resolved_path is None:
        resolved_path = find_model_file_in_folder(model_folder)
    if resolved_path is None:
        raise ToolError(
            f"No model file ({', '.join(MODEL_EXTENSIONS)}) found in {model_folder}. "
            "Add a model file to the folder or pass model_file_path."
        )
    token = await get_datarobot_access_token()
    client = DataRobotClient(token).get_client()
    out = deploy_custom_model_impl(
        client,
        model_folder=model_folder,
        model_file_path=resolved_path,
        name=name,
        target_type=target_type,
        target_name=target_name,
        positive_class_label=positive_class_label,
        negative_class_label=negative_class_label,
        class_labels=class_labels,
        deployment_label=deployment_label,
        execution_environment_id=execution_environment_id,
        description=description,
    )
    return out


@tool_metadata(tags={"predictive", "deployment", "read", "predictions", "history", "daria"})
async def get_prediction_history(
    *,
    deployment_id: Annotated[str, "The ID of the DataRobot deployment"],
    limit: Annotated[int, "Maximum number of prediction rows to return"] = 100,
    offset: Annotated[int, "Number of rows to skip for pagination"] = 0,
    start_time: Annotated[str, "ISO 8601 start time filter"] | None = None,
    end_time: Annotated[str, "ISO 8601 end time filter"] | None = None,
) -> dict[str, Any]:
    """Retrieve recent prediction results from a DataRobot deployment."""
    if not deployment_id:
        raise ToolError("Deployment ID must be provided")

    token = await get_datarobot_access_token()
    dr_module = DataRobotClient(token).get_client()
    rest_client = dr_module.client.get_client()

    params: dict = {"limit": limit, "offset": offset}
    if start_time:
        params["startTime"] = start_time
    if end_time:
        params["endTime"] = end_time

    try:
        response = rest_client.get(f"deployments/{deployment_id}/predictionResults/", params=params)
        data = response.json()
    except Exception as e:
        error_str = str(e)
        if "404" in error_str or "Not Found" in error_str:
            raise ToolError(
                f"Deployment '{deployment_id}' not found or has no prediction history. "
                "Please verify the deployment ID exists and you have access to it."
            )
        raise ToolError(
            f"Failed to retrieve prediction history for deployment '{deployment_id}': {error_str}"
        )
    rows = data.get("data", [])
    next_page = data.get("next")

    return {
        "deployment_id": deployment_id,
        "row_count": len(rows),
        "rows": rows,
        "has_more": next_page is not None,
    }
