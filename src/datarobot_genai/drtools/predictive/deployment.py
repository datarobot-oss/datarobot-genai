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
from datarobot_genai.drtools.core.clients.datarobot import DataRobotClient
from datarobot_genai.drtools.core.clients.datarobot import get_datarobot_access_token
from datarobot_genai.drtools.core.deployment_utils import MODEL_EXTENSIONS
from datarobot_genai.drtools.core.deployment_utils import REQUIRED_FILES
from datarobot_genai.drtools.core.deployment_utils import deploy_custom_model_impl
from datarobot_genai.drtools.core.deployment_utils import find_model_file_in_folder
from datarobot_genai.drtools.core.exceptions import ToolError

logger = logging.getLogger(__name__)


@tool_metadata(
    tags={"predictive", "deployment", "read", "management", "list", "daria"},
    description=(
        "[Deploy—discover deployments] Use when the user needs their MLOps deployments as "
        "id-to-label map. Read-only. Not modeling projects (list_projects), not logged "
        "prediction rows (get_prediction_history), not scoring payloads "
        "(predict_* / get_deployment_info)."
    ),
)
async def list_deployments() -> dict[str, Any]:
    token = await get_datarobot_access_token()
    client = DataRobotClient(token).get_client()
    deployments = client.Deployment.list()
    if not deployments:
        return {"deployments": []}
    deployments_dict = {d.id: d.label for d in deployments}
    return {"deployments": deployments_dict}


@tool_metadata(
    tags={"predictive", "deployment", "read", "model", "info", "daria"},
    description=(
        "[Deploy—model linkage] Use when the user wants the model record attached to a deployment "
        "(model id, project linkage). Read-only and narrow. For input feature names, types, and "
        "prediction contract, use get_deployment_info instead; for training leaderboard details, "
        "use get_model_details with project_id + model_id."
    ),
)
async def get_model_info_from_deployment(
    *,
    deployment_id: Annotated[str, "MLOps deployment id."],
) -> dict[str, Any]:
    if not deployment_id:
        raise ToolError("Deployment ID must be provided")
    token = await get_datarobot_access_token()
    client = DataRobotClient(token).get_client()
    deployment = client.Deployment.get(deployment_id)
    return deployment.model


@tool_metadata(
    tags={"predictive", "deployment", "write", "model", "create", "daria"},
    description=(
        "[Deploy—from leaderboard model] Use when the user wants a new live MLOps deployment "
        "from an existing trained leaderboard model_id (from list_models / get_best_model). "
        "Returns deployment id and label. Not batch or realtime scoring by itself, not custom "
        "folder deploy (deploy_custom_model when enabled), not listing deployments "
        "(list_deployments)."
    ),
)
async def deploy_model(
    *,
    model_id: Annotated[str, "Trained model id from list_models / leaderboard."],
    label: Annotated[str, "Human-readable deployment name shown in the UI."],
    description: Annotated[str, "Optional longer description for operators."] | None = None,
) -> dict[str, Any]:
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


# TODO: MODEL-23163 - This tool does not support remote MCP deployments, update or remove.
# @tool_metadata(tags={"predictive", "deployment", "write", "custom", "create"})
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


@tool_metadata(
    tags={"predictive", "deployment", "read", "predictions", "history", "daria"},
    description=(
        "[Deploy—prediction audit log] Use when the user asks for historical prediction rows "
        "already logged for a deployment (monitoring, audit). Read-only pagination; does not run "
        "new scores. For fresh scoring use predict_realtime, predict_by_ai_catalog, "
        "predict_from_project_data, etc."
    ),
)
async def get_prediction_history(
    *,
    deployment_id: Annotated[str, "MLOps deployment id."],
    limit: Annotated[int, "Max rows in this page."] = 100,
    offset: Annotated[int, "Rows to skip (pagination)."] = 0,
    start_time: Annotated[str, "Optional ISO 8601 lower bound on prediction time."] | None = None,
    end_time: Annotated[str, "Optional ISO 8601 upper bound on prediction time."] | None = None,
) -> dict[str, Any]:
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

    response = rest_client.get(f"deployments/{deployment_id}/predictionResults/", params=params)
    data = response.json()
    rows = data.get("data", [])
    next_page = data.get("next")

    return {
        "deployment_id": deployment_id,
        "row_count": len(rows),
        "rows": rows,
        "has_more": next_page is not None,
    }
