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

from fastmcp.exceptions import ToolError
from fastmcp.tools.tool import ToolResult

from datarobot_genai.drmcp import dr_mcp_integration_tool
from datarobot_genai.drmcp.tools.clients.datarobot import MODEL_EXTENSIONS
from datarobot_genai.drmcp.tools.clients.datarobot import REQUIRED_FILES
from datarobot_genai.drmcp.tools.clients.datarobot import DataRobotClient
from datarobot_genai.drmcp.tools.clients.datarobot import deploy_custom_model_impl
from datarobot_genai.drmcp.tools.clients.datarobot import find_model_file_in_folder
from datarobot_genai.drmcp.tools.clients.datarobot import get_datarobot_access_token

logger = logging.getLogger(__name__)


@dr_mcp_integration_tool(tags={"predictive", "deployment", "read", "management", "list"})
async def list_deployments() -> ToolResult:
    """List all DataRobot deployments for the authenticated user."""
    token = await get_datarobot_access_token()
    client = DataRobotClient(token).get_client()
    deployments = client.Deployment.list()
    if not deployments:
        return ToolResult(
            structured_content={"deployments": []},
        )
    deployments_dict = {d.id: d.label for d in deployments}
    return ToolResult(
        structured_content={"deployments": deployments_dict},
    )


@dr_mcp_integration_tool(tags={"predictive", "deployment", "read", "model", "info"})
async def get_model_info_from_deployment(
    *,
    deployment_id: Annotated[str, "The ID of the DataRobot deployment"] | None = None,
) -> ToolResult:
    """Retrieve model info associated with a given deployment ID."""
    if not deployment_id:
        raise ToolError("Deployment ID must be provided")

    token = await get_datarobot_access_token()
    client = DataRobotClient(token).get_client()
    deployment = client.Deployment.get(deployment_id)
    return ToolResult(
        structured_content=deployment.model,
    )


@dr_mcp_integration_tool(tags={"predictive", "deployment", "write", "model", "create"})
async def deploy_model(
    *,
    model_id: Annotated[str, "The ID of the DataRobot model to deploy"] | None = None,
    label: Annotated[str, "The label/name for the deployment"] | None = None,
    description: Annotated[str, "Optional description for the deployment"] | None = None,
) -> ToolResult:
    """Deploy a model by creating a new DataRobot deployment."""
    if not model_id:
        raise ToolError("Model ID must be provided")
    if not label:
        raise ToolError("Model label must be provided")

    token = await get_datarobot_access_token()
    client = DataRobotClient(token).get_client()
    try:
        prediction_servers = client.PredictionServer.list()
        if not prediction_servers:
            raise ToolError("No prediction servers available for deployment.")
        deployment = client.Deployment.create_from_learning_model(
            model_id=model_id,
            label=label,
            description=description,
            default_prediction_server_id=prediction_servers[0].id,
        )
        return ToolResult(
            structured_content={
                "deployment_id": deployment.id,
                "label": label,
            },
        )
    except Exception as e:
        raise ToolError(f"Error deploying model {model_id}: {type(e).__name__}: {e}")


@dr_mcp_integration_tool(tags={"predictive", "deployment", "write", "custom", "create"})
async def deploy_custom_model(
    *,
    model_folder: Annotated[
        str, "Path to directory with custom.py, requirements.txt, and optionally a model file"
    ]
    | None = None,
    name: Annotated[str, "Single name used for both custom model and deployment"] | None = None,
    target_type: Annotated[str, "Target type: binary, regression, or multiclass"] | None = None,
    target_name: Annotated[str, "Target column name"] | None = None,
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
) -> ToolResult:
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
    try:
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
        return ToolResult(structured_content=out)
    except Exception as e:
        raise ToolError(f"Deploy custom model failed: {type(e).__name__}: {e}")
