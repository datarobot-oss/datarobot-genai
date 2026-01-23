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
from typing import Annotated

from fastmcp.exceptions import ToolError
from fastmcp.tools.tool import ToolResult

from datarobot_genai.drmcp.core.clients import get_sdk_client
from datarobot_genai.drmcp.core.mcp_instance import dr_mcp_tool

logger = logging.getLogger(__name__)


@dr_mcp_tool(tags={"predictive", "deployment", "read", "management", "list"})
async def list_deployments() -> ToolResult:
    """List all DataRobot deployments for the authenticated user."""
    client = get_sdk_client()
    deployments = client.Deployment.list()
    if not deployments:
        return ToolResult(
            content="No deployments found.",
            structured_content={"deployments": []},
        )
    deployments_dict = {d.id: d.label for d in deployments}
    return ToolResult(
        content="\n".join(f"{d.id}: {d.label}" for d in deployments),
        structured_content={"deployments": deployments_dict},
    )


@dr_mcp_tool(tags={"predictive", "deployment", "read", "model", "info"})
async def get_model_info_from_deployment(
    *,
    deployment_id: Annotated[str, "The ID of the DataRobot deployment"] | None = None,
) -> ToolError | ToolResult:
    """Retrieve model info associated with a given deployment ID."""
    if not deployment_id:
        raise ToolError("Deployment ID must be provided")

    client = get_sdk_client()
    deployment = client.Deployment.get(deployment_id)
    return ToolResult(
        content=(
            f"Retrieved model info for deployment {deployment_id}, here are the details:\n"
            f"{json.dumps(deployment.model, indent=2)}"
        ),
        structured_content=deployment.model,
    )


@dr_mcp_tool(tags={"predictive", "deployment", "write", "model", "create"})
async def deploy_model(
    *,
    model_id: Annotated[str, "The ID of the DataRobot model to deploy"] | None = None,
    label: Annotated[str, "The label/name for the deployment"] | None = None,
    description: Annotated[str, "Optional description for the deployment"] | None = None,
) -> ToolError | ToolResult:
    """Deploy a model by creating a new DataRobot deployment."""
    if not model_id:
        raise ToolError("Model ID must be provided")
    if not label:
        raise ToolError("Model label must be provided")

    client = get_sdk_client()
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
            content=f"Created deployment {deployment.id} with label {label}",
            structured_content={
                "deployment_id": deployment.id,
                "label": label,
            },
        )
    except Exception as e:
        raise ToolError(f"Error deploying model {model_id}: {type(e).__name__}: {e}")
