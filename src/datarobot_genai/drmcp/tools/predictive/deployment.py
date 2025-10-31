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

from datarobot_genai.drmcp.core.clients import get_sdk_client
from datarobot_genai.drmcp.core.mcp_instance import dr_mcp_tool

logger = logging.getLogger(__name__)


@dr_mcp_tool(tags={"deployment", "management", "list"})
async def list_deployments() -> str:
    """
    List all DataRobot deployments for the authenticated user.

    Returns
    -------
        A string summary of the user's DataRobot deployments.
    """
    client = get_sdk_client()
    deployments = client.Deployment.list()
    if not deployments:
        logger.info("No deployments found")
        return "No deployments found."
    result = "\n".join(f"{d.id}: {d.label}" for d in deployments)
    logger.info(f"Found {len(deployments)} deployments")
    return result


@dr_mcp_tool(tags={"deployment", "model", "info"})
async def get_model_info_from_deployment(deployment_id: str) -> str:
    """
    Get model info associated with a given deployment ID.

    Args:
        deployment_id: The ID of the DataRobot deployment.

    Returns
    -------
        The model info associated with the deployment as a JSON string.
    """
    client = get_sdk_client()
    deployment = client.Deployment.get(deployment_id)
    logger.info(f"Retrieved model info for deployment {deployment_id}")
    return json.dumps(deployment.model, indent=2)


@dr_mcp_tool(tags={"deployment", "model", "create"})
async def deploy_model(model_id: str, label: str, description: str = "") -> str:
    """
    Deploy a model by creating a new DataRobot deployment.

    Args:
        model_id: The ID of the DataRobot model to deploy.
        label: The label/name for the deployment.
        description: Optional description for the deployment.

    Returns
    -------
        JSON string with deployment ID and label, or error message.
    """
    client = get_sdk_client()
    try:
        prediction_servers = client.PredictionServer.list()
        if not prediction_servers:
            logger.error("No prediction servers available")
            return json.dumps({"error": "No prediction servers available"})
        deployment = client.Deployment.create_from_learning_model(
            model_id=model_id,
            label=label,
            description=description,
            default_prediction_server_id=prediction_servers[0].id,
        )
        logger.info(f"Created deployment {deployment.id} with label {label}")
        return json.dumps({"deployment_id": deployment.id, "label": label})
    except Exception as e:
        logger.error(f"Error deploying model {model_id}: {type(e).__name__}: {e}")
        return json.dumps({"error": f"Error deploying model {model_id}: {type(e).__name__}: {e}"})
