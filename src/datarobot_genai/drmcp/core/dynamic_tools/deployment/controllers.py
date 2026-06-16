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

import datarobot as dr
from fastmcp.tools.tool import Tool

from datarobot_genai.drmcp.core.dynamic_tools.deployment.register import (
    register_tool_of_datarobot_deployment,
)
from datarobot_genai.drmcp.core.lineage.enums import LRSEnvVarIsNotSetError
from datarobot_genai.drmcp.core.lineage.manager import LineageManager
from datarobot_genai.drmcp.core.mcp_instance import mcp
from datarobot_genai.drmcputils.clients.datarobot import request_user_dr_sdk

logger = logging.getLogger(__name__)


async def register_tool_for_deployment_id(deployment_id: str) -> Tool:
    """Register a tool for a specific deployment ID.

    Args:
        deployment_id: The ID of the DataRobot deployment to register as a tool.

    Raises
    ------
        DynamicToolRegistrationError: If registration fails at any step.

    Returns
    -------
        The registered Tool instance.
    """
    with request_user_dr_sdk(headers_auth_only=True):
        deployment = dr.Deployment.get(deployment_id)
    registered_tool = await register_tool_of_datarobot_deployment(deployment)
    try:
        linear_manager = LineageManager(mcp)
        await linear_manager.sync_mcp_tools()
    except LRSEnvVarIsNotSetError as error:
        error_message = f"MCP item metadata is not sync. {str(error)}"
        logger.warning(error_message)

    return registered_tool


async def get_registered_tool_deployments() -> dict[str, str]:
    """Get the tool registered for the deployment in the MCP instance."""
    deployments = await mcp.get_deployment_mapping()
    return deployments


async def delete_registered_tool_deployment(deployment_id: str) -> bool:
    """Delete the tool registered for the deployment in the MCP instance."""
    deployments = await mcp.get_deployment_mapping()
    if deployment_id not in deployments:
        logger.debug(f"No tool registered for deployment {deployment_id}")
        return False

    tool_name = deployments[deployment_id]
    await mcp.remove_deployment_mapping(deployment_id)
    logger.info(f"Deleted tool {tool_name} for deployment {deployment_id}")
    try:
        linear_manager = LineageManager(mcp)
        await linear_manager.sync_mcp_tools()
    except LRSEnvVarIsNotSetError as error:
        error_message = f"MCP item metadata is not sync. {str(error)}"
        logger.warning(error_message)
    return True
