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

from datarobot_genai.drmcp.core.clients import get_api_client
from datarobot_genai.drmcp.core.dynamic_tools.deployment.config import create_deployment_tool_config
from datarobot_genai.drmcp.core.dynamic_tools.register import register_external_tool
from datarobot_genai.drmcp.core.exceptions import DynamicToolRegistrationError

logger = logging.getLogger(__name__)


async def register_tools_of_datarobot_deployments() -> None:
    """Register tools for all deployments tagged as 'tool' in DataRobot."""
    deployment_ids = get_datarobot_tool_deployments()
    logger.info(f"Found {len(deployment_ids)} tool deployments.")

    # Try to register each tool deployment, continue on failure
    for deployment_id in deployment_ids:
        try:
            deployment = dr.Deployment.get(deployment_id)
            await register_tool_of_datarobot_deployment(deployment)
        except DynamicToolRegistrationError:
            pass
        except Exception as exc:
            logger.error(f"Unexpected error for deployment {deployment_id}: {exc}")
            pass


async def register_tool_of_datarobot_deployment(
    deployment: dr.Deployment,
) -> Tool:
    """Register a single tool for a given deployment.

    Args:
        deployment: The tool deployment within DataRobot.

    Raises
    ------
        DynamicToolRegistrationError: If registration fails at any step.

    Returns
    -------
        The registered Tool instance.
    """
    logger.info(f"Found tool: id: {deployment.id}, label: {deployment.label}")

    try:
        # Create the configuration object with all necessary information
        # This includes fetching metadata and assembling all deployment-specific data
        config = create_deployment_tool_config(deployment)

        # Register using generic external tool registration with the config
        tool = await register_external_tool(config, deployment_id=deployment.id)

        return tool

    except Exception as exc:
        logger.error(f"Skipping deployment {deployment.id}. Registration failed: {exc}")
        raise DynamicToolRegistrationError("Registration failed. Could not create tool.") from exc


def get_datarobot_tool_deployments() -> list[str]:
    """Fetch deployments from DataRobot that are tagged as 'tool'."""
    # Replace this with dr.Deployment.list when the 3.9.0 version
    # of datarobot python SDK is released.
    deployments_data = dr.utils.pagination.unpaginate(
        initial_url="deployments/",
        initial_params={"tag_values": "tool", "tag_keys": "tool"},
        client=get_api_client(),
    )

    return [deployment["id"] for deployment in deployments_data]
