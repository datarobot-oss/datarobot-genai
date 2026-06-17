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

"""Configuration assembly for DataRobot deployment tools.

This module creates complete ExternalToolRegistrationConfig objects from DataRobot
deployments by combining data fetched via request_user_dr_client with the pure
assembly functions from drmcpbase.
"""

import datarobot as dr

from datarobot_genai.drmcp.core.dynamic_tools.deployment.metadata import get_mcp_tool_metadata
from datarobot_genai.drmcpbase.dynamic_tools.deployment.config import (
    assemble_deployment_tool_config,
)
from datarobot_genai.drmcpbase.dynamic_tools.deployment.config import build_deployment_auth_headers
from datarobot_genai.drmcpbase.dynamic_tools.deployment.config import get_deployment_base_url
from datarobot_genai.drmcpbase.dynamic_tools.external_tool import ExternalToolRegistrationConfig
from datarobot_genai.drmcputils.clients.datarobot import request_user_dr_client


def create_deployment_tool_config(
    deployment: dr.Deployment,
) -> ExternalToolRegistrationConfig:
    """Create an ExternalToolRegistrationConfig from deployment.

    Fetches the user's DR client (endpoint + token) and delegates to the
    pure drmcpbase assembly functions.

    Args:
        deployment: The DataRobot deployment object.

    Returns
    -------
        ExternalToolRegistrationConfig with all parameters needed for registration.
    """
    metadata = get_mcp_tool_metadata(deployment)

    with request_user_dr_client(headers_auth_only=False) as api_client:
        base_url = get_deployment_base_url(deployment, api_client.endpoint)
        auth_headers = build_deployment_auth_headers(deployment, api_client.token)

    return assemble_deployment_tool_config(deployment, metadata, base_url, auth_headers)
