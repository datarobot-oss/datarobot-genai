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

This module is responsible for creating complete ExternalToolRegistrationConfig
objects from DataRobot deployments. It handles all aspects of configuration
including metadata fetching, URL construction, authentication, and schema assembly.
"""

import re
from urllib.parse import urljoin

import datarobot as dr

from datarobot_genai.drmcp.core.clients import get_api_client
from datarobot_genai.drmcp.core.dynamic_tools.deployment.adapters.base import MetadataBase
from datarobot_genai.drmcp.core.dynamic_tools.deployment.metadata import get_mcp_tool_metadata
from datarobot_genai.drmcp.core.dynamic_tools.register import ExternalToolRegistrationConfig


def create_deployment_tool_config(
    deployment: dr.Deployment,
) -> ExternalToolRegistrationConfig:
    """Create an ExternalToolRegistrationConfig from deployment.

    This is the main public API of this module. It gathers all the information
    needed to register a DataRobot deployment as an external tool, handling both
    DRUM and standard deployments. It fetches metadata, extracts deployment-specific
    configuration, and assembles everything into a complete registration config.

    Args:
        deployment: The DataRobot deployment object.

    Returns
    -------
        ExternalToolRegistrationConfig with all parameters needed for registration.

    Raises
    ------
        Exception: If metadata fetching or config assembly fails.
    """
    # Fetch and validate metadata from the deployment
    metadata = get_mcp_tool_metadata(deployment)

    # Get deployment-specific infrastructure configuration
    base_url = _get_deployment_base_url(deployment)
    auth_headers = _get_deployment_auth_headers(deployment)

    # Merge metadata headers with deployment headers
    merged_headers = {**auth_headers, **metadata.headers}

    # Build endpoint path
    endpoint = metadata.endpoint.lstrip("/")

    # Generate tool name and description
    tool_name = _get_tool_name(deployment, metadata)
    tool_description = _get_tool_description(deployment, metadata)

    return ExternalToolRegistrationConfig(
        name=tool_name,
        title=deployment.label,
        description=tool_description,
        method=metadata.method,
        base_url=base_url,
        endpoint=endpoint,
        headers=merged_headers,
        input_schema=metadata.input_schema,
        tags=set(),  # Add missing tags parameter
    )


def _is_serverless_deployment(deployment: dr.Deployment) -> bool:
    """Check if deployment is serverless."""
    if not deployment.prediction_environment:
        return False
    return deployment.prediction_environment.get("platform") == "datarobotServerless"


def _get_deployment_base_url(deployment: dr.Deployment) -> str:
    """Get base URL for deployment prediction server.

    Args:
        deployment: DataRobot deployment instance

    Returns
    -------
        Formatted deployment URL including deployment ID path

    Raises
    ------
        ValueError: If prediction server cannot be determined
    """
    api_client = get_api_client()

    # Determine base URL based on deployment type
    if _is_serverless_deployment(deployment):
        base_url = api_client.endpoint
    elif "datarobot-nginx" in api_client.endpoint:
        # On-prem/ST SAAS environments
        base_url = "http://datarobot-prediction-server:80/predApi/v1.0"
    else:
        # Regular prediction server
        pred_server = deployment.default_prediction_server
        if not pred_server:
            raise ValueError(f"Deployment {deployment.id} has no default prediction server")

        url = pred_server["url"]
        if not url:
            raise ValueError(f"Deployment {deployment.id} prediction server has no URL")
        base_url = f"{url}/predApi/v1.0"

    merged_url = urljoin(base_url.rstrip("/") + "/", f"deployments/{deployment.id}/")
    return merged_url


def _get_deployment_auth_headers(deployment: dr.Deployment) -> dict[str, str]:
    """Get authentication headers for deployment.

    Args:
        deployment: DataRobot deployment instance

    Returns
    -------
        Dictionary of authentication headers
    """
    headers = {"Authorization": f"Bearer {get_api_client().token}"}

    # For non-serverless deployments, include datarobot-key
    if not _is_serverless_deployment(deployment):
        pred_server = deployment.default_prediction_server
        if pred_server:
            dr_key = pred_server.get("datarobot-key")
            if dr_key:
                headers["datarobot-key"] = dr_key

    return headers


def _get_tool_name(deployment: dr.Deployment, metadata: MetadataBase) -> str:
    """Generate tool name from deployment and metadata."""
    tool_name = deployment.label or metadata.name or f"deployment_{deployment.id}"
    return _convert_tool_string(tool_name)


def _get_additional_prediction_instructions(deployment_id: str) -> str:
    """Generate additional instructions for scoring prediction models, to make tool usage more
    reliable.
    """
    return f"""
    
Follow these steps in order:
1. Get deployment info: Call tools with the deployment_id="{deployment_id}" to learn about
   features and requirements.
2. Retrieve features: Use `get_deployment_features` to see all required and optional features
   with their importance scores.
3. Prepare data: Use `generate_prediction_data_template` to create the correctly structured
   CSV format.
4. Consider feature importance: For high-importance features, always provide values (infer or
   ask). Low-importance features can be left blank.
5. Validate: Run `validate_prediction_data` before submission to catch errors early.
6. Time series note: Ensure `datetime_column` and `series_id_columns` are properly formatted
   if applicable.

Parameter details and format requirements are specified in the input schema below."""


def _get_tool_description(deployment: dr.Deployment, metadata: MetadataBase) -> str:
    """Generate tool description from deployment and metadata.

    Args:
        deployment: The DataRobot deployment object.
        metadata: The metadata adapter containing tool information.

    Returns
    -------
        Complete tool description, optionally enhanced with workflow instructions
        for prediction endpoints.
    """
    base_description = deployment.description or metadata.description

    if metadata.endpoint.endswith("predictions"):
        additional_instructions = _get_additional_prediction_instructions(deployment.id)
        return f"{base_description}{additional_instructions}"

    return base_description


def _convert_tool_string(text: str | None) -> str:
    """Convert a string to a valid tool name format.

    Removes brackets, replaces spaces/hyphens with underscores, removes special
    characters, converts to lowercase, and cleans up multiple underscores.
    """
    if not text:
        return ""

    # Remove anything within brackets (including the brackets)
    text = re.sub(r"\[.*?\]", "", text)

    # Replace spaces with underscores
    text = text.replace(" ", "_")
    text = text.replace("-", "_")

    # Remove all non-alphanumeric characters except underscores
    text = re.sub(r"[^a-zA-Z0-9_]", "", text)

    # Convert to lowercase
    text = text.lower()

    # Clean up any multiple underscores that might result
    text = re.sub(r"_+", "_", text)

    # Remove leading/trailing underscores
    text = text.strip("_")

    return text
