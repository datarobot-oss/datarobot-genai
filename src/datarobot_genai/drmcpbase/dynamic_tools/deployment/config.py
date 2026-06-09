# Copyright 2026 DataRobot, Inc.
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

"""Pure configuration assembly for DataRobot deployment tools.

All functions in this module are pure: they take explicit arguments and do not
call request_user_dr_client or access any request context.
"""

import re
from urllib.parse import urljoin

import datarobot as dr

from datarobot_genai.drmcpbase.dynamic_tools.deployment.adapters.base import MetadataBase
from datarobot_genai.drmcpbase.dynamic_tools.external_tool import ExternalToolRegistrationConfig


def _is_serverless_deployment(deployment: dr.Deployment) -> bool:
    """Check if deployment is serverless."""
    if not deployment.prediction_environment:
        return False
    return deployment.prediction_environment.get("platform") == "datarobotServerless"


def get_deployment_base_url(deployment: dr.Deployment, datarobot_endpoint: str) -> str:
    """Compute the prediction base URL for a deployment.

    Args:
        deployment: The DataRobot deployment object.
        datarobot_endpoint: The DataRobot API endpoint (e.g. https://app.datarobot.com).

    Returns
    -------
        Deployment-scoped prediction URL.

    Raises
    ------
        ValueError: If prediction server cannot be determined.
    """
    if _is_serverless_deployment(deployment):
        base_url = datarobot_endpoint
    elif "datarobot-nginx" in datarobot_endpoint:
        # On-prem / ST SaaS environments
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

    return urljoin(base_url.rstrip("/") + "/", f"deployments/{deployment.id}/")


def build_deployment_auth_headers(deployment: dr.Deployment, token: str) -> dict[str, str]:
    """Build authentication headers for a deployment.

    Args:
        deployment: The DataRobot deployment object.
        token: The bearer token to use for authentication.

    Returns
    -------
        Dictionary of authentication headers.
    """
    headers = {"Authorization": f"Bearer {token}"}
    if not _is_serverless_deployment(deployment):
        pred_server = deployment.default_prediction_server
        if pred_server:
            dr_key = pred_server.get("datarobot-key")
            if dr_key:
                headers["datarobot-key"] = dr_key
    return headers


def _convert_tool_string(text: str | None) -> str:
    """Convert a string to a valid tool name format.

    Removes brackets, replaces spaces/hyphens with underscores, removes special
    characters, converts to lowercase, and cleans up multiple underscores.
    """
    if not text:
        return ""

    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace(" ", "_").replace("-", "_")
    text = re.sub(r"[^a-zA-Z0-9_]", "", text)
    text = text.lower()
    text = re.sub(r"_+", "_", text)
    text = text.strip("_")
    return text


def _get_tool_name(deployment: dr.Deployment, metadata: MetadataBase) -> str:
    """Generate tool name from deployment and metadata."""
    tool_name = deployment.label or metadata.name or f"deployment_{deployment.id}"
    return _convert_tool_string(tool_name)


def _get_additional_prediction_instructions(deployment_id: str) -> str:
    return f"""

Follow these steps in order:
1. Get deployment info: Call tools with the deployment_id="{deployment_id}" to learn about
   features and requirements.
2. Retrieve features: Use `deployment_get_features` to see all required and optional features
   with their importance scores.
3. Prepare data: Use `deployment_generate_prediction_sample` to create the correctly structured
   CSV format.
4. Consider feature importance: For high-importance features, always provide values (infer or
   ask). Low-importance features can be left blank.
5. Validate: Run `deployment_validate_prediction_data` before submission to catch errors early.
6. Time series note: Ensure `datetime_column` and `series_id_columns` are properly formatted
   if applicable.

Parameter details and format requirements are specified in the input schema below."""


def _get_tool_description(deployment: dr.Deployment, metadata: MetadataBase) -> str:
    """Generate tool description from deployment and metadata."""
    base_description = deployment.description or metadata.description

    if metadata.endpoint.endswith("predictions"):
        additional_instructions = _get_additional_prediction_instructions(deployment.id)
        return f"{base_description}{additional_instructions}"

    return base_description


def assemble_deployment_tool_config(
    deployment: dr.Deployment,
    metadata: MetadataBase,
    base_url: str,
    auth_headers: dict[str, str],
) -> ExternalToolRegistrationConfig:
    """Assemble an ExternalToolRegistrationConfig from pre-computed parts.

    This is the pure counterpart to create_deployment_tool_config in drmcp.
    All expensive lookups (URL, auth, metadata) are done by the caller and
    passed in as plain arguments.

    Args:
        deployment: The DataRobot deployment object.
        metadata: Resolved MetadataBase adapter.
        base_url: Pre-computed prediction base URL.
        auth_headers: Pre-computed authentication headers.

    Returns
    -------
        ExternalToolRegistrationConfig ready for tool creation.
    """
    merged_headers = {**auth_headers, **metadata.headers}
    endpoint = metadata.endpoint.lstrip("/")
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
        tags=set(),
    )
