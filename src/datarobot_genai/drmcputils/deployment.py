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

"""Pure helpers for DataRobot deployment prediction URLs and auth headers."""

from urllib.parse import urljoin

import datarobot as dr


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
