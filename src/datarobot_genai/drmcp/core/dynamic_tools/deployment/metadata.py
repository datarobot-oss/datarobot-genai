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
from typing import Any

import datarobot as dr
from datarobot.utils import from_api

from datarobot_genai.drmcpbase.dynamic_tools.deployment.adapters.base import MetadataBase
from datarobot_genai.drmcpbase.dynamic_tools.deployment.metadata import (
    _is_datarobot_structured_prediction,
)
from datarobot_genai.drmcpbase.dynamic_tools.deployment.metadata import build_mcp_tool_metadata
from datarobot_genai.drmcputils.clients.datarobot import request_user_dr_client

logger = logging.getLogger(__name__)


def _normalize_api_response(response: Any) -> dict[str, Any]:
    """Normalize API response to a dictionary (camelCase → snake_case)."""
    data = from_api(response.json())
    if isinstance(data, list):
        return data[0] if data else {}
    if isinstance(data, dict):
        return data
    return {}


def _fetch_deployment_metadata(deployment: dr.Deployment) -> dict[str, Any]:
    """Fetch metadata from deployment's directAccess/info endpoint."""
    with request_user_dr_client(headers_auth_only=False) as api_client:
        try:
            response = api_client.get(url=f"deployments/{deployment.id}/directAccess/info/")
            response.raise_for_status()
            return _normalize_api_response(response)
        except Exception as exc:
            logger.error(f"Failed to fetch metadata for deployment {deployment.id}: {exc}")
            raise RuntimeError(
                f"Could not retrieve metadata for deployment {deployment.id}"
            ) from exc


def _fetch_supports_chat_api(deployment: dr.Deployment) -> bool:
    """Return whether the deployment advertises chat completions support."""
    with request_user_dr_client(headers_auth_only=False) as api_client:
        try:
            response = api_client.get(url=f"deployments/{deployment.id}/capabilities/")
            response.raise_for_status()
            payload = response.json()
            capabilities = (payload or {}).get("data") or []
            for capability in capabilities:
                if not isinstance(capability, dict):
                    continue
                if capability.get("name") == "supports_chat_api":
                    return bool(capability.get("supported", False))
            return False
        except Exception as exc:
            logger.warning(
                f"Could not fetch capabilities for deployment {deployment.id}; "
                f"assuming chat API not supported: {exc}"
            )
            return False


def get_mcp_tool_metadata(deployment: dr.Deployment) -> MetadataBase:
    """Fetch and validate metadata for a given deployment.

    This method handles HTTP fetching via request_user_dr_client and delegates
    pure adapter selection to drmcpbase.build_mcp_tool_metadata.

    Args:
        deployment: The DataRobot deployment object.

    Raises
    ------
        RuntimeError: If metadata fetching fails.

    Returns
    -------
        MetadataBase adapter instance.
    """
    if _is_datarobot_structured_prediction(deployment):
        return build_mcp_tool_metadata(deployment, None, False)

    info_payload = _fetch_deployment_metadata(deployment)
    supports_chat_api = _fetch_supports_chat_api(deployment)
    return build_mcp_tool_metadata(deployment, info_payload, supports_chat_api)
