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

from datarobot_genai.drmcp.core.dynamic_tools.deployment.adapters.base import MetadataBase
from datarobot_genai.drmcp.core.dynamic_tools.deployment.adapters.default import Metadata
from datarobot_genai.drmcp.core.dynamic_tools.deployment.adapters.drum import DrumMetadataAdapter
from datarobot_genai.drmcp.core.dynamic_tools.deployment.adapters.drum import DrumTargetType
from datarobot_genai.drmcp.core.dynamic_tools.deployment.adapters.drum import is_drum
from datarobot_genai.drtools.core.clients.datarobot import request_user_dr_client

logger = logging.getLogger(__name__)


def _normalize_api_response(response: Any) -> dict[str, Any]:
    """Normalize API response to a dictionary.

    The API can return either a list or a dict. This function ensures
    we always get a dict, taking the first element if it's a list.

    Args:
        response: The raw API response object.

    Returns
    -------
        Normalized dictionary representation of the response.
    """
    data = from_api(response.json())
    if isinstance(data, list):
        return data[0] if data else {}
    if isinstance(data, dict):
        return data
    return {}


def _get_model_attribute(model: dict[str, Any], key: str, default: str = "") -> str:
    """Safely extract a string attribute from a model dictionary.

    Args:
        model: The deployment model dictionary.
        key: The attribute key to retrieve.
        default: Default value if key is not found.

    Returns
    -------
        The attribute value as a lowercase string.
    """
    value = model.get(key, default)
    return str(value).lower() if value else default


def _is_datarobot_structured_prediction(deployment: dr.Deployment) -> str | None:
    """Check if deployment is a DataRobot structured prediction model.

    DataRobot native predictive models with structured predictions don't support
    metadata fetching via API, so they need special handling.

    Args:
        deployment: The DataRobot deployment object.

    Returns
    -------
        Target type string if it's a DataRobot structured prediction, None otherwise.
    """
    if deployment.model is None:
        return None

    # deployment.model is a TypedDict at type-check time, but dict at runtime
    # Cast it to Dict[str, Any] to allow dynamic key access
    model_dict: dict[str, Any] = deployment.model  # type: ignore[assignment]
    target_type = _get_model_attribute(model_dict, "target_type")
    build_env = _get_model_attribute(model_dict, "build_environment_type")

    if not target_type or not build_env:
        return None

    if build_env == "datarobot" and target_type in DrumTargetType.prediction_types():
        return target_type

    return None


def _fetch_deployment_metadata(deployment: dr.Deployment) -> dict[str, Any]:
    """Fetch metadata from deployment's directAccess/info endpoint.

    Args:
        deployment: The DataRobot deployment object.

    Returns
    -------
        Normalized metadata dictionary.

    Raises
    ------
        RuntimeError: If API call fails.
    """
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
    """Return whether the deployment advertises chat completions support.

    Reads the ``supports_chat_api`` flag from the deployment's
    ``/capabilities/`` endpoint. Defaults to ``False`` on any failure or when
    the flag is absent so that endpoint routing for legacy deployments without
    this capability stays on ``/predictions``.

    Args:
        deployment: The DataRobot deployment object.

    Returns
    -------
        True only when the capability is present and explicitly true.
    """
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

    This method retrieves deployment metadata from the /directAccess/info/
    endpoint and validates it contains the required fields for tool registration.
    It uses a universal approach where DRUM deployments are treated as a
    special case and converted to a standard format.

    The returned metadata must contain at minimum:
    - input_schema: JSON schema describing the tool's input parameters
    - endpoint: The deployment endpoint path to call

    Args:
        deployment: The DataRobot deployment object.

    Raises
    ------
        RuntimeError: If metadata fetching fails.
        ValueError: If the deployment metadata is missing or invalid.

    Returns
    -------
        MetadataBase adapter instance to expose required
        metadata properties for tool registration in a standard way.
    """
    # Check if this is a DataRobot native structured prediction model
    # These don't support metadata API, so we create minimal metadata.
    # Native DR predictive models are never chat-capable, so we skip the
    # capabilities lookup for them.
    target_type = _is_datarobot_structured_prediction(deployment)
    if target_type:
        return DrumMetadataAdapter.from_target_type(target_type)

    # Fetch metadata from the deployment's info endpoint
    metadata = _fetch_deployment_metadata(deployment)

    # Inject the chat-API capability flag so adapters can route chat-style
    # deployments (e.g. Guarded RAG, LLM blueprints) to /chat/completions
    # instead of /predictions. Defaults to False when absent.
    metadata["supports_chat_api"] = _fetch_supports_chat_api(deployment)

    # Return appropriate adapter based on metadata type
    if is_drum(metadata):
        return DrumMetadataAdapter.from_deployment_metadata(metadata)

    return Metadata(metadata)
