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

"""Pure metadata assembly for DataRobot deployment tools.

All functions in this module are pure: they take pre-fetched data as arguments
and do not make any HTTP calls or access request context.
"""

from typing import Any

import datarobot as dr

from datarobot_genai.drmcpbase.dynamic_tools.deployment.adapters.base import MetadataBase
from datarobot_genai.drmcpbase.dynamic_tools.deployment.adapters.default import Metadata
from datarobot_genai.drmcpbase.dynamic_tools.deployment.adapters.drum import DrumMetadataAdapter
from datarobot_genai.drmcpbase.dynamic_tools.deployment.adapters.drum import DrumTargetType
from datarobot_genai.drmcpbase.dynamic_tools.deployment.adapters.drum import is_drum


def _get_model_attribute(model: dict[str, Any], key: str, default: str = "") -> str:
    """Safely extract a string attribute from a model dictionary."""
    value = model.get(key, default)
    return str(value).lower() if value else default


def _is_datarobot_structured_prediction(deployment: dr.Deployment) -> str | None:
    """Check if deployment is a DataRobot structured prediction model.

    DataRobot native predictive models with structured predictions don't support
    metadata fetching via API, so they need special handling.

    Returns
    -------
        Target type string if it's a DataRobot structured prediction, None otherwise.
    """
    if deployment.model is None:
        return None

    model_dict: dict[str, Any] = deployment.model  # type: ignore[assignment]
    target_type = _get_model_attribute(model_dict, "target_type")
    build_env = _get_model_attribute(model_dict, "build_environment_type")

    if not target_type or not build_env:
        return None

    if build_env == "datarobot" and target_type in DrumTargetType.prediction_types():
        return target_type

    return None


def build_mcp_tool_metadata(
    deployment: dr.Deployment,
    info_payload: dict[str, Any] | None,
    supports_chat_api: bool,
) -> MetadataBase:
    """Assemble a MetadataBase adapter from pre-fetched deployment data.

    This is the pure counterpart to get_mcp_tool_metadata in drmcp. It takes
    data already fetched from the DR API and selects the appropriate adapter.

    Args:
        deployment: The DataRobot deployment object.
        info_payload: The normalised response from directAccess/info/, or None
            when the deployment is a native DataRobot structured prediction model
            that does not expose this endpoint.
        supports_chat_api: Whether the deployment advertises chat completions
            support (from the capabilities endpoint). Ignored when info_payload
            is None.

    Returns
    -------
        MetadataBase adapter instance.

    Raises
    ------
        ValueError: If info_payload is required but not provided.
    """
    target_type = _is_datarobot_structured_prediction(deployment)
    if target_type:
        return DrumMetadataAdapter.from_target_type(target_type)

    if info_payload is None:
        raise ValueError(
            f"info_payload is required for deployment {deployment.id} "
            "which is not a native DataRobot structured prediction model."
        )

    metadata = dict(info_payload)
    metadata["supports_chat_api"] = supports_chat_api

    if is_drum(metadata):
        return DrumMetadataAdapter.from_deployment_metadata(metadata)

    return Metadata(metadata)
