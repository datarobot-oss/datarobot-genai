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

"""
The module is intentionally standalone: ``drmcp`` does not depend on
``datarobot_genai.core`` or on ``dragent``, so the few env var names and URL
patterns are restated here rather than shared. Keeping the MCP and agent trees
independently evolvable is worth more than removing this much duplication.
"""

from __future__ import annotations

import os
from enum import Enum
from enum import auto

from datarobot_genai.drmcp.core.constants import MCP_PATH_ENDPOINT
from datarobot_genai.drmcp.core.deployment_config import DeploymentConfig
from datarobot_genai.drmcp.core.deployment_config import MCPDeploymentType
from datarobot_genai.drmcp.core.routes_utils import prefix_mount_path

WORKLOAD_ID_ENV = "WORKLOAD_ID"
DEPLOYMENT_ID_ENV = "MLOPS_DEPLOYMENT_ID"
PUBLIC_ENDPOINT_ENV = "DATAROBOT_PUBLIC_API_ENDPOINT"
ENDPOINT_ENV = "DATAROBOT_ENDPOINT"

#: Segment the platform routes deployment traffic through.
DEPLOYMENT_DIRECT_ACCESS_SEGMENT = "directAccess"
#: Prefix the platform routes workload traffic through.
WORKLOAD_ENDPOINTS_SEGMENT = "endpoints/workloads"


def _env(name: str) -> str | None:
    """Read ``name``, treating blank and unset alike."""
    return os.getenv(name, "").strip() or None


def get_deployment_id() -> str | None:
    """Return the platform-injected deployment id, or None when not on a deployment."""
    return _env(DEPLOYMENT_ID_ENV)


def get_workload_id() -> str | None:
    """Return the platform-injected workload id, or None when not on a workload."""
    return _env(WORKLOAD_ID_ENV)


def resolve_datarobot_endpoint() -> str | None:
    """Return the externally reachable DataRobot API endpoint, or None when unset.

    Prefers ``DATAROBOT_PUBLIC_API_ENDPOINT`` over ``DATAROBOT_ENDPOINT``; see
    the module docstring for why that order matters and why there is no default.
    """
    return _env(PUBLIC_ENDPOINT_ENV) or _env(ENDPOINT_ENV)


def build_deployment_url(endpoint: str, deployment_id: str, path: str) -> str:
    """``{endpoint}/deployments/{deployment_id}/directAccess/{path}``."""
    base = endpoint.rstrip("/")
    return f"{base}/deployments/{deployment_id}/{DEPLOYMENT_DIRECT_ACCESS_SEGMENT}/{path}"


def build_workload_url(endpoint: str, workload_id: str, path: str) -> str:
    """``{endpoint}/endpoints/workloads/{workload_id}/{path}``."""
    base = endpoint.rstrip("/")
    return f"{base}/{WORKLOAD_ENDPOINTS_SEGMENT}/{workload_id}/{path}"


class GatewayType(Enum):
    """It is the gateway behind which MCP is deployed."""

    PUBLIC_API_BASED_GATEWAY = auto()
    NON_PUBLIC_API_BASED_GATEWAY = auto()

    def get_workload_deployment_url_segment(self, deployment_id: str) -> str:
        if self == GatewayType.PUBLIC_API_BASED_GATEWAY:
            url_segment = f"/endpoints/workloads/{deployment_id}/"
        else:
            url_segment = f"/workloads/{deployment_id}/"
        return url_segment.strip("/")

    def get_mlops_deployment_url_segment(self, deployment_id: str) -> str:
        if self == GatewayType.PUBLIC_API_BASED_GATEWAY:
            url_segment = f"/deployments/{deployment_id}/directAccess/"
            return url_segment.strip("/")
        else:
            raise ValueError(
                "MLOps deployment URL is not supported in a non-public-API-based gateway."
            )


class DeploymentEndpointResolver:
    def __init__(self) -> None:
        self.mcp_path_suffix = prefix_mount_path(MCP_PATH_ENDPOINT).strip("/")
        self.gateway_url = self.get_gateway_url()
        self.gateway_type = self.get_gateway_type()
        self.deployment_id = self.get_deployment_id()
        self.deployment_type = self.get_deployment_type()

    @staticmethod
    def get_gateway_url() -> str | None:
        gateway_url = (
            DeploymentConfig.DR_WORKLOAD_EXTERNAL_URL_HOST.get_from_os_env()
            or DeploymentConfig.get_datarobot_public_api_endpoint()
        )
        if gateway_url and "://" not in gateway_url:
            gateway_url = f"https://{gateway_url}"
        if gateway_url:
            gateway_url = gateway_url.rstrip("/")
        return gateway_url

    @staticmethod
    def get_gateway_type() -> GatewayType:
        if DeploymentConfig.DR_WORKLOAD_EXTERNAL_URL_HOST.get_from_os_env() is not None:
            return GatewayType.NON_PUBLIC_API_BASED_GATEWAY
        else:
            return GatewayType.PUBLIC_API_BASED_GATEWAY

    @staticmethod
    def get_deployment_id() -> str | None:
        return (
            DeploymentConfig.WORKLOAD_ID.get_from_os_env()
            or DeploymentConfig.MLOPS_DEPLOYMENT_ID.get_from_os_env()
        )

    @staticmethod
    def get_deployment_type() -> MCPDeploymentType:
        if DeploymentConfig.WORKLOAD_ID.get_from_os_env() is not None:
            return MCPDeploymentType.WORKLOAD
        else:
            return MCPDeploymentType.MLOPS

    def get_deployment_segment_url(self, deployment_id: str) -> str:
        return (
            self.gateway_type.get_workload_deployment_url_segment(deployment_id)
            if self.deployment_type == MCPDeploymentType.WORKLOAD
            else self.gateway_type.get_mlops_deployment_url_segment(deployment_id)
        )

    def get_deployment_url(self) -> str | None:
        if not self.gateway_url or not self.deployment_id:
            return None
        deployment_segment_url = self.get_deployment_segment_url(self.deployment_id)
        return f"{self.gateway_url}/{deployment_segment_url}/{self.mcp_path_suffix}"

    def get_well_known_protected_resource_metadata_url(self) -> str | None:
        if not self.gateway_url or not self.deployment_id:
            return None
        deployment_segment_url = self.get_deployment_segment_url(self.deployment_id)
        sub_path = prefix_mount_path(".well-known/oauth-protected-resource").lstrip("/")
        return f"{self.gateway_url}/{deployment_segment_url}/{sub_path}"
