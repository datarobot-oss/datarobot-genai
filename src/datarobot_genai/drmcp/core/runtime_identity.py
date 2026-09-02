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

"""Where this MCP server is reachable from outside.

Neither hosting mode knows its own public URL when the image is built, because
the id is assigned at deploy time. Both can compose it once running, from the id
DataRobot injects into the container and the platform's fixed routing patterns:

===========  =====================================================
deployment   ``{endpoint}/deployments/{id}/directAccess/{path}``
workload     ``{endpoint}/endpoints/workloads/{id}/{path}``
===========  =====================================================

Both are routed through the DataRobot API host, so composing the URL needs no
API call: there is nothing to time out, retry, or cache, and a server can always
answer the question about itself.

Two deliberate choices worth knowing about:

``DATAROBOT_PUBLIC_API_ENDPOINT`` wins over ``DATAROBOT_ENDPOINT``
    An on-prem install commonly points ``DATAROBOT_ENDPOINT`` at an internal
    cluster address. A resource identifier built from it is one no external
    client can reach, and RFC 9728 §7.3 has the client compare the metadata URL
    it fetched against ``resource``, so a wrong value fails discovery outright.

No fallback endpoint
    When neither variable is set this returns ``None`` rather than guessing a
    default host. The value is published as this server's identity, and a
    confidently wrong identity is worse than an absent one.

This module is intentionally standalone: ``drmcp`` does not depend on
``datarobot_genai.core`` or on ``dragent``, so the few env var names and URL
patterns are restated here rather than shared. Keeping the MCP and agent trees
independently evolvable is worth more than removing this much duplication.
"""

from __future__ import annotations

import os
from enum import Enum
from enum import auto

WORKLOAD_ID_ENV = "WORKLOAD_ID"
DEPLOYMENT_ID_ENV = "MLOPS_DEPLOYMENT_ID"
PUBLIC_ENDPOINT_ENV = "DATAROBOT_PUBLIC_API_ENDPOINT"
ENDPOINT_ENV = "DATAROBOT_ENDPOINT"

#: Segment the platform routes deployment traffic through.
DEPLOYMENT_DIRECT_ACCESS_SEGMENT = "directAccess"
#: Prefix the platform routes workload traffic through.
WORKLOAD_ENDPOINTS_SEGMENT = "endpoints/workloads"

#: Path this server is served from, relative to whichever prefix its mode uses.
DEFAULT_MCP_PATH = "mcp"


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


class DeploymentRelatedConfig(Enum):
    WORKLOAD_ID = auto()
    MLOPS_DEPLOYMENT_ID = auto()
    DATAROBOT_PUBLIC_API_ENDPOINT = auto()
    DATAROBOT_ENDPOINT = auto()
    DR_WORKLOAD_EXTERNAL_URL_HOST = auto()
    DR_WORKLOAD_EXTERNAL_URL_PREFIX = auto()

    def get_from_os_env(self) -> str | None:
        return os.getenv(self.name, "").strip() or None

    @staticmethod
    def get_datarobot_public_api_endpoint() -> str | None:
        return (
            DeploymentRelatedConfig.DATAROBOT_PUBLIC_API_ENDPOINT.get_from_os_env()
            or DeploymentRelatedConfig.DATAROBOT_ENDPOINT.get_from_os_env()
        )

    @staticmethod
    def get_workload_deployment_path_segment() -> str | None:
        return DeploymentRelatedConfig.DR_WORKLOAD_EXTERNAL_URL_PREFIX.get_from_os_env()


class DeploymentEndpointResolver:
    def __init__(self, mcp_path_segment: str = DEFAULT_MCP_PATH):
        self.mcp_path_segment = mcp_path_segment.strip("/")

    @staticmethod
    def get_workload_deployment_id() -> str | None:
        return DeploymentRelatedConfig.WORKLOAD_ID.get_from_os_env()

    @staticmethod
    def get_mlops_deployment_id() -> str | None:
        return DeploymentRelatedConfig.MLOPS_DEPLOYMENT_ID.get_from_os_env()

    @staticmethod
    def is_workload_deployment() -> bool:
        return DeploymentEndpointResolver.get_workload_deployment_id() is not None

    @staticmethod
    def get_deployment_id() -> str | None:
        return (
            DeploymentEndpointResolver.get_workload_deployment_id()
            or DeploymentEndpointResolver.get_mlops_deployment_id()
        )

    @staticmethod
    def get_non_public_api_based_gateway_url() -> str | None:
        return DeploymentRelatedConfig.DR_WORKLOAD_EXTERNAL_URL_HOST.get_from_os_env()

    @staticmethod
    def get_public_api_based_gateway_url() -> str | None:
        return DeploymentRelatedConfig.get_datarobot_public_api_endpoint()

    @staticmethod
    def is_deployed_behind_non_public_api_based_gateway() -> bool:
        return DeploymentEndpointResolver.get_non_public_api_based_gateway_url() is not None

    @staticmethod
    def get_gateway_url() -> str | None:
        gateway_url = (
            DeploymentEndpointResolver.get_non_public_api_based_gateway_url()
            or DeploymentEndpointResolver.get_public_api_based_gateway_url()
        )
        if gateway_url and "://" not in gateway_url:
            gateway_url = f"https://{gateway_url}"
        return gateway_url

    def build_url_of_mcp_mlops_deployment_behind_public_api_based_gateway(
        self, gateway_url: str, deployment_id: str
    ) -> str:
        gateway_url = gateway_url.rstrip("/")
        return f"{gateway_url}/deployments/{deployment_id}/directAccess/{self.mcp_path_segment}"

    def build_url_of_mcp_workload_deployment_behind_public_api_based_gateway(
        self, gateway_url: str, workload_id: str
    ) -> str:
        gateway_url = gateway_url.rstrip("/")
        return f"{gateway_url}/endpoints/workloads/{workload_id}/{self.mcp_path_segment}"

    def build_url_of_mcp_workload_deployment_behind_non_public_api_based_gateway(
        self, gateway_url: str, workload_deployment_segment: str
    ) -> str:
        gateway_url = gateway_url.rstrip("/")
        workload_deployment_segment = workload_deployment_segment.strip("/")
        url_segment = f"{workload_deployment_segment}/{self.mcp_path_segment}"
        return f"{gateway_url}/{url_segment}"

    @staticmethod
    def is_workload_deployment_behind_non_public_api_gateway() -> bool:
        return (
            DeploymentRelatedConfig.DR_WORKLOAD_EXTERNAL_URL_PREFIX.get_from_os_env() is not None
            and DeploymentRelatedConfig.DR_WORKLOAD_EXTERNAL_URL_HOST.get_from_os_env() is not None
        )

    def get_deployment_url(self) -> str | None:
        gateway_url = self.get_gateway_url()
        if not gateway_url:
            return None
        deployment_id = self.get_deployment_id()
        if not deployment_id:
            return None

        # DR_WORKLOAD_EXTERNAL_URL_HOST/_PREFIX are only ever set by the platform for
        # a genuine workload deployment (MLOps deployments are only ever valid behind
        # the public API gateway), so no separate deployment-type cross-check is
        # needed here — this trusts that platform-level invariant rather than
        # re-verifying it.
        if self.is_workload_deployment_behind_non_public_api_gateway():
            workload_deployment_url_segment = (
                DeploymentRelatedConfig.DR_WORKLOAD_EXTERNAL_URL_PREFIX.get_from_os_env()
            )
            return self.build_url_of_mcp_workload_deployment_behind_non_public_api_based_gateway(
                gateway_url,
                workload_deployment_url_segment,  # type: ignore[arg-type]
            )
        if self.is_workload_deployment():
            return self.build_url_of_mcp_workload_deployment_behind_public_api_based_gateway(
                gateway_url,
                deployment_id,
            )
        return self.build_url_of_mcp_mlops_deployment_behind_public_api_based_gateway(
            gateway_url,
            deployment_id,
        )
