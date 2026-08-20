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


def resolve_self_url(path: str = DEFAULT_MCP_PATH) -> str | None:
    """Return this server's own externally reachable URL for ``path``.

    Returns ``None`` when the process is not in a DataRobot-hosted runtime, or
    when the endpoint is unset. That is the local-development case: there is no
    platform-assigned id to compose a URL from, and the caller decides what to
    publish instead.
    """
    endpoint = resolve_datarobot_endpoint()
    if not endpoint:
        return None

    path = path.strip("/")
    if deployment_id := get_deployment_id():
        return build_deployment_url(endpoint, deployment_id, path)
    if workload_id := get_workload_id():
        return build_workload_url(endpoint, workload_id, path)
    return None
