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

import logging
from typing import Annotated
from typing import Any

import datarobot as dr
from datarobot.errors import ClientError

from datarobot_genai.drtools.core import tool_metadata
from datarobot_genai.drtools.core.clients.datarobot import ThreadSafeDataRobotClient
from datarobot_genai.drtools.core.clients.datarobot_workload import WorkloadApiClient
from datarobot_genai.drtools.core.exceptions import ToolError
from datarobot_genai.drtools.core.exceptions import ToolErrorKind
from datarobot_genai.drtools.pagination import clamp_limit
from datarobot_genai.drtools.pagination import merge_pagination_metadata
from datarobot_genai.drtools.predictive.client_exceptions import raise_tool_error_for_client_error

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Workload tools — read                                                #
# ------------------------------------------------------------------ #


@tool_metadata(
    tags={"workload", "datarobot", "list", "search"},
    description=(
        "[Workload—list] Use to discover workloads: returns id, name, status, "
        "artifactId, importance, and runtime for each workload. Supports "
        "Not for a single known workload id (workload_get), not for artifacts "
        "(artifact_list).\n\n"
        "Example: workload_list(limit=50) or workload_list(search='my-app', limit=20)."
    ),
)
async def workload_list(
    *,
    search: Annotated[
        str | None,
        "Optional server-side filter; matches workload name or id.",
    ] = None,
    limit: Annotated[
        int,
        "Maximum workloads to return (1–100). Default 100.",
    ] = 100,
    offset: Annotated[
        int,
        "Number of workloads to skip for pagination. Default 0.",
    ] = 0,
) -> dict[str, Any]:
    if offset < 0:
        raise ToolError(
            "Argument validation error: 'offset' must be >= 0.",
            kind=ToolErrorKind.VALIDATION,
        )

    clamped_limit, note = clamp_limit(limit)

    try:
        with ThreadSafeDataRobotClient().request_user_client():
            rest_client = dr.client.get_client()
            client = WorkloadApiClient(rest_client)
            result = client.list_workloads(limit=clamped_limit, offset=offset, search=search)
    except ClientError as exc:
        raise_tool_error_for_client_error(exc)

    data = result.get("data", []) or []
    return merge_pagination_metadata(
        {"workloads": data, "count": len(data)},
        result,
        note,
        offset=offset,
        limit=clamped_limit,
    )


@tool_metadata(
    tags={"workload", "datarobot", "get"},
    description=(
        "[Workload—get] Use when you have an exact workload id and need the "
        "full record: status, runtime, artifact reference, endpoint URL, "
        "creator, and timestamps. Not for discovery (workload_list), not for "
        "debugging pod conditions (proton_status_details).\n\n"
        "Example: workload_get(workload_id='wkld-abc123')."
    ),
)
async def workload_get(
    *,
    workload_id: Annotated[
        str,
        "The unique id of the workload (from workload_list or workload_create).",
    ],
) -> dict[str, Any]:
    if not workload_id or not workload_id.strip():
        raise ToolError(
            "Argument validation error: 'workload_id' cannot be empty.",
            kind=ToolErrorKind.VALIDATION,
        )

    try:
        with ThreadSafeDataRobotClient().request_user_client():
            rest_client = dr.client.get_client()
            client = WorkloadApiClient(rest_client)
            return client.get_workload(workload_id.strip())
    except ClientError as exc:
        raise_tool_error_for_client_error(exc)


# ------------------------------------------------------------------ #
# Bundle tools                                                         #
# ------------------------------------------------------------------ #


@tool_metadata(
    tags={"workload", "bundle", "datarobot", "list"},
    description=(
        "[Workload—bundle list] Use to discover available compute resource "
        "bundles (CPU count, memory, GPU type and VRAM). GPU type and VRAM "
        "are always determined by the bundle — there is no separate gpuType "
        "parameter. Use the bundle id when creating or updating a workload.\n\n"
        "Example: bundle_list()."
    ),
)
async def bundle_list() -> dict[str, Any]:
    try:
        with ThreadSafeDataRobotClient().request_user_client():
            rest_client = dr.client.get_client()
            client = WorkloadApiClient(rest_client)
            return client.list_bundles()
    except ClientError as exc:
        raise_tool_error_for_client_error(exc)
