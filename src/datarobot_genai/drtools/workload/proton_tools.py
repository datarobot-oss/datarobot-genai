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

from typing import Annotated
from typing import Any

from datarobot.errors import ClientError

from datarobot_genai.drmcputils.constants import LOG_LEVELS
from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.exceptions import ToolErrorKind
from datarobot_genai.drtools.core import tool_metadata
from datarobot_genai.drtools.core.clients.datarobot_workload import WorkloadApiClient
from datarobot_genai.drtools.core.utils import require_id
from datarobot_genai.drtools.pagination import clamp_limit
from datarobot_genai.drtools.pagination import merge_pagination_metadata
from datarobot_genai.drtools.predictive.client_exceptions import raise_tool_error_for_client_error

# ------------------------------------------------------------------ #
# Protons                                                              #
# ------------------------------------------------------------------ #


@tool_metadata(
    tags={"workload", "proton", "datarobot", "list"},
    description=(
        "[Workload—proton list] List all proton (deployed pod) instances for a "
        "workload. Each proton shows its id, status, replica count, and bundle. "
        "Use proton_status_details to inspect per-replica pod conditions.\n\n"
        "Example: proton_list(workload_id='wkld-abc')"
    ),
)
async def proton_list(
    *,
    workload_id: Annotated[str, "Id of the workload."],
    limit: Annotated[int, "Max protons to return (1–100). Default 20."] = 20,
    offset: Annotated[int, "Protons to skip for pagination. Default 0."] = 0,
) -> dict[str, Any]:
    wid = require_id(workload_id, "workload_id")
    if offset < 0:
        raise ToolError(
            "Argument validation error: 'offset' must be >= 0.",
            kind=ToolErrorKind.VALIDATION,
        )
    clamped_limit, note = clamp_limit(limit)
    try:
        result = WorkloadApiClient().list_protons(wid, limit=clamped_limit, offset=offset)
    except ClientError as exc:
        raise_tool_error_for_client_error(exc)

    data = result.get("data", []) or []
    return merge_pagination_metadata(
        {"protons": data, "count": len(data)},
        result,
        note,
        offset=offset,
        limit=clamped_limit,
    )


@tool_metadata(
    tags={"workload", "proton", "datarobot", "get"},
    description=(
        "[Workload—proton get] Retrieve a specific proton by id. Returns status, "
        "replica count, bundle, and timestamps. Use proton_list to discover "
        "proton ids for a workload.\n\n"
        "Example: proton_get(workload_id='wkld-abc', proton_id='ptn-xyz')"
    ),
)
async def proton_get(
    *,
    workload_id: Annotated[str, "Id of the workload."],
    proton_id: Annotated[str, "Id of the proton to retrieve."],
) -> dict[str, Any]:
    wid = require_id(workload_id, "workload_id")
    pid = require_id(proton_id, "proton_id")
    try:
        return WorkloadApiClient().get_proton(wid, pid)
    except ClientError as exc:
        raise_tool_error_for_client_error(exc)


@tool_metadata(
    tags={"workload", "proton", "datarobot", "status", "debug"},
    description=(
        "[Workload—proton status details] Get per-replica pod status for a "
        "proton: container readiness, restart count, pod phase, and conditions. "
        "Returns null when no status has been received yet (proton just scheduled). "
        "Use this to diagnose CrashLoopBackOff, OOMKilled, or image-pull errors.\n\n"
        "Example: proton_status_details(workload_id='wkld-abc', proton_id='ptn-xyz')"
    ),
)
async def proton_status_details(
    *,
    workload_id: Annotated[str, "Id of the workload."],
    proton_id: Annotated[str, "Id of the proton."],
) -> dict[str, Any]:
    wid = require_id(workload_id, "workload_id")
    pid = require_id(proton_id, "proton_id")
    try:
        result = WorkloadApiClient().get_proton_status_details(wid, pid)
    except ClientError as exc:
        raise_tool_error_for_client_error(exc)

    if result is None:
        return {"status": "pending", "message": "No status update received yet for this proton."}
    return result


# ------------------------------------------------------------------ #
# OTel logs                                                            #
# ------------------------------------------------------------------ #


@tool_metadata(
    tags={"workload", "datarobot", "logs", "otel"},
    description=(
        "[Workload—logs] Retrieve OTel log lines for a workload. Supports "
        "filtering by log level, time window, body text (includes/excludes), "
        "and OTel span/trace ids. Use this to debug application errors or "
        "trace request flows.\n\n"
        "Example: workload_logs(workload_id='wkld-abc')\n"
        "Example: workload_logs(workload_id='wkld-abc', level='error', "
        "start_time='2026-01-01T00:00:00Z')"
    ),
)
async def workload_logs(
    *,
    workload_id: Annotated[str, "Id of the workload."],
    limit: Annotated[int, "Max log lines to return (1–100). Default 100."] = 100,
    offset: Annotated[int, "Lines to skip for pagination. Default 0."] = 0,
    level: Annotated[
        str,
        "Minimum log level: 'debug' | 'info' | 'warn' | 'error'. Default 'debug'.",
    ] = "debug",
    start_time: Annotated[
        str | None, "ISO-8601 start of the log window (e.g. '2026-01-01T00:00:00Z')."
    ] = None,
    end_time: Annotated[str | None, "ISO-8601 end of the log window."] = None,
    includes: Annotated[
        list[str] | None,
        "Body text substrings that must be present (AND logic per entry).",
    ] = None,
    excludes: Annotated[
        list[str] | None,
        "Body text substrings that must be absent.",
    ] = None,
    span_id: Annotated[str | None, "Filter to a specific OTel span id."] = None,
    trace_id: Annotated[str | None, "Filter to a specific OTel trace id."] = None,
) -> dict[str, Any]:
    wid = require_id(workload_id, "workload_id")
    if offset < 0:
        raise ToolError(
            "Argument validation error: 'offset' must be >= 0.",
            kind=ToolErrorKind.VALIDATION,
        )
    if level.lower() not in LOG_LEVELS:
        raise ToolError(
            f"Argument validation error: 'level' must be one of {LOG_LEVELS}.",
            kind=ToolErrorKind.VALIDATION,
        )
    clamped_limit, note = clamp_limit(limit)
    try:
        result = WorkloadApiClient().list_workload_logs(
            wid,
            limit=clamped_limit,
            offset=offset,
            level=level.lower(),
            start_time=start_time,
            end_time=end_time,
            includes=includes,
            excludes=excludes,
            span_id=span_id,
            trace_id=trace_id,
        )
    except ClientError as exc:
        raise_tool_error_for_client_error(exc)

    data = result.get("data", []) or []
    return merge_pagination_metadata(
        {"logs": data, "count": len(data)},
        result,
        note,
        offset=offset,
        limit=clamped_limit,
    )
