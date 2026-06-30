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

"""Workload observability tools: stats, logs, activity, and proton inspection."""

from typing import Annotated
from typing import Any
from typing import Literal

from datarobot.errors import ClientError

from datarobot_genai.drmcputils.client_exceptions import raise_tool_error_for_client_error
from datarobot_genai.drmcputils.constants import LOG_LEVELS
from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.exceptions import ToolErrorKind
from datarobot_genai.drtools.core import tool_metadata
from datarobot_genai.drtools.core.clients.datarobot_workload import WorkloadApiClient
from datarobot_genai.drtools.core.utils import require_id
from datarobot_genai.drtools.pagination import clamp_limit
from datarobot_genai.drtools.pagination import merge_pagination_metadata

# ------------------------------------------------------------------ #
# workload_stats                                                       #
# ------------------------------------------------------------------ #


@tool_metadata(
    tags={"workload", "datarobot", "stats"},
    description=(
        "[Workload—stats] Get aggregated performance statistics for a workload: "
        "request count, error rate, response time quantile, slow requests. "
        "Optionally scope to a proton or time window.\n\n"
        "Example: workload_stats(workload_id='wkld-abc')\n"
        "Example: workload_stats(workload_id='wkld-abc', response_time_quantile=0.95)"
    ),
    display_name="Workload — Stats",
    description_ui=(
        "Get aggregated performance stats for a workload: request count, error "
        "rate, response time, and slow requests."
    ),
)
async def workload_stats(
    *,
    workload_id: Annotated[str, "Id of the workload."],
    proton_id: Annotated[
        str | None, "Scope stats to a specific proton id. Defaults to current proton."
    ] = None,
    start_time: Annotated[
        str | None, "ISO-8601 start of the stats window (e.g. '2026-01-01T00:00:00Z')."
    ] = None,
    end_time: Annotated[str | None, "ISO-8601 end of the stats window."] = None,
    response_time_quantile: Annotated[
        float, "Quantile for response time (0–1). Default 0.5 (median)."
    ] = 0.5,
    slow_requests_threshold: Annotated[
        int, "Threshold in ms above which a request is 'slow'. Default 2000."
    ] = 2000,
) -> dict[str, Any]:
    wid = require_id(workload_id, "workload_id")
    if not (0.0 <= response_time_quantile <= 1.0):
        raise ToolError(
            "Argument validation error: 'response_time_quantile' must be between 0 and 1.",
            kind=ToolErrorKind.VALIDATION,
        )
    if slow_requests_threshold < 0:
        raise ToolError(
            "Argument validation error: 'slow_requests_threshold' must be >= 0.",
            kind=ToolErrorKind.VALIDATION,
        )
    try:
        return WorkloadApiClient().get_workload_stats(
            wid,
            proton_id=proton_id,
            start_time=start_time,
            end_time=end_time,
            response_time_quantile=response_time_quantile,
            slow_requests_threshold=slow_requests_threshold,
        )
    except ClientError as exc:
        raise_tool_error_for_client_error(exc)


# ------------------------------------------------------------------ #
# workload_logs                                                        #
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
    display_name="Workload — Logs",
    description_ui=(
        "Retrieve OTel log lines for a workload, filtered by level, time window, "
        "body text, and span or trace id."
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


# ------------------------------------------------------------------ #
# workload_activity  (history / events / related)                     #
# ------------------------------------------------------------------ #

_ACTIVITY_RESULT_KEY: dict[str, str] = {
    "history": "history",
    "events": "events",
    "related": "related",
}


@tool_metadata(
    tags={"workload", "datarobot", "history", "events", "related"},
    description=(
        "[Workload—activity] Inspect a workload's activity. kind is one of:\n"
        "  'history' — artifact deployment history (which version was deployed, "
        "when, by whom).\n"
        "  'events'  — status-change and error events; useful for diagnosing why a "
        "workload failed to start or is in an unexpected state.\n"
        "  'related' — entities related to the workload (linked artifacts, "
        "repositories, and other connected resources).\n\n"
        "history and events are paginated (limit/offset); related ignores pagination.\n\n"
        "Example: workload_activity(workload_id='wkld-abc', kind='events')"
    ),
    display_name="Workload — Activity",
    description_ui=(
        "Inspect a workload's deployment history, status-change and error "
        "events, or related entities."
    ),
)
async def workload_activity(
    *,
    workload_id: Annotated[str, "Id of the workload."],
    kind: Annotated[
        Literal["history", "events", "related"],
        "Activity to fetch: 'history' | 'events' | 'related'.",
    ],
    limit: Annotated[int, "Max records to return (1–100). Default 20. Ignored for 'related'."] = 20,
    offset: Annotated[int, "Records to skip for pagination. Default 0. Ignored for 'related'."] = 0,
) -> dict[str, Any]:
    wid = require_id(workload_id, "workload_id")
    client = WorkloadApiClient()

    if kind == "related":
        try:
            return client.get_workload_related(wid)
        except ClientError as exc:
            raise_tool_error_for_client_error(exc)

    if offset < 0:
        raise ToolError(
            "Argument validation error: 'offset' must be >= 0.",
            kind=ToolErrorKind.VALIDATION,
        )
    clamped_limit, note = clamp_limit(limit)
    try:
        if kind == "history":
            result = client.list_workload_history(wid, limit=clamped_limit, offset=offset)
        else:  # events
            result = client.list_workload_events(wid, limit=clamped_limit, offset=offset)
    except ClientError as exc:
        raise_tool_error_for_client_error(exc)

    data = result.get("data", []) or []
    return merge_pagination_metadata(
        {_ACTIVITY_RESULT_KEY[kind]: data, "count": len(data)},
        result,
        note,
        offset=offset,
        limit=clamped_limit,
    )


# ------------------------------------------------------------------ #
# proton_get  (list / single / status details)                       #
# ------------------------------------------------------------------ #


@tool_metadata(
    tags={"workload", "proton", "datarobot", "get", "list", "status", "debug"},
    description=(
        "[Workload—proton get] Inspect proton (deployed pod) instances for a "
        "workload.\n"
        "  - Omit proton_id to LIST all protons (id, status, replica count, bundle); "
        "paginated.\n"
        "  - Set proton_id to GET a single proton.\n"
        "  - Set proton_id and include_status_details=True to also attach per-replica "
        "pod status (container readiness, restart count, pod phase, conditions) under "
        "'status_details' — use this to diagnose CrashLoopBackOff, OOMKilled, or "
        "image-pull errors. status_details is null when no status has arrived yet.\n\n"
        "Example (list):    proton_get(workload_id='wkld-abc')\n"
        "Example (details): proton_get(workload_id='wkld-abc', proton_id='ptn-xyz', "
        "include_status_details=True)"
    ),
    display_name="Workload — Get Proton",
    description_ui=(
        "Inspect proton pod instances for a workload, optionally with "
        "per-replica pod status details."
    ),
)
async def proton_get(
    *,
    workload_id: Annotated[str, "Id of the workload."],
    proton_id: Annotated[
        str | None, "Id of the proton. Omit to list all protons for the workload."
    ] = None,
    include_status_details: Annotated[
        bool,
        "When True and proton_id is set, attach per-replica pod status_details.",
    ] = False,
    limit: Annotated[int, "Max protons to return when listing (1–100). Default 20."] = 20,
    offset: Annotated[int, "Protons to skip for pagination when listing. Default 0."] = 0,
) -> dict[str, Any]:
    wid = require_id(workload_id, "workload_id")
    client = WorkloadApiClient()

    if proton_id is None:
        if offset < 0:
            raise ToolError(
                "Argument validation error: 'offset' must be >= 0.",
                kind=ToolErrorKind.VALIDATION,
            )
        clamped_limit, note = clamp_limit(limit)
        try:
            result = client.list_protons(wid, limit=clamped_limit, offset=offset)
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

    pid = require_id(proton_id, "proton_id")
    try:
        proton = client.get_proton(wid, pid)
        if include_status_details:
            details = client.get_proton_status_details(wid, pid)
            proton = dict(proton)
            proton["status_details"] = details
    except ClientError as exc:
        raise_tool_error_for_client_error(exc)
    return proton
