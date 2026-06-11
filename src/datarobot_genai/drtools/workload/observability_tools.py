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

from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.exceptions import ToolErrorKind
from datarobot_genai.drtools.core import tool_metadata
from datarobot_genai.drtools.core.clients.datarobot_workload import WorkloadApiClient
from datarobot_genai.drtools.pagination import clamp_limit
from datarobot_genai.drtools.pagination import merge_pagination_metadata
from datarobot_genai.drtools.predictive.client_exceptions import raise_tool_error_for_client_error


def _require_workload_id(workload_id: str) -> str:
    if not workload_id or not workload_id.strip():
        raise ToolError(
            "Argument validation error: 'workload_id' cannot be empty.",
            kind=ToolErrorKind.VALIDATION,
        )
    return workload_id.strip()


# ------------------------------------------------------------------ #
# Settings                                                             #
# ------------------------------------------------------------------ #


@tool_metadata(
    tags={"workload", "datarobot", "settings", "get"},
    description=(
        "[Workload—settings get] Retrieve the current runtime settings for a "
        "workload (containerGroups, replicaCount, resourceBundles). Use before "
        "calling workload_settings_update to inspect the current configuration.\n\n"
        "Example: workload_settings_get(workload_id='wkld-abc')"
    ),
)
async def workload_settings_get(
    *,
    workload_id: Annotated[str, "Id of the workload."],
) -> dict[str, Any]:
    wid = _require_workload_id(workload_id)
    try:
        return WorkloadApiClient().get_workload_settings(wid)
    except ClientError as exc:
        raise_tool_error_for_client_error(exc)


@tool_metadata(
    tags={"workload", "datarobot", "settings", "update"},
    description=(
        "[Workload—settings update] Update a workload's runtime settings by "
        "triggering a rolling replacement with the current artifact. Pass a "
        "runtime dict matching WorkloadRuntime: "
        '{"containerGroups": [{"name": "default", "replicaCount": 2, '
        '"resourceBundles": ["gpu.l4.small"]}]}. '
        "Returns the in-progress Replacement object (202).\n\n"
        "Example: workload_settings_update(workload_id='wkld-abc', "
        'runtime={"containerGroups": [{"name": "default", "replicaCount": 2}]})'
    ),
)
async def workload_settings_update(
    *,
    workload_id: Annotated[str, "Id of the workload."],
    runtime: Annotated[
        dict[str, Any],
        "WorkloadRuntime dict. Must contain 'containerGroups' list. "
        "Each group may have name, replicaCount, resourceBundles.",
    ],
) -> dict[str, Any]:
    wid = _require_workload_id(workload_id)
    if not runtime or not isinstance(runtime, dict):
        raise ToolError(
            "Argument validation error: 'runtime' must be a non-empty object.",
            kind=ToolErrorKind.VALIDATION,
        )
    if "containerGroups" not in runtime:
        raise ToolError(
            "Argument validation error: 'runtime' must contain 'containerGroups'.",
            kind=ToolErrorKind.VALIDATION,
        )
    try:
        return WorkloadApiClient().update_workload_settings(wid, runtime)
    except ClientError as exc:
        raise_tool_error_for_client_error(exc)


# ------------------------------------------------------------------ #
# Stats                                                                #
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
    wid = _require_workload_id(workload_id)
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
# History                                                              #
# ------------------------------------------------------------------ #


@tool_metadata(
    tags={"workload", "datarobot", "history"},
    description=(
        "[Workload—history] List the artifact deployment history for a workload: "
        "each entry shows which artifact version was deployed, when, and by whom.\n\n"
        "Example: workload_history(workload_id='wkld-abc')"
    ),
)
async def workload_history(
    *,
    workload_id: Annotated[str, "Id of the workload."],
    limit: Annotated[int, "Max records to return (1–100). Default 20."] = 20,
    offset: Annotated[int, "Records to skip for pagination. Default 0."] = 0,
) -> dict[str, Any]:
    wid = _require_workload_id(workload_id)
    if offset < 0:
        raise ToolError(
            "Argument validation error: 'offset' must be >= 0.",
            kind=ToolErrorKind.VALIDATION,
        )
    clamped_limit, note = clamp_limit(limit)
    try:
        result = WorkloadApiClient().list_workload_history(wid, limit=clamped_limit, offset=offset)
    except ClientError as exc:
        raise_tool_error_for_client_error(exc)

    data = result.get("data", []) or []
    return merge_pagination_metadata(
        {"history": data, "count": len(data)},
        result,
        note,
        offset=offset,
        limit=clamped_limit,
    )


# ------------------------------------------------------------------ #
# Events                                                               #
# ------------------------------------------------------------------ #


@tool_metadata(
    tags={"workload", "datarobot", "events"},
    description=(
        "[Workload—events] List status-change and error events for a workload. "
        "Useful for diagnosing why a workload failed to start or is in an "
        "unexpected state.\n\n"
        "Example: workload_events(workload_id='wkld-abc')"
    ),
)
async def workload_events(
    *,
    workload_id: Annotated[str, "Id of the workload."],
    limit: Annotated[int, "Max events to return (1–100). Default 20."] = 20,
    offset: Annotated[int, "Events to skip for pagination. Default 0."] = 0,
) -> dict[str, Any]:
    wid = _require_workload_id(workload_id)
    if offset < 0:
        raise ToolError(
            "Argument validation error: 'offset' must be >= 0.",
            kind=ToolErrorKind.VALIDATION,
        )
    clamped_limit, note = clamp_limit(limit)
    try:
        result = WorkloadApiClient().list_workload_events(wid, limit=clamped_limit, offset=offset)
    except ClientError as exc:
        raise_tool_error_for_client_error(exc)

    data = result.get("data", []) or []
    return merge_pagination_metadata(
        {"events": data, "count": len(data)},
        result,
        note,
        offset=offset,
        limit=clamped_limit,
    )


# ------------------------------------------------------------------ #
# Promote                                                              #
# ------------------------------------------------------------------ #


@tool_metadata(
    tags={"workload", "datarobot", "promote"},
    description=(
        "[Workload—promote] Lock the draft artifact currently running on a "
        "workload. The workload keeps running the same artifact; it is promoted "
        "from draft to locked and assigned a version number. Workload stats are "
        "reset and the event is recorded in history.\n\n"
        "Example: workload_promote(workload_id='wkld-abc')"
    ),
)
async def workload_promote(
    *,
    workload_id: Annotated[str, "Id of the workload whose artifact to promote."],
) -> dict[str, Any]:
    wid = _require_workload_id(workload_id)
    try:
        return WorkloadApiClient().promote_workload_artifact(wid)
    except ClientError as exc:
        raise_tool_error_for_client_error(exc)


# ------------------------------------------------------------------ #
# Related                                                              #
# ------------------------------------------------------------------ #


@tool_metadata(
    tags={"workload", "datarobot", "related"},
    description=(
        "[Workload—related] List entities related to a workload, such as linked "
        "artifacts, repositories, and other connected resources.\n\n"
        "Example: workload_related(workload_id='wkld-abc')"
    ),
)
async def workload_related(
    *,
    workload_id: Annotated[str, "Id of the workload."],
) -> dict[str, Any]:
    wid = _require_workload_id(workload_id)
    try:
        return WorkloadApiClient().get_workload_related(wid)
    except ClientError as exc:
        raise_tool_error_for_client_error(exc)
