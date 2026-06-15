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

from datarobot_genai.drmcputils.client_exceptions import raise_tool_error_for_client_error
from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.exceptions import ToolErrorKind
from datarobot_genai.drtools.core import tool_metadata
from datarobot_genai.drtools.core.clients.datarobot_workload import WorkloadApiClient
from datarobot_genai.drtools.core.utils import require_id

_REPLACEMENT_STRATEGIES = ("rolling",)


# ------------------------------------------------------------------ #
# workload_replacement_get                                             #
# ------------------------------------------------------------------ #


@tool_metadata(
    tags={"workload", "replacement", "datarobot", "get"},
    description=(
        "[Workload replacement—get] Retrieve the current rolling-update replacement "
        "status for a workload. Returns candidate artifact id, candidate proton ids, "
        "strategy, config, and timestamps.\n\n"
        "Example: workload_replacement_get(workload_id='wkld-abc')"
    ),
)
async def workload_replacement_get(
    *,
    workload_id: Annotated[str, "Id of the workload whose replacement to retrieve."],
) -> dict[str, Any]:
    wid = require_id(workload_id, "workload_id")
    try:
        return WorkloadApiClient().get_workload_replacement(wid)
    except ClientError as exc:
        raise_tool_error_for_client_error(exc)


# ------------------------------------------------------------------ #
# workload_replacement_create                                          #
# ------------------------------------------------------------------ #


@tool_metadata(
    tags={"workload", "replacement", "datarobot", "create"},
    description=(
        "[Workload replacement—create] Start a rolling replacement for a workload. "
        "Deploys a new artifact alongside the running version; traffic switches once "
        "the candidate is ready. Supports optional warmup/retention config and runtime "
        "override.\n\n"
        "Example: workload_replacement_create(workload_id='wkld-abc', "
        "artifact_id='art-xyz')\n"
        "Example: workload_replacement_create(workload_id='wkld-abc', "
        "artifact_id='art-xyz', warmup_duration_minutes=5, keep_old_version_minutes=2)"
    ),
)
async def workload_replacement_create(
    *,
    workload_id: Annotated[str, "Id of the workload to update."],
    artifact_id: Annotated[str, "Id of the artifact to deploy as the replacement."],
    strategy: Annotated[
        str, "Replacement strategy. Currently only 'rolling' is supported."
    ] = "rolling",
    warmup_duration_minutes: Annotated[
        int, "Minutes to keep the new version in warmup before switching traffic. Default 0."
    ] = 0,
    keep_old_version_minutes: Annotated[
        int, "Minutes to retain the old version after traffic switch. Default 0."
    ] = 0,
    runtime: Annotated[
        dict[str, Any] | None,
        "Optional runtime override for the replacement. If omitted the current runtime is reused.",
    ] = None,
) -> dict[str, Any]:
    wid = require_id(workload_id, "workload_id")
    if not artifact_id or not artifact_id.strip():
        raise ToolError(
            "Argument validation error: 'artifact_id' cannot be empty.",
            kind=ToolErrorKind.VALIDATION,
        )
    if strategy not in _REPLACEMENT_STRATEGIES:
        raise ToolError(
            f"Argument validation error: 'strategy' must be one of {_REPLACEMENT_STRATEGIES}.",
            kind=ToolErrorKind.VALIDATION,
        )
    if warmup_duration_minutes < 0:
        raise ToolError(
            "Argument validation error: 'warmup_duration_minutes' must be >= 0.",
            kind=ToolErrorKind.VALIDATION,
        )
    if keep_old_version_minutes < 0:
        raise ToolError(
            "Argument validation error: 'keep_old_version_minutes' must be >= 0.",
            kind=ToolErrorKind.VALIDATION,
        )
    payload: dict[str, Any] = {
        "artifactId": artifact_id.strip(),
        "strategy": strategy,
        "config": {
            "warmupDurationMinutes": warmup_duration_minutes,
            "keepOldVersionMinutes": keep_old_version_minutes,
        },
    }
    if runtime is not None:
        payload["runtime"] = runtime
    try:
        return WorkloadApiClient().create_workload_replacement(wid, payload)
    except ClientError as exc:
        raise_tool_error_for_client_error(exc)


# ------------------------------------------------------------------ #
# workload_replacement_delete                                          #
# ------------------------------------------------------------------ #


@tool_metadata(
    tags={"workload", "replacement", "datarobot", "delete"},
    description=(
        "[Workload replacement—delete] Cancel an in-progress rolling replacement. "
        "Stops the candidate deployment and reverts traffic to the original version.\n\n"
        "Example: workload_replacement_delete(workload_id='wkld-abc')"
    ),
)
async def workload_replacement_delete(
    *,
    workload_id: Annotated[str, "Id of the workload whose replacement to cancel."],
) -> dict[str, Any]:
    wid = require_id(workload_id, "workload_id")
    try:
        return WorkloadApiClient().delete_workload_replacement(wid)
    except ClientError as exc:
        raise_tool_error_for_client_error(exc)
