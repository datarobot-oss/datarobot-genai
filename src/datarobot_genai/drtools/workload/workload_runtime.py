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

"""Workload runtime tools: settings (scaling) and rolling replacements."""

from typing import Annotated
from typing import Any

from datarobot.errors import ClientError

from datarobot_genai.drmcputils.client_exceptions import raise_tool_error_for_client_error
from datarobot_genai.drmcputils.constants import REPLACEMENT_STRATEGIES
from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.exceptions import ToolErrorKind
from datarobot_genai.drtools.core import tool_metadata
from datarobot_genai.drtools.core.clients.datarobot_workload import WorkloadApiClient
from datarobot_genai.drtools.core.utils import require_id

# ------------------------------------------------------------------ #
# workload_settings  (get when runtime omitted, update when provided) #
# ------------------------------------------------------------------ #


@tool_metadata(
    tags={"workload", "datarobot", "settings", "get", "update"},
    description=(
        "[Workload—settings] Read or update a workload's runtime settings "
        "(containerGroups, replicaCount, resourceBundles). "
        "Omit 'runtime' to GET the current settings. Pass a 'runtime' dict to "
        "UPDATE: this triggers a rolling replacement with the current artifact and "
        "returns the in-progress Replacement object (202). Read first to inspect the "
        "current configuration before updating.\n\n"
        "Example (get):    workload_settings(workload_id='wkld-abc')\n"
        "Example (update): workload_settings(workload_id='wkld-abc', "
        'runtime={"containerGroups": [{"name": "default", "replicaCount": 2}]})'
    ),
    display_name="Workload — Settings",
    description_ui=(
        "Read or update a workload's runtime settings such as container groups, "
        "replica count, and resource bundles."
    ),
)
async def workload_settings(
    *,
    workload_id: Annotated[str, "Id of the workload."],
    runtime: Annotated[
        dict[str, Any] | None,
        "Omit to read current settings. To update, pass a WorkloadRuntime dict that "
        "must contain a 'containerGroups' list; each group may have name, "
        "replicaCount, resourceBundles.",
    ] = None,
) -> dict[str, Any]:
    wid = require_id(workload_id, "workload_id")
    client = WorkloadApiClient()

    if runtime is None:
        try:
            return client.get_workload_settings(wid)
        except ClientError as exc:
            raise_tool_error_for_client_error(exc)

    if not isinstance(runtime, dict) or not runtime:
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
        return client.update_workload_settings(wid, runtime)
    except ClientError as exc:
        raise_tool_error_for_client_error(exc)


# ------------------------------------------------------------------ #
# workload_artifact_replace  (get / create / cancel)                       #
# ------------------------------------------------------------------ #


@tool_metadata(
    tags={"workload", "replacement", "datarobot", "get", "create", "delete"},
    description=(
        "[Workload—replacement] Manage a rolling replacement (zero-downtime artifact "
        "swap) for a workload. Modes:\n"
        "  - read:   omit artifact_id and leave cancel=False — returns the current "
        "replacement status (candidate artifact, candidate proton IDs, strategy, "
        "config, timestamps).\n"
        "  - create: set artifact_id — deploys the new artifact alongside the running "
        "version; traffic switches once the candidate is ready. Supports optional "
        "warmup/retention config and runtime override.\n"
        "  - cancel: set cancel=True — stops the candidate deployment and reverts "
        "traffic to the original version.\n\n"
        "Example (read):   workload_artifact_replace(workload_id='wkld-abc')\n"
        "Example (create): workload_artifact_replace("
        "workload_id='wkld-abc', artifact_id='art-xyz')\n"
        "Example (cancel): workload_artifact_replace(workload_id='wkld-abc', cancel=True)"
    ),
    display_name="Workload — Replace artifact",
    description_ui=(
        "Read, create, or cancel a zero-downtime rolling replacement that swaps "
        "a workload's artifact."
    ),
)
async def workload_artifact_replace(
    *,
    workload_id: Annotated[str, "Id of the workload."],
    artifact_id: Annotated[
        str | None,
        "Id of the artifact to deploy as the replacement. Set to create a replacement.",
    ] = None,
    cancel: Annotated[
        bool, "Set True to cancel an in-progress replacement. Default False."
    ] = False,
    strategy: Annotated[
        str, "Replacement strategy (create only). Currently only 'rolling' is supported."
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
    client = WorkloadApiClient()

    if cancel and artifact_id and artifact_id.strip():
        raise ToolError(
            "Argument validation error: pass either artifact_id (create) or "
            "cancel=True (cancel), not both.",
            kind=ToolErrorKind.VALIDATION,
        )

    if cancel:
        try:
            return client.delete_workload_replacement(wid)
        except ClientError as exc:
            raise_tool_error_for_client_error(exc)

    if not artifact_id or not artifact_id.strip():
        try:
            return client.get_workload_replacement(wid)
        except ClientError as exc:
            raise_tool_error_for_client_error(exc)

    if strategy not in REPLACEMENT_STRATEGIES:
        raise ToolError(
            f"Argument validation error: 'strategy' must be one of {REPLACEMENT_STRATEGIES}.",
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
        if not isinstance(runtime, dict) or not runtime:
            raise ToolError(
                "Argument validation error: 'runtime' must be a non-empty object.",
                kind=ToolErrorKind.VALIDATION,
            )
        if "containerGroups" not in runtime:
            raise ToolError(
                "Argument validation error: 'runtime' must contain 'containerGroups'.",
                kind=ToolErrorKind.VALIDATION,
            )
        payload["runtime"] = runtime
    try:
        return client.create_workload_replacement(wid, payload)
    except ClientError as exc:
        raise_tool_error_for_client_error(exc)
