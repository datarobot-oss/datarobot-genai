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

"""Artifact image-build tools: read (list/get/logs) and actions (trigger/delete)."""

from typing import Annotated
from typing import Any
from typing import Literal

from datarobot.errors import ClientError

from datarobot_genai.drmcputils.client_exceptions import raise_tool_error_for_client_error
from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.exceptions import ToolErrorKind
from datarobot_genai.drtools.core import tool_metadata
from datarobot_genai.drtools.core.clients.datarobot_workload import WorkloadApiClient
from datarobot_genai.drtools.core.utils import require_id
from datarobot_genai.drtools.pagination import clamp_limit
from datarobot_genai.drtools.pagination import merge_pagination_metadata

# ------------------------------------------------------------------ #
# artifact_get_build  (list / single / logs)                          #
# ------------------------------------------------------------------ #


@tool_metadata(
    tags={"artifact", "build", "workload", "datarobot", "get", "list", "logs"},
    description=(
        "[Artifact build—get] Read image builds for an artifact.\n"
        "  - Omit build_id to LIST builds (status, timestamps, build ID); paginated.\n"
        "  - Set build_id to GET a single build's status and metadata.\n"
        "  - Set build_id and include_logs=True to also attach the raw build log "
        "output under 'logs' — useful for debugging build failures.\n\n"
        "Example (list): artifact_get_build(artifact_id='art-abc')\n"
        "Example (logs): artifact_get_build(artifact_id='art-abc', build_id='bld-xyz', "
        "include_logs=True)"
    ),
    display_name="Artifact — Get build",
    description_ui=(
        "Read image builds for an artifact: list them, or get a single build's "
        "status with optional log output."
    ),
)
async def artifact_get_build(
    *,
    artifact_id: Annotated[str, "Id of the artifact."],
    build_id: Annotated[
        str | None, "Id of the build. Omit to list all builds for the artifact."
    ] = None,
    include_logs: Annotated[
        bool, "When True and build_id is set, attach the raw build log output."
    ] = False,
    limit: Annotated[int, "Max builds to return when listing (1–100). Default 100."] = 100,
    offset: Annotated[int, "Builds to skip for pagination when listing. Default 0."] = 0,
) -> dict[str, Any]:
    aid = require_id(artifact_id, "artifact_id")
    client = WorkloadApiClient()

    if build_id is None:
        if offset < 0:
            raise ToolError(
                "Argument validation error: 'offset' must be >= 0.",
                kind=ToolErrorKind.VALIDATION,
            )
        clamped_limit, note = clamp_limit(limit)
        try:
            result = client.list_artifact_builds(aid, limit=clamped_limit, offset=offset)
        except ClientError as exc:
            raise_tool_error_for_client_error(exc)

        data = result.get("data", []) or []
        return merge_pagination_metadata(
            {"builds": data, "count": len(data)},
            result,
            note,
            offset=offset,
            limit=clamped_limit,
        )

    bid = require_id(build_id, "build_id")
    try:
        build = client.get_artifact_build(aid, bid)
        if include_logs:
            logs = client.get_artifact_build_logs(aid, bid)
            build = dict(build)
            build["logs"] = logs
    except ClientError as exc:
        raise_tool_error_for_client_error(exc)
    return build


# ------------------------------------------------------------------ #
# artifact_build_run_action  (trigger / delete)                           #
# ------------------------------------------------------------------ #


@tool_metadata(
    tags={"artifact", "build", "workload", "datarobot", "trigger", "delete"},
    description=(
        "[Artifact build—action] Run an action on artifact image builds. action is "
        "one of:\n"
        "  'trigger' — start an image build for a draft service artifact (codeRef). "
        "Locked artifacts are rejected. Returns the triggered build record.\n"
        "  'delete'  — cancel or delete a build (requires build_id). Locked artifacts "
        "cannot have their builds deleted.\n\n"
        "Example (trigger): artifact_build_run_action(artifact_id='art-abc', action='trigger')\n"
        "Example (delete):  artifact_build_run_action(artifact_id='art-abc', action='delete', "
        "build_id='bld-xyz')"
    ),
    display_name="Artifact Build — Run action",
    description_ui=(
        "Run an action on artifact image builds: trigger a build for a draft "
        "service artifact, or delete a build."
    ),
)
async def artifact_build_run_action(
    *,
    artifact_id: Annotated[str, "Id of the artifact."],
    action: Annotated[Literal["trigger", "delete"], "Action: 'trigger' | 'delete'."],
    build_id: Annotated[
        str | None, "Id of the build to cancel or delete. Required when action='delete'."
    ] = None,
) -> dict[str, Any]:
    aid = require_id(artifact_id, "artifact_id")
    client = WorkloadApiClient()

    if action == "trigger":
        try:
            return client.trigger_artifact_build(aid)
        except ClientError as exc:
            raise_tool_error_for_client_error(exc)

    if action == "delete":
        bid = require_id(build_id or "", "build_id")
        try:
            client.delete_artifact_build(aid, bid)
        except ClientError as exc:
            raise_tool_error_for_client_error(exc)
        return {"deleted": True, "artifact_id": aid, "build_id": bid}

    # pragma: no cover - guarded by Literal typing
    raise ToolError(
        f"Argument validation error: unknown action {action!r}.",
        kind=ToolErrorKind.VALIDATION,
    )
