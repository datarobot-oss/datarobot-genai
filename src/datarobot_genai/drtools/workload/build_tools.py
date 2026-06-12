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
from datarobot_genai.drtools.pagination import clamp_limit
from datarobot_genai.drtools.pagination import merge_pagination_metadata

# ------------------------------------------------------------------ #
# artifact_build_list                                                  #
# ------------------------------------------------------------------ #


@tool_metadata(
    tags={"artifact", "build", "workload", "datarobot", "list"},
    description=(
        "[Artifact build—list] List image builds for an artifact. Returns build "
        "status, timestamps, and build id. Supports pagination.\n\n"
        "Example: artifact_build_list(artifact_id='art-abc')"
    ),
)
async def artifact_build_list(
    *,
    artifact_id: Annotated[str, "Id of the artifact whose builds to list."],
    limit: Annotated[int, "Max builds to return (1–100). Default 100."] = 100,
    offset: Annotated[int, "Builds to skip for pagination. Default 0."] = 0,
) -> dict[str, Any]:
    aid = require_id(artifact_id, "artifact_id")
    if offset < 0:
        raise ToolError(
            "Argument validation error: 'offset' must be >= 0.",
            kind=ToolErrorKind.VALIDATION,
        )
    clamped_limit, note = clamp_limit(limit)
    try:
        result = WorkloadApiClient().list_artifact_builds(aid, limit=clamped_limit, offset=offset)
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


# ------------------------------------------------------------------ #
# artifact_build_trigger                                               #
# ------------------------------------------------------------------ #


@tool_metadata(
    tags={"artifact", "build", "workload", "datarobot", "trigger"},
    description=(
        "[Artifact build—trigger] Start an image build for a draft service artifact "
        "(codeRef). Locked artifacts are rejected. Returns the triggered build record.\n\n"
        "Example: artifact_build_trigger(artifact_id='art-abc')"
    ),
)
async def artifact_build_trigger(
    *,
    artifact_id: Annotated[str, "Id of the draft artifact to build."],
) -> dict[str, Any]:
    aid = require_id(artifact_id, "artifact_id")
    try:
        return WorkloadApiClient().trigger_artifact_build(aid)
    except ClientError as exc:
        raise_tool_error_for_client_error(exc)


# ------------------------------------------------------------------ #
# artifact_build_get                                                   #
# ------------------------------------------------------------------ #


@tool_metadata(
    tags={"artifact", "build", "workload", "datarobot", "get"},
    description=(
        "[Artifact build—get] Retrieve a specific build by id. Returns status, "
        "timestamps, and build metadata.\n\n"
        "Example: artifact_build_get(artifact_id='art-abc', build_id='bld-xyz')"
    ),
)
async def artifact_build_get(
    *,
    artifact_id: Annotated[str, "Id of the artifact."],
    build_id: Annotated[str, "Id of the build to retrieve."],
) -> dict[str, Any]:
    aid = require_id(artifact_id, "artifact_id")
    bid = require_id(build_id, "build_id")
    try:
        return WorkloadApiClient().get_artifact_build(aid, bid)
    except ClientError as exc:
        raise_tool_error_for_client_error(exc)


# ------------------------------------------------------------------ #
# artifact_build_logs                                                  #
# ------------------------------------------------------------------ #


@tool_metadata(
    tags={"artifact", "build", "workload", "datarobot", "logs"},
    description=(
        "[Artifact build—logs] Retrieve the raw build log output for a specific "
        "image build. Useful for debugging build failures.\n\n"
        "Example: artifact_build_logs(artifact_id='art-abc', build_id='bld-xyz')"
    ),
)
async def artifact_build_logs(
    *,
    artifact_id: Annotated[str, "Id of the artifact."],
    build_id: Annotated[str, "Id of the build whose logs to retrieve."],
) -> dict[str, Any]:
    aid = require_id(artifact_id, "artifact_id")
    bid = require_id(build_id, "build_id")
    try:
        logs = WorkloadApiClient().get_artifact_build_logs(aid, bid)
    except ClientError as exc:
        raise_tool_error_for_client_error(exc)
    return {"artifact_id": aid, "build_id": bid, "logs": logs}


# ------------------------------------------------------------------ #
# artifact_build_delete                                                #
# ------------------------------------------------------------------ #


@tool_metadata(
    tags={"artifact", "build", "workload", "datarobot", "delete"},
    description=(
        "[Artifact build—delete] Cancel or delete an image build for a draft artifact. "
        "Locked artifacts cannot have their builds deleted.\n\n"
        "Example: artifact_build_delete(artifact_id='art-abc', build_id='bld-xyz')"
    ),
)
async def artifact_build_delete(
    *,
    artifact_id: Annotated[str, "Id of the draft artifact."],
    build_id: Annotated[str, "Id of the build to cancel or delete."],
) -> dict[str, Any]:
    aid = require_id(artifact_id, "artifact_id")
    bid = require_id(build_id, "build_id")
    try:
        WorkloadApiClient().delete_artifact_build(aid, bid)
    except ClientError as exc:
        raise_tool_error_for_client_error(exc)
    return {"deleted": True, "artifact_id": aid, "build_id": bid}
