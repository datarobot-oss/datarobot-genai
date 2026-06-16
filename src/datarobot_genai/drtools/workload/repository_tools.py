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
from datarobot_genai.drmcputils.constants import ARTIFACT_TYPES
from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.exceptions import ToolErrorKind
from datarobot_genai.drtools.core import tool_metadata
from datarobot_genai.drtools.core.clients.datarobot_workload import WorkloadApiClient
from datarobot_genai.drtools.core.utils import require_id
from datarobot_genai.drtools.pagination import clamp_limit
from datarobot_genai.drtools.pagination import merge_pagination_metadata

# ------------------------------------------------------------------ #
# artifact_repository_list                                             #
# ------------------------------------------------------------------ #


@tool_metadata(
    tags={"artifact", "repository", "workload", "datarobot", "list"},
    description=(
        "[Artifact repository—list] Discover artifact repositories. Returns id, name, "
        "type, and timestamps. Supports pagination and optional search/type filters.\n\n"
        "Example: artifact_repository_list()\n"
        "Example: artifact_repository_list(search='my-registry', artifact_type='service')"
    ),
)
async def artifact_repository_list(
    *,
    search: Annotated[
        str | None, "Case-insensitive filter on name, description, and partial id."
    ] = None,
    artifact_type: Annotated[str | None, "Filter by type: 'service' or 'nim'."] = None,
    limit: Annotated[int, "Max repositories to return (1–100). Default 100."] = 100,
    offset: Annotated[int, "Repositories to skip for pagination. Default 0."] = 0,
) -> dict[str, Any]:
    if offset < 0:
        raise ToolError(
            "Argument validation error: 'offset' must be >= 0.",
            kind=ToolErrorKind.VALIDATION,
        )
    if artifact_type and artifact_type.lower() not in ARTIFACT_TYPES:
        raise ToolError(
            f"Argument validation error: 'artifact_type' must be one of {ARTIFACT_TYPES}.",
            kind=ToolErrorKind.VALIDATION,
        )
    clamped_limit, note = clamp_limit(limit)
    try:
        result = WorkloadApiClient().list_artifact_repositories(
            limit=clamped_limit,
            offset=offset,
            search=search,
            artifact_type=artifact_type.lower() if artifact_type else None,
        )
    except ClientError as exc:
        raise_tool_error_for_client_error(exc)

    data = result.get("data", []) or []
    return merge_pagination_metadata(
        {"repositories": data, "count": len(data)},
        result,
        note,
        offset=offset,
        limit=clamped_limit,
    )


# ------------------------------------------------------------------ #
# artifact_repository_get                                              #
# ------------------------------------------------------------------ #


@tool_metadata(
    tags={"artifact", "repository", "workload", "datarobot", "get"},
    description=(
        "[Artifact repository—get] Retrieve a single artifact repository by id. "
        "Returns full metadata including name, type, spec, and timestamps.\n\n"
        "Example: artifact_repository_get(repository_id='repo-abc123')"
    ),
)
async def artifact_repository_get(
    *,
    repository_id: Annotated[str, "Id of the artifact repository to retrieve."],
) -> dict[str, Any]:
    rid = require_id(repository_id, "repository_id")
    try:
        return WorkloadApiClient().get_artifact_repository(rid)
    except ClientError as exc:
        raise_tool_error_for_client_error(exc)


# ------------------------------------------------------------------ #
# artifact_repository_delete                                           #
# ------------------------------------------------------------------ #


@tool_metadata(
    tags={"artifact", "repository", "workload", "datarobot", "delete"},
    description=(
        "[Artifact repository—delete] Permanently delete an artifact repository. "
        "Artifacts within the repository are cascade-deleted unless they are locked "
        "or currently in use by a workload — those will cause a 409 conflict.\n\n"
        "Example: artifact_repository_delete(repository_id='repo-abc123')"
    ),
)
async def artifact_repository_delete(
    *,
    repository_id: Annotated[str, "Id of the artifact repository to delete."],
) -> dict[str, Any]:
    rid = require_id(repository_id, "repository_id")
    try:
        WorkloadApiClient().delete_artifact_repository(rid)
    except ClientError as exc:
        raise_tool_error_for_client_error(exc)
    return {"deleted": True, "repository_id": rid}
