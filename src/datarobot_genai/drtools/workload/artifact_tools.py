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

_ARTIFACT_STATUSES = ("draft", "locked")
_ARTIFACT_TYPES = ("service", "nim")


# ------------------------------------------------------------------ #
# artifact_list                                                        #
# ------------------------------------------------------------------ #


@tool_metadata(
    tags={"artifact", "workload", "datarobot", "list"},
    description=(
        "[Artifact—list] Discover artifacts: returns id, name, status (draft/locked), "
        "type, spec, and timestamps. Supports pagination and optional filters for "
        "status, type, repository, and search text. "
        "Not for a single known artifact id (artifact_get).\n\n"
        "Example: artifact_list()\n"
        "Example: artifact_list(search='my-service', status='draft')"
    ),
)
async def artifact_list(
    *,
    search: Annotated[
        str | None, "Case-insensitive filter on name, description, and partial id."
    ] = None,
    status: Annotated[str | None, "Filter by status: 'draft' or 'locked'."] = None,
    artifact_type: Annotated[str | None, "Filter by type: 'service' or 'nim'."] = None,
    repository_id: Annotated[str | None, "Filter by artifact repository id."] = None,
    limit: Annotated[int, "Max artifacts to return (1–100). Default 100."] = 100,
    offset: Annotated[int, "Artifacts to skip for pagination. Default 0."] = 0,
) -> dict[str, Any]:
    if offset < 0:
        raise ToolError(
            "Argument validation error: 'offset' must be >= 0.",
            kind=ToolErrorKind.VALIDATION,
        )
    if status and status.lower() not in _ARTIFACT_STATUSES:
        raise ToolError(
            f"Argument validation error: 'status' must be one of {_ARTIFACT_STATUSES}.",
            kind=ToolErrorKind.VALIDATION,
        )
    if artifact_type and artifact_type.lower() not in _ARTIFACT_TYPES:
        raise ToolError(
            f"Argument validation error: 'artifact_type' must be one of {_ARTIFACT_TYPES}.",
            kind=ToolErrorKind.VALIDATION,
        )
    clamped_limit, note = clamp_limit(limit)
    try:
        result = WorkloadApiClient().list_artifacts(
            limit=clamped_limit,
            offset=offset,
            search=search,
            status=status.lower() if status else None,
            artifact_type=artifact_type.lower() if artifact_type else None,
            repository_id=repository_id,
        )
    except ClientError as exc:
        raise_tool_error_for_client_error(exc)

    data = result.get("data", []) or []
    return merge_pagination_metadata(
        {"artifacts": data, "count": len(data)},
        result,
        note,
        offset=offset,
        limit=clamped_limit,
    )


# ------------------------------------------------------------------ #
# artifact_get                                                         #
# ------------------------------------------------------------------ #


@tool_metadata(
    tags={"artifact", "workload", "datarobot", "get"},
    description=(
        "[Artifact—get] Retrieve a single artifact by id: full spec, status, "
        "type, version, repository, and timestamps.\n\n"
        "Example: artifact_get(artifact_id='art-abc123')"
    ),
)
async def artifact_get(
    *,
    artifact_id: Annotated[str, "Id of the artifact to retrieve."],
) -> dict[str, Any]:
    aid = require_id(artifact_id, "artifact_id")
    try:
        return WorkloadApiClient().get_artifact(aid)
    except ClientError as exc:
        raise_tool_error_for_client_error(exc)


# ------------------------------------------------------------------ #
# artifact_create                                                      #
# ------------------------------------------------------------------ #


@tool_metadata(
    tags={"artifact", "workload", "datarobot", "create"},
    description=(
        "[Artifact—create] Create a new draft artifact from an InputArtifact payload. "
        "The payload must contain 'name' and 'spec'. 'spec' must have 'type' "
        "('service' or 'nim') and 'containerGroups'. Artifacts are always created "
        "as drafts.\n\n"
        "Example: artifact_create(payload={'name': 'my-svc', 'spec': {'type': 'service', "
        "'containerGroups': [{'containers': [{'name': 'main', 'imageUri': 'nginx:latest', "
        "'primary': true, 'port': 8080}]}]}})"
    ),
)
async def artifact_create(
    *,
    payload: Annotated[
        dict[str, Any],
        "InputArtifact payload. Must contain 'name' and 'spec' with 'type' and 'containerGroups'.",
    ],
) -> dict[str, Any]:
    if not payload or not isinstance(payload, dict):
        raise ToolError(
            "Argument validation error: 'payload' must be a non-empty object.",
            kind=ToolErrorKind.VALIDATION,
        )
    if not (payload.get("name") or "").strip():
        raise ToolError(
            "Argument validation error: payload must contain a non-empty 'name'.",
            kind=ToolErrorKind.VALIDATION,
        )
    if not payload.get("spec"):
        raise ToolError(
            "Argument validation error: payload must contain a 'spec' object.",
            kind=ToolErrorKind.VALIDATION,
        )
    try:
        return WorkloadApiClient().create_artifact(payload)
    except ClientError as exc:
        raise_tool_error_for_client_error(exc)


# ------------------------------------------------------------------ #
# artifact_update                                                      #
# ------------------------------------------------------------------ #


@tool_metadata(
    tags={"artifact", "workload", "datarobot", "update"},
    description=(
        "[Artifact—update] Partially update a draft artifact: name, description, "
        "or spec. Only the supplied fields are changed. At least one field is required. "
        "Locked artifacts cannot be updated.\n\n"
        "Example: artifact_update(artifact_id='art-abc', name='new-name')"
    ),
)
async def artifact_update(
    *,
    artifact_id: Annotated[str, "Id of the artifact to update."],
    name: Annotated[str | None, "New artifact name."] = None,
    description: Annotated[str | None, "New artifact description."] = None,
    spec: Annotated[
        dict[str, Any] | None,
        "New artifact spec (replaces current spec). Must include 'type' and 'containerGroups'.",
    ] = None,
) -> dict[str, Any]:
    aid = require_id(artifact_id, "artifact_id")
    patch: dict[str, Any] = {}
    if name is not None:
        stripped_name = name.strip()
        if not stripped_name:
            raise ToolError(
                "Argument validation error: 'name' cannot be empty.",
                kind=ToolErrorKind.VALIDATION,
            )
        patch["name"] = stripped_name
    if description is not None:
        patch["description"] = description.strip()
    if spec is not None:
        patch["spec"] = spec
    if not patch:
        raise ToolError(
            "Argument validation error: at least one of name, description, or spec must be set.",
            kind=ToolErrorKind.VALIDATION,
        )
    try:
        return WorkloadApiClient().patch_artifact(aid, patch)
    except ClientError as exc:
        raise_tool_error_for_client_error(exc)


# ------------------------------------------------------------------ #
# artifact_lock                                                        #
# ------------------------------------------------------------------ #


@tool_metadata(
    tags={"artifact", "workload", "datarobot", "lock"},
    description=(
        "[Artifact—lock] Lock a draft artifact so it can be versioned and used "
        "in production deployments. Once locked, an artifact cannot be updated "
        "or deleted. Use workload_promote to lock the artifact currently running "
        "on a workload instead.\n\n"
        "Example: artifact_lock(artifact_id='art-abc')"
    ),
)
async def artifact_lock(
    *,
    artifact_id: Annotated[str, "Id of the draft artifact to lock."],
) -> dict[str, Any]:
    aid = require_id(artifact_id, "artifact_id")
    try:
        return WorkloadApiClient().patch_artifact(aid, {"status": "locked"})
    except ClientError as exc:
        raise_tool_error_for_client_error(exc)


# ------------------------------------------------------------------ #
# artifact_clone                                                       #
# ------------------------------------------------------------------ #


@tool_metadata(
    tags={"artifact", "workload", "datarobot", "clone"},
    description=(
        "[Artifact—clone] Duplicate an existing artifact under a new name. "
        "The clone is created as a draft regardless of the original's status, "
        "allowing further modification before use.\n\n"
        "Example: artifact_clone(artifact_id='art-abc', name='my-service-v2')"
    ),
)
async def artifact_clone(
    *,
    artifact_id: Annotated[str, "Id of the artifact to clone."],
    name: Annotated[str, "Name for the cloned artifact."],
) -> dict[str, Any]:
    aid = require_id(artifact_id, "artifact_id")
    if not name or not name.strip():
        raise ToolError(
            "Argument validation error: 'name' cannot be empty.",
            kind=ToolErrorKind.VALIDATION,
        )
    try:
        return WorkloadApiClient().clone_artifact(aid, name.strip())
    except ClientError as exc:
        raise_tool_error_for_client_error(exc)


# ------------------------------------------------------------------ #
# artifact_delete                                                      #
# ------------------------------------------------------------------ #


@tool_metadata(
    tags={"artifact", "workload", "datarobot", "delete"},
    description=(
        "[Artifact—delete] Permanently delete a draft artifact. Locked artifacts "
        "and artifacts currently in use by a workload cannot be deleted.\n\n"
        "Example: artifact_delete(artifact_id='art-abc')"
    ),
)
async def artifact_delete(
    *,
    artifact_id: Annotated[str, "Id of the artifact to delete."],
) -> dict[str, Any]:
    aid = require_id(artifact_id, "artifact_id")
    try:
        WorkloadApiClient().delete_artifact(aid)
    except ClientError as exc:
        raise_tool_error_for_client_error(exc)
    return {"deleted": True, "artifact_id": aid}
