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

"""Artifact core management tools: read, create, update, and actions."""

from typing import Annotated
from typing import Any
from typing import Literal

from datarobot.errors import ClientError

from datarobot_genai.drmcputils.client_exceptions import raise_tool_error_for_client_error
from datarobot_genai.drmcputils.constants import ARTIFACT_STATUSES
from datarobot_genai.drmcputils.constants import ARTIFACT_TYPES
from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.exceptions import ToolErrorKind
from datarobot_genai.drtools.core import tool_metadata
from datarobot_genai.drtools.core.clients.datarobot_workload import WorkloadApiClient
from datarobot_genai.drtools.core.utils import require_id
from datarobot_genai.drtools.pagination import clamp_limit
from datarobot_genai.drtools.pagination import merge_pagination_metadata

# ------------------------------------------------------------------ #
# artifact_get  (list when no id, single when id)                     #
# ------------------------------------------------------------------ #


@tool_metadata(
    tags={"artifact", "workload", "datarobot", "get", "list"},
    description=(
        "[Artifact—get] Read artifacts.\n"
        "  - Omit artifact_id to LIST artifacts (id, name, status draft/locked, type, "
        "spec, timestamps); paginated with optional status / type / repository / search "
        "filters.\n"
        "  - Set artifact_id to GET a single artifact (full spec, status, type, version, "
        "repository, timestamps).\n\n"
        "Example (list): artifact_get(search='my-service', status='draft')\n"
        "Example (get):  artifact_get(artifact_id='art-abc123')"
    ),
)
async def artifact_get(
    *,
    artifact_id: Annotated[
        str | None, "Id of the artifact. Omit to list artifacts with filters."
    ] = None,
    search: Annotated[
        str | None, "List filter: case-insensitive match on name, description, and partial id."
    ] = None,
    status: Annotated[str | None, "List filter by status: 'draft' or 'locked'."] = None,
    artifact_type: Annotated[str | None, "List filter by type: 'service' or 'nim'."] = None,
    repository_id: Annotated[str | None, "List filter by artifact repository id."] = None,
    limit: Annotated[int, "Max artifacts to return when listing (1–100). Default 100."] = 100,
    offset: Annotated[int, "Artifacts to skip for pagination when listing. Default 0."] = 0,
) -> dict[str, Any]:
    if artifact_id is not None:
        aid = require_id(artifact_id, "artifact_id")
        try:
            return WorkloadApiClient().get_artifact(aid)
        except ClientError as exc:
            raise_tool_error_for_client_error(exc)

    if offset < 0:
        raise ToolError(
            "Argument validation error: 'offset' must be >= 0.",
            kind=ToolErrorKind.VALIDATION,
        )
    if status and status.lower() not in ARTIFACT_STATUSES:
        raise ToolError(
            f"Argument validation error: 'status' must be one of {ARTIFACT_STATUSES}.",
            kind=ToolErrorKind.VALIDATION,
        )
    if artifact_type and artifact_type.lower() not in ARTIFACT_TYPES:
        raise ToolError(
            f"Argument validation error: 'artifact_type' must be one of {ARTIFACT_TYPES}.",
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
# artifact_create                                                     #
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
# artifact_update                                                     #
# ------------------------------------------------------------------ #


@tool_metadata(
    tags={"artifact", "workload", "datarobot", "update"},
    description=(
        "[Artifact—update] Partially update a draft artifact: name, description, "
        "or spec. Only the supplied fields are changed. At least one field is required. "
        "Locked artifacts cannot be updated. To lock an artifact use "
        "artifact_action(action='lock').\n\n"
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
# artifact_action  (lock / clone / delete)                            #
# ------------------------------------------------------------------ #


@tool_metadata(
    tags={"artifact", "workload", "datarobot", "lock", "clone", "delete"},
    description=(
        "[Artifact—action] Run an action on an artifact. action is one of:\n"
        "  'lock'   — lock a draft artifact so it can be versioned and used in "
        "production. Once locked it cannot be updated or deleted. (To lock the "
        "artifact currently running on a workload, use workload_action "
        "action='promote'.)\n"
        "  'clone'  — duplicate the artifact under a new name (requires 'name'). The "
        "clone is always a draft, regardless of the original's status.\n"
        "  'delete' — permanently delete a draft artifact. Locked artifacts and "
        "artifacts in use by a workload cannot be deleted.\n\n"
        "Example (lock):   artifact_action(artifact_id='art-abc', action='lock')\n"
        "Example (clone):  artifact_action(artifact_id='art-abc', action='clone', name='svc-v2')\n"
        "Example (delete): artifact_action(artifact_id='art-abc', action='delete')"
    ),
)
async def artifact_action(
    *,
    artifact_id: Annotated[str, "Id of the artifact to act on."],
    action: Annotated[
        Literal["lock", "clone", "delete"],
        "Action: 'lock' | 'clone' | 'delete'.",
    ],
    name: Annotated[
        str | None, "Name for the cloned artifact. Required when action='clone'."
    ] = None,
) -> dict[str, Any]:
    aid = require_id(artifact_id, "artifact_id")
    client = WorkloadApiClient()

    try:
        if action == "lock":
            return client.patch_artifact(aid, {"status": "locked"})
        if action == "clone":
            if not name or not name.strip():
                raise ToolError(
                    "Argument validation error: 'name' is required when action='clone'.",
                    kind=ToolErrorKind.VALIDATION,
                )
            return client.clone_artifact(aid, name.strip())
        if action == "delete":
            client.delete_artifact(aid)
            return {"deleted": True, "artifact_id": aid}
    except ClientError as exc:
        raise_tool_error_for_client_error(exc)

    # pragma: no cover - guarded by Literal typing
    raise ToolError(
        f"Argument validation error: unknown action {action!r}.",
        kind=ToolErrorKind.VALIDATION,
    )
