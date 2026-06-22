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

"""Workload discovery, creation, and lifecycle tools.

All tools issue a single REST call and return immediately. Asynchronous
operations (start / stop) never block: they return a note directing the agent
to call ``workload_get`` with a ``target_status`` for a non-blocking status
check.
"""

from typing import Annotated
from typing import Any
from typing import Literal

from datarobot.errors import ClientError

from datarobot_genai.drmcputils.client_exceptions import raise_tool_error_for_client_error
from datarobot_genai.drmcputils.constants import IMPORTANCE_VALUES
from datarobot_genai.drmcputils.constants import WORKLOAD_TERMINAL_FAILURE_STATUS
from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.exceptions import ToolErrorKind
from datarobot_genai.drtools.core import tool_metadata
from datarobot_genai.drtools.core.clients.datarobot_workload import WorkloadApiClient
from datarobot_genai.drtools.core.utils import require_id
from datarobot_genai.drtools.pagination import clamp_limit
from datarobot_genai.drtools.pagination import merge_pagination_metadata

# ------------------------------------------------------------------ #
# workload_list                                                        #
# ------------------------------------------------------------------ #


@tool_metadata(
    tags={"workload", "datarobot", "list", "search"},
    description=(
        "[Workload—list] Discover workloads: returns id, name, status, "
        "artifactId, importance, and runtime for each. Use workload_get for a "
        "single known workload id, not this.\n\n"
        "Example: workload_list(limit=50) or workload_list(search='my-app', limit=20)."
    ),
)
async def workload_list(
    *,
    search: Annotated[
        str | None,
        "Optional server-side filter; matches workload name or id.",
    ] = None,
    limit: Annotated[int, "Maximum workloads to return (1–100). Default 100."] = 100,
    offset: Annotated[int, "Number of workloads to skip for pagination. Default 0."] = 0,
) -> dict[str, Any]:
    if offset < 0:
        raise ToolError(
            "Argument validation error: 'offset' must be >= 0.",
            kind=ToolErrorKind.VALIDATION,
        )

    clamped_limit, note = clamp_limit(limit)

    try:
        result = WorkloadApiClient().list_workloads(
            limit=clamped_limit, offset=offset, search=search
        )
    except ClientError as exc:
        raise_tool_error_for_client_error(exc)

    data = result.get("data", []) or []
    return merge_pagination_metadata(
        {"workloads": data, "count": len(data)},
        result,
        note,
        offset=offset,
        limit=clamped_limit,
    )


# ------------------------------------------------------------------ #
# workload_get  (single record + optional non-blocking status check)  #
# ------------------------------------------------------------------ #


@tool_metadata(
    tags={"workload", "datarobot", "get", "status"},
    description=(
        "[Workload—get] Fetch a single workload by id: status, runtime, artifact "
        "reference, endpoint URL, creator, and timestamps. This is also the status "
        "check used after workload_action(start/stop): pass target_status to get a "
        "target_reached flag. It performs ONE non-blocking fetch — it does not wait "
        "or poll. Call it again yourself every few seconds until target_reached is "
        "true. Raises if the workload enters 'errored'.\n\n"
        "Example: workload_get(workload_id='wkld-abc')\n"
        "Example (status check): workload_get(workload_id='wkld-abc', target_status='running')"
    ),
)
async def workload_get(
    *,
    workload_id: Annotated[
        str,
        "The unique id of the workload (from workload_list or workload_create).",
    ],
    target_status: Annotated[
        str | None,
        "Optional status to compare against, e.g. 'running', 'stopped'. When set, "
        "the response includes target_reached. Does not block.",
    ] = None,
) -> dict[str, Any]:
    wid = require_id(workload_id, "workload_id")
    if target_status is not None:
        target = target_status.strip()
        if not target:
            raise ToolError(
                "Argument validation error: 'target_status' cannot be empty.",
                kind=ToolErrorKind.VALIDATION,
            )
    else:
        target = None

    try:
        obj = WorkloadApiClient().get_workload(wid)
    except ClientError as exc:
        raise_tool_error_for_client_error(exc)

    if target is None:
        return obj

    status = obj.get("status")
    if status == WORKLOAD_TERMINAL_FAILURE_STATUS:
        raise ToolError(
            f"Workload {wid} entered terminal status {status!r}.",
            kind=ToolErrorKind.UPSTREAM,
        )
    return {
        "workload_id": wid,
        "status": status,
        "target_reached": status == target,
        "raw": obj,
    }


# ------------------------------------------------------------------ #
# workload_create_payload                                              #
# ------------------------------------------------------------------ #


@tool_metadata(
    tags={"workload", "datarobot", "payload", "helper"},
    description=(
        "[Workload—build create payload] Build a valid workload create payload "
        "without making an API call. Pass the returned payload dict to "
        "workload_create. Two modes — pick exactly one:\n"
        "  (1) existing artifact: set artifact_id.\n"
        "  (2) inline artifact: set artifact_name, image_uri, port, cpu, memory_bytes.\n\n"
        "Example (existing): workload_create_payload(name='my-wl', artifact_id='art-xyz')\n"
        "Example (inline):   workload_create_payload(name='echo', artifact_name='echo', "
        "image_uri='hashicorp/http-echo:0.2.3', port=8080, cpu=1, memory_bytes=134217728)"
    ),
)
async def workload_create_payload(
    *,
    name: Annotated[str, "Workload display name. Required."],
    artifact_id: Annotated[
        str | None,
        "ID of an existing artifact to deploy. Use this OR inline fields, not both.",
    ] = None,
    description: Annotated[str | None, "Optional workload description."] = None,
    importance: Annotated[
        str | None,
        "Importance level: 'low' | 'moderate' | 'high' | 'critical'. Default 'low'.",
    ] = "low",
    resource_bundle_id: Annotated[
        str | None,
        "Bundle id from bundle_list. Sets runtime.containerGroups[0].resourceBundles.",
    ] = None,
    replica_count: Annotated[int | None, "Fixed replica count (>= 1). Default 1."] = 1,
    artifact_name: Annotated[
        str | None,
        "Inline artifact name. Required when artifact_id is not provided.",
    ] = None,
    artifact_description: Annotated[str | None, "Inline artifact description."] = None,
    image_uri: Annotated[
        str | None,
        "Docker image URI (e.g. nginx:latest). Required for inline artifact.",
    ] = None,
    port: Annotated[
        int | None,
        "Primary container port (1024–65535). Required for inline artifact.",
    ] = None,
    cpu: Annotated[int | None, "CPU cores (>= 1). Required for inline artifact."] = None,
    memory_bytes: Annotated[
        int | None,
        "Memory in bytes (e.g. 134217728 = 128 MiB). Required for inline artifact.",
    ] = None,
    gpu: Annotated[int | None, "GPU count. Default 0."] = 0,
    environment_vars: Annotated[
        list[dict[str, str]] | None,
        "Env vars as [{name, value}, ...] for the inline artifact container.",
    ] = None,
) -> dict[str, Any]:
    use_existing = bool(artifact_id and str(artifact_id).strip())
    inline_required = [artifact_name, image_uri, port, cpu, memory_bytes]
    use_inline = all(v is not None for v in inline_required) and all(
        str(v).strip() for v in [artifact_name, image_uri]
    )

    if use_existing and use_inline:
        raise ToolError(
            "Argument validation error: provide either artifact_id or inline fields, not both.",
            kind=ToolErrorKind.VALIDATION,
        )
    if not use_existing and not use_inline:
        raise ToolError(
            "Argument validation error: provide either artifact_id or all inline fields "
            "(artifact_name, image_uri, port, cpu, memory_bytes).",
            kind=ToolErrorKind.VALIDATION,
        )
    if not name or not str(name).strip():
        raise ToolError(
            "Argument validation error: 'name' is required.",
            kind=ToolErrorKind.VALIDATION,
        )
    if importance and importance.lower() not in IMPORTANCE_VALUES:
        raise ToolError(
            f"Argument validation error: 'importance' must be one of {IMPORTANCE_VALUES}.",
            kind=ToolErrorKind.VALIDATION,
        )
    if use_inline:
        if port is not None and not (1024 <= port <= 65535):
            raise ToolError(
                "Argument validation error: 'port' must be between 1024 and 65535.",
                kind=ToolErrorKind.VALIDATION,
            )
        if cpu is not None and cpu < 1:
            raise ToolError(
                "Argument validation error: 'cpu' must be >= 1.",
                kind=ToolErrorKind.VALIDATION,
            )
        if memory_bytes is not None and memory_bytes < 0:
            raise ToolError(
                "Argument validation error: 'memory_bytes' must be >= 0.",
                kind=ToolErrorKind.VALIDATION,
            )
    if replica_count is not None and replica_count < 1:
        raise ToolError(
            "Argument validation error: 'replica_count' must be >= 1.",
            kind=ToolErrorKind.VALIDATION,
        )

    payload: dict[str, Any] = {"name": name.strip()}
    if description:
        payload["description"] = description.strip()
    payload["importance"] = (importance or "low").lower()

    if use_existing:
        assert artifact_id is not None
        payload["artifactId"] = artifact_id.strip()
    else:
        assert image_uri is not None
        assert artifact_name is not None
        container: dict[str, Any] = {
            "name": "main",
            "imageUri": image_uri.strip(),
            "primary": True,
            "port": port,
            "resourceRequest": {"cpu": cpu, "memory": memory_bytes, "gpu": gpu or 0},
        }
        if environment_vars:
            container["environmentVars"] = [
                {"name": str(e.get("name", "")), "value": str(e.get("value", ""))}
                for e in environment_vars
                if isinstance(e, dict)
            ]
        artifact: dict[str, Any] = {
            "name": artifact_name.strip(),
            "spec": {"type": "service", "containerGroups": [{"containers": [container]}]},
        }
        if artifact_description:
            artifact["description"] = artifact_description.strip()
        payload["artifact"] = artifact

    group_runtime: dict[str, Any] = {"name": "default", "replicaCount": replica_count or 1}
    if resource_bundle_id and str(resource_bundle_id).strip():
        group_runtime["resourceBundles"] = [resource_bundle_id.strip()]
    payload["runtime"] = {"containerGroups": [group_runtime]}

    return {"payload": payload, "usage": "Pass payload to workload_create(payload=...)."}


# ------------------------------------------------------------------ #
# workload_create                                                      #
# ------------------------------------------------------------------ #


@tool_metadata(
    tags={"workload", "datarobot", "create"},
    description=(
        "[Workload—create] Create a new workload from a payload dict. "
        "Use workload_create_payload to build the payload first. "
        "The payload must contain 'name' and exactly one of 'artifactId' or 'artifact'.\n\n"
        "Example: workload_create(payload=workload_create_payload(...)['payload'])"
    ),
)
async def workload_create(
    *,
    payload: Annotated[
        dict[str, Any],
        "Create payload with 'name' and one of 'artifactId' or 'artifact'. "
        "Build with workload_create_payload.",
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
    if bool(payload.get("artifactId")) == bool(payload.get("artifact")):
        raise ToolError(
            "Argument validation error: payload must contain exactly one of "
            "'artifactId' or 'artifact'.",
            kind=ToolErrorKind.VALIDATION,
        )

    try:
        return WorkloadApiClient().create_workload(payload)
    except ClientError as exc:
        raise_tool_error_for_client_error(exc)


# ------------------------------------------------------------------ #
# workload_update                                                      #
# ------------------------------------------------------------------ #


@tool_metadata(
    tags={"workload", "datarobot", "update"},
    description=(
        "[Workload—update] Partially update a workload's name, description, or "
        "importance. Only supplied fields are changed. At least one field is required.\n\n"
        "Example: workload_update(workload_id='wkld-abc', name='new-name', importance='high')"
    ),
)
async def workload_update(
    *,
    workload_id: Annotated[str, "Id of the workload to update."],
    name: Annotated[str | None, "New workload name."] = None,
    description: Annotated[str | None, "New workload description."] = None,
    importance: Annotated[
        str | None, "New importance: 'low' | 'moderate' | 'high' | 'critical'."
    ] = None,
) -> dict[str, Any]:
    wid = require_id(workload_id, "workload_id")
    if not any([name, description, importance]):
        raise ToolError(
            "Argument validation error: at least one of name, description, or importance must be set.",  # noqa: E501
            kind=ToolErrorKind.VALIDATION,
        )
    if importance and importance.lower() not in IMPORTANCE_VALUES:
        raise ToolError(
            f"Argument validation error: 'importance' must be one of {IMPORTANCE_VALUES}.",
            kind=ToolErrorKind.VALIDATION,
        )

    patch: dict[str, Any] = {}
    if name is not None:
        patch["name"] = name.strip()
    if description is not None:
        patch["description"] = description.strip()
    if importance is not None:
        patch["importance"] = importance.lower()

    try:
        return WorkloadApiClient().patch_workload(wid, patch)
    except ClientError as exc:
        raise_tool_error_for_client_error(exc)


# ------------------------------------------------------------------ #
# workload_action  (start / stop / delete / promote)                  #
# ------------------------------------------------------------------ #

_ACTION_TARGET_STATUS: dict[str, str] = {"start": "running", "stop": "stopped"}


@tool_metadata(
    tags={"workload", "datarobot", "start", "stop", "delete", "promote"},
    description=(
        "[Workload—action] Run a lifecycle action on a workload. action is one of:\n"
        "  'start'   — request a stopped workload to start (returns immediately, 202).\n"
        "  'stop'    — request a running workload to stop (returns immediately, 202).\n"
        "  'delete'  — permanently delete a workload (it must be stopped first).\n"
        "  'promote' — lock the running draft artifact, assigning it a version "
        "(stats reset; recorded in history).\n\n"
        "start and stop are asynchronous and do NOT wait for completion: poll "
        "workload_get(workload_id, target_status=...) yourself until target_reached.\n\n"
        "Example: workload_action(workload_id='wkld-abc', action='start')"
    ),
)
async def workload_action(
    *,
    workload_id: Annotated[str, "Id of the workload to act on."],
    action: Annotated[
        Literal["start", "stop", "delete", "promote"],
        "Lifecycle action: 'start' | 'stop' | 'delete' | 'promote'.",
    ],
) -> dict[str, Any]:
    wid = require_id(workload_id, "workload_id")
    client = WorkloadApiClient()

    try:
        if action == "start":
            raw = client.start_workload(wid)
        elif action == "stop":
            raw = client.stop_workload(wid)
        elif action == "delete":
            client.delete_workload(wid)
            return {"deleted": True, "workload_id": wid}
        elif action == "promote":
            return client.promote_workload_artifact(wid)
        else:  # pragma: no cover - guarded by Literal typing
            raise ToolError(
                f"Argument validation error: unknown action {action!r}.",
                kind=ToolErrorKind.VALIDATION,
            )
    except ClientError as exc:
        raise_tool_error_for_client_error(exc)

    target_status = _ACTION_TARGET_STATUS[action]
    return {
        "workload_id": wid,
        "accepted": raw,
        "note": (
            f"This tool only requests the workload {action}; it does NOT wait for "
            f"completion. Call workload_get(workload_id={wid!r}, "
            f"target_status={target_status!r}) every few seconds until target_reached "
            "is true. It raises if the workload enters 'errored'."
        ),
    }


# ------------------------------------------------------------------ #
# bundle_list                                                          #
# ------------------------------------------------------------------ #


@tool_metadata(
    tags={"workload", "bundle", "datarobot", "list"},
    description=(
        "[Workload—bundle list] Discover available compute resource bundles "
        "(CPU count, memory, GPU type and VRAM). GPU type and VRAM are always "
        "determined by the bundle — there is no separate gpuType parameter. Use "
        "the bundle id when creating or updating a workload.\n\n"
        "Example: bundle_list()."
    ),
)
async def bundle_list() -> dict[str, Any]:
    try:
        return WorkloadApiClient().list_bundles()
    except ClientError as exc:
        raise_tool_error_for_client_error(exc)
