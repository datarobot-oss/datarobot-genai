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
from datarobot_genai.drmcputils.constants import IMPORTANCE_VALUES
from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.exceptions import ToolErrorKind
from datarobot_genai.drtools.core import tool_metadata
from datarobot_genai.drtools.core.clients.datarobot_workload import WorkloadApiClient


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


@tool_metadata(
    tags={"workload", "datarobot", "start"},
    description=(
        "[Workload—start] Start a stopped workload. Returns immediately after the "
        "request is accepted (202). Use workload_wait_for_status with "
        "target_status='running' to block until live.\n\n"
        "Example: workload_start(workload_id='wkld-abc')"
    ),
)
async def workload_start(
    *,
    workload_id: Annotated[str, "Id of the workload to start."],
) -> dict[str, Any]:
    if not workload_id or not workload_id.strip():
        raise ToolError(
            "Argument validation error: 'workload_id' cannot be empty.",
            kind=ToolErrorKind.VALIDATION,
        )
    try:
        return WorkloadApiClient().start_workload(workload_id.strip())
    except ClientError as exc:
        raise_tool_error_for_client_error(exc)


@tool_metadata(
    tags={"workload", "datarobot", "stop"},
    description=(
        "[Workload—stop] Stop a running workload. By default waits until the "
        "workload reaches 'stopped' status (up to timeout_seconds). Set "
        "wait_stopped=False to return immediately after the stop request.\n\n"
        "Example: workload_stop(workload_id='wkld-abc')"
    ),
)
async def workload_stop(
    *,
    workload_id: Annotated[str, "Id of the workload to stop."],
    wait_stopped: Annotated[bool, "Poll until status is 'stopped'. Default true."] = True,
    timeout_seconds: Annotated[
        int, "Max seconds to wait when wait_stopped is true. Default 120."
    ] = 120,
) -> dict[str, Any]:
    if not workload_id or not workload_id.strip():
        raise ToolError(
            "Argument validation error: 'workload_id' cannot be empty.",
            kind=ToolErrorKind.VALIDATION,
        )
    if timeout_seconds < 1:
        raise ToolError(
            "Argument validation error: 'timeout_seconds' must be >= 1.",
            kind=ToolErrorKind.VALIDATION,
        )

    client = WorkloadApiClient()
    try:
        result = client.stop_workload(workload_id.strip())
    except ClientError as exc:
        raise_tool_error_for_client_error(exc)

    if wait_stopped:
        try:
            result = await client.wait_for_workload_status(
                workload_id.strip(), "stopped", timeout_seconds=timeout_seconds
            )
        except (RuntimeError, TimeoutError) as exc:
            raise ToolError(str(exc), kind=ToolErrorKind.UPSTREAM) from exc
        except ClientError as exc:
            raise_tool_error_for_client_error(exc)

    return result


@tool_metadata(
    tags={"workload", "datarobot", "delete"},
    description=(
        "[Workload—delete] Permanently delete a workload. The workload must be "
        "stopped first — use workload_stop then workload_delete.\n\n"
        "Example: workload_delete(workload_id='wkld-abc')"
    ),
)
async def workload_delete(
    *,
    workload_id: Annotated[str, "Id of the workload to delete."],
) -> dict[str, Any]:
    if not workload_id or not workload_id.strip():
        raise ToolError(
            "Argument validation error: 'workload_id' cannot be empty.",
            kind=ToolErrorKind.VALIDATION,
        )
    try:
        WorkloadApiClient().delete_workload(workload_id.strip())
    except ClientError as exc:
        raise_tool_error_for_client_error(exc)

    return {"deleted": True, "workload_id": workload_id.strip()}


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
    if not workload_id or not workload_id.strip():
        raise ToolError(
            "Argument validation error: 'workload_id' cannot be empty.",
            kind=ToolErrorKind.VALIDATION,
        )
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
        return WorkloadApiClient().patch_workload(workload_id.strip(), patch)
    except ClientError as exc:
        raise_tool_error_for_client_error(exc)


@tool_metadata(
    tags={"workload", "datarobot", "wait"},
    description=(
        "[Workload—wait for status] Poll a workload until it reaches a target "
        "status, raises if it enters 'errored', or times out. "
        "Common targets: 'running' (after start), 'stopped' (after stop).\n\n"
        "Example: workload_wait_for_status(workload_id='wkld-abc', target_status='running')"
    ),
)
async def workload_wait_for_status(
    *,
    workload_id: Annotated[str, "Id of the workload to poll."],
    target_status: Annotated[str, "Status to wait for, e.g. 'running', 'stopped', 'initializing'."],
    timeout_seconds: Annotated[
        int, "Max seconds to wait before raising a timeout error. Default 600."
    ] = 600,
) -> dict[str, Any]:
    if not workload_id or not workload_id.strip():
        raise ToolError(
            "Argument validation error: 'workload_id' cannot be empty.",
            kind=ToolErrorKind.VALIDATION,
        )
    if not target_status or not target_status.strip():
        raise ToolError(
            "Argument validation error: 'target_status' cannot be empty.",
            kind=ToolErrorKind.VALIDATION,
        )
    if timeout_seconds < 1:
        raise ToolError(
            "Argument validation error: 'timeout_seconds' must be >= 1.",
            kind=ToolErrorKind.VALIDATION,
        )

    try:
        return await WorkloadApiClient().wait_for_workload_status(
            workload_id.strip(),
            target_status.strip(),
            timeout_seconds=timeout_seconds,
        )
    except (RuntimeError, TimeoutError) as exc:
        raise ToolError(str(exc), kind=ToolErrorKind.UPSTREAM) from exc
    except ClientError as exc:
        raise_tool_error_for_client_error(exc)
