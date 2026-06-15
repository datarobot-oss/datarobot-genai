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

"""Unit tests for workload tools (PR1 — read-only surface)."""

from collections.abc import Iterator
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from datarobot.errors import ClientError

from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.exceptions import ToolErrorKind
from datarobot_genai.drtools.workload import artifact_tools
from datarobot_genai.drtools.workload import build_tools
from datarobot_genai.drtools.workload import lifecycle_tools
from datarobot_genai.drtools.workload import observability_tools
from datarobot_genai.drtools.workload import proton_tools
from datarobot_genai.drtools.workload import read_tools
from datarobot_genai.drtools.workload import replacement_tools
from datarobot_genai.drtools.workload import repository_tools

# ------------------------------------------------------------------ #
# Shared fixtures                                                       #
# ------------------------------------------------------------------ #


@pytest.fixture
def mock_rest_client() -> MagicMock:
    return MagicMock()


@pytest.fixture
def patched_dr_client(mock_rest_client: MagicMock) -> Iterator[MagicMock]:
    with patch(
        "datarobot_genai.drtools.core.clients.datarobot_workload.request_user_dr_client"
    ) as mock_cm:
        mock_cm.return_value.__enter__.return_value = mock_rest_client
        mock_cm.return_value.__exit__.return_value = False
        yield mock_rest_client


# ------------------------------------------------------------------ #
# workload_list                                                         #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_workload_list_success(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.return_value = MagicMock(
        json=lambda: {
            "data": [
                {"id": "wkld-1", "name": "Alpha", "status": "running"},
                {"id": "wkld-2", "name": "Beta", "status": "stopped"},
            ],
            "count": 2,
            "totalCount": 2,
            "next": None,
            "previous": None,
        }
    )

    result = await read_tools.workload_list()

    assert result["count"] == 2
    assert result["total_count"] == 2
    assert result["workloads"][0]["id"] == "wkld-1"
    assert result["workloads"][1]["name"] == "Beta"
    patched_dr_client.get.assert_called_once_with("workloads/", params={"limit": 100, "offset": 0})


@pytest.mark.asyncio
async def test_workload_list_with_search(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.return_value = MagicMock(
        json=lambda: {"data": [{"id": "wkld-1", "name": "my-app"}], "count": 1}
    )

    result = await read_tools.workload_list(search="my-app", limit=10, offset=5)

    patched_dr_client.get.assert_called_once_with(
        "workloads/", params={"limit": 10, "offset": 5, "search": "my-app"}
    )
    assert result["count"] == 1
    assert result["offset"] == 5
    assert result["limit"] == 10


@pytest.mark.asyncio
async def test_workload_list_clamps_limit(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.return_value = MagicMock(json=lambda: {"data": [], "count": 0})

    result = await read_tools.workload_list(limit=500)

    # limit clamped to 100
    patched_dr_client.get.assert_called_once_with("workloads/", params={"limit": 100, "offset": 0})
    assert result["limit"] == 100
    assert "note" in result


@pytest.mark.asyncio
async def test_workload_list_negative_offset_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await read_tools.workload_list(offset=-1)
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_workload_list_client_error(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.side_effect = ClientError(
        "500 Internal Server Error", status_code=500, json={}
    )

    with pytest.raises(ToolError) as exc_info:
        await read_tools.workload_list()
    assert exc_info.value.kind is ToolErrorKind.UPSTREAM


@pytest.mark.asyncio
async def test_workload_list_404_raises_not_found(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.side_effect = ClientError("404 Not Found", status_code=404, json={})

    with pytest.raises(ToolError) as exc_info:
        await read_tools.workload_list()
    assert exc_info.value.kind is ToolErrorKind.NOT_FOUND


# ------------------------------------------------------------------ #
# workload_get                                                          #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_workload_get_success(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.return_value = MagicMock(
        json=lambda: {
            "id": "wkld-abc",
            "name": "my-workload",
            "status": "running",
            "artifactId": "art-xyz",
        }
    )

    result = await read_tools.workload_get(workload_id="wkld-abc")

    patched_dr_client.get.assert_called_once_with("workloads/wkld-abc")
    assert result["id"] == "wkld-abc"
    assert result["status"] == "running"


@pytest.mark.asyncio
async def test_workload_get_strips_whitespace(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.return_value = MagicMock(json=lambda: {"id": "wkld-abc"})

    await read_tools.workload_get(workload_id="  wkld-abc  ")

    patched_dr_client.get.assert_called_once_with("workloads/wkld-abc")


@pytest.mark.asyncio
async def test_workload_get_empty_id_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await read_tools.workload_get(workload_id="")
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_workload_get_not_found(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.side_effect = ClientError("404 Not Found", status_code=404, json={})

    with pytest.raises(ToolError) as exc_info:
        await read_tools.workload_get(workload_id="wkld-missing")
    assert exc_info.value.kind is ToolErrorKind.NOT_FOUND


# ------------------------------------------------------------------ #
# bundle_list                                                           #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_bundle_list_success(patched_dr_client: MagicMock) -> None:
    bundles_payload = {
        "data": [
            {"id": "cpu.small", "cpuCount": 1, "memoryBytes": 2147483648, "gpuCount": 0},
            {"id": "gpu.l4.small", "cpuCount": 4, "memoryBytes": 17179869184, "gpuCount": 1},
        ]
    }
    patched_dr_client.get.return_value = MagicMock(json=lambda: bundles_payload)

    result = await read_tools.bundle_list()

    patched_dr_client.get.assert_called_once_with("mlops/compute/bundles")
    assert result == bundles_payload


@pytest.mark.asyncio
async def test_bundle_list_client_error(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.side_effect = ClientError(
        "500 Internal Server Error", status_code=500, json={}
    )

    with pytest.raises(ToolError) as exc_info:
        await read_tools.bundle_list()
    assert exc_info.value.kind is ToolErrorKind.UPSTREAM


# ------------------------------------------------------------------ #
# workload_create_payload                                              #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_create_payload_existing_artifact() -> None:
    result = await lifecycle_tools.workload_create_payload(name="my-wl", artifact_id="art-abc")
    p = result["payload"]
    assert p["name"] == "my-wl"
    assert p["artifactId"] == "art-abc"
    assert "artifact" not in p
    assert p["importance"] == "low"
    assert p["runtime"]["containerGroups"][0]["replicaCount"] == 1


@pytest.mark.asyncio
async def test_create_payload_inline_artifact() -> None:
    result = await lifecycle_tools.workload_create_payload(
        name="test-echo",
        artifact_name="echo",
        image_uri="hashicorp/http-echo:0.2.3",
        port=8080,
        cpu=1,
        memory_bytes=134217728,
        importance="high",
        resource_bundle_id="cpu.small",
    )
    p = result["payload"]
    assert p["name"] == "test-echo"
    assert p["importance"] == "high"
    assert "artifactId" not in p
    container = p["artifact"]["spec"]["containerGroups"][0]["containers"][0]
    assert container["imageUri"] == "hashicorp/http-echo:0.2.3"
    assert container["port"] == 8080
    assert container["resourceRequest"]["cpu"] == 1
    assert p["runtime"]["containerGroups"][0]["resourceBundles"] == ["cpu.small"]


@pytest.mark.asyncio
async def test_create_payload_both_modes_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await lifecycle_tools.workload_create_payload(
            name="x",
            artifact_id="art-abc",
            artifact_name="echo",
            image_uri="nginx:latest",
            port=8080,
            cpu=1,
            memory_bytes=134217728,
        )
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_create_payload_neither_mode_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await lifecycle_tools.workload_create_payload(name="x")
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_create_payload_invalid_importance_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await lifecycle_tools.workload_create_payload(
            name="wl", artifact_id="art-1", importance="ultra"
        )
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_create_payload_invalid_port_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await lifecycle_tools.workload_create_payload(
            name="wl", artifact_name="x", image_uri="x:1", port=80, cpu=1, memory_bytes=1
        )
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_create_payload_env_vars() -> None:
    result = await lifecycle_tools.workload_create_payload(
        name="app-wl",
        artifact_name="app",
        image_uri="my-image:latest",
        port=8080,
        cpu=2,
        memory_bytes=268435456,
        environment_vars=[{"name": "FOO", "value": "bar"}],
    )
    container = result["payload"]["artifact"]["spec"]["containerGroups"][0]["containers"][0]
    assert container["environmentVars"] == [{"name": "FOO", "value": "bar"}]


# ------------------------------------------------------------------ #
# workload_create                                                      #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_workload_create_success(patched_dr_client: MagicMock) -> None:
    patched_dr_client.post.return_value = MagicMock(
        json=lambda: {"id": "wkld-new", "name": "my-wl", "status": "initializing"}
    )

    result = await lifecycle_tools.workload_create(
        payload={"name": "my-wl", "artifactId": "art-abc", "runtime": {}}
    )

    patched_dr_client.post.assert_called_once_with(
        "workloads/", json={"name": "my-wl", "artifactId": "art-abc", "runtime": {}}
    )
    assert result["id"] == "wkld-new"


@pytest.mark.asyncio
async def test_workload_create_empty_payload_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await lifecycle_tools.workload_create(payload={})
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_workload_create_missing_name_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await lifecycle_tools.workload_create(payload={"artifactId": "art-1"})
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_workload_create_both_artifact_fields_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await lifecycle_tools.workload_create(
            payload={"name": "wl", "artifactId": "art-1", "artifact": {"name": "x"}}
        )
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_workload_create_client_error(patched_dr_client: MagicMock) -> None:
    patched_dr_client.post.side_effect = ClientError("422", status_code=422, json={})
    with pytest.raises(ToolError) as exc_info:
        await lifecycle_tools.workload_create(payload={"name": "wl", "artifactId": "art-1"})
    assert exc_info.value.kind is ToolErrorKind.UPSTREAM


# ------------------------------------------------------------------ #
# workload_start                                                        #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_workload_start_success(patched_dr_client: MagicMock) -> None:
    patched_dr_client.post.return_value = MagicMock(
        content=b'{"accepted":true}', json=lambda: {"accepted": True}
    )

    result = await lifecycle_tools.workload_start(workload_id="wkld-abc")

    patched_dr_client.post.assert_called_once_with("workloads/wkld-abc/start")
    assert result == {"accepted": True}


@pytest.mark.asyncio
async def test_workload_start_empty_id_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await lifecycle_tools.workload_start(workload_id="")
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


# ------------------------------------------------------------------ #
# workload_stop                                                         #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_workload_stop_no_wait(patched_dr_client: MagicMock) -> None:
    patched_dr_client.post.return_value = MagicMock(
        content=b'{"accepted":true}', json=lambda: {"accepted": True}
    )

    result = await lifecycle_tools.workload_stop(workload_id="wkld-abc", wait_stopped=False)

    patched_dr_client.post.assert_called_once_with("workloads/wkld-abc/stop")
    assert result == {"accepted": True}


@pytest.mark.asyncio
async def test_workload_stop_with_wait(patched_dr_client: MagicMock) -> None:
    patched_dr_client.post.return_value = MagicMock(content=b"{}", json=lambda: {})
    patched_dr_client.get.return_value = MagicMock(
        json=lambda: {"id": "wkld-abc", "status": "stopped"}
    )

    result = await lifecycle_tools.workload_stop(workload_id="wkld-abc", wait_stopped=True)

    assert result["status"] == "stopped"


@pytest.mark.asyncio
async def test_workload_stop_timeout_raises(patched_dr_client: MagicMock) -> None:
    patched_dr_client.post.return_value = MagicMock(content=b"{}", json=lambda: {})
    patched_dr_client.get.return_value = MagicMock(
        json=lambda: {"id": "wkld-abc", "status": "running"}
    )

    with pytest.raises(ToolError) as exc_info:
        await lifecycle_tools.workload_stop(
            workload_id="wkld-abc", wait_stopped=True, timeout_seconds=1
        )
    assert exc_info.value.kind is ToolErrorKind.UPSTREAM


@pytest.mark.asyncio
async def test_workload_stop_errored_raises(patched_dr_client: MagicMock) -> None:
    patched_dr_client.post.return_value = MagicMock(content=b"{}", json=lambda: {})
    patched_dr_client.get.return_value = MagicMock(
        json=lambda: {"id": "wkld-abc", "status": "errored"}
    )

    with pytest.raises(ToolError) as exc_info:
        await lifecycle_tools.workload_stop(
            workload_id="wkld-abc", wait_stopped=True, timeout_seconds=30
        )
    assert exc_info.value.kind is ToolErrorKind.UPSTREAM


@pytest.mark.asyncio
async def test_workload_stop_wait_client_error(patched_dr_client: MagicMock) -> None:
    patched_dr_client.post.return_value = MagicMock(content=b"{}", json=lambda: {})
    patched_dr_client.get.side_effect = ClientError("404 Not Found", status_code=404, json={})

    with pytest.raises(ToolError) as exc_info:
        await lifecycle_tools.workload_stop(workload_id="wkld-abc", wait_stopped=True)
    assert exc_info.value.kind is ToolErrorKind.NOT_FOUND


# ------------------------------------------------------------------ #
# workload_delete                                                       #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_workload_delete_success(patched_dr_client: MagicMock) -> None:
    patched_dr_client.delete.return_value = MagicMock()

    result = await lifecycle_tools.workload_delete(workload_id="wkld-abc")

    patched_dr_client.delete.assert_called_once_with("workloads/wkld-abc")
    assert result == {"deleted": True, "workload_id": "wkld-abc"}


@pytest.mark.asyncio
async def test_workload_delete_empty_id_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await lifecycle_tools.workload_delete(workload_id="")
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


# ------------------------------------------------------------------ #
# workload_update                                                       #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_workload_update_name(patched_dr_client: MagicMock) -> None:
    patched_dr_client.patch.return_value = MagicMock(
        json=lambda: {"id": "wkld-abc", "name": "new-name"}
    )

    result = await lifecycle_tools.workload_update(workload_id="wkld-abc", name="new-name")

    patched_dr_client.patch.assert_called_once_with("workloads/wkld-abc", json={"name": "new-name"})
    assert result["name"] == "new-name"


@pytest.mark.asyncio
async def test_workload_update_multiple_fields(patched_dr_client: MagicMock) -> None:
    patched_dr_client.patch.return_value = MagicMock(json=lambda: {"id": "wkld-abc"})

    await lifecycle_tools.workload_update(
        workload_id="wkld-abc", name="x", description="desc", importance="high"
    )

    patched_dr_client.patch.assert_called_once_with(
        "workloads/wkld-abc", json={"name": "x", "description": "desc", "importance": "high"}
    )


@pytest.mark.asyncio
async def test_workload_update_no_fields_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await lifecycle_tools.workload_update(workload_id="wkld-abc")
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_workload_update_invalid_importance_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await lifecycle_tools.workload_update(workload_id="wkld-abc", importance="ultra")
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


# ------------------------------------------------------------------ #
# workload_wait_for_status                                             #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_workload_wait_for_status_immediate(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.return_value = MagicMock(
        json=lambda: {"id": "wkld-abc", "status": "running"}
    )

    result = await lifecycle_tools.workload_wait_for_status(
        workload_id="wkld-abc", target_status="running"
    )

    assert result["status"] == "running"


@pytest.mark.asyncio
async def test_workload_wait_for_status_errored_raises(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.return_value = MagicMock(
        json=lambda: {"id": "wkld-abc", "status": "errored"}
    )

    with pytest.raises(ToolError) as exc_info:
        await lifecycle_tools.workload_wait_for_status(
            workload_id="wkld-abc", target_status="running"
        )
    assert exc_info.value.kind is ToolErrorKind.UPSTREAM


@pytest.mark.asyncio
async def test_workload_wait_for_status_empty_target_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await lifecycle_tools.workload_wait_for_status(workload_id="wkld-abc", target_status="")
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_workload_wait_for_status_zero_timeout_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await lifecycle_tools.workload_wait_for_status(
            workload_id="wkld-abc", target_status="running", timeout_seconds=0
        )
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


# ================================================================== #
# PR3 — settings + observability                                       #
# ================================================================== #

# ------------------------------------------------------------------ #
# workload_settings_get                                                #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_workload_settings_get_success(patched_dr_client: MagicMock) -> None:
    payload = {"runtime": {"containerGroups": [{"name": "default", "replicaCount": 1}]}}
    patched_dr_client.get.return_value = MagicMock(json=lambda: payload)

    result = await observability_tools.workload_settings_get(workload_id="wkld-abc")

    patched_dr_client.get.assert_called_once_with("workloads/wkld-abc/settings")
    assert result == payload


@pytest.mark.asyncio
async def test_workload_settings_get_empty_id_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await observability_tools.workload_settings_get(workload_id="")
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_workload_settings_get_client_error(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.side_effect = ClientError("404", status_code=404, json={})
    with pytest.raises(ToolError) as exc_info:
        await observability_tools.workload_settings_get(workload_id="wkld-abc")
    assert exc_info.value.kind is ToolErrorKind.NOT_FOUND


# ------------------------------------------------------------------ #
# workload_settings_update                                             #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_workload_settings_update_success(patched_dr_client: MagicMock) -> None:
    replacement = {"id": "repl-1", "status": "in_progress"}
    patched_dr_client.patch.return_value = MagicMock(json=lambda: replacement)
    runtime = {"containerGroups": [{"name": "default", "replicaCount": 2}]}

    result = await observability_tools.workload_settings_update(
        workload_id="wkld-abc", runtime=runtime
    )

    patched_dr_client.patch.assert_called_once_with(
        "workloads/wkld-abc/settings", json={"runtime": runtime}
    )
    assert result == replacement


@pytest.mark.asyncio
async def test_workload_settings_update_missing_container_groups_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await observability_tools.workload_settings_update(
            workload_id="wkld-abc", runtime={"replicaCount": 2}
        )
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_workload_settings_update_empty_runtime_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await observability_tools.workload_settings_update(workload_id="wkld-abc", runtime={})
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


# ------------------------------------------------------------------ #
# workload_stats                                                        #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_workload_stats_success(patched_dr_client: MagicMock) -> None:
    stats = {"requestCount": 100, "errorRate": 0.01, "responseTimeMs": 42.0}
    patched_dr_client.get.return_value = MagicMock(json=lambda: stats)

    result = await observability_tools.workload_stats(workload_id="wkld-abc")

    patched_dr_client.get.assert_called_once_with(
        "workloads/wkld-abc/stats",
        params={
            "responseTimeQuantile": 0.5,
            "slowRequestsThreshold": 2000,
        },
    )
    assert result == stats


@pytest.mark.asyncio
async def test_workload_stats_with_options(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.return_value = MagicMock(json=lambda: {})

    await observability_tools.workload_stats(
        workload_id="wkld-abc",
        proton_id="ptn-1",
        start_time="2026-01-01T00:00:00Z",
        end_time="2026-01-02T00:00:00Z",
        response_time_quantile=0.95,
        slow_requests_threshold=1000,
    )

    patched_dr_client.get.assert_called_once_with(
        "workloads/wkld-abc/stats",
        params={
            "responseTimeQuantile": 0.95,
            "slowRequestsThreshold": 1000,
            "protonId": "ptn-1",
            "startTime": "2026-01-01T00:00:00Z",
            "endTime": "2026-01-02T00:00:00Z",
        },
    )


@pytest.mark.asyncio
async def test_workload_stats_invalid_quantile_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await observability_tools.workload_stats(workload_id="wkld-abc", response_time_quantile=1.5)
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


# ------------------------------------------------------------------ #
# workload_history                                                      #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_workload_history_success(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.return_value = MagicMock(
        json=lambda: {
            "data": [{"artifactId": "art-1", "deployedAt": "2026-01-01T00:00:00Z"}],
            "count": 1,
            "totalCount": 1,
        }
    )

    result = await observability_tools.workload_history(workload_id="wkld-abc")

    patched_dr_client.get.assert_called_once_with(
        "workloads/wkld-abc/history", params={"limit": 20, "offset": 0}
    )
    assert result["count"] == 1
    assert result["history"][0]["artifactId"] == "art-1"


@pytest.mark.asyncio
async def test_workload_history_negative_offset_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await observability_tools.workload_history(workload_id="wkld-abc", offset=-1)
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


# ------------------------------------------------------------------ #
# workload_events                                                       #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_workload_events_success(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.return_value = MagicMock(
        json=lambda: {
            "data": [{"type": "status_change", "status": "running"}],
            "count": 1,
            "totalCount": 1,
        }
    )

    result = await observability_tools.workload_events(workload_id="wkld-abc")

    patched_dr_client.get.assert_called_once_with(
        "workloads/wkld-abc/events", params={"limit": 20, "offset": 0}
    )
    assert result["count"] == 1
    assert result["events"][0]["type"] == "status_change"


@pytest.mark.asyncio
async def test_workload_events_client_error(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.side_effect = ClientError("500", status_code=500, json={})
    with pytest.raises(ToolError) as exc_info:
        await observability_tools.workload_events(workload_id="wkld-abc")
    assert exc_info.value.kind is ToolErrorKind.UPSTREAM


# ------------------------------------------------------------------ #
# workload_promote                                                      #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_workload_promote_success(patched_dr_client: MagicMock) -> None:
    patched_dr_client.post.return_value = MagicMock(
        content=b'{"id":"wkld-abc","status":"running","artifactId":"art-locked"}',
        json=lambda: {"id": "wkld-abc", "status": "running", "artifactId": "art-locked"},
    )

    result = await observability_tools.workload_promote(workload_id="wkld-abc")

    patched_dr_client.post.assert_called_once_with("workloads/wkld-abc/promote")
    assert result["artifactId"] == "art-locked"


@pytest.mark.asyncio
async def test_workload_promote_empty_body(patched_dr_client: MagicMock) -> None:
    patched_dr_client.post.return_value = MagicMock(content=b"", json=lambda: {})

    result = await observability_tools.workload_promote(workload_id="wkld-abc")

    patched_dr_client.post.assert_called_once_with("workloads/wkld-abc/promote")
    assert result == {}


@pytest.mark.asyncio
async def test_workload_promote_empty_id_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await observability_tools.workload_promote(workload_id="")
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_workload_promote_client_error(patched_dr_client: MagicMock) -> None:
    patched_dr_client.post.side_effect = ClientError("422", status_code=422, json={})
    with pytest.raises(ToolError) as exc_info:
        await observability_tools.workload_promote(workload_id="wkld-abc")
    assert exc_info.value.kind is ToolErrorKind.UPSTREAM


# ------------------------------------------------------------------ #
# workload_related                                                      #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_workload_related_success(patched_dr_client: MagicMock) -> None:
    payload = {"artifacts": [{"id": "art-1", "name": "my-artifact"}]}
    patched_dr_client.get.return_value = MagicMock(json=lambda: payload)

    result = await observability_tools.workload_related(workload_id="wkld-abc")

    patched_dr_client.get.assert_called_once_with("workloads/wkld-abc/related")
    assert result == payload


@pytest.mark.asyncio
async def test_workload_related_empty_id_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await observability_tools.workload_related(workload_id="")
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


# ================================================================== #
# Protons + OTel logs                                           #
# ================================================================== #

# ------------------------------------------------------------------ #
# proton_list                                                          #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_proton_list_success(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.return_value = MagicMock(
        json=lambda: {
            "data": [{"id": "ptn-1", "status": "running"}, {"id": "ptn-2", "status": "stopped"}],
            "count": 2,
            "totalCount": 2,
        }
    )

    result = await proton_tools.proton_list(workload_id="wkld-abc")

    patched_dr_client.get.assert_called_once_with(
        "workloads/wkld-abc/protons", params={"limit": 20, "offset": 0}
    )
    assert result["count"] == 2
    assert result["protons"][0]["id"] == "ptn-1"


@pytest.mark.asyncio
async def test_proton_list_empty_id_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await proton_tools.proton_list(workload_id="")
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_proton_list_negative_offset_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await proton_tools.proton_list(workload_id="wkld-abc", offset=-1)
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_proton_list_client_error(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.side_effect = ClientError("404", status_code=404, json={})
    with pytest.raises(ToolError) as exc_info:
        await proton_tools.proton_list(workload_id="wkld-abc")
    assert exc_info.value.kind is ToolErrorKind.NOT_FOUND


# ------------------------------------------------------------------ #
# proton_get                                                           #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_proton_get_success(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.return_value = MagicMock(
        json=lambda: {"id": "ptn-1", "status": "running", "replicaCount": 1}
    )

    result = await proton_tools.proton_get(workload_id="wkld-abc", proton_id="ptn-1")

    patched_dr_client.get.assert_called_once_with("workloads/wkld-abc/protons/ptn-1")
    assert result["id"] == "ptn-1"


@pytest.mark.asyncio
async def test_proton_get_empty_proton_id_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await proton_tools.proton_get(workload_id="wkld-abc", proton_id="")
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_proton_get_strips_whitespace(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.return_value = MagicMock(json=lambda: {"id": "ptn-1"})

    await proton_tools.proton_get(workload_id="  wkld-abc  ", proton_id="  ptn-1  ")

    patched_dr_client.get.assert_called_once_with("workloads/wkld-abc/protons/ptn-1")


# ------------------------------------------------------------------ #
# proton_status_details                                                #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_proton_status_details_success(patched_dr_client: MagicMock) -> None:
    snapshot = {"replicas": [{"name": "pod-0", "phase": "Running", "ready": True}]}
    patched_dr_client.get.return_value = MagicMock(
        content=b'{"replicas": []}', json=lambda: snapshot
    )

    result = await proton_tools.proton_status_details(workload_id="wkld-abc", proton_id="ptn-1")

    patched_dr_client.get.assert_called_once_with("workloads/wkld-abc/protons/ptn-1/statusDetails")
    assert result == snapshot


@pytest.mark.asyncio
async def test_proton_status_details_no_status_yet(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.return_value = MagicMock(content=b"", json=lambda: None)

    result = await proton_tools.proton_status_details(workload_id="wkld-abc", proton_id="ptn-1")

    assert result["status"] == "pending"


@pytest.mark.asyncio
async def test_proton_status_details_empty_id_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await proton_tools.proton_status_details(workload_id="wkld-abc", proton_id="")
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


# ------------------------------------------------------------------ #
# workload_logs                                                         #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_workload_logs_success(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.return_value = MagicMock(
        json=lambda: {
            "data": [{"body": "Server started", "severityText": "INFO"}],
            "count": 1,
            "totalCount": 1,
        }
    )

    result = await proton_tools.workload_logs(workload_id="wkld-abc")

    patched_dr_client.get.assert_called_once_with(
        "otel/workload/wkld-abc/logs/",
        params=[("limit", 100), ("offset", 0), ("level", "debug")],
    )
    assert result["count"] == 1
    assert result["logs"][0]["body"] == "Server started"


@pytest.mark.asyncio
async def test_workload_logs_with_filters(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.return_value = MagicMock(
        json=lambda: {"data": [], "count": 0, "totalCount": 0}
    )

    await proton_tools.workload_logs(
        workload_id="wkld-abc",
        level="error",
        start_time="2026-01-01T00:00:00Z",
        end_time="2026-01-02T00:00:00Z",
        includes=["ERROR", "FATAL"],
        excludes=["healthcheck"],
        span_id="span-1",
        trace_id="trace-1",
    )

    patched_dr_client.get.assert_called_once_with(
        "otel/workload/wkld-abc/logs/",
        params=[
            ("limit", 100),
            ("offset", 0),
            ("level", "error"),
            ("startTime", "2026-01-01T00:00:00Z"),
            ("endTime", "2026-01-02T00:00:00Z"),
            ("includes", "ERROR"),
            ("includes", "FATAL"),
            ("excludes", "healthcheck"),
            ("spanId", "span-1"),
            ("traceId", "trace-1"),
        ],
    )


@pytest.mark.asyncio
async def test_workload_logs_invalid_level_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await proton_tools.workload_logs(workload_id="wkld-abc", level="verbose")
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_workload_logs_negative_offset_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await proton_tools.workload_logs(workload_id="wkld-abc", offset=-1)
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_workload_logs_client_error(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.side_effect = ClientError("500", status_code=500, json={})
    with pytest.raises(ToolError) as exc_info:
        await proton_tools.workload_logs(workload_id="wkld-abc")
    assert exc_info.value.kind is ToolErrorKind.UPSTREAM


# ------------------------------------------------------------------ #
# artifact_build_list                                                  #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_artifact_build_list_success(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.return_value = MagicMock(
        json=lambda: {
            "data": [
                {"id": "bld-1", "status": "success"},
                {"id": "bld-2", "status": "running"},
            ],
            "count": 2,
            "totalCount": 2,
        }
    )

    result = await build_tools.artifact_build_list(artifact_id="art-abc")

    patched_dr_client.get.assert_called_once_with(
        "artifacts/art-abc/builds", params={"limit": 100, "offset": 0}
    )
    assert result["count"] == 2
    assert result["builds"][0]["id"] == "bld-1"


@pytest.mark.asyncio
async def test_artifact_build_list_empty_id_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await build_tools.artifact_build_list(artifact_id="")
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_artifact_build_list_negative_offset_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await build_tools.artifact_build_list(artifact_id="art-abc", offset=-1)
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_artifact_build_list_not_found(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.side_effect = ClientError("404", status_code=404, json={})
    with pytest.raises(ToolError) as exc_info:
        await build_tools.artifact_build_list(artifact_id="art-missing")
    assert exc_info.value.kind is ToolErrorKind.NOT_FOUND


# ------------------------------------------------------------------ #
# artifact_build_trigger                                               #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_artifact_build_trigger_success(patched_dr_client: MagicMock) -> None:
    patched_dr_client.post.return_value = MagicMock(json=lambda: {"buildIds": ["bld-new"]})

    result = await build_tools.artifact_build_trigger(artifact_id="art-abc")

    patched_dr_client.post.assert_called_once_with("artifacts/art-abc/builds")
    assert result["buildIds"] == ["bld-new"]


@pytest.mark.asyncio
async def test_artifact_build_trigger_empty_id_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await build_tools.artifact_build_trigger(artifact_id="")
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_artifact_build_trigger_locked_error(patched_dr_client: MagicMock) -> None:
    patched_dr_client.post.side_effect = ClientError("422", status_code=422, json={})
    with pytest.raises(ToolError) as exc_info:
        await build_tools.artifact_build_trigger(artifact_id="art-locked")
    assert exc_info.value.kind is ToolErrorKind.UPSTREAM


# ------------------------------------------------------------------ #
# artifact_build_get                                                   #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_artifact_build_get_success(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.return_value = MagicMock(
        json=lambda: {"id": "bld-xyz", "status": "success"}
    )

    result = await build_tools.artifact_build_get(artifact_id="art-abc", build_id="bld-xyz")

    patched_dr_client.get.assert_called_once_with("artifacts/art-abc/builds/bld-xyz")
    assert result["id"] == "bld-xyz"


@pytest.mark.asyncio
async def test_artifact_build_get_empty_build_id_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await build_tools.artifact_build_get(artifact_id="art-abc", build_id="")
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


# ------------------------------------------------------------------ #
# artifact_build_logs                                                  #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_artifact_build_logs_success(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.return_value = MagicMock(text="Step 1/3: FROM python:3.11\n")

    result = await build_tools.artifact_build_logs(artifact_id="art-abc", build_id="bld-xyz")

    patched_dr_client.get.assert_called_once_with("artifacts/art-abc/builds/bld-xyz/logs")
    assert result == {
        "artifact_id": "art-abc",
        "build_id": "bld-xyz",
        "logs": "Step 1/3: FROM python:3.11\n",
    }


@pytest.mark.asyncio
async def test_artifact_build_logs_not_found(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.side_effect = ClientError("404", status_code=404, json={})
    with pytest.raises(ToolError) as exc_info:
        await build_tools.artifact_build_logs(artifact_id="art-abc", build_id="bld-xyz")
    assert exc_info.value.kind is ToolErrorKind.NOT_FOUND


# ------------------------------------------------------------------ #
# artifact_build_delete                                                #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_artifact_build_delete_success(patched_dr_client: MagicMock) -> None:
    patched_dr_client.delete.return_value = MagicMock()

    result = await build_tools.artifact_build_delete(artifact_id="art-abc", build_id="bld-xyz")

    patched_dr_client.delete.assert_called_once_with("artifacts/art-abc/builds/bld-xyz")
    assert result == {"deleted": True, "artifact_id": "art-abc", "build_id": "bld-xyz"}


@pytest.mark.asyncio
async def test_artifact_build_delete_empty_ids_raise() -> None:
    with pytest.raises(ToolError) as exc_info:
        await build_tools.artifact_build_delete(artifact_id="", build_id="bld-xyz")
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


# ------------------------------------------------------------------ #
# artifact_repository_list                                             #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_artifact_repository_list_success(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.return_value = MagicMock(
        json=lambda: {
            "data": [{"id": "repo-1", "name": "my-registry"}],
            "count": 1,
            "totalCount": 1,
        }
    )

    result = await repository_tools.artifact_repository_list()

    patched_dr_client.get.assert_called_once_with(
        "artifactRepositories", params={"limit": 100, "offset": 0}
    )
    assert result["count"] == 1
    assert result["repositories"][0]["id"] == "repo-1"


@pytest.mark.asyncio
async def test_artifact_repository_list_with_filters(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.return_value = MagicMock(
        json=lambda: {"data": [], "count": 0, "totalCount": 0}
    )

    await repository_tools.artifact_repository_list(search="ecr", artifact_type="nim")

    patched_dr_client.get.assert_called_once_with(
        "artifactRepositories",
        params={"limit": 100, "offset": 0, "search": "ecr", "type": "nim"},
    )


@pytest.mark.asyncio
async def test_artifact_repository_list_invalid_type_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await repository_tools.artifact_repository_list(artifact_type="docker")
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_artifact_repository_list_negative_offset_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await repository_tools.artifact_repository_list(offset=-1)
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


# ------------------------------------------------------------------ #
# artifact_repository_get                                              #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_artifact_repository_get_success(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.return_value = MagicMock(
        json=lambda: {"id": "repo-abc", "name": "my-registry"}
    )

    result = await repository_tools.artifact_repository_get(repository_id="repo-abc")

    patched_dr_client.get.assert_called_once_with("artifactRepositories/repo-abc")
    assert result["id"] == "repo-abc"


@pytest.mark.asyncio
async def test_artifact_repository_get_empty_id_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await repository_tools.artifact_repository_get(repository_id="")
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_artifact_repository_get_not_found(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.side_effect = ClientError("404", status_code=404, json={})
    with pytest.raises(ToolError) as exc_info:
        await repository_tools.artifact_repository_get(repository_id="repo-missing")
    assert exc_info.value.kind is ToolErrorKind.NOT_FOUND


# ------------------------------------------------------------------ #
# artifact_repository_delete                                           #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_artifact_repository_delete_success(patched_dr_client: MagicMock) -> None:
    patched_dr_client.delete.return_value = MagicMock()

    result = await repository_tools.artifact_repository_delete(repository_id="repo-abc")

    patched_dr_client.delete.assert_called_once_with("artifactRepositories/repo-abc")
    assert result == {"deleted": True, "repository_id": "repo-abc"}


@pytest.mark.asyncio
async def test_artifact_repository_delete_empty_id_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await repository_tools.artifact_repository_delete(repository_id="")
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_artifact_repository_delete_conflict(patched_dr_client: MagicMock) -> None:
    patched_dr_client.delete.side_effect = ClientError("409 Conflict", status_code=409, json={})
    with pytest.raises(ToolError) as exc_info:
        await repository_tools.artifact_repository_delete(repository_id="repo-in-use")
    assert exc_info.value.kind is ToolErrorKind.UPSTREAM


# ================================================================== #
# PR5 — artifacts core                                                 #
# ================================================================== #

# ------------------------------------------------------------------ #
# artifact_list                                                        #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_artifact_list_success(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.return_value = MagicMock(
        json=lambda: {
            "data": [
                {"id": "art-1", "name": "my-service", "status": "draft"},
                {"id": "art-2", "name": "my-nim", "status": "locked"},
            ],
            "count": 2,
            "totalCount": 2,
        }
    )

    result = await artifact_tools.artifact_list()

    patched_dr_client.get.assert_called_once_with("artifacts/", params={"limit": 100, "offset": 0})
    assert result["count"] == 2
    assert result["artifacts"][0]["id"] == "art-1"


@pytest.mark.asyncio
async def test_artifact_list_with_filters(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.return_value = MagicMock(
        json=lambda: {"data": [], "count": 0, "totalCount": 0}
    )

    await artifact_tools.artifact_list(
        search="my-svc", status="draft", artifact_type="service", repository_id="repo-1"
    )

    patched_dr_client.get.assert_called_once_with(
        "artifacts/",
        params={
            "limit": 100,
            "offset": 0,
            "search": "my-svc",
            "status": "draft",
            "type": "service",
            "repositoryId": "repo-1",
        },
    )


@pytest.mark.asyncio
async def test_artifact_list_invalid_status_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await artifact_tools.artifact_list(status="pending")
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_artifact_list_invalid_type_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await artifact_tools.artifact_list(artifact_type="docker")
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_artifact_list_negative_offset_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await artifact_tools.artifact_list(offset=-1)
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


# ------------------------------------------------------------------ #
# artifact_get                                                         #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_artifact_get_success(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.return_value = MagicMock(
        json=lambda: {"id": "art-abc", "name": "my-service", "status": "draft"}
    )

    result = await artifact_tools.artifact_get(artifact_id="art-abc")

    patched_dr_client.get.assert_called_once_with("artifacts/art-abc")
    assert result["id"] == "art-abc"


@pytest.mark.asyncio
async def test_artifact_get_empty_id_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await artifact_tools.artifact_get(artifact_id="")
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_artifact_get_not_found(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.side_effect = ClientError("404", status_code=404, json={})
    with pytest.raises(ToolError) as exc_info:
        await artifact_tools.artifact_get(artifact_id="art-missing")
    assert exc_info.value.kind is ToolErrorKind.NOT_FOUND


# ------------------------------------------------------------------ #
# artifact_create                                                      #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_artifact_create_success(patched_dr_client: MagicMock) -> None:
    patched_dr_client.post.return_value = MagicMock(
        json=lambda: {"id": "art-new", "name": "my-svc", "status": "draft"}
    )
    payload = {
        "name": "my-svc",
        "spec": {
            "type": "service",
            "containerGroups": [{"containers": [{"name": "main", "imageUri": "nginx:latest"}]}],
        },
    }

    result = await artifact_tools.artifact_create(payload=payload)

    patched_dr_client.post.assert_called_once_with("artifacts/", json=payload)
    assert result["id"] == "art-new"


@pytest.mark.asyncio
async def test_artifact_create_missing_name_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await artifact_tools.artifact_create(payload={"spec": {"type": "service"}})
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_artifact_create_missing_spec_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await artifact_tools.artifact_create(payload={"name": "my-svc"})
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_artifact_create_client_error(patched_dr_client: MagicMock) -> None:
    patched_dr_client.post.side_effect = ClientError("422", status_code=422, json={})
    with pytest.raises(ToolError) as exc_info:
        await artifact_tools.artifact_create(payload={"name": "x", "spec": {"type": "service"}})
    assert exc_info.value.kind is ToolErrorKind.UPSTREAM


# ------------------------------------------------------------------ #
# artifact_update                                                      #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_artifact_update_name(patched_dr_client: MagicMock) -> None:
    patched_dr_client.patch.return_value = MagicMock(
        json=lambda: {"id": "art-abc", "name": "new-name"}
    )

    result = await artifact_tools.artifact_update(artifact_id="art-abc", name="new-name")

    patched_dr_client.patch.assert_called_once_with("artifacts/art-abc", json={"name": "new-name"})
    assert result["name"] == "new-name"


@pytest.mark.asyncio
async def test_artifact_update_multiple_fields(patched_dr_client: MagicMock) -> None:
    patched_dr_client.patch.return_value = MagicMock(json=lambda: {"id": "art-abc"})

    await artifact_tools.artifact_update(artifact_id="art-abc", name="x", description="desc")

    patched_dr_client.patch.assert_called_once_with(
        "artifacts/art-abc", json={"name": "x", "description": "desc"}
    )


@pytest.mark.asyncio
async def test_artifact_update_no_fields_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await artifact_tools.artifact_update(artifact_id="art-abc")
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


# ------------------------------------------------------------------ #
# artifact_lock                                                        #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_artifact_lock_success(patched_dr_client: MagicMock) -> None:
    patched_dr_client.patch.return_value = MagicMock(
        json=lambda: {"id": "art-abc", "status": "locked"}
    )

    result = await artifact_tools.artifact_lock(artifact_id="art-abc")

    patched_dr_client.patch.assert_called_once_with("artifacts/art-abc", json={"status": "locked"})
    assert result["status"] == "locked"


@pytest.mark.asyncio
async def test_artifact_lock_empty_id_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await artifact_tools.artifact_lock(artifact_id="")
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


# ------------------------------------------------------------------ #
# artifact_clone                                                       #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_artifact_clone_success(patched_dr_client: MagicMock) -> None:
    patched_dr_client.post.return_value = MagicMock(
        json=lambda: {"id": "art-new", "name": "my-service-v2", "status": "draft"}
    )

    result = await artifact_tools.artifact_clone(artifact_id="art-abc", name="my-service-v2")

    patched_dr_client.post.assert_called_once_with(
        "artifacts/art-abc/clone", json={"name": "my-service-v2"}
    )
    assert result["name"] == "my-service-v2"


@pytest.mark.asyncio
async def test_artifact_clone_empty_name_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await artifact_tools.artifact_clone(artifact_id="art-abc", name="")
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_artifact_clone_not_found(patched_dr_client: MagicMock) -> None:
    patched_dr_client.post.side_effect = ClientError("404", status_code=404, json={})
    with pytest.raises(ToolError) as exc_info:
        await artifact_tools.artifact_clone(artifact_id="art-missing", name="clone")
    assert exc_info.value.kind is ToolErrorKind.NOT_FOUND


# ------------------------------------------------------------------ #
# artifact_delete                                                      #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_artifact_delete_success(patched_dr_client: MagicMock) -> None:
    patched_dr_client.delete.return_value = MagicMock()

    result = await artifact_tools.artifact_delete(artifact_id="art-abc")

    patched_dr_client.delete.assert_called_once_with("artifacts/art-abc")
    assert result == {"deleted": True, "artifact_id": "art-abc"}


@pytest.mark.asyncio
async def test_artifact_delete_empty_id_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await artifact_tools.artifact_delete(artifact_id="")
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_artifact_delete_locked_error(patched_dr_client: MagicMock) -> None:
    patched_dr_client.delete.side_effect = ClientError("409 Conflict", status_code=409, json={})
    with pytest.raises(ToolError) as exc_info:
        await artifact_tools.artifact_delete(artifact_id="art-locked")
    assert exc_info.value.kind is ToolErrorKind.UPSTREAM


# ================================================================== #
# PR7 — workload replacement (rolling update)                         #
# ================================================================== #

# ------------------------------------------------------------------ #
# workload_replacement_get                                             #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_workload_replacement_get_success(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.return_value = MagicMock(
        json=lambda: {
            "candidateArtifactId": "art-new",
            "strategy": "rolling",
            "config": {"warmupDurationMinutes": 5, "keepOldVersionMinutes": 2},
        }
    )

    result = await replacement_tools.workload_replacement_get(workload_id="wkld-abc")

    patched_dr_client.get.assert_called_once_with("workloads/wkld-abc/replacement")
    assert result["candidateArtifactId"] == "art-new"


@pytest.mark.asyncio
async def test_workload_replacement_get_empty_id_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await replacement_tools.workload_replacement_get(workload_id="")
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_workload_replacement_get_not_found(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.side_effect = ClientError("404", status_code=404, json={})
    with pytest.raises(ToolError) as exc_info:
        await replacement_tools.workload_replacement_get(workload_id="wkld-missing")
    assert exc_info.value.kind is ToolErrorKind.NOT_FOUND


# ------------------------------------------------------------------ #
# workload_replacement_create                                          #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_workload_replacement_create_success(patched_dr_client: MagicMock) -> None:
    patched_dr_client.post.return_value = MagicMock(
        json=lambda: {"candidateArtifactId": "art-xyz", "strategy": "rolling"}
    )

    result = await replacement_tools.workload_replacement_create(
        workload_id="wkld-abc", artifact_id="art-xyz"
    )

    patched_dr_client.post.assert_called_once_with(
        "workloads/wkld-abc/replacement",
        json={
            "artifactId": "art-xyz",
            "strategy": "rolling",
            "config": {"warmupDurationMinutes": 0, "keepOldVersionMinutes": 0},
        },
    )
    assert result["candidateArtifactId"] == "art-xyz"


@pytest.mark.asyncio
async def test_workload_replacement_create_with_config(patched_dr_client: MagicMock) -> None:
    patched_dr_client.post.return_value = MagicMock(json=lambda: {"candidateArtifactId": "art-xyz"})

    await replacement_tools.workload_replacement_create(
        workload_id="wkld-abc",
        artifact_id="art-xyz",
        warmup_duration_minutes=5,
        keep_old_version_minutes=2,
    )

    patched_dr_client.post.assert_called_once_with(
        "workloads/wkld-abc/replacement",
        json={
            "artifactId": "art-xyz",
            "strategy": "rolling",
            "config": {"warmupDurationMinutes": 5, "keepOldVersionMinutes": 2},
        },
    )


@pytest.mark.asyncio
async def test_workload_replacement_create_with_runtime(patched_dr_client: MagicMock) -> None:
    patched_dr_client.post.return_value = MagicMock(json=lambda: {"candidateArtifactId": "art-xyz"})
    runtime = {"containerGroups": [{"resourceBundles": [{"bundleId": "bundle-1"}]}]}

    await replacement_tools.workload_replacement_create(
        workload_id="wkld-abc", artifact_id="art-xyz", runtime=runtime
    )

    call_kwargs = patched_dr_client.post.call_args
    assert call_kwargs[1]["json"]["runtime"] == runtime


@pytest.mark.asyncio
async def test_workload_replacement_create_empty_workload_id_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await replacement_tools.workload_replacement_create(workload_id="", artifact_id="art-xyz")
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_workload_replacement_create_empty_artifact_id_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await replacement_tools.workload_replacement_create(workload_id="wkld-abc", artifact_id="")
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_workload_replacement_create_invalid_strategy_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await replacement_tools.workload_replacement_create(
            workload_id="wkld-abc", artifact_id="art-xyz", strategy="bluegreen"
        )
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_workload_replacement_create_negative_warmup_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await replacement_tools.workload_replacement_create(
            workload_id="wkld-abc", artifact_id="art-xyz", warmup_duration_minutes=-1
        )
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_workload_replacement_create_client_error(patched_dr_client: MagicMock) -> None:
    patched_dr_client.post.side_effect = ClientError("400", status_code=400, json={})
    with pytest.raises(ToolError) as exc_info:
        await replacement_tools.workload_replacement_create(
            workload_id="wkld-abc", artifact_id="art-xyz"
        )
    assert exc_info.value.kind is ToolErrorKind.UPSTREAM


# ------------------------------------------------------------------ #
# workload_replacement_delete                                          #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_workload_replacement_delete_success(patched_dr_client: MagicMock) -> None:
    patched_dr_client.delete.return_value = MagicMock(
        content=b'{"status": "cancelled"}',
        json=lambda: {"status": "cancelled"},
    )

    result = await replacement_tools.workload_replacement_delete(workload_id="wkld-abc")

    patched_dr_client.delete.assert_called_once_with("workloads/wkld-abc/replacement")
    assert result["status"] == "cancelled"


@pytest.mark.asyncio
async def test_workload_replacement_delete_empty_id_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await replacement_tools.workload_replacement_delete(workload_id="")
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_workload_replacement_delete_not_found(patched_dr_client: MagicMock) -> None:
    patched_dr_client.delete.side_effect = ClientError("404", status_code=404, json={})
    with pytest.raises(ToolError) as exc_info:
        await replacement_tools.workload_replacement_delete(workload_id="wkld-missing")
    assert exc_info.value.kind is ToolErrorKind.NOT_FOUND
