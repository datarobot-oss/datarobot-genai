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

"""Unit tests for the consolidated workload tools.

Style: GIVEN preconditions / WHEN behavior under test / THEN expected outcomes.
"""

from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from datarobot.errors import ClientError

from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.exceptions import ToolErrorKind
from datarobot_genai.drtools.workload import artifact_builds
from datarobot_genai.drtools.workload import artifact_repositories
from datarobot_genai.drtools.workload import artifacts
from datarobot_genai.drtools.workload import workload_observability
from datarobot_genai.drtools.workload import workload_runtime
from datarobot_genai.drtools.workload import workloads


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
# workload_list                                                        #
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

    result = await workloads.workload_list()

    assert result["count"] == 2
    assert result["total_count"] == 2
    assert result["workloads"][0]["id"] == "wkld-1"
    patched_dr_client.get.assert_called_once_with("workloads/", params={"limit": 100, "offset": 0})


@pytest.mark.asyncio
async def test_workload_list_with_search(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.return_value = MagicMock(
        json=lambda: {"data": [{"id": "wkld-1", "name": "my-app"}], "count": 1}
    )

    result = await workloads.workload_list(search="my-app", limit=10, offset=5)

    patched_dr_client.get.assert_called_once_with(
        "workloads/", params={"limit": 10, "offset": 5, "search": "my-app"}
    )
    assert result["offset"] == 5
    assert result["limit"] == 10


@pytest.mark.asyncio
async def test_workload_list_clamps_limit(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.return_value = MagicMock(json=lambda: {"data": [], "count": 0})

    result = await workloads.workload_list(limit=500)

    patched_dr_client.get.assert_called_once_with("workloads/", params={"limit": 100, "offset": 0})
    assert result["limit"] == 100
    assert "note" in result


@pytest.mark.asyncio
async def test_workload_list_negative_offset_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await workloads.workload_list(offset=-1)
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_workload_list_client_error(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.side_effect = ClientError("500", status_code=500, json={})
    with pytest.raises(ToolError) as exc_info:
        await workloads.workload_list()
    assert exc_info.value.kind is ToolErrorKind.UPSTREAM


# ------------------------------------------------------------------ #
# workload_get  (single record + non-blocking status check)           #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_workload_get_success(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.return_value = MagicMock(
        json=lambda: {"id": "wkld-abc", "name": "my-workload", "status": "running"}
    )

    result = await workloads.workload_get(workload_id="wkld-abc")

    patched_dr_client.get.assert_called_once_with("workloads/wkld-abc")
    assert result["id"] == "wkld-abc"
    assert result["status"] == "running"


@pytest.mark.asyncio
async def test_workload_get_strips_whitespace(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.return_value = MagicMock(json=lambda: {"id": "wkld-abc"})

    await workloads.workload_get(workload_id="  wkld-abc  ")

    patched_dr_client.get.assert_called_once_with("workloads/wkld-abc")


@pytest.mark.asyncio
async def test_workload_get_empty_id_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await workloads.workload_get(workload_id="")
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_workload_get_not_found(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.side_effect = ClientError("404 Not Found", status_code=404, json={})
    with pytest.raises(ToolError) as exc_info:
        await workloads.workload_get(workload_id="wkld-missing")
    assert exc_info.value.kind is ToolErrorKind.NOT_FOUND


@pytest.mark.asyncio
async def test_workload_get_target_reached(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.return_value = MagicMock(
        json=lambda: {"id": "wkld-abc", "status": "running"}
    )

    result = await workloads.workload_get(workload_id="wkld-abc", target_status="running")

    assert result["status"] == "running"
    assert result["target_reached"] is True
    assert result["raw"]["id"] == "wkld-abc"


@pytest.mark.asyncio
async def test_workload_get_target_not_reached(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.return_value = MagicMock(
        json=lambda: {"id": "wkld-abc", "status": "initializing"}
    )

    result = await workloads.workload_get(workload_id="wkld-abc", target_status="running")

    assert result["target_reached"] is False


@pytest.mark.asyncio
async def test_workload_get_target_errored_raises(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.return_value = MagicMock(
        json=lambda: {"id": "wkld-abc", "status": "errored"}
    )

    with pytest.raises(ToolError) as exc_info:
        await workloads.workload_get(workload_id="wkld-abc", target_status="running")
    assert exc_info.value.kind is ToolErrorKind.UPSTREAM


@pytest.mark.asyncio
async def test_workload_get_empty_target_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await workloads.workload_get(workload_id="wkld-abc", target_status="   ")
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


# ------------------------------------------------------------------ #
# workload_bundle_list                                                          #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_bundle_list_success(patched_dr_client: MagicMock) -> None:
    bundles_payload = {"data": [{"id": "cpu.small", "cpuCount": 1}]}
    patched_dr_client.get.return_value = MagicMock(json=lambda: bundles_payload)

    result = await workloads.workload_bundle_list()

    patched_dr_client.get.assert_called_once_with("mlops/compute/bundles")
    assert result == bundles_payload


@pytest.mark.asyncio
async def test_bundle_list_client_error(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.side_effect = ClientError("500", status_code=500, json={})
    with pytest.raises(ToolError) as exc_info:
        await workloads.workload_bundle_list()
    assert exc_info.value.kind is ToolErrorKind.UPSTREAM


# ------------------------------------------------------------------ #
# workload_create_payload_build                                              #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_create_payload_existing_artifact() -> None:
    result = await workloads.workload_create_payload_build(name="my-wl", artifact_id="art-abc")
    p = result["payload"]
    assert p["name"] == "my-wl"
    assert p["artifactId"] == "art-abc"
    assert "artifact" not in p
    assert p["importance"] == "low"
    assert p["runtime"]["containerGroups"][0]["replicaCount"] == 1


@pytest.mark.asyncio
async def test_create_payload_inline_artifact() -> None:
    result = await workloads.workload_create_payload_build(
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
    assert p["importance"] == "high"
    assert "artifactId" not in p
    container = p["artifact"]["spec"]["containerGroups"][0]["containers"][0]
    assert container["imageUri"] == "hashicorp/http-echo:0.2.3"
    assert container["port"] == 8080
    assert p["runtime"]["containerGroups"][0]["resourceBundles"] == ["cpu.small"]


@pytest.mark.asyncio
async def test_create_payload_both_modes_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await workloads.workload_create_payload_build(
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
        await workloads.workload_create_payload_build(name="x")
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_create_payload_invalid_importance_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await workloads.workload_create_payload_build(
            name="wl", artifact_id="art-1", importance="ultra"
        )
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_create_payload_invalid_port_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await workloads.workload_create_payload_build(
            name="wl", artifact_name="x", image_uri="x:1", port=80, cpu=1, memory_bytes=1
        )
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_create_payload_env_vars() -> None:
    result = await workloads.workload_create_payload_build(
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

    result = await workloads.workload_create(
        payload={"name": "my-wl", "artifactId": "art-abc", "runtime": {}}
    )

    patched_dr_client.post.assert_called_once_with(
        "workloads/", json={"name": "my-wl", "artifactId": "art-abc", "runtime": {}}
    )
    assert result["id"] == "wkld-new"


@pytest.mark.asyncio
async def test_workload_create_missing_name_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await workloads.workload_create(payload={"artifactId": "art-1"})
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_workload_create_both_artifact_fields_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await workloads.workload_create(
            payload={"name": "wl", "artifactId": "art-1", "artifact": {"name": "x"}}
        )
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_workload_create_client_error(patched_dr_client: MagicMock) -> None:
    patched_dr_client.post.side_effect = ClientError("422", status_code=422, json={})
    with pytest.raises(ToolError) as exc_info:
        await workloads.workload_create(payload={"name": "wl", "artifactId": "art-1"})
    assert exc_info.value.kind is ToolErrorKind.UPSTREAM


# ------------------------------------------------------------------ #
# workload_update                                                      #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_workload_update_name(patched_dr_client: MagicMock) -> None:
    patched_dr_client.patch.return_value = MagicMock(
        json=lambda: {"id": "wkld-abc", "name": "new-name"}
    )

    result = await workloads.workload_update(workload_id="wkld-abc", name="new-name")

    patched_dr_client.patch.assert_called_once_with("workloads/wkld-abc", json={"name": "new-name"})
    assert result["name"] == "new-name"


@pytest.mark.asyncio
async def test_workload_update_multiple_fields(patched_dr_client: MagicMock) -> None:
    patched_dr_client.patch.return_value = MagicMock(json=lambda: {"id": "wkld-abc"})

    await workloads.workload_update(
        workload_id="wkld-abc", name="x", description="desc", importance="high"
    )

    patched_dr_client.patch.assert_called_once_with(
        "workloads/wkld-abc", json={"name": "x", "description": "desc", "importance": "high"}
    )


@pytest.mark.asyncio
async def test_workload_update_no_fields_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await workloads.workload_update(workload_id="wkld-abc")
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_workload_update_invalid_importance_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await workloads.workload_update(workload_id="wkld-abc", importance="ultra")
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


# ------------------------------------------------------------------ #
# workload_action_run  (start / stop / delete / promote)                  #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_workload_action_start(patched_dr_client: MagicMock) -> None:
    patched_dr_client.post.return_value = MagicMock(
        content=b'{"accepted":true}', json=lambda: {"accepted": True}
    )

    result = await workloads.workload_action_run(workload_id="wkld-abc", action="start")

    patched_dr_client.post.assert_called_once_with("workloads/wkld-abc/start")
    assert result["workload_id"] == "wkld-abc"
    assert result["accepted"] == {"accepted": True}
    assert "workload_get(" in result["note"]
    assert "running" in result["note"]


@pytest.mark.asyncio
async def test_workload_action_stop(patched_dr_client: MagicMock) -> None:
    patched_dr_client.post.return_value = MagicMock(
        content=b'{"accepted":true}', json=lambda: {"accepted": True}
    )

    result = await workloads.workload_action_run(workload_id="wkld-abc", action="stop")

    patched_dr_client.post.assert_called_once_with("workloads/wkld-abc/stop")
    assert "workload_get(" in result["note"]
    assert "stopped" in result["note"]


@pytest.mark.asyncio
async def test_workload_action_delete(patched_dr_client: MagicMock) -> None:
    patched_dr_client.delete.return_value = MagicMock()

    result = await workloads.workload_action_run(workload_id="wkld-abc", action="delete")

    patched_dr_client.delete.assert_called_once_with("workloads/wkld-abc")
    assert result == {"deleted": True, "workload_id": "wkld-abc"}


@pytest.mark.asyncio
async def test_workload_action_promote(patched_dr_client: MagicMock) -> None:
    patched_dr_client.post.return_value = MagicMock(
        content=b'{"id":"wkld-abc","artifactId":"art-locked"}',
        json=lambda: {"id": "wkld-abc", "artifactId": "art-locked"},
    )

    result = await workloads.workload_action_run(workload_id="wkld-abc", action="promote")

    patched_dr_client.post.assert_called_once_with("workloads/wkld-abc/promote")
    assert result["artifactId"] == "art-locked"


@pytest.mark.asyncio
async def test_workload_action_promote_empty_body(patched_dr_client: MagicMock) -> None:
    patched_dr_client.post.return_value = MagicMock(content=b"", json=lambda: {})

    result = await workloads.workload_action_run(workload_id="wkld-abc", action="promote")

    assert result == {}


@pytest.mark.asyncio
async def test_workload_action_empty_id_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await workloads.workload_action_run(workload_id="", action="start")
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_workload_action_stop_not_found(patched_dr_client: MagicMock) -> None:
    patched_dr_client.post.side_effect = ClientError("404 Not Found", status_code=404, json={})
    with pytest.raises(ToolError) as exc_info:
        await workloads.workload_action_run(workload_id="wkld-abc", action="stop")
    assert exc_info.value.kind is ToolErrorKind.NOT_FOUND


# ================================================================== #
# workload_runtime — settings + replacement                            #
# ================================================================== #

# ------------------------------------------------------------------ #
# workload_settings  (read when runtime omitted, update when set)     #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_workload_settings_read(patched_dr_client: MagicMock) -> None:
    payload = {"runtime": {"containerGroups": [{"name": "default", "replicaCount": 1}]}}
    patched_dr_client.get.return_value = MagicMock(json=lambda: payload)

    result = await workload_runtime.workload_settings(workload_id="wkld-abc")

    patched_dr_client.get.assert_called_once_with("workloads/wkld-abc/settings")
    assert result == payload


@pytest.mark.asyncio
async def test_workload_settings_update(patched_dr_client: MagicMock) -> None:
    replacement = {"id": "repl-1", "status": "in_progress"}
    patched_dr_client.patch.return_value = MagicMock(json=lambda: replacement)
    runtime = {"containerGroups": [{"name": "default", "replicaCount": 2}]}

    result = await workload_runtime.workload_settings(workload_id="wkld-abc", runtime=runtime)

    patched_dr_client.patch.assert_called_once_with(
        "workloads/wkld-abc/settings", json={"runtime": runtime}
    )
    assert result == replacement


@pytest.mark.asyncio
async def test_workload_settings_empty_id_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await workload_runtime.workload_settings(workload_id="")
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_workload_settings_update_missing_container_groups_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await workload_runtime.workload_settings(
            workload_id="wkld-abc", runtime={"replicaCount": 2}
        )
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_workload_settings_read_client_error(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.side_effect = ClientError("404", status_code=404, json={})
    with pytest.raises(ToolError) as exc_info:
        await workload_runtime.workload_settings(workload_id="wkld-abc")
    assert exc_info.value.kind is ToolErrorKind.NOT_FOUND


# ------------------------------------------------------------------ #
# workload_artifact_replace  (read / create / cancel)                      #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_workload_replacement_read(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.return_value = MagicMock(
        json=lambda: {"candidateArtifactId": "art-new", "strategy": "rolling"}
    )

    result = await workload_runtime.workload_artifact_replace(workload_id="wkld-abc")

    patched_dr_client.get.assert_called_once_with("workloads/wkld-abc/replacement")
    assert result["candidateArtifactId"] == "art-new"


@pytest.mark.asyncio
async def test_workload_replacement_create(patched_dr_client: MagicMock) -> None:
    patched_dr_client.post.return_value = MagicMock(
        json=lambda: {"candidateArtifactId": "art-xyz", "strategy": "rolling"}
    )

    result = await workload_runtime.workload_artifact_replace(
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

    await workload_runtime.workload_artifact_replace(
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
    runtime = {"containerGroups": [{"resourceBundles": ["bundle-1"]}]}

    await workload_runtime.workload_artifact_replace(
        workload_id="wkld-abc", artifact_id="art-xyz", runtime=runtime
    )

    assert patched_dr_client.post.call_args[1]["json"]["runtime"] == runtime


@pytest.mark.asyncio
async def test_workload_replacement_cancel(patched_dr_client: MagicMock) -> None:
    patched_dr_client.delete.return_value = MagicMock(
        content=b'{"status": "cancelled"}', json=lambda: {"status": "cancelled"}
    )

    result = await workload_runtime.workload_artifact_replace(workload_id="wkld-abc", cancel=True)

    patched_dr_client.delete.assert_called_once_with("workloads/wkld-abc/replacement")
    assert result["status"] == "cancelled"


@pytest.mark.asyncio
async def test_workload_replacement_create_and_cancel_conflict_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await workload_runtime.workload_artifact_replace(
            workload_id="wkld-abc", artifact_id="art-xyz", cancel=True
        )
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_workload_replacement_empty_id_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await workload_runtime.workload_artifact_replace(workload_id="")
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_workload_replacement_invalid_strategy_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await workload_runtime.workload_artifact_replace(
            workload_id="wkld-abc", artifact_id="art-xyz", strategy="bluegreen"
        )
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_workload_replacement_negative_warmup_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await workload_runtime.workload_artifact_replace(
            workload_id="wkld-abc", artifact_id="art-xyz", warmup_duration_minutes=-1
        )
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_workload_replacement_missing_container_groups_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await workload_runtime.workload_artifact_replace(
            workload_id="wkld-abc", artifact_id="art-xyz", runtime={"replicaCount": 2}
        )
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_workload_replacement_create_client_error(patched_dr_client: MagicMock) -> None:
    patched_dr_client.post.side_effect = ClientError("400", status_code=400, json={})
    with pytest.raises(ToolError) as exc_info:
        await workload_runtime.workload_artifact_replace(
            workload_id="wkld-abc", artifact_id="art-xyz"
        )
    assert exc_info.value.kind is ToolErrorKind.UPSTREAM


# ================================================================== #
# workload_observability — stats / logs / activity / workload_proton_get        #
# ================================================================== #

# ------------------------------------------------------------------ #
# workload_stats_get                                                       #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_workload_stats_success(patched_dr_client: MagicMock) -> None:
    stats = {"requestCount": 100, "errorRate": 0.01}
    patched_dr_client.get.return_value = MagicMock(json=lambda: stats)

    result = await workload_observability.workload_stats_get(workload_id="wkld-abc")

    patched_dr_client.get.assert_called_once_with(
        "workloads/wkld-abc/stats",
        params={"responseTimeQuantile": 0.5, "slowRequestsThreshold": 2000},
    )
    assert result == stats


@pytest.mark.asyncio
async def test_workload_stats_with_options(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.return_value = MagicMock(json=lambda: {})

    await workload_observability.workload_stats_get(
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
        await workload_observability.workload_stats_get(
            workload_id="wkld-abc", response_time_quantile=1.5
        )
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


# ------------------------------------------------------------------ #
# workload_logs_get                                                        #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_workload_logs_success(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.return_value = MagicMock(
        json=lambda: {"data": [{"body": "Server started"}], "count": 1, "totalCount": 1}
    )

    result = await workload_observability.workload_logs_get(workload_id="wkld-abc")

    patched_dr_client.get.assert_called_once_with(
        "otel/workload/wkld-abc/logs/",
        params={"limit": 100, "offset": 0, "level": "debug"},
    )
    assert result["logs"][0]["body"] == "Server started"


@pytest.mark.asyncio
async def test_workload_logs_with_filters(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.return_value = MagicMock(json=lambda: {"data": [], "count": 0})

    await workload_observability.workload_logs_get(
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
        params={
            "limit": 100,
            "offset": 0,
            "level": "error",
            "startTime": "2026-01-01T00:00:00Z",
            "endTime": "2026-01-02T00:00:00Z",
            "includes": ["ERROR", "FATAL"],
            "excludes": ["healthcheck"],
            "spanId": "span-1",
            "traceId": "trace-1",
        },
    )


@pytest.mark.asyncio
async def test_workload_logs_params_pass_sdk_to_api_validation() -> None:
    """Regression: the DataRobot SDK's REST client runs every ``params``
    through ``to_api()``, which asserts ``isinstance(data, dict)`` and raises
    ``AssertionError: Wrong type`` otherwise. list_workload_logs used to build params as a
    list[tuple[str, Any]], which fails that assertion unconditionally. patched_dr_client
    mocks client.get() directly and would not have caught this, since it never calls the
    real to_api(); this test runs params through the real SDK function instead.
    """
    from datarobot.utils import to_api

    captured: dict[str, Any] = {}

    class _ToApiValidatingClient:
        def get(self, url: str, params: Any = None, **kwargs: Any) -> MagicMock:
            captured["url"] = url
            captured["params"] = to_api(params)
            return MagicMock(json=lambda: {"data": [], "count": 0})

    with patch(
        "datarobot_genai.drtools.core.clients.datarobot_workload.request_user_dr_client"
    ) as mock_cm:
        mock_cm.return_value.__enter__.return_value = _ToApiValidatingClient()
        mock_cm.return_value.__exit__.return_value = False

        await workload_observability.workload_logs_get(
            workload_id="wkld-abc",
            includes=["ERROR", "FATAL"],
            excludes=["healthcheck"],
        )

    assert captured["url"] == "otel/workload/wkld-abc/logs/"
    assert captured["params"]["includes"] == ["ERROR", "FATAL"]
    assert captured["params"]["excludes"] == ["healthcheck"]


@pytest.mark.asyncio
async def test_workload_logs_invalid_level_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await workload_observability.workload_logs_get(workload_id="wkld-abc", level="verbose")
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_workload_logs_negative_offset_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await workload_observability.workload_logs_get(workload_id="wkld-abc", offset=-1)
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


# ------------------------------------------------------------------ #
# workload_activity_get  (history / events / related)                     #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_workload_activity_history(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.return_value = MagicMock(
        json=lambda: {"data": [{"artifactId": "art-1"}], "count": 1, "totalCount": 1}
    )

    result = await workload_observability.workload_activity_get(
        workload_id="wkld-abc", kind="history"
    )

    patched_dr_client.get.assert_called_once_with(
        "workloads/wkld-abc/history", params={"limit": 20, "offset": 0}
    )
    assert result["history"][0]["artifactId"] == "art-1"


@pytest.mark.asyncio
async def test_workload_activity_events(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.return_value = MagicMock(
        json=lambda: {"data": [{"type": "status_change"}], "count": 1, "totalCount": 1}
    )

    result = await workload_observability.workload_activity_get(
        workload_id="wkld-abc", kind="events"
    )

    patched_dr_client.get.assert_called_once_with(
        "workloads/wkld-abc/events", params={"limit": 20, "offset": 0}
    )
    assert result["events"][0]["type"] == "status_change"


@pytest.mark.asyncio
async def test_workload_activity_related(patched_dr_client: MagicMock) -> None:
    payload = {"artifacts": [{"id": "art-1"}]}
    patched_dr_client.get.return_value = MagicMock(json=lambda: payload)

    result = await workload_observability.workload_activity_get(
        workload_id="wkld-abc", kind="related"
    )

    patched_dr_client.get.assert_called_once_with("workloads/wkld-abc/related")
    assert result == payload


@pytest.mark.asyncio
async def test_workload_activity_negative_offset_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await workload_observability.workload_activity_get(
            workload_id="wkld-abc", kind="history", offset=-1
        )
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_workload_activity_empty_id_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await workload_observability.workload_activity_get(workload_id="", kind="events")
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_workload_activity_events_client_error(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.side_effect = ClientError("500", status_code=500, json={})
    with pytest.raises(ToolError) as exc_info:
        await workload_observability.workload_activity_get(workload_id="wkld-abc", kind="events")
    assert exc_info.value.kind is ToolErrorKind.UPSTREAM


# ------------------------------------------------------------------ #
# workload_proton_get  (list / single / status details)                       #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_proton_get_list(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.return_value = MagicMock(
        json=lambda: {
            "data": [{"id": "ptn-1", "status": "running"}],
            "count": 1,
            "totalCount": 1,
        }
    )

    result = await workload_observability.workload_proton_get(workload_id="wkld-abc")

    patched_dr_client.get.assert_called_once_with(
        "workloads/wkld-abc/protons", params={"limit": 20, "offset": 0}
    )
    assert result["protons"][0]["id"] == "ptn-1"


@pytest.mark.asyncio
async def test_proton_get_single(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.return_value = MagicMock(
        json=lambda: {"id": "ptn-1", "status": "running"}
    )

    result = await workload_observability.workload_proton_get(
        workload_id="wkld-abc", proton_id="ptn-1"
    )

    patched_dr_client.get.assert_called_once_with("workloads/wkld-abc/protons/ptn-1")
    assert result["id"] == "ptn-1"


@pytest.mark.asyncio
async def test_proton_get_with_status_details(patched_dr_client: MagicMock) -> None:
    snapshot = {"replicas": [{"name": "pod-0", "phase": "Running", "ready": True}]}
    patched_dr_client.get.side_effect = [
        MagicMock(json=lambda: {"id": "ptn-1", "status": "running"}),
        MagicMock(content=b'{"replicas": []}', json=lambda: snapshot),
    ]

    result = await workload_observability.workload_proton_get(
        workload_id="wkld-abc", proton_id="ptn-1", include_status_details=True
    )

    assert patched_dr_client.get.call_args_list[0][0][0] == "workloads/wkld-abc/protons/ptn-1"
    assert (
        patched_dr_client.get.call_args_list[1][0][0]
        == "workloads/wkld-abc/protons/ptn-1/statusDetails"
    )
    assert result["status_details"] == snapshot


@pytest.mark.asyncio
async def test_proton_get_status_details_pending(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.side_effect = [
        MagicMock(json=lambda: {"id": "ptn-1"}),
        MagicMock(content=b"", json=lambda: None),
    ]

    result = await workload_observability.workload_proton_get(
        workload_id="wkld-abc", proton_id="ptn-1", include_status_details=True
    )

    assert result["status_details"] is None


@pytest.mark.asyncio
async def test_proton_get_empty_workload_id_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await workload_observability.workload_proton_get(workload_id="")
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_proton_get_blank_proton_id_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await workload_observability.workload_proton_get(workload_id="wkld-abc", proton_id="   ")
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_proton_get_list_negative_offset_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await workload_observability.workload_proton_get(workload_id="wkld-abc", offset=-1)
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


# ================================================================== #
# artifacts — get / create / update / action                          #
# ================================================================== #

# ------------------------------------------------------------------ #
# artifact_get  (list when no id, single when id)                     #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_artifact_get_list(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.return_value = MagicMock(
        json=lambda: {
            "data": [{"id": "art-1", "status": "draft"}],
            "count": 1,
            "totalCount": 1,
        }
    )

    result = await artifacts.artifact_get()

    patched_dr_client.get.assert_called_once_with("artifacts/", params={"limit": 100, "offset": 0})
    assert result["artifacts"][0]["id"] == "art-1"


@pytest.mark.asyncio
async def test_artifact_get_list_with_filters(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.return_value = MagicMock(json=lambda: {"data": [], "count": 0})

    await artifacts.artifact_get(
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
async def test_artifact_get_single(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.return_value = MagicMock(
        json=lambda: {"id": "art-abc", "status": "draft"}
    )

    result = await artifacts.artifact_get(artifact_id="art-abc")

    patched_dr_client.get.assert_called_once_with("artifacts/art-abc")
    assert result["id"] == "art-abc"


@pytest.mark.asyncio
async def test_artifact_get_blank_id_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await artifacts.artifact_get(artifact_id="   ")
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_artifact_get_invalid_status_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await artifacts.artifact_get(status="pending")
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_artifact_get_invalid_type_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await artifacts.artifact_get(artifact_type="docker")
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_artifact_get_negative_offset_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await artifacts.artifact_get(offset=-1)
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_artifact_get_single_not_found(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.side_effect = ClientError("404", status_code=404, json={})
    with pytest.raises(ToolError) as exc_info:
        await artifacts.artifact_get(artifact_id="art-missing")
    assert exc_info.value.kind is ToolErrorKind.NOT_FOUND


# ------------------------------------------------------------------ #
# artifact_create                                                     #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_artifact_create_success(patched_dr_client: MagicMock) -> None:
    payload = {
        "name": "my-svc",
        "spec": {"type": "service", "containerGroups": [{"containers": []}]},
    }
    patched_dr_client.post.return_value = MagicMock(
        json=lambda: {"id": "art-new", "status": "draft"}
    )

    result = await artifacts.artifact_create(payload=payload)

    patched_dr_client.post.assert_called_once_with("artifacts/", json=payload)
    assert result["id"] == "art-new"


@pytest.mark.asyncio
async def test_artifact_create_missing_name_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await artifacts.artifact_create(payload={"spec": {"type": "service"}})
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_artifact_create_missing_spec_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await artifacts.artifact_create(payload={"name": "my-svc"})
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_artifact_create_client_error(patched_dr_client: MagicMock) -> None:
    patched_dr_client.post.side_effect = ClientError("422", status_code=422, json={})
    with pytest.raises(ToolError) as exc_info:
        await artifacts.artifact_create(payload={"name": "x", "spec": {"type": "service"}})
    assert exc_info.value.kind is ToolErrorKind.UPSTREAM


# ------------------------------------------------------------------ #
# artifact_update                                                     #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_artifact_update_name(patched_dr_client: MagicMock) -> None:
    patched_dr_client.patch.return_value = MagicMock(
        json=lambda: {"id": "art-abc", "name": "new-name"}
    )

    result = await artifacts.artifact_update(artifact_id="art-abc", name="new-name")

    patched_dr_client.patch.assert_called_once_with("artifacts/art-abc", json={"name": "new-name"})
    assert result["name"] == "new-name"


@pytest.mark.asyncio
async def test_artifact_update_multiple_fields(patched_dr_client: MagicMock) -> None:
    patched_dr_client.patch.return_value = MagicMock(json=lambda: {"id": "art-abc"})

    await artifacts.artifact_update(artifact_id="art-abc", name="x", description="desc")

    patched_dr_client.patch.assert_called_once_with(
        "artifacts/art-abc", json={"name": "x", "description": "desc"}
    )


@pytest.mark.asyncio
async def test_artifact_update_no_fields_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await artifacts.artifact_update(artifact_id="art-abc")
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


# ------------------------------------------------------------------ #
# artifact_action_run  (lock / clone / delete)                            #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_artifact_action_lock(patched_dr_client: MagicMock) -> None:
    patched_dr_client.patch.return_value = MagicMock(
        json=lambda: {"id": "art-abc", "status": "locked"}
    )

    result = await artifacts.artifact_action_run(artifact_id="art-abc", action="lock")

    patched_dr_client.patch.assert_called_once_with("artifacts/art-abc", json={"status": "locked"})
    assert result["status"] == "locked"


@pytest.mark.asyncio
async def test_artifact_action_clone(patched_dr_client: MagicMock) -> None:
    patched_dr_client.post.return_value = MagicMock(
        json=lambda: {"id": "art-new", "name": "svc-v2", "status": "draft"}
    )

    result = await artifacts.artifact_action_run(
        artifact_id="art-abc", action="clone", name="svc-v2"
    )

    patched_dr_client.post.assert_called_once_with(
        "artifacts/art-abc/clone", json={"name": "svc-v2"}
    )
    assert result["name"] == "svc-v2"


@pytest.mark.asyncio
async def test_artifact_action_clone_missing_name_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await artifacts.artifact_action_run(artifact_id="art-abc", action="clone")
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_artifact_action_delete(patched_dr_client: MagicMock) -> None:
    patched_dr_client.delete.return_value = MagicMock()

    result = await artifacts.artifact_action_run(artifact_id="art-abc", action="delete")

    patched_dr_client.delete.assert_called_once_with("artifacts/art-abc")
    assert result == {"deleted": True, "artifact_id": "art-abc"}


@pytest.mark.asyncio
async def test_artifact_action_empty_id_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await artifacts.artifact_action_run(artifact_id="", action="lock")
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_artifact_action_delete_conflict(patched_dr_client: MagicMock) -> None:
    patched_dr_client.delete.side_effect = ClientError("409 Conflict", status_code=409, json={})
    with pytest.raises(ToolError) as exc_info:
        await artifacts.artifact_action_run(artifact_id="art-locked", action="delete")
    assert exc_info.value.kind is ToolErrorKind.UPSTREAM


# ================================================================== #
# artifact_builds — get / action                                      #
# ================================================================== #


@pytest.mark.asyncio
async def test_artifact_build_get_list(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.return_value = MagicMock(
        json=lambda: {
            "data": [{"id": "bld-1", "status": "success"}],
            "count": 1,
            "totalCount": 1,
        }
    )

    result = await artifact_builds.artifact_get_build(artifact_id="art-abc")

    patched_dr_client.get.assert_called_once_with(
        "artifacts/art-abc/builds", params={"limit": 100, "offset": 0}
    )
    assert result["builds"][0]["id"] == "bld-1"


@pytest.mark.asyncio
async def test_artifact_build_get_single(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.return_value = MagicMock(
        json=lambda: {"id": "bld-xyz", "status": "success"}
    )

    result = await artifact_builds.artifact_get_build(artifact_id="art-abc", build_id="bld-xyz")

    patched_dr_client.get.assert_called_once_with("artifacts/art-abc/builds/bld-xyz")
    assert result["id"] == "bld-xyz"


@pytest.mark.asyncio
async def test_artifact_build_get_with_logs(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.side_effect = [
        MagicMock(json=lambda: {"id": "bld-xyz", "status": "success"}),
        MagicMock(text="Step 1/3: FROM python:3.11\n"),
    ]

    result = await artifact_builds.artifact_get_build(
        artifact_id="art-abc", build_id="bld-xyz", include_logs=True
    )

    assert patched_dr_client.get.call_args_list[1][0][0] == "artifacts/art-abc/builds/bld-xyz/logs"
    assert result["logs"] == "Step 1/3: FROM python:3.11\n"


@pytest.mark.asyncio
async def test_artifact_build_get_empty_artifact_id_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await artifact_builds.artifact_get_build(artifact_id="")
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_artifact_build_get_blank_build_id_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await artifact_builds.artifact_get_build(artifact_id="art-abc", build_id="   ")
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_artifact_build_get_list_not_found(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.side_effect = ClientError("404", status_code=404, json={})
    with pytest.raises(ToolError) as exc_info:
        await artifact_builds.artifact_get_build(artifact_id="art-missing")
    assert exc_info.value.kind is ToolErrorKind.NOT_FOUND


@pytest.mark.asyncio
async def test_artifact_build_action_trigger(patched_dr_client: MagicMock) -> None:
    patched_dr_client.post.return_value = MagicMock(json=lambda: {"buildIds": ["bld-new"]})

    result = await artifact_builds.artifact_build_run_action(
        artifact_id="art-abc", action="trigger"
    )

    patched_dr_client.post.assert_called_once_with("artifacts/art-abc/builds")
    assert result["buildIds"] == ["bld-new"]


@pytest.mark.asyncio
async def test_artifact_build_action_trigger_locked_error(patched_dr_client: MagicMock) -> None:
    patched_dr_client.post.side_effect = ClientError("422", status_code=422, json={})
    with pytest.raises(ToolError) as exc_info:
        await artifact_builds.artifact_build_run_action(artifact_id="art-locked", action="trigger")
    assert exc_info.value.kind is ToolErrorKind.UPSTREAM


@pytest.mark.asyncio
async def test_artifact_build_action_delete(patched_dr_client: MagicMock) -> None:
    patched_dr_client.delete.return_value = MagicMock()

    result = await artifact_builds.artifact_build_run_action(
        artifact_id="art-abc", action="delete", build_id="bld-xyz"
    )

    patched_dr_client.delete.assert_called_once_with("artifacts/art-abc/builds/bld-xyz")
    assert result == {"deleted": True, "artifact_id": "art-abc", "build_id": "bld-xyz"}


@pytest.mark.asyncio
async def test_artifact_build_action_delete_missing_build_id_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await artifact_builds.artifact_build_run_action(artifact_id="art-abc", action="delete")
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


# ================================================================== #
# artifact_repositories — get / delete                                #
# ================================================================== #


@pytest.mark.asyncio
async def test_artifact_repository_get_list(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.return_value = MagicMock(
        json=lambda: {"data": [{"id": "repo-1"}], "count": 1, "totalCount": 1}
    )

    result = await artifact_repositories.artifact_repository_get()

    patched_dr_client.get.assert_called_once_with(
        "artifactRepositories", params={"limit": 100, "offset": 0}
    )
    assert result["repositories"][0]["id"] == "repo-1"


@pytest.mark.asyncio
async def test_artifact_repository_get_list_with_filters(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.return_value = MagicMock(json=lambda: {"data": [], "count": 0})

    await artifact_repositories.artifact_repository_get(search="ecr", artifact_type="nim")

    patched_dr_client.get.assert_called_once_with(
        "artifactRepositories",
        params={"limit": 100, "offset": 0, "search": "ecr", "type": "nim"},
    )


@pytest.mark.asyncio
async def test_artifact_repository_get_single(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.return_value = MagicMock(
        json=lambda: {"id": "repo-abc", "name": "my-registry"}
    )

    result = await artifact_repositories.artifact_repository_get(repository_id="repo-abc")

    patched_dr_client.get.assert_called_once_with("artifactRepositories/repo-abc")
    assert result["id"] == "repo-abc"


@pytest.mark.asyncio
async def test_artifact_repository_get_blank_id_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await artifact_repositories.artifact_repository_get(repository_id="   ")
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_artifact_repository_get_invalid_type_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await artifact_repositories.artifact_repository_get(artifact_type="docker")
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_artifact_repository_get_single_not_found(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.side_effect = ClientError("404", status_code=404, json={})
    with pytest.raises(ToolError) as exc_info:
        await artifact_repositories.artifact_repository_get(repository_id="repo-missing")
    assert exc_info.value.kind is ToolErrorKind.NOT_FOUND


@pytest.mark.asyncio
async def test_artifact_repository_delete_success(patched_dr_client: MagicMock) -> None:
    patched_dr_client.delete.return_value = MagicMock()

    result = await artifact_repositories.artifact_repository_delete(repository_id="repo-abc")

    patched_dr_client.delete.assert_called_once_with("artifactRepositories/repo-abc")
    assert result == {"deleted": True, "repository_id": "repo-abc"}


@pytest.mark.asyncio
async def test_artifact_repository_delete_empty_id_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await artifact_repositories.artifact_repository_delete(repository_id="")
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_artifact_repository_delete_conflict(patched_dr_client: MagicMock) -> None:
    patched_dr_client.delete.side_effect = ClientError("409 Conflict", status_code=409, json={})
    with pytest.raises(ToolError) as exc_info:
        await artifact_repositories.artifact_repository_delete(repository_id="repo-in-use")
    assert exc_info.value.kind is ToolErrorKind.UPSTREAM
