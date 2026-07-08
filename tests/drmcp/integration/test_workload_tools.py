# Copyright 2026 DataRobot, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Integration tests for DataRobot Workload API MCP tools."""

import json

import pytest
from mcp.types import TextContent

from datarobot_genai.drmcp.test_utils.mcp_utils_integration import integration_test_mcp_session
from datarobot_genai.drmcp.test_utils.mcp_utils_integration import (
    integration_test_server_params_with_env,
)
from datarobot_genai.drmcp.test_utils.stubs.workload_stubs import STUB_ARTIFACT_ID
from datarobot_genai.drmcp.test_utils.stubs.workload_stubs import STUB_BUILD_ID
from datarobot_genai.drmcp.test_utils.stubs.workload_stubs import STUB_BUNDLE_ID
from datarobot_genai.drmcp.test_utils.stubs.workload_stubs import STUB_PROTON_ID
from datarobot_genai.drmcp.test_utils.stubs.workload_stubs import STUB_REPOSITORY_ID
from datarobot_genai.drmcp.test_utils.stubs.workload_stubs import STUB_WORKLOAD_ID

_EXPECTED_TOOLS = frozenset(
    {
        "workload_list",
        "workload_get",
        "workload_create_payload_build",
        "workload_create",
        "workload_update",
        "workload_action_run",
        "workload_bundle_list",
        "artifact_get",
        "artifact_create",
        "artifact_update",
        "artifact_action_run",
        "artifact_repository_get",
        "artifact_repository_delete",
        "artifact_get_build",
        "artifact_build_run_action",
        "workload_settings",
        "workload_artifact_replace",
        "workload_stats_get",
        "workload_logs_get",
        "workload_activity_get",
        "workload_proton_get",
    }
)


def _workload_server_params():
    """Return server params with workload tools enabled."""
    return integration_test_server_params_with_env({"ENABLE_WORKLOAD_TOOLS": "true"})


def _parse_result(result: object) -> dict:
    assert not getattr(result, "isError", True)
    content = getattr(result, "content", [])
    assert len(content) > 0
    assert isinstance(content[0], TextContent)
    return json.loads(content[0].text)


@pytest.mark.asyncio
class TestMCPWorkloadToolsRegistration:
    """Verify workload tools are registered in the MCP server."""

    async def test_tools_registered(self) -> None:
        async with integration_test_mcp_session(server_params=_workload_server_params()) as session:
            result = await session.list_tools()
            tool_names = {t.name for t in result.tools}
            missing = _EXPECTED_TOOLS - tool_names
            assert not missing, f"workload tools not registered: {missing}"


@pytest.mark.asyncio
class TestMCPWorkloadDiscoveryIntegration:
    """Integration tests for workload_list, workload_get, and workload_bundle_list."""

    async def test_workload_list_returns_workloads(self) -> None:
        async with integration_test_mcp_session(server_params=_workload_server_params()) as session:
            data = _parse_result(await session.call_tool("workload_list", {}))
            assert "workloads" in data
            assert data["count"] >= 1
            workload_ids = [w["id"] for w in data["workloads"]]
            assert STUB_WORKLOAD_ID in workload_ids

    async def test_workload_list_with_search(self) -> None:
        async with integration_test_mcp_session(server_params=_workload_server_params()) as session:
            data = _parse_result(
                await session.call_tool("workload_list", {"search": "Alpha", "limit": 10})
            )
            assert data["count"] >= 1
            for workload in data["workloads"]:
                assert "alpha" in workload["name"].lower()

    async def test_workload_get_success(self) -> None:
        async with integration_test_mcp_session(server_params=_workload_server_params()) as session:
            data = _parse_result(
                await session.call_tool("workload_get", {"workload_id": STUB_WORKLOAD_ID})
            )
            assert data["id"] == STUB_WORKLOAD_ID
            assert data["status"] == "running"

    async def test_workload_get_target_status(self) -> None:
        async with integration_test_mcp_session(server_params=_workload_server_params()) as session:
            data = _parse_result(
                await session.call_tool(
                    "workload_get",
                    {"workload_id": STUB_WORKLOAD_ID, "target_status": "running"},
                )
            )
            assert data["target_reached"] is True
            assert data["status"] == "running"

    async def test_workload_get_missing_id(self) -> None:
        async with integration_test_mcp_session(server_params=_workload_server_params()) as session:
            result = await session.call_tool("workload_get", {})
            assert result.isError

    async def test_bundle_list_returns_bundles(self) -> None:
        async with integration_test_mcp_session(server_params=_workload_server_params()) as session:
            data = _parse_result(await session.call_tool("workload_bundle_list", {}))
            assert "data" in data
            bundle_ids = [b["id"] for b in data["data"]]
            assert STUB_BUNDLE_ID in bundle_ids


@pytest.mark.asyncio
class TestMCPWorkloadCreateIntegration:
    """Integration tests for workload_create_payload_build and workload_create."""

    async def test_workload_create_payload_existing_artifact(self) -> None:
        async with integration_test_mcp_session(server_params=_workload_server_params()) as session:
            data = _parse_result(
                await session.call_tool(
                    "workload_create_payload_build",
                    {"name": "integration-wl", "artifact_id": STUB_ARTIFACT_ID},
                )
            )
            assert "payload" in data
            assert data["payload"]["artifactId"] == STUB_ARTIFACT_ID

    async def test_workload_create_payload_inline_artifact(self) -> None:
        async with integration_test_mcp_session(server_params=_workload_server_params()) as session:
            data = _parse_result(
                await session.call_tool(
                    "workload_create_payload_build",
                    {
                        "name": "echo-wl",
                        "artifact_name": "echo",
                        "image_uri": "hashicorp/http-echo:0.2.3",
                        "port": 8080,
                        "cpu": 1,
                        "memory_bytes": 134217728,
                    },
                )
            )
            container = data["payload"]["artifact"]["spec"]["containerGroups"][0]["containers"][0]
            assert container["imageUri"] == "hashicorp/http-echo:0.2.3"

    async def test_workload_create_success(self) -> None:
        async with integration_test_mcp_session(server_params=_workload_server_params()) as session:
            payload_data = _parse_result(
                await session.call_tool(
                    "workload_create_payload_build",
                    {"name": "new-wl", "artifact_id": STUB_ARTIFACT_ID},
                )
            )
            data = _parse_result(
                await session.call_tool("workload_create", {"payload": payload_data["payload"]})
            )
            assert data["id"] == STUB_WORKLOAD_ID
            assert data["status"] == "initializing"


@pytest.mark.asyncio
class TestMCPWorkloadLifecycleIntegration:
    """Integration tests for workload_update and workload_action_run."""

    async def test_workload_update_name(self) -> None:
        async with integration_test_mcp_session(server_params=_workload_server_params()) as session:
            data = _parse_result(
                await session.call_tool(
                    "workload_update",
                    {"workload_id": STUB_WORKLOAD_ID, "name": "renamed-workload"},
                )
            )
            assert data["name"] == "renamed-workload"

    async def test_workload_action_start(self) -> None:
        async with integration_test_mcp_session(server_params=_workload_server_params()) as session:
            data = _parse_result(
                await session.call_tool(
                    "workload_action_run",
                    {"workload_id": STUB_WORKLOAD_ID, "action": "start"},
                )
            )
            assert data["workload_id"] == STUB_WORKLOAD_ID
            assert "note" in data
            assert "workload_get(" in data["note"]

    async def test_workload_action_delete(self) -> None:
        async with integration_test_mcp_session(server_params=_workload_server_params()) as session:
            data = _parse_result(
                await session.call_tool(
                    "workload_action_run",
                    {"workload_id": STUB_WORKLOAD_ID, "action": "delete"},
                )
            )
            assert data == {"deleted": True, "workload_id": STUB_WORKLOAD_ID}


@pytest.mark.asyncio
class TestMCPArtifactToolsIntegration:
    """Integration tests for artifact_get, artifact_create, artifact_update, artifact_action_run."""

    async def test_artifact_get_list(self) -> None:
        async with integration_test_mcp_session(server_params=_workload_server_params()) as session:
            data = _parse_result(await session.call_tool("artifact_get", {}))
            assert "artifacts" in data
            artifact_ids = [a["id"] for a in data["artifacts"]]
            assert STUB_ARTIFACT_ID in artifact_ids

    async def test_artifact_get_single(self) -> None:
        async with integration_test_mcp_session(server_params=_workload_server_params()) as session:
            data = _parse_result(
                await session.call_tool("artifact_get", {"artifact_id": STUB_ARTIFACT_ID})
            )
            assert data["id"] == STUB_ARTIFACT_ID
            assert data["status"] == "draft"

    async def test_artifact_create_success(self) -> None:
        async with integration_test_mcp_session(server_params=_workload_server_params()) as session:
            data = _parse_result(
                await session.call_tool(
                    "artifact_create",
                    {
                        "payload": {
                            "name": "integration-artifact",
                            "spec": {
                                "type": "service",
                                "containerGroups": [
                                    {
                                        "containers": [
                                            {
                                                "name": "main",
                                                "imageUri": "nginx:latest",
                                                "primary": True,
                                                "port": 8080,
                                            }
                                        ]
                                    }
                                ],
                            },
                        }
                    },
                )
            )
            assert data["status"] == "draft"
            assert "id" in data

    async def test_artifact_update_name(self) -> None:
        async with integration_test_mcp_session(server_params=_workload_server_params()) as session:
            data = _parse_result(
                await session.call_tool(
                    "artifact_update",
                    {"artifact_id": STUB_ARTIFACT_ID, "name": "updated-artifact"},
                )
            )
            assert data["name"] == "updated-artifact"

    async def test_artifact_action_lock(self) -> None:
        async with integration_test_mcp_session(server_params=_workload_server_params()) as session:
            data = _parse_result(
                await session.call_tool(
                    "artifact_action_run",
                    {"artifact_id": STUB_ARTIFACT_ID, "action": "lock"},
                )
            )
            assert data["status"] == "locked"

    async def test_artifact_action_clone(self) -> None:
        async with integration_test_mcp_session(server_params=_workload_server_params()) as session:
            data = _parse_result(
                await session.call_tool(
                    "artifact_action_run",
                    {
                        "artifact_id": STUB_ARTIFACT_ID,
                        "action": "clone",
                        "name": "artifact-clone",
                    },
                )
            )
            assert data["name"] == "artifact-clone"
            assert data["status"] == "draft"


@pytest.mark.asyncio
class TestMCPArtifactRepositoryToolsIntegration:
    """Integration tests for artifact_repository_get and artifact_repository_delete."""

    async def test_artifact_repository_get_list(self) -> None:
        async with integration_test_mcp_session(server_params=_workload_server_params()) as session:
            data = _parse_result(await session.call_tool("artifact_repository_get", {}))
            assert "repositories" in data
            repo_ids = [r["id"] for r in data["repositories"]]
            assert STUB_REPOSITORY_ID in repo_ids

    async def test_artifact_repository_get_single(self) -> None:
        async with integration_test_mcp_session(server_params=_workload_server_params()) as session:
            data = _parse_result(
                await session.call_tool(
                    "artifact_repository_get", {"repository_id": STUB_REPOSITORY_ID}
                )
            )
            assert data["id"] == STUB_REPOSITORY_ID

    async def test_artifact_repository_delete(self) -> None:
        async with integration_test_mcp_session(server_params=_workload_server_params()) as session:
            data = _parse_result(
                await session.call_tool(
                    "artifact_repository_delete", {"repository_id": STUB_REPOSITORY_ID}
                )
            )
            assert data == {"deleted": True, "repository_id": STUB_REPOSITORY_ID}


@pytest.mark.asyncio
class TestMCPArtifactBuildToolsIntegration:
    """Integration tests for artifact_get_build and artifact_build_run_action."""

    async def test_artifact_build_get_list(self) -> None:
        async with integration_test_mcp_session(server_params=_workload_server_params()) as session:
            data = _parse_result(
                await session.call_tool("artifact_get_build", {"artifact_id": STUB_ARTIFACT_ID})
            )
            assert "builds" in data
            build_ids = [b["id"] for b in data["builds"]]
            assert STUB_BUILD_ID in build_ids

    async def test_artifact_build_get_with_logs(self) -> None:
        async with integration_test_mcp_session(server_params=_workload_server_params()) as session:
            data = _parse_result(
                await session.call_tool(
                    "artifact_get_build",
                    {
                        "artifact_id": STUB_ARTIFACT_ID,
                        "build_id": STUB_BUILD_ID,
                        "include_logs": True,
                    },
                )
            )
            assert data["id"] == STUB_BUILD_ID
            assert "logs" in data
            assert "FROM python" in data["logs"]

    async def test_artifact_build_action_trigger(self) -> None:
        async with integration_test_mcp_session(server_params=_workload_server_params()) as session:
            data = _parse_result(
                await session.call_tool(
                    "artifact_build_run_action",
                    {"artifact_id": STUB_ARTIFACT_ID, "action": "trigger"},
                )
            )
            assert STUB_BUILD_ID in data["buildIds"]


@pytest.mark.asyncio
class TestMCPWorkloadRuntimeToolsIntegration:
    """Integration tests for workload_settings and workload_artifact_replace."""

    async def test_workload_settings_read(self) -> None:
        async with integration_test_mcp_session(server_params=_workload_server_params()) as session:
            data = _parse_result(
                await session.call_tool("workload_settings", {"workload_id": STUB_WORKLOAD_ID})
            )
            assert "runtime" in data
            assert data["runtime"]["containerGroups"][0]["replicaCount"] == 1

    async def test_workload_settings_update(self) -> None:
        async with integration_test_mcp_session(server_params=_workload_server_params()) as session:
            runtime = {"containerGroups": [{"name": "default", "replicaCount": 2}]}
            data = _parse_result(
                await session.call_tool(
                    "workload_settings",
                    {"workload_id": STUB_WORKLOAD_ID, "runtime": runtime},
                )
            )
            assert data["status"] == "in_progress"

    async def test_workload_replacement_read(self) -> None:
        async with integration_test_mcp_session(server_params=_workload_server_params()) as session:
            data = _parse_result(
                await session.call_tool(
                    "workload_artifact_replace", {"workload_id": STUB_WORKLOAD_ID}
                )
            )
            assert "candidateArtifactId" in data

    async def test_workload_replacement_create(self) -> None:
        async with integration_test_mcp_session(server_params=_workload_server_params()) as session:
            data = _parse_result(
                await session.call_tool(
                    "workload_artifact_replace",
                    {
                        "workload_id": STUB_WORKLOAD_ID,
                        "artifact_id": STUB_ARTIFACT_ID,
                    },
                )
            )
            assert data["candidateArtifactId"] == STUB_ARTIFACT_ID


@pytest.mark.asyncio
class TestMCPWorkloadObservabilityToolsIntegration:
    """Integration tests for stats, logs, activity, and workload_proton_get."""

    async def test_workload_stats(self) -> None:
        async with integration_test_mcp_session(server_params=_workload_server_params()) as session:
            data = _parse_result(
                await session.call_tool("workload_stats_get", {"workload_id": STUB_WORKLOAD_ID})
            )
            assert data["requestCount"] == 120
            assert "errorRate" in data

    async def test_workload_logs(self) -> None:
        async with integration_test_mcp_session(server_params=_workload_server_params()) as session:
            data = _parse_result(
                await session.call_tool("workload_logs_get", {"workload_id": STUB_WORKLOAD_ID})
            )
            assert "logs" in data
            assert data["count"] >= 1

    async def test_workload_activity_history(self) -> None:
        async with integration_test_mcp_session(server_params=_workload_server_params()) as session:
            data = _parse_result(
                await session.call_tool(
                    "workload_activity_get",
                    {"workload_id": STUB_WORKLOAD_ID, "kind": "history"},
                )
            )
            assert "history" in data
            assert data["history"][0]["artifactId"] == STUB_ARTIFACT_ID

    async def test_workload_activity_events(self) -> None:
        async with integration_test_mcp_session(server_params=_workload_server_params()) as session:
            data = _parse_result(
                await session.call_tool(
                    "workload_activity_get",
                    {"workload_id": STUB_WORKLOAD_ID, "kind": "events"},
                )
            )
            assert "events" in data

    async def test_workload_activity_related(self) -> None:
        async with integration_test_mcp_session(server_params=_workload_server_params()) as session:
            data = _parse_result(
                await session.call_tool(
                    "workload_activity_get",
                    {"workload_id": STUB_WORKLOAD_ID, "kind": "related"},
                )
            )
            assert "artifacts" in data

    async def test_proton_get_list(self) -> None:
        async with integration_test_mcp_session(server_params=_workload_server_params()) as session:
            data = _parse_result(
                await session.call_tool("workload_proton_get", {"workload_id": STUB_WORKLOAD_ID})
            )
            assert "protons" in data
            proton_ids = [p["id"] for p in data["protons"]]
            assert STUB_PROTON_ID in proton_ids

    async def test_proton_get_with_status_details(self) -> None:
        async with integration_test_mcp_session(server_params=_workload_server_params()) as session:
            data = _parse_result(
                await session.call_tool(
                    "workload_proton_get",
                    {
                        "workload_id": STUB_WORKLOAD_ID,
                        "proton_id": STUB_PROTON_ID,
                        "include_status_details": True,
                    },
                )
            )
            assert data["id"] == STUB_PROTON_ID
            assert "status_details" in data
            assert data["status_details"]["replicas"][0]["phase"] == "Running"
