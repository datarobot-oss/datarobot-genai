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
from unittest.mock import Mock
from unittest.mock import patch

import datarobot as dr
import pytest
from datarobot.errors import ClientError

from datarobot_genai.drtools.core.clients.datarobot import ThreadSafeDataRobotClient
from datarobot_genai.drtools.core.exceptions import ToolError
from datarobot_genai.drtools.core.exceptions import ToolErrorKind
from datarobot_genai.drtools.workload import tools

# ------------------------------------------------------------------ #
# Shared fixtures                                                       #
# ------------------------------------------------------------------ #


@pytest.fixture
def mock_request_user_client() -> Iterator[Mock]:
    with patch.object(ThreadSafeDataRobotClient, "request_user_client") as mock_cm:
        yield mock_cm


@pytest.fixture
def mock_rest_client() -> MagicMock:
    return MagicMock()


@pytest.fixture
def patched_dr_client(
    mock_request_user_client: Mock,  # noqa: ARG001 — activates the context manager patch
    mock_rest_client: MagicMock,
) -> Iterator[MagicMock]:
    with patch.object(dr.client, "get_client", return_value=mock_rest_client):
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

    result = await tools.workload_list()

    assert result["count"] == 2
    assert result["workloads"][0]["id"] == "wkld-1"
    assert result["workloads"][1]["name"] == "Beta"
    patched_dr_client.get.assert_called_once_with("workloads/", params={"limit": 100, "offset": 0})


@pytest.mark.asyncio
async def test_workload_list_with_search(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.return_value = MagicMock(
        json=lambda: {"data": [{"id": "wkld-1", "name": "my-app"}], "count": 1}
    )

    result = await tools.workload_list(search="my-app", limit=10, offset=5)

    patched_dr_client.get.assert_called_once_with(
        "workloads/", params={"limit": 10, "offset": 5, "search": "my-app"}
    )
    assert result["count"] == 1
    assert result["offset"] == 5
    assert result["limit"] == 10


@pytest.mark.asyncio
async def test_workload_list_clamps_limit(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.return_value = MagicMock(json=lambda: {"data": [], "count": 0})

    result = await tools.workload_list(limit=500)

    # limit clamped to 100
    patched_dr_client.get.assert_called_once_with("workloads/", params={"limit": 100, "offset": 0})
    assert result["limit"] == 100
    assert "note" in result


@pytest.mark.asyncio
async def test_workload_list_negative_offset_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await tools.workload_list(offset=-1)
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_workload_list_client_error(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.side_effect = ClientError(
        "500 Internal Server Error", status_code=500, json={}
    )

    with pytest.raises(ToolError) as exc_info:
        await tools.workload_list()
    assert exc_info.value.kind is ToolErrorKind.UPSTREAM


@pytest.mark.asyncio
async def test_workload_list_404_raises_not_found(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.side_effect = ClientError("404 Not Found", status_code=404, json={})

    with pytest.raises(ToolError) as exc_info:
        await tools.workload_list()
    assert exc_info.value.kind is ToolErrorKind.NOT_FOUND


# ------------------------------------------------------------------ #
# workload_search                                                       #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_workload_search_success(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.return_value = MagicMock(
        json=lambda: {
            "data": [{"id": "wkld-1", "name": "mcp-server"}],
            "count": 1,
        }
    )

    result = await tools.workload_search(query="mcp-server")

    patched_dr_client.get.assert_called_once_with(
        "workloads/", params={"limit": 20, "offset": 0, "search": "mcp-server"}
    )
    assert result["count"] == 1
    assert result["workloads"][0]["id"] == "wkld-1"


@pytest.mark.asyncio
async def test_workload_search_empty_query_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await tools.workload_search(query="")
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_workload_search_whitespace_query_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await tools.workload_search(query="   ")
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_workload_search_negative_offset_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await tools.workload_search(query="anything", offset=-5)
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_workload_search_client_error(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.side_effect = ClientError(
        "503 Service Unavailable", status_code=503, json={}
    )

    with pytest.raises(ToolError) as exc_info:
        await tools.workload_search(query="foo")
    assert exc_info.value.kind is ToolErrorKind.UPSTREAM


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

    result = await tools.workload_get(workload_id="wkld-abc")

    patched_dr_client.get.assert_called_once_with("workloads/wkld-abc")
    assert result["id"] == "wkld-abc"
    assert result["status"] == "running"


@pytest.mark.asyncio
async def test_workload_get_strips_whitespace(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.return_value = MagicMock(json=lambda: {"id": "wkld-abc"})

    await tools.workload_get(workload_id="  wkld-abc  ")

    patched_dr_client.get.assert_called_once_with("workloads/wkld-abc")


@pytest.mark.asyncio
async def test_workload_get_empty_id_raises() -> None:
    with pytest.raises(ToolError) as exc_info:
        await tools.workload_get(workload_id="")
    assert exc_info.value.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_workload_get_not_found(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.side_effect = ClientError("404 Not Found", status_code=404, json={})

    with pytest.raises(ToolError) as exc_info:
        await tools.workload_get(workload_id="wkld-missing")
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

    result = await tools.bundle_list()

    patched_dr_client.get.assert_called_once_with("mlops/compute/bundles")
    assert result == bundles_payload


@pytest.mark.asyncio
async def test_bundle_list_client_error(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.side_effect = ClientError(
        "500 Internal Server Error", status_code=500, json={}
    )

    with pytest.raises(ToolError) as exc_info:
        await tools.bundle_list()
    assert exc_info.value.kind is ToolErrorKind.UPSTREAM
