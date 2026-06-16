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

"""Unit tests for OpenAPI tools."""

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from datarobot.errors import ClientError

from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.exceptions import ToolErrorKind
from datarobot_genai.drtools.openapi import tools as openapi_tools


@pytest.fixture
def mock_rest_client() -> MagicMock:
    return MagicMock()


@pytest.fixture
def patched_dr_client(mock_rest_client: MagicMock) -> Iterator[MagicMock]:
    with patch("datarobot_genai.drtools.core.clients.datarobot.request_user_dr_client") as mock_cm:
        mock_cm.return_value.__enter__.return_value = mock_rest_client
        mock_cm.return_value.__exit__.return_value = False
        yield mock_rest_client


# ------------------------------------------------------------------ #
# read_openapi_spec                                                     #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_read_openapi_spec_remote_success(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.return_value = MagicMock(
        json=lambda: {"openapi": "3.1.0", "info": {"title": "DataRobot API"}, "paths": {}}
    )

    result = await openapi_tools.read_openapi_spec()

    patched_dr_client.get.assert_called_once_with("openapi.json")
    assert result["openapi"] == "3.1.0"


@pytest.mark.asyncio
async def test_read_openapi_spec_custom_remote_path(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.return_value = MagicMock(
        json=lambda: {"openapi": "3.1.0", "paths": {"/workloads/": {}}}
    )

    result = await openapi_tools.read_openapi_spec(remote_path="workload-openapi.json")

    patched_dr_client.get.assert_called_once_with("workload-openapi.json")
    assert "/workloads/" in result["paths"]


@pytest.mark.asyncio
async def test_read_openapi_spec_local_path_not_found() -> None:
    with pytest.raises(ToolError) as exc_info:
        await openapi_tools.read_openapi_spec(local_path="/nonexistent/openapi.yaml")
    assert exc_info.value.kind is ToolErrorKind.NOT_FOUND


@pytest.mark.asyncio
async def test_read_openapi_spec_local_path_json(tmp_path: Path) -> None:
    import json as _json

    spec = {"openapi": "3.1.0", "info": {"title": "Test"}, "paths": {}}
    spec_file = tmp_path / "spec.json"
    spec_file.write_text(_json.dumps(spec))

    result = await openapi_tools.read_openapi_spec(local_path=str(spec_file))
    assert result["info"]["title"] == "Test"


@pytest.mark.asyncio
async def test_read_openapi_spec_local_path_yaml(tmp_path: Path) -> None:
    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text("openapi: '3.1.0'\ninfo:\n  title: YAMLTest\npaths: {}\n")

    result = await openapi_tools.read_openapi_spec(local_path=str(spec_file))
    assert result["info"]["title"] == "YAMLTest"


@pytest.mark.asyncio
async def test_read_openapi_spec_local_path_invalid_json_raises(tmp_path: Path) -> None:
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{ this is not valid json !!!")

    with pytest.raises(ToolError) as exc_info:
        await openapi_tools.read_openapi_spec(local_path=str(bad_file))
    assert exc_info.value.kind is ToolErrorKind.UPSTREAM


@pytest.mark.asyncio
async def test_read_openapi_spec_remote_non_client_error_falls_through(
    patched_dr_client: MagicMock,
) -> None:
    patched_dr_client.get.side_effect = ValueError("unexpected json decode error")
    with pytest.raises(ToolError) as exc_info:
        await openapi_tools.read_openapi_spec()
    assert exc_info.value.kind is ToolErrorKind.NOT_FOUND


@pytest.mark.asyncio
async def test_read_openapi_spec_remote_error_raises(patched_dr_client: MagicMock) -> None:
    patched_dr_client.get.side_effect = ClientError("503", status_code=503, json={})
    with pytest.raises(ToolError) as exc_info:
        await openapi_tools.read_openapi_spec()
    assert exc_info.value.kind is ToolErrorKind.NOT_FOUND
