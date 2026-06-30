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

"""Integration tests for DataRobot Files API MCP tools."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest
from mcp.types import TextContent

from datarobot_genai.drmcp.test_utils.mcp_utils_integration import integration_test_mcp_session
from datarobot_genai.drmcp.test_utils.mcp_utils_integration import (
    integration_test_server_params_with_env,
)
from datarobot_genai.drmcp.test_utils.stubs.files_api_stubs import STUB_CATALOG_ID
from datarobot_genai.drmcp.test_utils.stubs.files_api_stubs import STUB_CATALOG_ID_2
from datarobot_genai.drmcp.test_utils.stubs.files_api_stubs import STUB_IMPORT_STATUS_ID
from datarobot_genai.drmcp.test_utils.stubs.files_api_stubs import STUB_IMPORT_STATUS_INPROGRESS

_EXPECTED_TOOLS = frozenset(
    {
        "file_list",
        "file_info",
        "file_read",
        "file_sign",
        "file_write",
        "file_upload",
        "file_manage",
        "file_import",
        "file_get_status",
    }
)


def _files_api_server_params(*, local_allowed_root: str | None = None) -> object:
    """Return server params with Files API tools enabled."""
    env: dict[str, str] = {"ENABLE_FILES_API_TOOLS": "true"}
    if local_allowed_root is not None:
        env["FILES_API_LOCAL_ALLOWED_ROOTS"] = local_allowed_root
    return integration_test_server_params_with_env(env)


def _parse_result(result: object) -> dict:
    assert not getattr(result, "isError", True)
    content = getattr(result, "content", [])
    assert len(content) > 0
    assert isinstance(content[0], TextContent)
    return json.loads(content[0].text)


def _parse_error(result: object) -> str:
    assert getattr(result, "isError", False)
    content = getattr(result, "content", [])
    assert len(content) > 0
    assert isinstance(content[0], TextContent)
    return content[0].text


@pytest.mark.asyncio
class TestMCPFilesApiToolsRegistration:
    """Verify Files API tools are registered in the MCP server."""

    async def test_tools_registered(self) -> None:
        async with integration_test_mcp_session(
            server_params=_files_api_server_params()
        ) as session:
            result = await session.list_tools()
            tool_names = {t.name for t in result.tools}
            missing = _EXPECTED_TOOLS - tool_names
            assert not missing, f"files_api tools not registered: {missing}"


@pytest.mark.asyncio
class TestMCPFilesApiReadIntegration:
    """Integration tests for file_list, file_info, file_read, and file_sign."""

    async def test_file_list_root_lists_catalog_items(self) -> None:
        async with integration_test_mcp_session(
            server_params=_files_api_server_params()
        ) as session:
            data = _parse_result(await session.call_tool("file_list", {"path": "dr://"}))
            assert data["path"] == "dr://"
            assert data["count"] >= 2
            names = {e["name"] for e in data["entries"]}
            assert f"{STUB_CATALOG_ID}/" in names
            assert f"{STUB_CATALOG_ID_2}/" in names

    async def test_file_list_catalog_contents(self) -> None:
        async with integration_test_mcp_session(
            server_params=_files_api_server_params()
        ) as session:
            data = _parse_result(
                await session.call_tool("file_list", {"path": f"dr://{STUB_CATALOG_ID}/data/"})
            )
            assert data["count"] >= 2
            file_names = {e["name"].split("/")[-1] for e in data["entries"]}
            assert "employees.csv" in file_names
            assert "report.pdf" in file_names

    async def test_file_list_glob_csv(self) -> None:
        async with integration_test_mcp_session(
            server_params=_files_api_server_params()
        ) as session:
            data = _parse_result(
                await session.call_tool(
                    "file_list",
                    {"pattern": f"dr://{STUB_CATALOG_ID}/**/*.csv"},
                )
            )
            assert data["count"] >= 1
            assert all(entry["name"].endswith(".csv") for entry in data["entries"])

    async def test_file_list_recursive(self) -> None:
        async with integration_test_mcp_session(
            server_params=_files_api_server_params()
        ) as session:
            data = _parse_result(
                await session.call_tool(
                    "file_list",
                    {"path": f"dr://{STUB_CATALOG_ID}/", "recursive": True},
                )
            )
            assert data["total_count"] >= 3

    async def test_file_list_as_tree(self) -> None:
        async with integration_test_mcp_session(
            server_params=_files_api_server_params()
        ) as session:
            data = _parse_result(
                await session.call_tool(
                    "file_list",
                    {"path": f"dr://{STUB_CATALOG_ID}/", "as_tree": True},
                )
            )
            assert "tree" in data
            assert "employees.csv" in data["tree"]

    async def test_file_info_file(self) -> None:
        async with integration_test_mcp_session(
            server_params=_files_api_server_params()
        ) as session:
            path = f"dr://{STUB_CATALOG_ID}/notes.txt"
            data = _parse_result(await session.call_tool("file_info", {"path": path}))
            assert data["name"] == f"{STUB_CATALOG_ID}/notes.txt"
            assert data["type"] == "file"
            assert data["size"] > 0

    async def test_file_read_utf8(self) -> None:
        async with integration_test_mcp_session(
            server_params=_files_api_server_params()
        ) as session:
            path = f"dr://{STUB_CATALOG_ID}/notes.txt"
            data = _parse_result(await session.call_tool("file_read", {"path": path}))
            assert data["encoding"] == "utf-8"
            assert "hello world" in data["content"]
            assert data["bytes_read"] == data["total_size"]

    async def test_file_read_binary_base64(self) -> None:
        async with integration_test_mcp_session(
            server_params=_files_api_server_params()
        ) as session:
            path = f"dr://{STUB_CATALOG_ID}/data/report.pdf"
            data = _parse_result(await session.call_tool("file_read", {"path": path}))
            assert data["encoding"] == "base64"
            decoded = base64.b64decode(data["content"])
            assert decoded.startswith(b"%PDF")

    async def test_file_read_byte_range(self) -> None:
        async with integration_test_mcp_session(
            server_params=_files_api_server_params()
        ) as session:
            path = f"dr://{STUB_CATALOG_ID}/notes.txt"
            data = _parse_result(
                await session.call_tool("file_read", {"path": path, "offset": 0, "length": 5})
            )
            assert data["encoding"] == "utf-8"
            assert data["content"] == "hello"
            assert data["bytes_read"] == 5

    async def test_file_sign(self) -> None:
        async with integration_test_mcp_session(
            server_params=_files_api_server_params()
        ) as session:
            path = f"dr://{STUB_CATALOG_ID}/data/report.pdf"
            data = _parse_result(
                await session.call_tool("file_sign", {"path": path, "expiration": 300})
            )
            assert data["path"] == path
            assert data["expiration"] == 300
            assert data["url"].startswith("https://")

    async def test_file_read_directory_rejected(self) -> None:
        async with integration_test_mcp_session(
            server_params=_files_api_server_params()
        ) as session:
            result = await session.call_tool("file_read", {"path": f"dr://{STUB_CATALOG_ID}/"})
            assert result.isError
            assert "directory" in _parse_error(result).lower()


@pytest.mark.asyncio
class TestMCPFilesApiMutationsIntegration:
    """Integration tests for file_write, file_manage, and file_upload."""

    async def test_file_write_and_read_roundtrip(self) -> None:
        async with integration_test_mcp_session(
            server_params=_files_api_server_params()
        ) as session:
            path = f"dr://{STUB_CATALOG_ID}/integration/new.txt"
            write_data = _parse_result(
                await session.call_tool(
                    "file_write",
                    {"path": path, "content": "integration test", "mode": "create"},
                )
            )
            assert write_data["bytes_written"] == len(b"integration test")
            read_data = _parse_result(await session.call_tool("file_read", {"path": path}))
            assert read_data["content"] == "integration test"

    async def test_file_manage_create_dir(self) -> None:
        async with integration_test_mcp_session(
            server_params=_files_api_server_params()
        ) as session:
            data = _parse_result(await session.call_tool("file_manage", {"action": "create_dir"}))
            assert data["created"] is True
            assert data["catalog_id"]
            assert data["path"] == f"dr://{data['catalog_id']}/"

    async def test_file_manage_copy(self) -> None:
        async with integration_test_mcp_session(
            server_params=_files_api_server_params()
        ) as session:
            source = f"dr://{STUB_CATALOG_ID}/notes.txt"
            target = f"dr://{STUB_CATALOG_ID}/notes-copy.txt"
            data = _parse_result(
                await session.call_tool(
                    "file_manage",
                    {"action": "copy", "path": source, "target_path": target},
                )
            )
            assert data["copied"] is True
            read_data = _parse_result(await session.call_tool("file_read", {"path": target}))
            assert "hello world" in read_data["content"]

    async def test_file_manage_move(self) -> None:
        async with integration_test_mcp_session(
            server_params=_files_api_server_params()
        ) as session:
            source = f"dr://{STUB_CATALOG_ID_2}/readme.txt"
            target = f"dr://{STUB_CATALOG_ID_2}/readme-moved.txt"
            data = _parse_result(
                await session.call_tool(
                    "file_manage",
                    {"action": "move", "path": source, "target_path": target},
                )
            )
            assert data["moved"] is True
            read_data = _parse_result(await session.call_tool("file_read", {"path": target}))
            assert "catalog two" in read_data["content"]

    async def test_file_manage_clone(self) -> None:
        async with integration_test_mcp_session(
            server_params=_files_api_server_params()
        ) as session:
            data = _parse_result(
                await session.call_tool(
                    "file_manage",
                    {"action": "clone", "path": f"dr://{STUB_CATALOG_ID}/"},
                )
            )
            assert data["cloned"] is True
            clone_path = f"dr://{data['catalog_id']}/notes.txt"
            read_data = _parse_result(await session.call_tool("file_read", {"path": clone_path}))
            assert "hello world" in read_data["content"]

    async def test_file_upload_from_local(self, tmp_path: Path) -> None:
        local_file = tmp_path / "upload.txt"
        local_file.write_bytes(b"uploaded via integration test")
        async with integration_test_mcp_session(
            server_params=_files_api_server_params(local_allowed_root=str(tmp_path))
        ) as session:
            dest = f"dr://{STUB_CATALOG_ID}/uploads/"
            data = _parse_result(
                await session.call_tool(
                    "file_upload",
                    {"local_path": str(local_file), "path": dest},
                )
            )
            assert data["uploaded"] is True
            assert data["file_count"] == 1
            read_data = _parse_result(
                await session.call_tool(
                    "file_read",
                    {"path": f"dr://{STUB_CATALOG_ID}/uploads/upload.txt"},
                )
            )
            assert read_data["content"] == "uploaded via integration test"


@pytest.mark.asyncio
class TestMCPFilesApiImportIntegration:
    """Integration tests for file_import and file_get_status."""

    async def test_file_import_from_url(self) -> None:
        async with integration_test_mcp_session(
            server_params=_files_api_server_params()
        ) as session:
            data = _parse_result(
                await session.call_tool(
                    "file_import",
                    {
                        "path": f"dr://{STUB_CATALOG_ID}/imports/",
                        "source": "url",
                        "url": "https://example.com/data.zip",
                    },
                )
            )
            assert data["source"] == "url"
            assert data["status_id"]
            assert "file_get_status" in data["note"]

    async def test_file_import_from_data_source(self) -> None:
        async with integration_test_mcp_session(
            server_params=_files_api_server_params()
        ) as session:
            data = _parse_result(
                await session.call_tool(
                    "file_import",
                    {
                        "path": f"dr://{STUB_CATALOG_ID}/imports/",
                        "source": "data_source",
                        "data_source_id": "stub_data_source_id",
                    },
                )
            )
            assert data["source"] == "data_source"
            assert data["status_id"]

    async def test_file_get_status_in_progress(self) -> None:
        async with integration_test_mcp_session(
            server_params=_files_api_server_params()
        ) as session:
            data = _parse_result(
                await session.call_tool("file_get_status", {"status_id": STUB_IMPORT_STATUS_ID})
            )
            assert data["status_id"] == STUB_IMPORT_STATUS_ID
            assert data["status"] == STUB_IMPORT_STATUS_INPROGRESS

    async def test_file_get_status_target_not_reached(self) -> None:
        async with integration_test_mcp_session(
            server_params=_files_api_server_params()
        ) as session:
            data = _parse_result(
                await session.call_tool(
                    "file_get_status",
                    {
                        "status_id": STUB_IMPORT_STATUS_ID,
                        "target_status": "completed",
                    },
                )
            )
            assert data["target_reached"] is False

    async def test_file_import_missing_url_rejected(self) -> None:
        async with integration_test_mcp_session(
            server_params=_files_api_server_params()
        ) as session:
            result = await session.call_tool(
                "file_import",
                {"path": f"dr://{STUB_CATALOG_ID}/imports/", "source": "url"},
            )
            assert result.isError
            assert "url" in _parse_error(result).lower()
