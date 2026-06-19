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

"""Unit tests for DataRobotFileSystemStore.

The fsspec backend is replaced with a fake filesystem; we assert FileInfo ->
FileEntry mapping and the normalisation of SDK/OS errors to ToolError.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Any
from unittest.mock import patch

import pytest
from datarobot.enums import FilesOverwriteStrategy
from datarobot.errors import ClientError

from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.exceptions import ToolErrorKind
from datarobot_genai.drmcputils.files.file_system_store import DataRobotFileSystemStore
from datarobot_genai.drmcputils.files.file_system_store import FileEntry
from datarobot_genai.drmcputils.files.file_system_store import FileSystemStore
from datarobot_genai.drmcputils.files.file_system_store import extract_status_id
from datarobot_genai.drmcputils.files.file_system_store import is_terminal_import_failure_status
from datarobot_genai.drmcputils.files.file_system_store import normalize_status_location


class FakeFS:
    """Stand-in for datarobot.fs.DataRobotFileSystem with scripted return values."""

    def __init__(self, **behavior: Any) -> None:
        self._behavior = behavior

    def _resolve(self, key: str, default: Any = None) -> Any:
        value = self._behavior.get(key, default)
        if isinstance(value, Exception):
            raise value
        return value

    def ls(self, path: str, detail: bool = True) -> Any:
        return self._resolve("ls", [])

    def find(
        self, path: str, maxdepth: Any = None, withdirs: bool = False, detail: bool = True
    ) -> Any:
        return self._resolve("find", {})

    def glob(self, path: str, maxdepth: Any = None, detail: bool = True) -> Any:
        return self._resolve("glob", {})

    def tree(self, path: str, **kwargs: Any) -> Any:
        return self._resolve("tree", "")

    def info(self, path: str) -> Any:
        return self._resolve("info", {})

    def cat_file(self, path: str, start: Any = None, end: Any = None) -> Any:
        return self._resolve("cat_file", b"")

    def sign(self, path: str, expiration: int = 100) -> Any:
        return self._resolve("sign", "")

    def pipe_file(self, path: str, value: bytes, mode: str = "overwrite") -> Any:
        self.recorded = ("pipe_file", path, value, mode)
        return self._resolve("pipe_file", None)

    def create_catalog_item_dir(self) -> Any:
        return self._resolve("create_catalog_item_dir", "")

    def rm(self, path: str, recursive: bool = False, maxdepth: Any = None) -> Any:
        self.recorded = ("rm", path, recursive, maxdepth)
        return self._resolve("rm", None)

    def copy(self, path1: str, path2: str, recursive: bool = False, maxdepth: Any = None) -> Any:
        self.recorded = ("copy", path1, path2, recursive, maxdepth)
        return self._resolve("copy", None)

    def mv(self, path1: str, path2: str, recursive: bool = False, maxdepth: Any = None) -> Any:
        self.recorded = ("mv", path1, path2, recursive, maxdepth)
        return self._resolve("mv", None)

    def clone_catalog_item_dir(self, path_or_id: str, files_to_omit: Any = None) -> Any:
        self.recorded = ("clone", path_or_id, files_to_omit)
        return self._resolve("clone_catalog_item_dir", "")

    def _split_path(self, path: str) -> tuple[str, str]:
        stripped = path.removeprefix("dr://").strip("/")
        catalog_id, _, rest = stripped.partition("/")
        return catalog_id, rest

    def _get_files_wrapper_for_folder_id(self, catalog_id: str) -> FakeFilesWrapper:
        self.recorded = ("files_wrapper", catalog_id)
        return self._files_wrapper


class FakeFilesEntity:
    def __init__(self, async_location: str) -> None:
        self._async_status_location = async_location


class FakeFilesWrapper:
    def __init__(self, *, async_location: str = "status/job123/") -> None:
        self._async_location = async_location
        self.last_call: tuple[str, dict[str, Any]] | None = None

    def upload_from_url(self, **kwargs: Any) -> FakeFilesEntity:
        self.last_call = ("upload_from_url", kwargs)
        return FakeFilesEntity(self._async_location)

    def upload_from_data_source(self, **kwargs: Any) -> FakeFilesEntity:
        self.last_call = ("upload_from_data_source", kwargs)
        return FakeFilesEntity(self._async_location)


@contextmanager
def _noop_sdk(**_: Any) -> Iterator[None]:
    yield None


def _store_and_fs(**behavior: Any) -> tuple[DataRobotFileSystemStore, FakeFS]:
    """Patch the SDK + fsspec backend; return the store and the shared fake fs."""
    fs = FakeFS(**behavior)
    fs._files_wrapper = FakeFilesWrapper(
        async_location=behavior.get("async_location", "status/job123/")
    )
    patch(
        "datarobot_genai.drmcputils.files.file_system_store.request_user_dr_sdk",
        _noop_sdk,
    ).start()
    patch("datarobot.fs.DataRobotFileSystem", lambda *a, **k: fs).start()
    return DataRobotFileSystemStore(), fs


def _store_with(**behavior: Any) -> DataRobotFileSystemStore:
    """Patch the request-scoped SDK and the fsspec backend, returning a live store."""
    store, _ = _store_and_fs(**behavior)
    return store


@pytest.fixture(autouse=True)
def _cleanup_patches() -> Iterator[None]:
    yield
    patch.stopall()


def test_store_satisfies_protocol() -> None:
    assert isinstance(DataRobotFileSystemStore(), FileSystemStore)


def test_file_entry_from_info_serializes_datetime() -> None:
    entry = FileEntry.from_info(
        {
            "name": "abc/file.csv",
            "size": 10,
            "type": "file",
            "format": "csv",
            "created_at": datetime(2026, 1, 2, 3, 4, 5),
        }
    )
    assert entry.created_at == "2026-01-02T03:04:05"
    assert entry.to_dict() == {
        "name": "abc/file.csv",
        "type": "file",
        "size": 10,
        "format": "csv",
        "created_at": "2026-01-02T03:04:05",
    }


async def test_ls_maps_file_info_to_entries() -> None:
    store = _store_with(
        ls=[
            {"name": "abc/", "size": 0, "type": "directory", "format": None, "created_at": None},
            {"name": "abc/x.txt", "size": 5, "type": "file", "format": "txt", "created_at": None},
        ]
    )
    entries = await store.ls("dr://abc/")
    assert [e.name for e in entries] == ["abc/", "abc/x.txt"]
    assert entries[0].type == "directory"
    assert entries[1].size == 5


async def test_find_and_glob_use_detail_dict_values() -> None:
    info = {"name": "abc/d/y.csv", "size": 3, "type": "file", "format": "csv", "created_at": None}
    store = _store_with(find={"abc/d/y.csv": info}, glob={"abc/d/y.csv": info})
    assert (await store.find("dr://abc/"))[0].name == "abc/d/y.csv"
    assert (await store.glob("dr://abc/**/*.csv"))[0].format == "csv"


async def test_read_returns_bytes() -> None:
    store = _store_with(cat_file=b"hello")
    assert await store.read("dr://abc/x.txt", start=0, end=5) == b"hello"


async def test_sign_returns_url() -> None:
    store = _store_with(sign="https://signed.example/x")
    assert await store.sign("dr://abc/x.txt") == "https://signed.example/x"


async def test_not_found_error_maps_to_tool_error() -> None:
    store = _store_with(info=FileNotFoundError("nope"))
    with pytest.raises(ToolError) as exc:
        await store.info("dr://abc/missing")
    assert exc.value.kind == ToolErrorKind.NOT_FOUND


async def test_value_error_maps_to_validation() -> None:
    store = _store_with(info=ValueError("bad path"))
    with pytest.raises(ToolError) as exc:
        await store.info("dr://")
    assert exc.value.kind == ToolErrorKind.VALIDATION


async def test_permission_error_maps_to_upstream() -> None:
    store = _store_with(ls=PermissionError("denied"))
    with pytest.raises(ToolError) as exc:
        await store.ls("dr://abc/")
    assert exc.value.kind == ToolErrorKind.UPSTREAM


async def test_client_error_maps_via_helper() -> None:
    store = _store_with(info=ClientError("boom", 404))
    with pytest.raises(ToolError) as exc:
        await store.info("dr://abc/x")
    assert exc.value.kind == ToolErrorKind.NOT_FOUND


async def test_write_delegates_to_pipe_file() -> None:
    store, fs = _store_and_fs()
    await store.write("dr://abc/x.txt", b"data", mode="create")
    assert fs.recorded == ("pipe_file", "dr://abc/x.txt", b"data", "create")


async def test_create_dir_returns_catalog_id() -> None:
    store, _ = _store_and_fs(create_catalog_item_dir="cat1")
    assert await store.create_dir() == "cat1"


async def test_delete_copy_move_delegate() -> None:
    store, fs = _store_and_fs()
    await store.delete("dr://abc/x", recursive=True, maxdepth=2)
    assert fs.recorded == ("rm", "dr://abc/x", True, 2)
    await store.copy("dr://abc/a", "dr://abc/b", recursive=True)
    assert fs.recorded == ("copy", "dr://abc/a", "dr://abc/b", True, None)
    await store.move("dr://abc/a", "dr://abc/b")
    assert fs.recorded == ("mv", "dr://abc/a", "dr://abc/b", False, None)


async def test_clone_returns_new_catalog_id() -> None:
    store, fs = _store_and_fs(clone_catalog_item_dir="clone1")
    assert await store.clone("dr://abc/", files_to_omit=["s.txt"]) == "clone1"
    assert fs.recorded == ("clone", "dr://abc/", ["s.txt"])


async def test_import_from_url_returns_status_id() -> None:
    store, fs = _store_and_fs()
    status_id = await store.import_from_url(
        "dr://abc123/data/",
        "https://example.com/file.zip",
        unpack_archive=False,
        overwrite="replace",
    )
    assert status_id == "job123"
    assert fs.recorded == ("files_wrapper", "abc123")
    assert fs._files_wrapper.last_call == (
        "upload_from_url",
        {
            "url": "https://example.com/file.zip",
            "use_archive_contents": False,
            "overwrite": FilesOverwriteStrategy.REPLACE,
            "wait_for_completion": False,
            "prefix": "data/",
        },
    )


async def test_import_from_data_source_returns_status_id() -> None:
    store, fs = _store_and_fs(async_location="https://host/api/v2/status/ds-job/")
    status_id = await store.import_from_data_source(
        "dr://abc123/in/",
        "ds-456",
        credential_id="cred-1",
    )
    assert status_id == "ds-job"
    assert fs._files_wrapper.last_call[0] == "upload_from_data_source"
    assert fs._files_wrapper.last_call[1]["data_source_id"] == "ds-456"
    assert fs._files_wrapper.last_call[1]["credential_id"] == "cred-1"
    assert fs._files_wrapper.last_call[1]["wait_for_completion"] is False


async def test_get_status_fetches_json_payload() -> None:
    store = _store_with()

    class FakeResponse:
        status_code = 200

        def json(self) -> dict[str, str]:
            return {"status": "INPROGRESS"}

    fake_client = type("C", (), {"get": lambda _s, _loc, allow_redirects=False: FakeResponse()})()
    with patch(
        "datarobot_genai.drmcputils.files.file_system_store.dr.client.get_client",
        return_value=fake_client,
    ):
        payload = await store.get_status("job123")
    assert payload == {"status": "INPROGRESS"}


def test_status_helpers() -> None:
    assert normalize_status_location("job123") == "status/job123/"
    assert normalize_status_location("status/job123/") == "status/job123/"
    assert normalize_status_location("https://host/api/v2/status/job123/") == "status/job123/"
    assert extract_status_id("status/job123/") == "job123"
    assert extract_status_id("https://host/api/v2/status/job123/") == "job123"
    assert is_terminal_import_failure_status("ERROR")
    assert is_terminal_import_failure_status("aborted")
    assert not is_terminal_import_failure_status("completed")


async def test_get_status_accepts_full_async_url() -> None:
    store = _store_with()
    requested: list[str] = []

    class FakeResponse:
        status_code = 200

        def json(self) -> dict[str, str]:
            return {"status": "INPROGRESS"}

    fake_client = type(
        "C",
        (),
        {
            "get": lambda _s, loc, allow_redirects=False: requested.append(loc) or FakeResponse(),
        },
    )()
    with patch(
        "datarobot_genai.drmcputils.files.file_system_store.dr.client.get_client",
        return_value=fake_client,
    ):
        await store.get_status("https://host/api/v2/status/job123/")
    assert requested == ["status/job123/"]


async def test_get_status_307_without_location_raises_tool_error() -> None:
    store = _store_with()

    class RedirectResponse:
        status_code = 307
        headers: dict[str, str] = {}

    fake_client = type(
        "C",
        (),
        {"get": lambda _s, _loc, allow_redirects=False: RedirectResponse()},
    )()
    with patch(
        "datarobot_genai.drmcputils.files.file_system_store.dr.client.get_client",
        return_value=fake_client,
    ):
        with pytest.raises(ToolError, match="307") as exc:
            await store.get_status("job123")
    assert exc.value.kind == ToolErrorKind.UPSTREAM
