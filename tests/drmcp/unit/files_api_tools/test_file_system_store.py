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
from datarobot.errors import ClientError

from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.exceptions import ToolErrorKind
from datarobot_genai.drmcputils.files.file_system_store import DataRobotFileSystemStore
from datarobot_genai.drmcputils.files.file_system_store import FileEntry
from datarobot_genai.drmcputils.files.file_system_store import FileSystemStore


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


@contextmanager
def _noop_sdk(**_: Any) -> Iterator[None]:
    yield None


def _store_with(**behavior: Any) -> DataRobotFileSystemStore:
    """Patch the request-scoped SDK and the fsspec backend, returning a live store."""
    patch(
        "datarobot_genai.drmcputils.files.file_system_store.request_user_dr_sdk",
        _noop_sdk,
    ).start()
    patch("datarobot.fs.DataRobotFileSystem", lambda *a, **k: FakeFS(**behavior)).start()
    return DataRobotFileSystemStore()


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
