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

"""Unit tests for the read-only Files API tools.

The store is replaced with an in-memory fake so tests cover argument
validation, listing/pagination, encoding, and the inline-size guard.
"""

from __future__ import annotations

import base64
from typing import Any

import pytest

from datarobot_genai.drmcputils.constants import MAX_INLINE_SIZE
from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.files.file_system_store import FileEntry
from datarobot_genai.drtools.files_api import read_tools as tools_mod


class FakeStore:
    """Async in-memory FileSystemStore double recording the last call."""

    def __init__(self, **scripted: Any) -> None:
        self._scripted = scripted
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def _record(self, name: str, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((name, args, kwargs))
        return self._scripted.get(name)

    async def ls(self, path: str, *, detail: bool = True) -> list[FileEntry]:
        return self._record("ls", path, detail=detail) or []

    async def find(
        self, path: str, *, maxdepth: Any = None, withdirs: bool = False
    ) -> list[FileEntry]:
        return self._record("find", path, maxdepth=maxdepth) or []

    async def glob(self, pattern: str, *, maxdepth: Any = None) -> list[FileEntry]:
        return self._record("glob", pattern, maxdepth=maxdepth) or []

    async def tree(self, path: str, **kwargs: Any) -> str:
        return self._record("tree", path, **kwargs) or ""

    async def info(self, path: str) -> FileEntry:
        return self._record("info", path)

    async def read(self, path: str, *, start: Any = None, end: Any = None) -> bytes:
        return self._record("read", path, start=start, end=end)

    async def sign(self, path: str, *, expiration: int = 100) -> str:
        return self._record("sign", path, expiration=expiration)


def _use_store(monkeypatch: pytest.MonkeyPatch, store: FakeStore) -> FakeStore:
    monkeypatch.setattr(tools_mod, "_get_store", lambda: store)
    return store


def _entry(name: str, type_: str = "file", size: int = 1) -> FileEntry:
    return FileEntry(name=name, type=type_, size=size, format=None, created_at=None)


# ------------------------------------------------------------------ #
# file_list                                                            #
# ------------------------------------------------------------------ #


async def test_file_list_root_lists_catalog_items(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _use_store(monkeypatch, FakeStore(ls=[_entry("abc/", "directory", 0)]))
    result = await tools_mod.file_list()
    assert result["path"] == "dr://"
    assert result["count"] == 1
    assert result["entries"][0]["type"] == "directory"
    assert store.calls[0][0] == "ls"


async def test_file_list_pattern_uses_glob(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _use_store(monkeypatch, FakeStore(glob=[_entry("abc/a.csv")]))
    result = await tools_mod.file_list(pattern="dr://abc/**/*.csv")
    assert result["path"] == "dr://abc/**/*.csv"
    assert store.calls[0][0] == "glob"


async def test_file_list_recursive_uses_find(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _use_store(monkeypatch, FakeStore(find=[_entry("abc/a.txt")]))
    await tools_mod.file_list(path="dr://abc/", recursive=True)
    assert store.calls[0][0] == "find"


async def test_file_list_as_tree_returns_string(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _use_store(monkeypatch, FakeStore(tree="abc/\n  a.txt"))
    result = await tools_mod.file_list(path="dr://abc/", as_tree=True)
    assert result["tree"] == "abc/\n  a.txt"
    assert "entries" not in result
    assert store.calls[0][0] == "tree"


async def test_file_list_paginates(monkeypatch: pytest.MonkeyPatch) -> None:
    entries = [_entry(f"abc/{i}.txt") for i in range(5)]
    _use_store(monkeypatch, FakeStore(ls=entries))
    result = await tools_mod.file_list(path="dr://abc/", limit=2, offset=1)
    assert result["count"] == 2
    assert result["total_count"] == 5
    assert [e["name"] for e in result["entries"]] == ["abc/1.txt", "abc/2.txt"]
    assert result["offset"] == 1
    assert result["limit"] == 2


async def test_file_list_clamps_limit_with_note(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_store(monkeypatch, FakeStore(ls=[_entry("abc/a.txt")]))
    result = await tools_mod.file_list(path="dr://abc/", limit=9999)
    assert result["limit"] == 100
    assert "note" in result


@pytest.mark.parametrize(
    "kwargs",
    [
        {"offset": -1},
        {"pattern": "   "},
        {"maxdepth": 0},
    ],
)
async def test_file_list_validation_errors(
    monkeypatch: pytest.MonkeyPatch, kwargs: dict[str, Any]
) -> None:
    _use_store(monkeypatch, FakeStore(ls=[]))
    with pytest.raises(ToolError):
        await tools_mod.file_list(path="dr://abc/", **kwargs)


async def test_file_list_recursive_at_root_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_store(monkeypatch, FakeStore(ls=[]))
    with pytest.raises(ToolError, match="recursive listing at 'dr://'"):
        await tools_mod.file_list(path="dr://", recursive=True)


async def test_file_list_root_as_tree_ignores_recursive(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _use_store(monkeypatch, FakeStore(tree="abc/\n  a.txt"))
    result = await tools_mod.file_list(path="dr://", as_tree=True, recursive=True)
    assert result["tree"] == "abc/\n  a.txt"
    assert store.calls[0][0] == "tree"


async def test_file_list_root_pattern_ignores_recursive(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _use_store(monkeypatch, FakeStore(glob=[_entry("abc/a.csv")]))
    result = await tools_mod.file_list(path="dr://", pattern="dr://abc/**/*.csv", recursive=True)
    assert result["path"] == "dr://abc/**/*.csv"
    assert store.calls[0][0] == "glob"


async def test_file_list_includes_browse_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    entries = [_entry(f"abc/{i}.txt") for i in range(3)]
    _use_store(monkeypatch, FakeStore(ls=entries))
    result = await tools_mod.file_list(path="dr://abc/", limit=2)
    assert "hint" in result
    assert "pattern=" in result["hint"]


# ------------------------------------------------------------------ #
# file_info                                                            #
# ------------------------------------------------------------------ #


async def test_file_info_success(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_store(monkeypatch, FakeStore(info=_entry("abc/x.csv", size=20)))
    result = await tools_mod.file_info(path="dr://abc/x.csv")
    assert result["name"] == "abc/x.csv"
    assert result["size"] == 20


@pytest.mark.parametrize("path", ["", "   ", "dr://", "dr://  "])
async def test_file_info_rejects_root_or_empty(monkeypatch: pytest.MonkeyPatch, path: str) -> None:
    _use_store(monkeypatch, FakeStore())
    with pytest.raises(ToolError):
        await tools_mod.file_info(path=path)


# ------------------------------------------------------------------ #
# file_read                                                            #
# ------------------------------------------------------------------ #


async def test_file_read_utf8(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_store(
        monkeypatch,
        FakeStore(info=_entry("abc/x.txt", size=5), read=b"hello"),
    )
    result = await tools_mod.file_read(path="dr://abc/x.txt")
    assert result["encoding"] == "utf-8"
    assert result["content"] == "hello"
    assert result["bytes_read"] == 5
    assert result["total_size"] == 5


async def test_file_read_binary_is_base64(monkeypatch: pytest.MonkeyPatch) -> None:
    blob = b"\xff\xfe\x00"
    _use_store(monkeypatch, FakeStore(info=_entry("abc/x.bin", size=3), read=blob))
    result = await tools_mod.file_read(path="dr://abc/x.bin")
    assert result["encoding"] == "base64"
    assert base64.b64decode(result["content"]) == blob


async def test_file_read_range_passes_offsets(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _use_store(monkeypatch, FakeStore(info=_entry("abc/x.bin", size=1000), read=b"ab"))
    await tools_mod.file_read(path="dr://abc/x.bin", offset=10, length=2)
    read_call = next(c for c in store.calls if c[0] == "read")
    assert read_call[2] == {"start": 10, "end": 12}


async def test_file_read_directory_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_store(monkeypatch, FakeStore(info=_entry("abc/", "directory", 0)))
    with pytest.raises(ToolError):
        await tools_mod.file_read(path="dr://abc/")


async def test_file_read_oversize_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_store(
        monkeypatch,
        FakeStore(info=_entry("abc/big.bin", size=MAX_INLINE_SIZE + 1)),
    )
    with pytest.raises(ToolError):
        await tools_mod.file_read(path="dr://abc/big.bin")


@pytest.mark.parametrize("kwargs", [{"offset": -1}, {"length": 0}, {"length": -5}])
async def test_file_read_validation_errors(
    monkeypatch: pytest.MonkeyPatch, kwargs: dict[str, Any]
) -> None:
    _use_store(monkeypatch, FakeStore(info=_entry("abc/x.txt", size=5), read=b"hi"))
    with pytest.raises(ToolError):
        await tools_mod.file_read(path="dr://abc/x.txt", **kwargs)


# ------------------------------------------------------------------ #
# file_sign                                                            #
# ------------------------------------------------------------------ #


async def test_file_sign_success(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _use_store(monkeypatch, FakeStore(sign="https://signed/x"))
    result = await tools_mod.file_sign(path="dr://abc/x.pdf", expiration=300)
    assert result["url"] == "https://signed/x"
    assert result["expiration"] == 300
    assert store.calls[0][2] == {"expiration": 300}


async def test_file_sign_rejects_root(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_store(monkeypatch, FakeStore())
    with pytest.raises(ToolError):
        await tools_mod.file_sign(path="dr://")


async def test_file_sign_rejects_nonpositive_expiration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_store(monkeypatch, FakeStore(sign="x"))
    with pytest.raises(ToolError):
        await tools_mod.file_sign(path="dr://abc/x.pdf", expiration=0)
