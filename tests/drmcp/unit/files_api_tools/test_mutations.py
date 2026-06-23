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

"""Unit tests for the write/structural Files API tools (file_write, file_manage)."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import pytest

from datarobot_genai.drmcputils.constants import MAX_INLINE_SIZE
from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drtools.files_api import common_utils
from datarobot_genai.drtools.files_api import mutations_tools as mut_mod


class FakeStore:
    """Async FileSystemStore double recording calls; create_dir/clone return scripted ids."""

    def __init__(self, *, new_catalog_id: str = "newcat") -> None:
        self._new_catalog_id = new_catalog_id
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def _record(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append((name, args, kwargs))

    async def write(self, path: str, data: bytes, *, mode: str = "overwrite") -> None:
        self._record("write", path, data, mode=mode)

    async def upload(
        self,
        local_path: Any,
        dest: Any,
        *,
        recursive: bool = False,
        maxdepth: Any = None,
        overwrite: str = "rename",
    ) -> None:
        self._record(
            "upload", local_path, dest, recursive=recursive, maxdepth=maxdepth, overwrite=overwrite
        )

    async def create_dir(self) -> str:
        self._record("create_dir")
        return self._new_catalog_id

    async def delete(self, path: str, *, recursive: bool = False, maxdepth: Any = None) -> None:
        self._record("delete", path, recursive=recursive, maxdepth=maxdepth)

    async def copy(
        self, source: str, dest: str, *, recursive: bool = False, maxdepth: Any = None
    ) -> None:
        self._record("copy", source, dest, recursive=recursive, maxdepth=maxdepth)

    async def move(
        self, source: str, dest: str, *, recursive: bool = False, maxdepth: Any = None
    ) -> None:
        self._record("move", source, dest, recursive=recursive, maxdepth=maxdepth)

    async def clone(self, path_or_id: str, *, files_to_omit: Any = None) -> str:
        self._record("clone", path_or_id, files_to_omit=files_to_omit)
        return self._new_catalog_id


def _use_store(monkeypatch: pytest.MonkeyPatch, store: FakeStore) -> FakeStore:
    monkeypatch.setattr(mut_mod, "_get_store", lambda: store)
    return store


# ------------------------------------------------------------------ #
# file_write                                                           #
# ------------------------------------------------------------------ #


async def test_file_write_utf8(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _use_store(monkeypatch, FakeStore())
    result = await mut_mod.file_write(path="dr://abc/notes.txt", content="hello")
    assert result == {
        "path": "dr://abc/notes.txt",
        "bytes_written": 5,
        "mode": "overwrite",
    }
    name, args, kwargs = store.calls[0]
    assert name == "write"
    assert args[1] == b"hello"
    assert kwargs == {"mode": "overwrite"}


async def test_file_write_base64(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _use_store(monkeypatch, FakeStore())
    blob = b"\x00\x01\x02"
    encoded = base64.b64encode(blob).decode("ascii")
    result = await mut_mod.file_write(
        path="dr://abc/x.bin", content=encoded, encoding="base64", mode="create"
    )
    assert result["bytes_written"] == 3
    assert result["mode"] == "create"
    assert store.calls[0][1][1] == blob


async def test_file_write_invalid_base64(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_store(monkeypatch, FakeStore())
    with pytest.raises(ToolError):
        await mut_mod.file_write(path="dr://abc/x.bin", content="not!base64!", encoding="base64")


async def test_file_write_oversize_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_store(monkeypatch, FakeStore())
    big = "a" * (MAX_INLINE_SIZE + 1)
    with pytest.raises(ToolError):
        await mut_mod.file_write(path="dr://abc/big.txt", content=big)


@pytest.mark.parametrize("path", ["", "dr://"])
async def test_file_write_rejects_root_or_empty(monkeypatch: pytest.MonkeyPatch, path: str) -> None:
    _use_store(monkeypatch, FakeStore())
    with pytest.raises(ToolError):
        await mut_mod.file_write(path=path, content="x")


# ------------------------------------------------------------------ #
# local-disk access (allowlist) + file_upload                         #
# ------------------------------------------------------------------ #


@pytest.fixture
def allow_local(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Allow local access rooted at a tmp dir and clear caches between tests."""
    monkeypatch.setenv("FILES_API_LOCAL_ALLOWED_ROOTS", str(tmp_path))
    return tmp_path


async def test_file_upload_single_file(monkeypatch: pytest.MonkeyPatch, allow_local: Path) -> None:
    store = _use_store(monkeypatch, FakeStore())
    local = allow_local / "report.pdf"
    local.write_bytes(b"%PDF-1.4 data")
    result = await mut_mod.file_upload(local_path=str(local), path="dr://abc/docs/")
    assert result["uploaded"] is True
    assert result["file_count"] == 1
    assert result["total_bytes"] == len(b"%PDF-1.4 data")
    assert result["overwrite"] == "rename"
    name, args, kwargs = store.calls[0]
    assert name == "upload"
    assert args == (str(local), "dr://abc/docs/")
    assert kwargs == {"recursive": False, "maxdepth": None, "overwrite": "rename"}


async def test_file_upload_directory_recursive(
    monkeypatch: pytest.MonkeyPatch, allow_local: Path
) -> None:
    store = _use_store(monkeypatch, FakeStore())
    (allow_local / "a.txt").write_bytes(b"aa")
    sub = allow_local / "sub"
    sub.mkdir()
    (sub / "b.txt").write_bytes(b"bbb")
    result = await mut_mod.file_upload(
        local_path=str(allow_local), path="dr://abc/data/", recursive=True, overwrite="replace"
    )
    assert result["file_count"] == 2
    assert result["total_bytes"] == 5
    assert store.calls[0][2]["overwrite"] == "replace"
    assert store.calls[0][2]["recursive"] is True


async def test_file_upload_glob(monkeypatch: pytest.MonkeyPatch, allow_local: Path) -> None:
    _use_store(monkeypatch, FakeStore())
    (allow_local / "x.csv").write_bytes(b"1,2")
    (allow_local / "y.csv").write_bytes(b"3,4")
    (allow_local / "z.txt").write_bytes(b"nope")
    result = await mut_mod.file_upload(
        local_path=str(allow_local / "*.csv"), path="dr://abc/csv/", recursive=True
    )
    assert result["file_count"] == 2


async def test_file_upload_directory_without_recursive_rejected(
    monkeypatch: pytest.MonkeyPatch, allow_local: Path
) -> None:
    _use_store(monkeypatch, FakeStore())
    with pytest.raises(ToolError, match="recursive=True"):
        await mut_mod.file_upload(local_path=str(allow_local), path="dr://abc/data/")


async def test_file_upload_no_match_rejected(
    monkeypatch: pytest.MonkeyPatch, allow_local: Path
) -> None:
    _use_store(monkeypatch, FakeStore())
    with pytest.raises(ToolError, match="No files matched"):
        await mut_mod.file_upload(
            local_path=str(allow_local / "*.csv"), path="dr://abc/csv/", recursive=True
        )


async def test_file_upload_missing_path_rejected(
    monkeypatch: pytest.MonkeyPatch, allow_local: Path
) -> None:
    _use_store(monkeypatch, FakeStore())
    with pytest.raises(ToolError, match="No files matched"):
        await mut_mod.file_upload(local_path=str(allow_local / "nope.txt"), path="dr://abc/")


async def test_file_upload_disabled_without_allowlist(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(common_utils, "_allowed_local_roots", lambda: [])
    _use_store(monkeypatch, FakeStore())
    local = tmp_path / "a.txt"
    local.write_bytes(b"x")
    with pytest.raises(ToolError, match="disabled"):
        await mut_mod.file_upload(local_path=str(local), path="dr://abc/")


async def test_file_upload_outside_allowlist_rejected(
    monkeypatch: pytest.MonkeyPatch, allow_local: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    _use_store(monkeypatch, FakeStore())
    outside = tmp_path_factory.mktemp("outside") / "secret.txt"
    outside.write_bytes(b"secret")
    with pytest.raises(ToolError, match="outside the allowed"):
        await mut_mod.file_upload(local_path=str(outside), path="dr://abc/")


async def test_file_upload_rejects_bad_maxdepth(
    monkeypatch: pytest.MonkeyPatch, allow_local: Path
) -> None:
    _use_store(monkeypatch, FakeStore())
    with pytest.raises(ToolError, match="maxdepth"):
        await mut_mod.file_upload(
            local_path=str(allow_local), path="dr://abc/", recursive=True, maxdepth=0
        )


# ------------------------------------------------------------------ #
# file_manage                                                          #
# ------------------------------------------------------------------ #


async def test_file_manage_create_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _use_store(monkeypatch, FakeStore(new_catalog_id="cat42"))
    result = await mut_mod.file_manage(action="create_dir")
    assert result == {"created": True, "catalog_id": "cat42", "path": "dr://cat42/"}
    assert store.calls[0][0] == "create_dir"


async def test_file_manage_delete_recursive(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _use_store(monkeypatch, FakeStore())
    result = await mut_mod.file_manage(
        action="delete", path="dr://abc/old/", recursive=True, maxdepth=2
    )
    assert result == {"deleted": True, "path": "dr://abc/old/"}
    assert store.calls[0] == ("delete", ("dr://abc/old/",), {"recursive": True, "maxdepth": 2})


async def test_file_manage_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _use_store(monkeypatch, FakeStore())
    result = await mut_mod.file_manage(
        action="copy", path="dr://abc/a.txt", target_path="dr://abc/b.txt"
    )
    assert result == {"copied": True, "source": "dr://abc/a.txt", "target": "dr://abc/b.txt"}
    assert store.calls[0][0] == "copy"


async def test_file_manage_move(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _use_store(monkeypatch, FakeStore())
    result = await mut_mod.file_manage(
        action="move", path="dr://abc/a.txt", target_path="dr://abc/b.txt"
    )
    assert result == {"moved": True, "source": "dr://abc/a.txt", "target": "dr://abc/b.txt"}
    assert store.calls[0][0] == "move"


async def test_file_manage_clone(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _use_store(monkeypatch, FakeStore(new_catalog_id="clone99"))
    result = await mut_mod.file_manage(
        action="clone", path="dr://abc/", files_to_omit=["secret.txt"]
    )
    assert result["cloned"] is True
    assert result["catalog_id"] == "clone99"
    assert result["path"] == "dr://clone99/"
    assert store.calls[0] == ("clone", ("dr://abc/",), {"files_to_omit": ["secret.txt"]})


async def test_file_manage_copy_requires_target(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_store(monkeypatch, FakeStore())
    with pytest.raises(ToolError):
        await mut_mod.file_manage(action="copy", path="dr://abc/a.txt")


async def test_file_manage_delete_requires_path(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_store(monkeypatch, FakeStore())
    with pytest.raises(ToolError):
        await mut_mod.file_manage(action="delete")


async def test_file_manage_rejects_bad_maxdepth(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_store(monkeypatch, FakeStore())
    with pytest.raises(ToolError):
        await mut_mod.file_manage(action="delete", path="dr://abc/x", recursive=True, maxdepth=0)
