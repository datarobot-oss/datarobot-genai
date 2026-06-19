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

"""Unit tests for async Files API import tools (file_import, file_get_status)."""

from __future__ import annotations

from typing import Any

import pytest

from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drtools.files_api import imports as imports_mod


class FakeStore:
    """Records import/status calls and returns scripted payloads."""

    def __init__(self, **scripted: Any) -> None:
        self._scripted = scripted
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def _record(self, name: str, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((name, args, kwargs))
        return self._scripted.get(name)

    async def import_from_url(self, path: str, url: str, **kwargs: Any) -> str:
        return self._record("import_from_url", path, url, **kwargs) or "job-url"

    async def import_from_data_source(self, path: str, data_source_id: str, **kwargs: Any) -> str:
        return self._record("import_from_data_source", path, data_source_id, **kwargs) or "job-ds"

    async def get_status(self, status_id: str) -> dict[str, Any]:
        return self._record("get_status", status_id) or {"status": "INPROGRESS"}


def _use_store(monkeypatch: pytest.MonkeyPatch, store: FakeStore) -> FakeStore:
    monkeypatch.setattr(imports_mod, "_get_store", lambda: store)
    return store


async def test_file_import_from_url(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _use_store(monkeypatch, FakeStore())
    result = await imports_mod.file_import(
        path="dr://abc123/data/",
        source="url",
        url="https://example.com/file.zip",
        unpack_archive=False,
        overwrite="replace",
    )
    assert result["status_id"] == "job-url"
    assert result["source"] == "url"
    assert "file_get_status" in result["note"]
    name, args, kwargs = store.calls[0]
    assert name == "import_from_url"
    assert args == ("dr://abc123/data/", "https://example.com/file.zip")
    assert kwargs == {"unpack_archive": False, "overwrite": "replace"}


async def test_file_import_from_data_source(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _use_store(monkeypatch, FakeStore())
    result = await imports_mod.file_import(
        path="dr://abc123/in/",
        source="data_source",
        data_source_id="ds-1",
        credential_id="cred-1",
    )
    assert result["status_id"] == "job-ds"
    name, args, kwargs = store.calls[0]
    assert name == "import_from_data_source"
    assert args == ("dr://abc123/in/", "ds-1")
    assert kwargs["credential_id"] == "cred-1"


async def test_file_import_requires_url_for_url_source(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_store(monkeypatch, FakeStore())
    with pytest.raises(ToolError, match="url"):
        await imports_mod.file_import(path="dr://abc123/data/", source="url")


async def test_file_import_requires_data_source_id(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_store(monkeypatch, FakeStore())
    with pytest.raises(ToolError, match="data_source_id"):
        await imports_mod.file_import(path="dr://abc123/data/", source="data_source")


async def test_file_get_status_without_target(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _use_store(monkeypatch, FakeStore(get_status={"status": "INPROGRESS", "message": "x"}))
    result = await imports_mod.file_get_status(status_id="job123")
    assert result == {
        "status_id": "job123",
        "status": "INPROGRESS",
        "raw": {"status": "INPROGRESS", "message": "x"},
    }
    assert store.calls[0][0] == "get_status"


async def test_file_get_status_target_reached(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_store(monkeypatch, FakeStore(get_status={"status": "completed"}))
    result = await imports_mod.file_get_status(status_id="job123", target_status="completed")
    assert result["target_reached"] is True


async def test_file_get_status_raises_on_terminal_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_store(monkeypatch, FakeStore(get_status={"status": "ERROR"}))
    with pytest.raises(ToolError, match="terminal status"):
        await imports_mod.file_get_status(status_id="job123")
