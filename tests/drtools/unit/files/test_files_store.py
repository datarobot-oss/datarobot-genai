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

"""Unit tests for the Files-API-backed BlobStore.

The DataRobot Files SDK and the per-request client context manager are mocked,
so these tests are hermetic (no network, no credentials).
"""

import functools
import inspect
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from datarobot.errors import ClientError
from datarobot.models import Files

from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.exceptions import ToolErrorKind
from datarobot_genai.drmcputils.files import store as store_mod
from datarobot_genai.drmcputils.files.store import BlobRef
from datarobot_genai.drmcputils.files.store import DataRobotFilesBlobStore


@pytest.fixture
def fake_dr(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch the ``datarobot`` SDK and per-request client CM inside the store module."""
    fake = MagicMock(name="datarobot")
    monkeypatch.setattr(store_mod, "dr", fake)

    @contextmanager
    def _fake_sdk(**_kwargs: object):
        yield fake

    monkeypatch.setattr(store_mod, "request_user_dr_sdk", _fake_sdk)
    return fake


def _files_obj(*, id: str, name: str, tags: list[str] | None = None) -> SimpleNamespace:
    """Build a stand-in for an SDK ``Files`` / ``FilesCatalogSearch`` object.

    Uses ``SimpleNamespace`` (not ``MagicMock``) so absent attributes really are
    absent — important for exercising the ``name``/``catalog_name`` fallbacks.
    """
    return SimpleNamespace(id=id, name=name, tags=list(tags or []))


@pytest.mark.asyncio
async def test_put_uploads_and_returns_ref(fake_dr: MagicMock) -> None:
    fake_dr.models.Files.create_from_file.return_value = _files_obj(
        id="abc12", name="payload.parquet", tags=["panel"]
    )

    ref = await DataRobotFilesBlobStore().put(
        b"hello",
        name="payload.parquet",
        content_type="application/octet-stream",
        tags=["panel"],
    )

    assert ref == BlobRef(files_id="abc12", name="payload.parquet", tags=("panel",))
    _, kwargs = fake_dr.models.Files.create_from_file.call_args
    assert kwargs["use_archive_contents"] is False
    assert kwargs["tags"] == ["panel"]
    filelike = kwargs["filelike"]
    assert filelike.name == "payload.parquet"
    assert filelike.getvalue() == b"hello"


@pytest.mark.asyncio
async def test_put_does_not_expose_content_type_or_size(fake_dr: MagicMock) -> None:
    """content_type is advisory and size is not a field — neither leaks onto the ref."""
    fake_dr.models.Files.create_from_file.return_value = _files_obj(id="id1", name="n")

    ref = await DataRobotFilesBlobStore().put(b"payload-bytes", name="n", content_type="text/csv")

    assert ref == BlobRef(files_id="id1", name="n", tags=())
    assert not hasattr(ref, "content_type")
    assert not hasattr(ref, "size")


@pytest.mark.asyncio
async def test_get_downloads_bytes(fake_dr: MagicMock) -> None:
    container = MagicMock()
    container.download.side_effect = lambda filelike=None, **_kw: filelike.write(b"world")
    fake_dr.models.Files.get.return_value = container

    data = await DataRobotFilesBlobStore().get("abc12")

    assert data == b"world"
    fake_dr.models.Files.get.assert_called_once_with("abc12")


@pytest.mark.asyncio
async def test_get_accepts_blobref(fake_dr: MagicMock) -> None:
    container = MagicMock()
    container.download.side_effect = lambda filelike=None, **_kw: filelike.write(b"x")
    fake_dr.models.Files.get.return_value = container

    await DataRobotFilesBlobStore().get(BlobRef(files_id="zzz99", name="n"))

    fake_dr.models.Files.get.assert_called_once_with("zzz99")


@pytest.mark.asyncio
async def test_delete_uses_files_id(fake_dr: MagicMock) -> None:
    await DataRobotFilesBlobStore().delete(BlobRef(files_id="abc12", name="n"))
    fake_dr.models.Files.delete.assert_called_once_with("abc12")


@pytest.mark.asyncio
async def test_list_maps_results(fake_dr: MagicMock) -> None:
    fake_dr.models.Files.search_catalog.return_value = [
        _files_obj(id="id1", name="one", tags=["t"])
    ]

    refs = await DataRobotFilesBlobStore().list(search="one", limit=10)

    assert refs == [BlobRef(files_id="id1", name="one", tags=("t",))]
    _, kwargs = fake_dr.models.Files.search_catalog.call_args
    assert kwargs["search"] == "one"
    assert kwargs["limit"] == 10


@pytest.mark.asyncio
async def test_list_falls_back_to_catalog_name(fake_dr: MagicMock) -> None:
    """``FilesCatalogSearch`` exposes ``catalog_name`` (mapped to ``name``); cover the fallback."""
    item = SimpleNamespace(id="id2", catalog_name="from-catalog", tags=[])
    fake_dr.models.Files.search_catalog.return_value = [item]

    refs = await DataRobotFilesBlobStore().list()

    assert refs == [BlobRef(files_id="id2", name="from-catalog", tags=())]


@pytest.mark.asyncio
async def test_list_logs_when_page_limit_is_hit(
    fake_dr: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    """When the result count reaches the limit, warn that more may exist."""
    fake_dr.models.Files.search_catalog.return_value = [
        _files_obj(id=f"id{i}", name=f"n{i}") for i in range(3)
    ]

    with caplog.at_level("DEBUG", logger=store_mod.logger.name):
        refs = await DataRobotFilesBlobStore().list(limit=3)

    assert len(refs) == 3
    assert any("page limit" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_put_and_list_refs_are_equal(fake_dr: MagicMock) -> None:
    """The asymmetry fix: a ref from put() equals the same blob's ref from list()."""
    stored = _files_obj(id="p1", name="panel.json", tags=["dr_panel"])
    fake_dr.models.Files.create_from_file.return_value = stored
    fake_dr.models.Files.search_catalog.return_value = [stored]

    store = DataRobotFilesBlobStore()
    put_ref = await store.put(
        b"{}", name="panel.json", content_type="application/json", tags=["dr_panel"]
    )
    [list_ref] = await store.list(search="panel.json")

    assert put_ref == list_ref
    # frozen + tuple tags => the ref stays hashable / dedupable.
    assert {put_ref, list_ref} == {put_ref}


@pytest.mark.parametrize(
    ("op", "sdk_attr"),
    [
        ("put", "create_from_file"),
        ("get", "get"),
        ("delete", "delete"),
        ("list", "search_catalog"),
    ],
)
@pytest.mark.asyncio
async def test_client_error_becomes_tool_error(fake_dr: MagicMock, op: str, sdk_attr: str) -> None:
    getattr(fake_dr.models.Files, sdk_attr).side_effect = ClientError("nope", 403)
    store = DataRobotFilesBlobStore()
    calls = {
        "put": functools.partial(store.put, b"x", name="n"),
        "get": functools.partial(store.get, "id"),
        "delete": functools.partial(store.delete, "id"),
        "list": functools.partial(store.list),
    }

    with pytest.raises(ToolError) as excinfo:
        await calls[op]()
    assert excinfo.value.kind == ToolErrorKind.UPSTREAM


@pytest.mark.asyncio
async def test_404_maps_to_not_found(fake_dr: MagicMock) -> None:
    fake_dr.models.Files.get.side_effect = ClientError("missing", 404)
    with pytest.raises(ToolError) as excinfo:
        await DataRobotFilesBlobStore().get("missing-id")
    assert excinfo.value.kind == ToolErrorKind.NOT_FOUND


# --- Contract: our SDK call kwargs must match the *real* Files API signatures ---
#
# Every test above mocks the SDK, so a renamed/removed kwarg in a future
# `datarobot` release would not be caught there. These bind the exact arguments
# the store passes against the installed SDK's real signatures — hermetic (no
# network), but they fail loudly on signature drift.


def test_create_from_file_signature_matches() -> None:
    inspect.signature(Files.create_from_file).bind(
        filelike=MagicMock(), tags=["t"], use_archive_contents=False
    )


def test_search_catalog_signature_matches() -> None:
    inspect.signature(Files.search_catalog).bind(search="q", tags=["t"], limit=10, offset=0)


def test_get_signature_matches() -> None:
    inspect.signature(Files.get).bind("files-id")


def test_delete_signature_matches() -> None:
    inspect.signature(Files.delete).bind("files-id")


def test_download_signature_matches() -> None:
    # `download` is an instance method: bind a placeholder self plus our kwarg.
    inspect.signature(Files.download).bind(MagicMock(), filelike=MagicMock())
