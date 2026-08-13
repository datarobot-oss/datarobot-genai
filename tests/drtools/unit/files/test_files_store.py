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

"""Unit tests for the Files-API-backed shared-container BlobStore.

The DataRobot Files SDK and the per-request client context manager are mocked,
so these tests are hermetic (no network, no credentials).
"""

import inspect
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from datarobot.enums import FilesOverwriteStrategy
from datarobot.errors import ClientError
from datarobot.models import Files

from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.exceptions import ToolErrorKind
from datarobot_genai.drmcputils.files import store as store_mod
from datarobot_genai.drmcputils.files.store import CONTAINER_NAME
from datarobot_genai.drmcputils.files.store import CONTAINER_TAG
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


def _catalog_entry(*, id: str, tags: list[str] | None = None) -> SimpleNamespace:
    """Build a stand-in for an SDK ``FilesCatalogSearch`` result."""
    return SimpleNamespace(id=id, tags=list(tags or [CONTAINER_TAG]))


def _contained_file(*, name: str, size: int = 1) -> SimpleNamespace:
    """Build a stand-in for an SDK ``File`` (``name`` is the container-relative path)."""
    return SimpleNamespace(name=name, size=size)


def _with_container(fake_dr: MagicMock, *, container_id: str = "c1") -> MagicMock:
    """Wire the fake SDK so container discovery finds one existing container."""
    fake_dr.models.Files.search_catalog.return_value = [_catalog_entry(id=container_id)]
    container = MagicMock(name="container")
    container.id = container_id
    fake_dr.models.Files.get.return_value = container
    return container


# --- put -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_put_uploads_into_the_shared_container(fake_dr: MagicMock) -> None:
    container = _with_container(fake_dr)

    ref = await DataRobotFilesBlobStore().put(
        b"hello", path="main/_shared/abc.json", content_type="application/json"
    )

    assert ref == BlobRef(path="main/_shared/abc.json", container_id="c1", size=5)
    args, kwargs = container.upload_file.call_args
    # The folder travels via ``prefix`` (joined server-side after filename
    # sanitization); only the basename is the uploaded filename.
    assert kwargs["prefix"] == "main/_shared/"
    assert kwargs["file_name"] == "abc.json"
    assert kwargs["use_archive_contents"] is False
    assert kwargs["overwrite"] is FilesOverwriteStrategy.REPLACE
    assert args[0].getvalue() == b"hello"
    fake_dr.models.Files.create_empty_catalog_item_dir.assert_not_called()


@pytest.mark.asyncio
async def test_put_creates_the_container_on_first_write(fake_dr: MagicMock) -> None:
    # No container exists yet (both before and after the create re-find).
    fake_dr.models.Files.search_catalog.return_value = []
    created = MagicMock(name="created")
    created.id = "new1"
    fake_dr.models.Files.create_empty_catalog_item_dir.return_value = created

    ref = await DataRobotFilesBlobStore().put(b"x", path="a/b/c.json")

    assert ref.container_id == "new1"
    created.modify.assert_called_once_with(name=CONTAINER_NAME, tags=[CONTAINER_TAG])
    created.upload_file.assert_called_once()


@pytest.mark.asyncio
async def test_container_discovery_prefers_the_oldest_and_requires_exact_tag(
    fake_dr: MagicMock,
) -> None:
    # A create race left two containers; a loose tag match also sneaks in.
    fake_dr.models.Files.search_catalog.return_value = [
        _catalog_entry(id="bbb"),
        _catalog_entry(id="zzz", tags=["unrelated_tag"]),
        _catalog_entry(id="aaa"),
    ]
    container = MagicMock(name="container")
    container.id = "aaa"
    fake_dr.models.Files.get.return_value = container

    await DataRobotFilesBlobStore().put(b"x", path="a/b.json")

    fake_dr.models.Files.get.assert_called_once_with("aaa")


@pytest.mark.asyncio
async def test_container_is_cached_per_store_instance(fake_dr: MagicMock) -> None:
    _with_container(fake_dr)
    store = DataRobotFilesBlobStore()
    await store.put(b"x", path="a/b.json")
    await store.list(prefix="a/")
    fake_dr.models.Files.search_catalog.assert_called_once()


# --- get -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_downloads_by_path(fake_dr: MagicMock) -> None:
    container = _with_container(fake_dr)
    container.download.side_effect = lambda file_name=None, filelike=None: filelike.write(b"world")

    data = await DataRobotFilesBlobStore().get("main/_shared/abc.json")

    assert data == b"world"
    _, kwargs = container.download.call_args
    assert kwargs["file_name"] == "main/_shared/abc.json"


@pytest.mark.asyncio
async def test_get_bare_id_falls_back_to_the_legacy_standalone_container(
    fake_dr: MagicMock,
) -> None:
    legacy = MagicMock(name="legacy")
    legacy.download.side_effect = lambda filelike=None, **_kw: filelike.write(b"old")
    fake_dr.models.Files.get.return_value = legacy

    data = await DataRobotFilesBlobStore().get("legacyid1")

    assert data == b"old"
    fake_dr.models.Files.get.assert_called_once_with("legacyid1")
    # The shared container is never resolved for a legacy read.
    fake_dr.models.Files.search_catalog.assert_not_called()


@pytest.mark.asyncio
async def test_get_by_path_without_a_container_is_not_found(fake_dr: MagicMock) -> None:
    fake_dr.models.Files.search_catalog.return_value = []
    with pytest.raises(ToolError) as excinfo:
        await DataRobotFilesBlobStore().get("a/b.json")
    assert excinfo.value.kind == ToolErrorKind.NOT_FOUND


# --- delete ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_batches_container_paths(fake_dr: MagicMock) -> None:
    container = _with_container(fake_dr)
    await DataRobotFilesBlobStore().delete(["a/b/c.json", "a/b/c.payload"])
    container.delete_files.assert_called_once_with(["a/b/c.json", "a/b/c.payload"])


@pytest.mark.asyncio
async def test_delete_accepts_a_single_path(fake_dr: MagicMock) -> None:
    container = _with_container(fake_dr)
    await DataRobotFilesBlobStore().delete("a/b/c.json")
    container.delete_files.assert_called_once_with(["a/b/c.json"])


@pytest.mark.asyncio
async def test_delete_bare_ids_use_the_legacy_layout(fake_dr: MagicMock) -> None:
    await DataRobotFilesBlobStore().delete(["legacyid1", "legacyid2"])
    assert [c.args for c in fake_dr.models.Files.delete.call_args_list] == [
        ("legacyid1",),
        ("legacyid2",),
    ]


@pytest.mark.asyncio
async def test_delete_by_path_without_a_container_is_a_noop(fake_dr: MagicMock) -> None:
    fake_dr.models.Files.search_catalog.return_value = []
    await DataRobotFilesBlobStore().delete("a/b.json")  # nothing was ever stored
    fake_dr.models.Files.delete.assert_not_called()


# --- move ------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_move_renames_in_place(fake_dr: MagicMock) -> None:
    container = _with_container(fake_dr)
    await DataRobotFilesBlobStore().move("staging/c/x.json", "main/c/x.json")
    container.rename_files.assert_called_once_with(
        "staging/c/x.json", "main/c/x.json", overwrite=FilesOverwriteStrategy.REPLACE
    )


# --- list ------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_queries_by_prefix_and_maps_results(fake_dr: MagicMock) -> None:
    container = _with_container(fake_dr)
    container.list_contained_files.return_value = [
        _contained_file(name="main/_shared/a.json", size=12),
    ]

    refs = await DataRobotFilesBlobStore().list(prefix="main/_shared/", limit=10, offset=0)

    assert refs == [BlobRef(path="main/_shared/a.json", container_id="c1", size=12)]
    _, kwargs = container.list_contained_files.call_args
    assert kwargs["prefix"] == "main/_shared/"
    assert kwargs["limit"] == 10
    assert kwargs["offset"] == 0
    assert kwargs["recursive"] is True


@pytest.mark.asyncio
async def test_list_without_a_container_returns_empty(fake_dr: MagicMock) -> None:
    fake_dr.models.Files.search_catalog.return_value = []
    assert await DataRobotFilesBlobStore().list(prefix="main/") == []


# --- error mapping ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_upstream_client_error_becomes_tool_error(fake_dr: MagicMock) -> None:
    container = _with_container(fake_dr)
    container.upload_file.side_effect = ClientError("nope", 403)
    with pytest.raises(ToolError) as excinfo:
        await DataRobotFilesBlobStore().put(b"x", path="a/b.json")
    assert excinfo.value.kind == ToolErrorKind.UPSTREAM


@pytest.mark.asyncio
async def test_404_maps_to_not_found(fake_dr: MagicMock) -> None:
    container = _with_container(fake_dr)
    container.download.side_effect = ClientError("missing", 404)
    with pytest.raises(ToolError) as excinfo:
        await DataRobotFilesBlobStore().get("a/missing.json")
    assert excinfo.value.kind == ToolErrorKind.NOT_FOUND


@pytest.mark.parametrize("op", ["get", "delete", "move", "list"])
@pytest.mark.asyncio
async def test_container_resolution_errors_are_mapped(fake_dr: MagicMock, op: str) -> None:
    fake_dr.models.Files.search_catalog.side_effect = ClientError("nope", 403)
    store = DataRobotFilesBlobStore()
    calls = {
        "get": lambda: store.get("a/b.json"),
        "delete": lambda: store.delete("a/b.json"),
        "move": lambda: store.move("a/b.json", "c/b.json"),
        "list": lambda: store.list(prefix="a/"),
    }
    with pytest.raises(ToolError) as excinfo:
        await calls[op]()
    assert excinfo.value.kind == ToolErrorKind.UPSTREAM


# --- Contract: our SDK call kwargs must match the *real* Files API signatures ---
#
# Every test above mocks the SDK, so a renamed/removed kwarg in a future
# `datarobot` release would not be caught there. These bind the exact arguments
# the store passes against the installed SDK's real signatures — hermetic (no
# network), but they fail loudly on signature drift.


def test_search_catalog_signature_matches() -> None:
    inspect.signature(Files.search_catalog).bind(tags=[CONTAINER_TAG], limit=100)


def test_get_signature_matches() -> None:
    inspect.signature(Files.get).bind("files-id")


def test_delete_signature_matches() -> None:
    inspect.signature(Files.delete).bind("files-id")


def test_create_empty_catalog_item_dir_signature_matches() -> None:
    inspect.signature(Files.create_empty_catalog_item_dir).bind()


def test_modify_signature_matches() -> None:
    inspect.signature(Files.modify).bind(MagicMock(), name=CONTAINER_NAME, tags=[CONTAINER_TAG])


def test_upload_file_signature_matches() -> None:
    inspect.signature(Files.upload_file).bind(
        MagicMock(),
        MagicMock(),
        prefix="a/b/",
        file_name="c.json",
        use_archive_contents=False,
        overwrite=FilesOverwriteStrategy.REPLACE,
        read_timeout=600,
        max_wait=600,
    )


def test_download_signature_matches() -> None:
    inspect.signature(Files.download).bind(MagicMock(), file_name="a/b.json", filelike=MagicMock())


def test_list_contained_files_signature_matches() -> None:
    inspect.signature(Files.list_contained_files).bind(
        MagicMock(), prefix="a/", limit=100, offset=0, recursive=True
    )


def test_rename_files_signature_matches() -> None:
    inspect.signature(Files.rename_files).bind(
        MagicMock(), "a/b.json", "c/b.json", overwrite=FilesOverwriteStrategy.REPLACE
    )


def test_delete_files_signature_matches() -> None:
    inspect.signature(Files.delete_files).bind(MagicMock(), ["a/b.json"])
