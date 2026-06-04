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

from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest
from datarobot.errors import ClientError

from datarobot_genai.drtools.core.exceptions import ToolError
from datarobot_genai.drtools.files import BlobRef
from datarobot_genai.drtools.files import DataRobotFilesBlobStore
from datarobot_genai.drtools.files import store as store_mod


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


@pytest.mark.asyncio
async def test_put_uploads_and_returns_ref(fake_dr: MagicMock) -> None:
    uploaded = MagicMock()
    uploaded.id = "abc12"
    uploaded.name = "payload.parquet"
    fake_dr.models.Files.create_from_file.return_value = uploaded

    ref = await DataRobotFilesBlobStore().put(
        b"hello",
        name="payload.parquet",
        content_type="application/octet-stream",
        tags=["panel"],
    )

    assert ref == BlobRef(
        files_id="abc12",
        name="payload.parquet",
        content_type="application/octet-stream",
        size=5,
    )
    _, kwargs = fake_dr.models.Files.create_from_file.call_args
    assert kwargs["use_archive_contents"] is False
    assert kwargs["tags"] == ["panel"]
    filelike = kwargs["filelike"]
    assert filelike.name == "payload.parquet"
    assert filelike.getvalue() == b"hello"


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
    item = MagicMock()
    item.id = "id1"
    item.name = "one"
    fake_dr.models.Files.search_catalog.return_value = [item]

    refs = await DataRobotFilesBlobStore().list(search="one", limit=10)

    assert refs == [BlobRef(files_id="id1", name="one")]
    _, kwargs = fake_dr.models.Files.search_catalog.call_args
    assert kwargs["search"] == "one"
    assert kwargs["limit"] == 10


@pytest.mark.asyncio
async def test_client_error_becomes_tool_error(fake_dr: MagicMock) -> None:
    fake_dr.models.Files.create_from_file.side_effect = ClientError("forbidden", 403)
    with pytest.raises(ToolError):
        await DataRobotFilesBlobStore().put(b"x", name="n")
