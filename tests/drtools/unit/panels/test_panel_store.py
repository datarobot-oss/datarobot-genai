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

"""Unit tests for PanelStore (backed by an in-memory FakeBlobStore)."""

import pytest

from datarobot_genai.drmcputils.files.store import BlobStore
from datarobot_genai.drmcputils.panels.models import Dataset
from datarobot_genai.drmcputils.panels.models import PanelType
from datarobot_genai.drmcputils.panels.models import Text
from datarobot_genai.drmcputils.panels.store import PanelStore

from .conftest import FakeBlobStore


def test_fake_blob_store_satisfies_protocol(fake_blob_store: FakeBlobStore) -> None:
    # runtime_checkable Protocol — guards against drift between fake and contract.
    assert isinstance(fake_blob_store, BlobStore)


async def test_create_list_get_delete_roundtrip(fake_blob_store: FakeBlobStore) -> None:
    store = PanelStore(fake_blob_store)

    created = await store.create(Text(title="Hello", text="world"), source="staging")
    assert created.id is not None
    assert created.type == PanelType.TEXT
    assert created.updated_at is not None

    got = await store.get(created.id)
    assert isinstance(got, Text)
    assert got.title == "Hello"
    assert got.text == "world"
    assert got.id == created.id

    listed = await store.list(source="staging")
    assert [p.id for p in listed] == [created.id]
    # Source isolation: the panel is not visible from 'main'.
    assert await store.list(source="main") == []

    await store.delete(created.id)
    assert await store.list(source="staging") == []


async def test_create_with_payload_stores_and_cleans_up_payload_blob(
    fake_blob_store: FakeBlobStore,
) -> None:
    store = PanelStore(fake_blob_store)

    created = await store.create(
        Dataset(title="DS"),
        source="main",
        payload=b"PARQUET-BYTES",
        payload_name="d.parquet",
        content_type="application/octet-stream",
    )
    assert created.payload_files_id is not None
    assert created.payload_name == "d.parquet"
    # Payload is retrievable as its own blob.
    assert await fake_blob_store.get(created.payload_files_id) == b"PARQUET-BYTES"

    # Delete removes both the manifest and the payload blob.
    await store.delete(created.id)
    assert created.payload_files_id not in fake_blob_store.blobs
    assert created.id not in fake_blob_store.blobs


async def test_create_cleans_up_payload_when_manifest_put_fails(
    fake_blob_store: FakeBlobStore,
) -> None:
    store = PanelStore(fake_blob_store)
    original_put = fake_blob_store.put
    calls = {"n": 0}

    async def _put_fail_second(data: bytes, **kwargs: object):
        calls["n"] += 1
        if calls["n"] == 2:  # first put = payload, second = manifest
            raise RuntimeError("manifest write failed")
        return await original_put(data, **kwargs)

    fake_blob_store.put = _put_fail_second  # type: ignore[method-assign]
    panel = Dataset(title="DS")
    with pytest.raises(RuntimeError):
        await store.create(panel, source="main", payload=b"DATA", payload_name="d.parquet")
    # No orphan payload blob is left behind and the panel keeps no stale ref.
    assert fake_blob_store.blobs == {}
    assert panel.payload_files_id is None


async def test_delete_propagates_transient_get_errors(fake_blob_store: FakeBlobStore) -> None:
    store = PanelStore(fake_blob_store)
    created = await store.create(Dataset(title="DS"), source="main", payload=b"DATA")
    original_get = fake_blob_store.get

    async def _get_boom(ref: object):
        raise RuntimeError("transient backend error")

    fake_blob_store.get = _get_boom  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        await store.delete(created.id)
    # Nothing was deleted: the caller can retry without orphaning the payload.
    assert created.id in fake_blob_store.blobs
    assert created.payload_files_id in fake_blob_store.blobs
    fake_blob_store.get = original_get  # type: ignore[method-assign]


async def test_delete_with_corrupt_manifest_still_deletes_manifest(
    fake_blob_store: FakeBlobStore,
) -> None:
    store = PanelStore(fake_blob_store)
    created = await store.create(Dataset(title="DS"), source="main")
    data, ref, tags = fake_blob_store.blobs[created.id]
    fake_blob_store.blobs[created.id] = (b"not-json", ref, tags)
    await store.delete(created.id)
    assert created.id not in fake_blob_store.blobs


async def test_list_pages_with_offset(fake_blob_store: FakeBlobStore) -> None:
    store = PanelStore(fake_blob_store)
    for i in range(3):
        await store.create(Dataset(title=f"DS{i}"), source="main")
    first = await store.list(source="main", limit=2, offset=0)
    rest = await store.list(source="main", limit=2, offset=2)
    assert len(first) == 2
    assert len(rest) == 1
    assert {p.id for p in first} | {p.id for p in rest} == {
        p.id for p in await store.list(source="main")
    }


async def test_list_skips_corrupt_manifests(fake_blob_store: FakeBlobStore) -> None:
    store = PanelStore(fake_blob_store)
    good = await store.create(Dataset(title="good"), source="main")
    bad = await store.create(Dataset(title="bad"), source="main")
    data, ref, tags = fake_blob_store.blobs[bad.id]
    fake_blob_store.blobs[bad.id] = (b'{"title": "no type field"}', ref, tags)
    listed = await store.list(source="main")
    assert [p.id for p in listed] == [good.id]


async def test_delete_handles_manifest_missing_type_key(
    fake_blob_store: FakeBlobStore,
) -> None:
    store = PanelStore(fake_blob_store)
    created = await store.create(Dataset(title="DS"), source="main")
    data, ref, tags = fake_blob_store.blobs[created.id]
    fake_blob_store.blobs[created.id] = (b'{"title": "decodes but no type"}', ref, tags)
    await store.delete(created.id)
    assert created.id not in fake_blob_store.blobs
