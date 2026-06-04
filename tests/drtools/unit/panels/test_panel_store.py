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

from datarobot_genai.drtools.files.store import BlobStore
from datarobot_genai.drtools.panels.models import Dataset
from datarobot_genai.drtools.panels.models import PanelType
from datarobot_genai.drtools.panels.models import Text
from datarobot_genai.drtools.panels.store import PanelStore

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
