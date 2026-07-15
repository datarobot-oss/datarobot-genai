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
from datarobot_genai.drmcputils.panels.store import normalize_conversation_id

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
    # Blobs live under the folder-style "panels/" prefix (registry structure).
    assert created.payload_name == "panels/d.parquet"
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


# --- Conversation scoping ---------------------------------------------------


class TestNormalizeConversationId:
    def test_none_and_empty_normalize_to_none(self) -> None:
        # GIVEN no conversation id (missing, empty, or whitespace)
        # THEN normalization yields None (unscoped)
        assert normalize_conversation_id(None) is None
        assert normalize_conversation_id("") is None
        assert normalize_conversation_id("   ") is None

    def test_hyphens_become_underscores_for_files_tag_safety(self) -> None:
        # The Files API rejects tags containing "-" (422), so ids are made tag-safe.
        assert normalize_conversation_id("abc-123-def") == "abc_123_def"

    def test_non_alphanumeric_characters_are_replaced(self) -> None:
        assert normalize_conversation_id("a/b:c d.e") == "a_b_c_d_e"

    def test_length_is_capped_at_128(self) -> None:
        normalized = normalize_conversation_id("x" * 300)
        assert normalized is not None
        assert len(normalized) == 128

    def test_clean_ids_pass_through(self) -> None:
        assert normalize_conversation_id("65f0c0ffee15c0ffee15c0de") == "65f0c0ffee15c0ffee15c0de"


async def test_scoped_create_uses_conversation_folder_name_and_tag(
    fake_blob_store: FakeBlobStore,
) -> None:
    # GIVEN a store scoped to a conversation
    store = PanelStore(fake_blob_store, conversation_id="conv-1")
    assert store.conversation_id == "conv_1"

    # WHEN a panel with a payload is created
    created = await store.create(
        Text(title="T", text="hi"), source="staging", payload=b"P", payload_name="t.bin"
    )

    # THEN both blobs live under the panels/<conversation_id>/ folder prefix
    _, manifest_ref, manifest_tags = fake_blob_store.blobs[created.id]
    assert manifest_ref.name == "panels/conv_1/panel-text.json"
    _, payload_ref, payload_tags = fake_blob_store.blobs[created.payload_files_id]
    assert payload_ref.name == "panels/conv_1/t.bin"
    # AND both carry the conversation tag for scoped discovery
    assert "dr_panel_conversation:conv_1" in manifest_tags
    assert "dr_panel_conversation:conv_1" in payload_tags
    # AND the panel records its owning conversation
    assert created.conversation_id == "conv_1"


async def test_unscoped_create_uses_root_panels_folder_without_conversation_tag(
    fake_blob_store: FakeBlobStore,
) -> None:
    store = PanelStore(fake_blob_store)
    created = await store.create(Text(title="T", text="hi"), source="staging")
    _, manifest_ref, manifest_tags = fake_blob_store.blobs[created.id]
    assert manifest_ref.name == "panels/panel-text.json"
    assert not any(t.startswith("dr_panel_conversation:") for t in manifest_tags)
    assert created.conversation_id is None


async def test_list_is_scoped_to_the_conversation(fake_blob_store: FakeBlobStore) -> None:
    # GIVEN panels created in two conversations plus one unscoped panel
    store_a = PanelStore(fake_blob_store, conversation_id="conv-a")
    store_b = PanelStore(fake_blob_store, conversation_id="conv-b")
    unscoped = PanelStore(fake_blob_store)
    in_a = await store_a.create(Text(title="A", text="a"), source="staging")
    in_b = await store_b.create(Text(title="B", text="b"), source="staging")
    loose = await unscoped.create(Text(title="L", text="l"), source="staging")

    # WHEN each conversation lists its staging panels
    listed_a = await store_a.list(source="staging")
    listed_b = await store_b.list(source="staging")

    # THEN each sees only its own panels (no cross-conversation leak)
    assert [p.id for p in listed_a] == [in_a.id]
    assert [p.id for p in listed_b] == [in_b.id]
    # AND the unscoped store keeps the legacy global view (existing panels stay reachable)
    assert {p.id for p in await unscoped.list(source="staging")} == {in_a.id, in_b.id, loose.id}


async def test_scoped_get_reads_any_panel_by_id(fake_blob_store: FakeBlobStore) -> None:
    # Panel ids are globally unique blob ids; get() is not list-scoped, so panels
    # created before conversation scoping existed stay reachable by id.
    unscoped = PanelStore(fake_blob_store)
    legacy = await unscoped.create(Text(title="old", text="x"), source="main")
    scoped = PanelStore(fake_blob_store, conversation_id="conv-a")
    got = await scoped.get(legacy.id)
    assert got.id == legacy.id
    assert got.conversation_id is None


async def test_move_preserves_panel_id_and_payload(fake_blob_store: FakeBlobStore) -> None:
    # GIVEN a staging panel with a payload
    store = PanelStore(fake_blob_store, conversation_id="conv-a")
    created = await store.create(
        Dataset(title="DS"), source="staging", payload=b"DATA", payload_name="d.parquet"
    )

    # WHEN it is moved (promoted) to main
    moved = await store.move(created.id, to_source="main")

    # THEN the id is preserved (the move is an in-place retag, not copy+delete)
    assert moved.id == created.id
    assert [p.id for p in await store.list(source="main")] == [created.id]
    assert await store.list(source="staging") == []
    # AND the payload blob moved with it and is still readable
    assert await store.get_payload(created.id) == b"DATA"
    _, _, payload_tags = fake_blob_store.blobs[created.payload_files_id]
    assert "dr_panel_source:main" in payload_tags
    assert "dr_panel_source:staging" not in payload_tags


async def test_move_keeps_the_panels_own_conversation_scope(
    fake_blob_store: FakeBlobStore,
) -> None:
    # GIVEN a panel created inside a conversation
    scoped = PanelStore(fake_blob_store, conversation_id="conv-a")
    created = await scoped.create(Text(title="T", text="x"), source="staging")

    # WHEN an unscoped store performs the move (e.g. an admin/global consumer)
    unscoped = PanelStore(fake_blob_store)
    moved = await unscoped.move(created.id, to_source="main")

    # THEN the panel stays in its original conversation
    assert moved.id == created.id
    assert [p.id for p in await scoped.list(source="main")] == [created.id]
    _, _, tags = fake_blob_store.blobs[created.id]
    assert "dr_panel_conversation:conv_a" in tags


async def test_move_retains_type_tag(fake_blob_store: FakeBlobStore) -> None:
    store = PanelStore(fake_blob_store)
    created = await store.create(Dataset(title="DS"), source="staging")
    await store.move(created.id, to_source="main")
    _, _, tags = fake_blob_store.blobs[created.id]
    assert "dr_panel" in tags
    assert "dr_panel_type:dataset" in tags


async def test_move_rejects_missing_panel_id(fake_blob_store: FakeBlobStore) -> None:
    store = PanelStore(fake_blob_store)
    with pytest.raises(KeyError):
        await store.move("does-not-exist", to_source="main")
