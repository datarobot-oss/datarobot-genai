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

import json

import pytest

from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.exceptions import ToolErrorKind
from datarobot_genai.drmcputils.files.store import BlobStore
from datarobot_genai.drmcputils.panels.models import Dataset
from datarobot_genai.drmcputils.panels.models import PanelType
from datarobot_genai.drmcputils.panels.models import Text
from datarobot_genai.drmcputils.panels.store import PanelStore
from datarobot_genai.drmcputils.panels.store import normalize_conversation_id

from .conftest import FakeBlobStore


def _container_paths(fake: FakeBlobStore) -> set[str]:
    return set(fake.container)


def test_fake_blob_store_satisfies_protocol(fake_blob_store: FakeBlobStore) -> None:
    # runtime_checkable Protocol — guards against drift between fake and contract.
    assert isinstance(fake_blob_store, BlobStore)


async def test_create_list_get_delete_roundtrip(fake_blob_store: FakeBlobStore) -> None:
    store = PanelStore(fake_blob_store)

    created = await store.create(Text(title="Hello", text="world"), source="staging")
    assert created.id is not None
    assert created.type == PanelType.TEXT
    assert created.updated_at is not None
    # The manifest lives at <source>/<scope>/<panel_id>.json.
    assert _container_paths(fake_blob_store) == {f"staging/_shared/{created.id}.json"}

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
    assert fake_blob_store.container == {}


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
    # payload_files_id + payload_path form the Files download reference;
    # payload_name stays the caller's display name.
    assert created.payload_files_id == fake_blob_store.container_id
    assert created.payload_name == "d.parquet"
    assert created.payload_path == f"main/_shared/{created.id}.payload"
    assert await store.get_payload(created.id) == b"PARQUET-BYTES"

    # Delete removes both the manifest and the payload blob.
    await store.delete(created.id)
    assert fake_blob_store.container == {}


async def test_manifest_does_not_persist_derived_fields(fake_blob_store: FakeBlobStore) -> None:
    # id and payload_path are derived from blob paths on load; persisting them
    # would force a manifest rewrite on every move.
    store = PanelStore(fake_blob_store)
    created = await store.create(Dataset(title="DS"), source="main", payload=b"DATA")
    manifest = json.loads(fake_blob_store.container[f"main/_shared/{created.id}.json"])
    assert "id" not in manifest
    assert "payload_path" not in manifest
    assert manifest["payload_files_id"] == fake_blob_store.container_id


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
    assert fake_blob_store.container == {}
    assert panel.payload_files_id is None
    assert panel.payload_path is None


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


async def test_list_pages_count_panels_not_blobs(fake_blob_store: FakeBlobStore) -> None:
    # Payload blobs share the listing prefix; limit/offset must count panels.
    store = PanelStore(fake_blob_store)
    for i in range(3):
        await store.create(Dataset(title=f"DS{i}"), source="main", payload=b"DATA")
    assert len(await store.list(source="main", limit=2)) == 2
    assert len(await store.list(source="main", limit=2, offset=2)) == 1


async def test_list_skips_corrupt_manifests(fake_blob_store: FakeBlobStore) -> None:
    store = PanelStore(fake_blob_store)
    good = await store.create(Dataset(title="good"), source="main")
    bad = await store.create(Dataset(title="bad"), source="main")
    fake_blob_store.container[f"main/_shared/{bad.id}.json"] = b'{"title": "no type field"}'
    listed = await store.list(source="main")
    assert [p.id for p in listed] == [good.id]


async def test_delete_with_corrupt_manifest_still_deletes(
    fake_blob_store: FakeBlobStore,
) -> None:
    # Container-layout deletes go by path and never read the manifest, so a
    # corrupt manifest cannot block deletion.
    store = PanelStore(fake_blob_store)
    created = await store.create(Dataset(title="DS"), source="main")
    fake_blob_store.container[f"main/_shared/{created.id}.json"] = b"not-json"
    await store.delete(created.id)
    assert fake_blob_store.container == {}


# --- Validation ---------------------------------------------------------------


async def test_get_rejects_path_like_panel_ids(fake_blob_store: FakeBlobStore) -> None:
    store = PanelStore(fake_blob_store)
    for bad_id in ("../../etc/passwd", "a/b", "", "x" * 129):
        with pytest.raises(ToolError) as excinfo:
            await store.get(bad_id)
        assert excinfo.value.kind == ToolErrorKind.VALIDATION


async def test_create_and_list_reject_path_like_sources(fake_blob_store: FakeBlobStore) -> None:
    store = PanelStore(fake_blob_store)
    with pytest.raises(ToolError) as excinfo:
        await store.create(Text(title="T", text="x"), source="main/../other")
    assert excinfo.value.kind == ToolErrorKind.VALIDATION
    with pytest.raises(ToolError):
        await store.list(source="")


# --- Conversation scoping ---------------------------------------------------


class TestNormalizeConversationId:
    def test_none_and_empty_normalize_to_none(self) -> None:
        # GIVEN no conversation id (missing, empty, or whitespace)
        # THEN normalization yields None (unscoped)
        assert normalize_conversation_id(None) is None
        assert normalize_conversation_id("") is None
        assert normalize_conversation_id("   ") is None

    def test_hyphens_become_underscores(self) -> None:
        # Ids become path segments (and historically tags, which reject "-").
        assert normalize_conversation_id("abc-123-def") == "abc_123_def"

    def test_non_alphanumeric_characters_are_replaced(self) -> None:
        assert normalize_conversation_id("a/b:c d.e") == "a_b_c_d_e"

    def test_length_is_capped_at_128(self) -> None:
        normalized = normalize_conversation_id("x" * 300)
        assert normalized is not None
        assert len(normalized) == 128

    def test_clean_ids_pass_through(self) -> None:
        assert normalize_conversation_id("65f0c0ffee15c0ffee15c0de") == "65f0c0ffee15c0ffee15c0de"


async def test_scoped_create_places_blobs_under_the_conversation_path(
    fake_blob_store: FakeBlobStore,
) -> None:
    # GIVEN a store scoped to a conversation
    store = PanelStore(fake_blob_store, conversation_id="conv-1")
    assert store.conversation_id == "conv_1"

    # WHEN a panel with a payload is created
    created = await store.create(
        Text(title="T", text="hi"), source="staging", payload=b"P", payload_name="t.bin"
    )

    # THEN both blobs live under the <source>/<conversation_id>/ path
    assert _container_paths(fake_blob_store) == {
        f"staging/conv_1/{created.id}.json",
        f"staging/conv_1/{created.id}.payload",
    }
    # AND the panel records its owning conversation
    assert created.conversation_id == "conv_1"


async def test_unscoped_create_uses_the_shared_scope(
    fake_blob_store: FakeBlobStore,
) -> None:
    store = PanelStore(fake_blob_store)
    created = await store.create(Text(title="T", text="hi"), source="staging")
    assert _container_paths(fake_blob_store) == {f"staging/_shared/{created.id}.json"}
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
    # Panel ids are globally unique; get() is not list-scoped, so any
    # conversation's store can resolve a panel it holds the id for.
    unscoped = PanelStore(fake_blob_store)
    other = await unscoped.create(Text(title="old", text="x"), source="main")
    scoped = PanelStore(fake_blob_store, conversation_id="conv-a")
    got = await scoped.get(other.id)
    assert got.id == other.id
    assert got.conversation_id is None


async def test_move_preserves_panel_id_and_payload(fake_blob_store: FakeBlobStore) -> None:
    # GIVEN a staging panel with a payload
    store = PanelStore(fake_blob_store, conversation_id="conv-a")
    created = await store.create(
        Dataset(title="DS"), source="staging", payload=b"DATA", payload_name="d.parquet"
    )

    # WHEN it is moved (promoted) to main
    moved = await store.move(created.id, to_source="main")

    # THEN the id is preserved (the move is an in-place path rename, not copy+delete)
    assert moved.id == created.id
    assert [p.id for p in await store.list(source="main")] == [created.id]
    assert await store.list(source="staging") == []
    # AND the payload blob moved with it and is still readable
    assert await store.get_payload(created.id) == b"DATA"
    assert _container_paths(fake_blob_store) == {
        f"main/conv_a/{created.id}.json",
        f"main/conv_a/{created.id}.payload",
    }


async def test_move_keeps_the_panels_own_conversation_scope(
    fake_blob_store: FakeBlobStore,
) -> None:
    # GIVEN a panel created inside a conversation
    scoped = PanelStore(fake_blob_store, conversation_id="conv-a")
    created = await scoped.create(Text(title="T", text="x"), source="staging")

    # WHEN an unscoped store performs the move (e.g. an admin/global consumer)
    unscoped = PanelStore(fake_blob_store)
    moved = await unscoped.move(created.id, to_source="main")

    # THEN the panel stays in its original conversation (not the mover's scope)
    assert moved.id == created.id
    assert [p.id for p in await scoped.list(source="main")] == [created.id]
    assert _container_paths(fake_blob_store) == {f"main/conv_a/{created.id}.json"}


async def test_move_to_the_same_source_is_a_noop(fake_blob_store: FakeBlobStore) -> None:
    store = PanelStore(fake_blob_store)
    created = await store.create(Text(title="T", text="x"), source="main")
    moved = await store.move(created.id, to_source="main")
    assert moved.id == created.id
    assert _container_paths(fake_blob_store) == {f"main/_shared/{created.id}.json"}


async def test_move_keeps_manifest_content(fake_blob_store: FakeBlobStore) -> None:
    store = PanelStore(fake_blob_store)
    created = await store.create(Dataset(title="DS"), source="staging")
    moved = await store.move(created.id, to_source="main")
    assert isinstance(moved, Dataset)
    assert moved.title == "DS"


async def test_move_rolls_payload_back_when_manifest_move_fails(
    fake_blob_store: FakeBlobStore,
) -> None:
    store = PanelStore(fake_blob_store)
    created = await store.create(Dataset(title="DS"), source="staging", payload=b"DATA")
    original_move = fake_blob_store.move
    calls = {"n": 0}

    async def _move_fail_second(from_path: str, to_path: str):
        calls["n"] += 1
        if calls["n"] == 2:  # first move = payload, second = manifest
            raise RuntimeError("manifest move failed")
        return await original_move(from_path, to_path)

    fake_blob_store.move = _move_fail_second  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        await store.move(created.id, to_source="main")
    # The panel is intact in its original source, payload co-located.
    fake_blob_store.move = original_move  # type: ignore[method-assign]
    assert _container_paths(fake_blob_store) == {
        f"staging/_shared/{created.id}.json",
        f"staging/_shared/{created.id}.payload",
    }
    assert await store.get_payload(created.id) == b"DATA"


async def test_move_rejects_missing_panel_id(fake_blob_store: FakeBlobStore) -> None:
    store = PanelStore(fake_blob_store)
    with pytest.raises(ToolError) as excinfo:
        await store.move("doesnotexist", to_source="main")
    assert excinfo.value.kind == ToolErrorKind.NOT_FOUND


# --- Legacy panels (pre shared-container layout) ------------------------------


def _seed_legacy_panel(
    fake: FakeBlobStore, *, manifest_id: str = "legacymanifest1", with_payload: bool = False
) -> str:
    manifest = {
        "type": "dataset",
        "title": "old",
        "payload_files_id": "legacypayload1" if with_payload else None,
        "payload_name": "d.parquet" if with_payload else None,
    }
    fake.legacy[manifest_id] = json.dumps(manifest).encode("utf-8")
    if with_payload:
        fake.legacy["legacypayload1"] = b"LEGACY-DATA"
    return manifest_id


async def test_legacy_panel_stays_reachable_by_id(fake_blob_store: FakeBlobStore) -> None:
    panel_id = _seed_legacy_panel(fake_blob_store, with_payload=True)
    store = PanelStore(fake_blob_store, conversation_id="conv-a")
    got = await store.get(panel_id)
    assert got.id == panel_id
    assert got.title == "old"
    # The payload path is unknown for legacy panels; payload_files_id alone
    # identifies the standalone payload container.
    assert got.payload_path is None
    assert await store.get_payload(got) == b"LEGACY-DATA"


async def test_legacy_panel_delete_removes_manifest_and_payload(
    fake_blob_store: FakeBlobStore,
) -> None:
    panel_id = _seed_legacy_panel(fake_blob_store, with_payload=True)
    await PanelStore(fake_blob_store).delete(panel_id)
    assert fake_blob_store.legacy == {}


async def test_legacy_panel_delete_propagates_transient_get_errors(
    fake_blob_store: FakeBlobStore,
) -> None:
    panel_id = _seed_legacy_panel(fake_blob_store, with_payload=True)
    store = PanelStore(fake_blob_store)
    original_get = fake_blob_store.get

    async def _get_boom(path: str):
        raise RuntimeError("transient backend error")

    fake_blob_store.get = _get_boom  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        await store.delete(panel_id)
    # Nothing was deleted: the caller can retry without orphaning the payload.
    fake_blob_store.get = original_get  # type: ignore[method-assign]
    assert set(fake_blob_store.legacy) == {panel_id, "legacypayload1"}


async def test_legacy_panel_delete_with_corrupt_manifest_deletes_manifest_only(
    fake_blob_store: FakeBlobStore,
) -> None:
    fake_blob_store.legacy["legacymanifest1"] = b"not-json"
    await PanelStore(fake_blob_store).delete("legacymanifest1")
    assert "legacymanifest1" not in fake_blob_store.legacy


async def test_legacy_panel_cannot_be_moved(fake_blob_store: FakeBlobStore) -> None:
    panel_id = _seed_legacy_panel(fake_blob_store)
    with pytest.raises(ToolError) as excinfo:
        await PanelStore(fake_blob_store).move(panel_id, to_source="main")
    assert excinfo.value.kind == ToolErrorKind.NOT_FOUND
    # The legacy blob is untouched.
    assert panel_id in fake_blob_store.legacy
