# Copyright 2025 DataRobot, Inc.
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

"""Tests for MemoryAPI."""

import pytest

from datarobot_genai.drmcp.core.resource_store.memory import MemoryAPI
from datarobot_genai.drmcp.core.resource_store.store import ResourceStore


@pytest.mark.asyncio
class TestMemoryAPI:
    """Tests for MemoryAPI."""

    async def test_write(self, store: ResourceStore) -> None:
        """Test writing a memory entry."""
        memory = MemoryAPI(store)
        resource_id = await memory.write(
            scope_id="user_123",
            kind="note",
            content="Remember to buy milk",
            metadata={"tag": "shopping"},
        )
        assert resource_id is not None

        # Verify stored
        result = await store.get(resource_id)
        assert result is not None
        resource, data = result
        assert resource.scope.type == "memory"
        assert resource.scope.id == "user_123"
        assert resource.kind == "note"
        assert resource.lifetime == "persistent"
        assert data == "Remember to buy milk"
        assert resource.metadata["tag"] == "shopping"

    async def test_read(self, store: ResourceStore) -> None:
        """Test reading a memory entry."""
        memory = MemoryAPI(store)
        resource_id = await memory.write(
            scope_id="user_123",
            kind="preference",
            content='{"theme": "dark"}',
            metadata={"category": "ui"},
        )

        result = await memory.read(resource_id)
        assert result is not None
        assert result["id"] == resource_id
        assert result["kind"] == "preference"
        assert result["content"] == '{"theme": "dark"}'
        assert result["metadata"]["category"] == "ui"

    async def test_read_nonexistent(self, store: ResourceStore) -> None:
        """Test reading non-existent memory."""
        memory = MemoryAPI(store)
        result = await memory.read("nonexistent")
        assert result is None

    async def test_search_by_scope(self, store: ResourceStore) -> None:
        """Test searching memories by scope."""
        memory = MemoryAPI(store)

        await memory.write("user_123", "note", "Note 1", {"tag": "important"})
        await memory.write("user_123", "note", "Note 2", {"tag": "unimportant"})
        await memory.write("user_456", "note", "Note 3", {"tag": "important"})

        results = await memory.search("user_123")
        assert len(results) == 2

    async def test_search_by_kind(self, store: ResourceStore) -> None:
        """Test searching memories by kind."""
        memory = MemoryAPI(store)

        await memory.write("user_123", "note", "Note 1")
        await memory.write("user_123", "preference", "Pref 1")
        await memory.write("user_123", "note", "Note 2")

        results = await memory.search("user_123", kind="note")
        assert len(results) == 2
        assert all(r["kind"] == "note" for r in results)

    async def test_search_by_metadata(self, store: ResourceStore) -> None:
        """Test searching memories by metadata."""
        memory = MemoryAPI(store)

        await memory.write("user_123", "note", "Note 1", {"tag": "important", "category": "work"})
        await memory.write(
            "user_123", "note", "Note 2", {"tag": "important", "category": "personal"}
        )
        await memory.write("user_123", "note", "Note 3", {"tag": "unimportant"})

        results = await memory.search("user_123", metadata={"tag": "important"})
        assert len(results) == 2

    async def test_delete(self, store: ResourceStore) -> None:
        """Test deleting a memory entry."""
        memory = MemoryAPI(store)
        resource_id = await memory.write("user_123", "note", "To be deleted")

        assert await memory.read(resource_id) is not None

        success = await memory.delete(resource_id)
        assert success is True

        assert await memory.read(resource_id) is None

    async def test_delete_nonexistent(self, store: ResourceStore) -> None:
        """Test deleting non-existent memory."""
        memory = MemoryAPI(store)
        success = await memory.delete("nonexistent")
        assert success is False

    async def test_delete_non_memory_resource(self, store: ResourceStore) -> None:
        """Test that delete only works for memory resources."""
        memory = MemoryAPI(store)

        # Create a non-memory resource
        from datarobot_genai.drmcp.core.resource_store.models import Scope  # noqa: PLC0415

        await store.put(
            scope=Scope(type="conversation", id="conv_123"),
            kind="message",
            data="test",
            lifetime="ephemeral",
            contentType="text/plain",
            resource_id="non_memory_res",
        )

        success = await memory.delete("non_memory_res")
        assert success is False
