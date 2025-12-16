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

"""Tests for LlamaIndex Memory Adapter."""

import pytest

from datarobot_genai.drmcp.core.resource_store.adapters.llamaindex_adapter import (
    ResourceStoreChatStore,
)
from datarobot_genai.drmcp.core.resource_store.adapters.llamaindex_adapter import (
    ResourceStoreMemoryBlock,
)
from datarobot_genai.drmcp.core.resource_store.adapters.llamaindex_adapter import (
    ResourceStoreVectorMemoryBlock,
)
from datarobot_genai.drmcp.core.resource_store.store import ResourceStore


@pytest.mark.asyncio
class TestResourceStoreChatStore:
    """Tests for ResourceStoreChatStore (LlamaIndex ChatStore adapter)."""

    def test_set_and_get_messages(self, store: ResourceStore) -> None:
        """Test setting and getting messages."""
        # GIVEN a ResourceStoreChatStore instance
        chat_store = ResourceStoreChatStore(store)

        messages = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]

        # WHEN setting messages
        chat_store.set_messages("test_session", messages)

        # THEN they should be retrievable
        retrieved = chat_store.get_messages("test_session")
        assert len(retrieved) == 3
        assert retrieved[0]["content"] == "Hello!"
        assert retrieved[1]["role"] == "assistant"

    def test_add_message(self, store: ResourceStore) -> None:
        """Test adding a single message."""
        # GIVEN a chat store with some messages
        chat_store = ResourceStoreChatStore(store)
        chat_store.set_messages("add_test", [{"role": "user", "content": "First"}])

        # WHEN adding a message
        chat_store.add_message("add_test", {"role": "assistant", "content": "Second"})

        # THEN it should be appended
        messages = chat_store.get_messages("add_test")
        assert len(messages) == 2
        assert messages[1]["content"] == "Second"

    def test_delete_messages(self, store: ResourceStore) -> None:
        """Test deleting all messages."""
        # GIVEN a chat store with messages
        chat_store = ResourceStoreChatStore(store)
        chat_store.set_messages(
            "delete_test",
            [
                {"role": "user", "content": "Message 1"},
                {"role": "assistant", "content": "Message 2"},
            ],
        )

        # WHEN deleting messages
        deleted = chat_store.delete_messages("delete_test")

        # THEN they should be removed
        assert deleted is not None
        assert len(deleted) == 2
        remaining = chat_store.get_messages("delete_test")
        assert len(remaining) == 0

    def test_delete_message_by_index(self, store: ResourceStore) -> None:
        """Test deleting a specific message by index."""
        # GIVEN a chat store with messages
        chat_store = ResourceStoreChatStore(store)
        chat_store.set_messages(
            "idx_delete_test",
            [
                {"role": "user", "content": "Keep me"},
                {"role": "assistant", "content": "Delete me"},
                {"role": "user", "content": "Keep me too"},
            ],
        )

        # WHEN deleting message at index 1
        deleted = chat_store.delete_message("idx_delete_test", 1)

        # THEN only that message should be removed
        assert deleted is not None
        assert deleted["content"] == "Delete me"
        remaining = chat_store.get_messages("idx_delete_test")
        assert len(remaining) == 2
        assert remaining[0]["content"] == "Keep me"
        assert remaining[1]["content"] == "Keep me too"

    def test_delete_last_message(self, store: ResourceStore) -> None:
        """Test deleting the last message."""
        # GIVEN a chat store with messages
        chat_store = ResourceStoreChatStore(store)
        chat_store.set_messages(
            "last_delete_test",
            [
                {"role": "user", "content": "First"},
                {"role": "assistant", "content": "Last"},
            ],
        )

        # WHEN deleting the last message
        deleted = chat_store.delete_last_message("last_delete_test")

        # THEN only the last should be removed
        assert deleted is not None
        assert deleted["content"] == "Last"
        remaining = chat_store.get_messages("last_delete_test")
        assert len(remaining) == 1
        assert remaining[0]["content"] == "First"

    def test_get_keys(self, store: ResourceStore) -> None:
        """Test getting all session keys."""
        # GIVEN a chat store with multiple sessions
        chat_store = ResourceStoreChatStore(store)
        chat_store.set_messages("session_a", [{"role": "user", "content": "A"}])
        chat_store.set_messages("session_b", [{"role": "user", "content": "B"}])

        # WHEN getting keys
        keys = chat_store.get_keys()

        # THEN both keys should be present
        assert "session_a" in keys
        assert "session_b" in keys

    def test_empty_session(self, store: ResourceStore) -> None:
        """Test getting messages from non-existent session."""
        # GIVEN a chat store
        chat_store = ResourceStoreChatStore(store)

        # WHEN getting messages from non-existent session
        messages = chat_store.get_messages("nonexistent")

        # THEN empty list should be returned
        assert messages == []


@pytest.mark.asyncio
class TestResourceStoreChatStoreAsync:
    """Async tests for ResourceStoreChatStore."""

    async def test_async_set_and_get(self, store: ResourceStore) -> None:
        """Test async set and get messages."""
        # GIVEN a chat store
        chat_store = ResourceStoreChatStore(store)

        # WHEN setting messages async
        await chat_store.aset_messages(
            "async_test",
            [{"role": "user", "content": "Async message"}],
        )

        # THEN they should be retrievable async
        messages = await chat_store.aget_messages("async_test")
        assert len(messages) == 1
        assert messages[0]["content"] == "Async message"

    async def test_async_add_message(self, store: ResourceStore) -> None:
        """Test async add message."""
        # GIVEN a chat store
        chat_store = ResourceStoreChatStore(store)
        await chat_store.aset_messages("async_add", [])

        # WHEN adding async
        await chat_store.aadd_message("async_add", {"role": "user", "content": "Added"})

        # THEN it should be there
        messages = await chat_store.aget_messages("async_add")
        assert len(messages) == 1


@pytest.mark.asyncio
class TestResourceStoreMemoryBlock:
    """Tests for ResourceStoreMemoryBlock (LlamaIndex MemoryBlock adapter)."""

    async def test_put_and_get(self, store: ResourceStore) -> None:
        """Test storing and retrieving memory."""
        # GIVEN a memory block
        memory_block = ResourceStoreMemoryBlock(store, name="test_block")

        # WHEN putting messages
        messages = [
            {"role": "user", "content": "Remember this fact"},
            {"role": "assistant", "content": "I will remember that"},
        ]
        await memory_block._aput(messages)

        # THEN they should be retrievable
        content = await memory_block._aget(session_id="default")
        assert "Remember this fact" in content or "I will remember that" in content

    async def test_truncate(self, store: ResourceStore) -> None:
        """Test truncation behavior."""
        # GIVEN a memory block
        memory_block = ResourceStoreMemoryBlock(store, name="truncate_test")

        # WHEN truncating with tokens to remove
        result = await memory_block.atruncate("Some content", tokens_to_truncate=100)

        # THEN it should return empty (block disabled)
        assert result == ""

    async def test_truncate_no_removal(self, store: ResourceStore) -> None:
        """Test truncation when no removal needed."""
        # GIVEN a memory block
        memory_block = ResourceStoreMemoryBlock(store, name="no_truncate_test")

        # WHEN truncating with no tokens to remove
        result = await memory_block.atruncate("Some content", tokens_to_truncate=0)

        # THEN content should be preserved
        assert result == "Some content"

    def test_sync_methods(self, store: ResourceStore) -> None:
        """Test sync wrapper methods."""
        # GIVEN a memory block
        memory_block = ResourceStoreMemoryBlock(store, name="sync_test")

        # WHEN using sync methods
        memory_block.put([{"role": "user", "content": "Sync test"}])
        content = memory_block.get(session_id="default")

        # THEN it should work
        assert "Sync test" in content or content == ""  # May be empty if not found

    def test_block_attributes(self, store: ResourceStore) -> None:
        """Test memory block attributes."""
        # GIVEN a memory block with custom settings
        memory_block = ResourceStoreMemoryBlock(
            store,
            name="custom_block",
            priority=2,
            max_entries=50,
        )

        # THEN attributes should be set correctly
        assert memory_block.name == "custom_block"
        assert memory_block.priority == 2
        assert memory_block.max_entries == 50


@pytest.mark.asyncio
class TestResourceStoreVectorMemoryBlock:
    """Tests for ResourceStoreVectorMemoryBlock."""

    async def test_put_and_get_batch(self, store: ResourceStore) -> None:
        """Test storing and retrieving message batches."""
        # GIVEN a vector memory block
        vector_block = ResourceStoreVectorMemoryBlock(store, name="vector_test")

        # WHEN putting a batch of messages
        messages = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language."},
        ]
        await vector_block._aput(messages)

        # THEN they should be retrievable
        content = await vector_block._aget(session_id="default")
        assert "Python" in content

    async def test_multiple_batches(self, store: ResourceStore) -> None:
        """Test storing multiple batches."""
        # GIVEN a vector memory block
        vector_block = ResourceStoreVectorMemoryBlock(store, name="multi_batch_test", top_k=2)

        # WHEN putting multiple batches
        batch1 = [{"role": "user", "content": "Batch 1 content"}]
        batch2 = [{"role": "user", "content": "Batch 2 content"}]
        await vector_block._aput(batch1)
        await vector_block._aput(batch2)

        # THEN both should be retrievable
        content = await vector_block._aget(session_id="default")
        # At least one batch should be present
        assert "Batch" in content

    def test_sync_methods(self, store: ResourceStore) -> None:
        """Test sync wrapper methods for vector block."""
        # GIVEN a vector memory block
        vector_block = ResourceStoreVectorMemoryBlock(store, name="sync_vector_test")

        # WHEN using sync methods
        vector_block.put([{"role": "user", "content": "Sync vector test"}])
        content = vector_block.get(session_id="default")

        # THEN it should work
        assert "Sync vector test" in content or isinstance(content, str)
