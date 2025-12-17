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

"""Tests for NVIDIA NAT Memory Adapter."""

import pytest

from datarobot_genai.drmcp.core.resource_store.adapters.nat_adapter import MemoryItem
from datarobot_genai.drmcp.core.resource_store.adapters.nat_adapter import ResourceStoreMemoryConfig
from datarobot_genai.drmcp.core.resource_store.adapters.nat_adapter import ResourceStoreMemoryEditor
from datarobot_genai.drmcp.core.resource_store.adapters.nat_adapter import create_nat_memory_editor
from datarobot_genai.drmcp.core.resource_store.store import ResourceStore

# Test constants
TEST_SCORE = 0.95


@pytest.mark.asyncio
class TestResourceStoreMemoryEditor:
    """Tests for ResourceStoreMemoryEditor (NAT MemoryEditor adapter)."""

    async def test_add_and_search_items(self, store: ResourceStore) -> None:
        """Test adding and searching memory items."""
        # GIVEN a memory editor
        config = ResourceStoreMemoryConfig(user_id="test_user")
        editor = ResourceStoreMemoryEditor(config, store=store)

        # WHEN adding items
        items = [
            MemoryItem(content="User prefers dark mode"),
            MemoryItem(content="User is a Python developer"),
            MemoryItem(content="User likes coffee"),
        ]
        ids = await editor.add_items(items)

        # THEN IDs should be returned
        assert len(ids) == 3

        # AND items should be searchable
        results = await editor.search("Python", top_k=5)
        assert len(results) >= 1
        assert any("Python" in r.content for r in results)

    async def test_search_no_results(self, store: ResourceStore) -> None:
        """Test search with no matching results."""
        # GIVEN a memory editor with no data
        config = ResourceStoreMemoryConfig(user_id="empty_user")
        editor = ResourceStoreMemoryEditor(config, store=store)

        # WHEN searching
        results = await editor.search("nonexistent query")

        # THEN no results should be returned
        assert len(results) == 0

    async def test_search_with_top_k(self, store: ResourceStore) -> None:
        """Test search respects top_k limit."""
        # GIVEN a memory editor with multiple items
        config = ResourceStoreMemoryConfig(user_id="topk_user")
        editor = ResourceStoreMemoryEditor(config, store=store)

        # Add multiple items with "test" in content
        items = [MemoryItem(content=f"Test item number {i}") for i in range(10)]
        await editor.add_items(items)

        # WHEN searching with top_k
        results = await editor.search("test", top_k=3)

        # THEN only top_k results should be returned
        assert len(results) <= 3

    async def test_remove_specific_items(self, store: ResourceStore) -> None:
        """Test removing specific items by ID."""
        # GIVEN a memory editor with items
        config = ResourceStoreMemoryConfig(user_id="remove_user")
        editor = ResourceStoreMemoryEditor(config, store=store)

        items = [
            MemoryItem(content="Keep this"),
            MemoryItem(content="Delete this"),
        ]
        ids = await editor.add_items(items)

        # WHEN removing specific item
        removed = await editor.remove_items(item_ids=[ids[1]])

        # THEN only that item should be removed
        assert removed == 1

        # AND the other item should remain
        remaining = await editor.list_items()
        assert len(remaining) == 1
        assert remaining[0].content == "Keep this"

    async def test_remove_all_items(self, store: ResourceStore) -> None:
        """Test removing all items."""
        # GIVEN a memory editor with items
        config = ResourceStoreMemoryConfig(user_id="remove_all_user")
        editor = ResourceStoreMemoryEditor(config, store=store)

        items = [
            MemoryItem(content="Item 1"),
            MemoryItem(content="Item 2"),
        ]
        await editor.add_items(items)

        # WHEN removing all items
        removed = await editor.remove_items()

        # THEN all should be removed
        assert removed == 2
        remaining = await editor.list_items()
        assert len(remaining) == 0

    async def test_get_item(self, store: ResourceStore) -> None:
        """Test getting a specific item by ID."""
        # GIVEN a memory editor with an item
        config = ResourceStoreMemoryConfig(user_id="get_user")
        editor = ResourceStoreMemoryEditor(config, store=store)

        items = [MemoryItem(content="Specific item", metadata={"key": "value"})]
        ids = await editor.add_items(items)

        # WHEN getting the item
        item = await editor.get_item(ids[0])

        # THEN it should be returned with correct data
        assert item is not None
        assert item.content == "Specific item"
        assert item.metadata.get("key") == "value"

    async def test_get_item_not_found(self, store: ResourceStore) -> None:
        """Test getting a non-existent item."""
        # GIVEN a memory editor
        config = ResourceStoreMemoryConfig(user_id="notfound_user")
        editor = ResourceStoreMemoryEditor(config, store=store)

        # WHEN getting non-existent item
        item = await editor.get_item("nonexistent_id")

        # THEN None should be returned
        assert item is None

    async def test_update_item(self, store: ResourceStore) -> None:
        """Test updating an existing item."""
        # GIVEN a memory editor with an item
        config = ResourceStoreMemoryConfig(user_id="update_user")
        editor = ResourceStoreMemoryEditor(config, store=store)

        items = [MemoryItem(content="Original content")]
        ids = await editor.add_items(items)

        # WHEN updating the item
        success = await editor.update_item(ids[0], content="Updated content")

        # THEN it should succeed
        assert success is True

        # AND content should be updated
        item = await editor.get_item(ids[0])
        assert item is not None
        assert item.content == "Updated content"

    async def test_update_item_not_found(self, store: ResourceStore) -> None:
        """Test updating a non-existent item."""
        # GIVEN a memory editor
        config = ResourceStoreMemoryConfig(user_id="update_notfound_user")
        editor = ResourceStoreMemoryEditor(config, store=store)

        # WHEN updating non-existent item
        success = await editor.update_item("nonexistent_id", content="New content")

        # THEN it should return False
        assert success is False

    async def test_list_items(self, store: ResourceStore) -> None:
        """Test listing all items."""
        # GIVEN a memory editor with items
        config = ResourceStoreMemoryConfig(user_id="list_user")
        editor = ResourceStoreMemoryEditor(config, store=store)

        items = [
            MemoryItem(content="Item A"),
            MemoryItem(content="Item B"),
            MemoryItem(content="Item C"),
        ]
        await editor.add_items(items)

        # WHEN listing items
        all_items = await editor.list_items()

        # THEN all should be returned
        assert len(all_items) == 3

    async def test_list_items_with_pagination(self, store: ResourceStore) -> None:
        """Test listing items with pagination."""
        # GIVEN a memory editor with items
        config = ResourceStoreMemoryConfig(user_id="paginate_user")
        editor = ResourceStoreMemoryEditor(config, store=store)

        items = [MemoryItem(content=f"Item {i}") for i in range(5)]
        await editor.add_items(items)

        # WHEN listing with limit
        page1 = await editor.list_items(limit=2, offset=0)
        page2 = await editor.list_items(limit=2, offset=2)

        # THEN pagination should work
        assert len(page1) == 2
        assert len(page2) == 2

    async def test_clear(self, store: ResourceStore) -> None:
        """Test clearing all items."""
        # GIVEN a memory editor with items
        config = ResourceStoreMemoryConfig(user_id="clear_user")
        editor = ResourceStoreMemoryEditor(config, store=store)

        items = [MemoryItem(content=f"Item {i}") for i in range(3)]
        await editor.add_items(items)

        # WHEN clearing
        removed = await editor.clear()

        # THEN all should be removed
        assert removed == 3
        remaining = await editor.list_items()
        assert len(remaining) == 0

    async def test_user_isolation(self, store: ResourceStore) -> None:
        """Test that different users have isolated memories."""
        # GIVEN two editors with different user IDs
        editor1 = ResourceStoreMemoryEditor(
            ResourceStoreMemoryConfig(user_id="user_a"), store=store
        )
        editor2 = ResourceStoreMemoryEditor(
            ResourceStoreMemoryConfig(user_id="user_b"), store=store
        )

        # WHEN adding items to each
        await editor1.add_items([MemoryItem(content="User A secret")])
        await editor2.add_items([MemoryItem(content="User B secret")])

        # THEN each should only see their own
        user_a_items = await editor1.list_items()
        user_b_items = await editor2.list_items()

        assert len(user_a_items) == 1
        assert len(user_b_items) == 1
        assert user_a_items[0].content == "User A secret"
        assert user_b_items[0].content == "User B secret"


@pytest.mark.asyncio
class TestMemoryItem:
    """Tests for MemoryItem dataclass."""

    def test_to_dict(self) -> None:
        """Test converting MemoryItem to dictionary."""
        # GIVEN a memory item
        item = MemoryItem(
            id="test_id",
            content="Test content",
            metadata={"key": "value"},
            score=TEST_SCORE,
        )

        # WHEN converting to dict
        result = item.to_dict()

        # THEN all fields should be present
        assert result["id"] == "test_id"
        assert result["content"] == "Test content"
        assert result["metadata"]["key"] == "value"
        assert result["score"] == TEST_SCORE

    def test_from_dict(self) -> None:
        """Test creating MemoryItem from dictionary."""
        # GIVEN a dictionary
        data = {
            "id": "test_id",
            "content": "Test content",
            "metadata": {"key": "value"},
            "score": 0.95,
        }

        # WHEN creating from dict
        item = MemoryItem.from_dict(data)

        # THEN fields should be set correctly
        assert item.id == "test_id"
        assert item.content == "Test content"
        assert item.metadata["key"] == "value"
        assert item.score == TEST_SCORE


@pytest.mark.asyncio
class TestSyncWrappers:
    """Tests for sync wrapper methods."""

    def test_add_items_sync(self, store: ResourceStore) -> None:
        """Test sync add_items wrapper."""
        # GIVEN a memory editor
        editor = create_nat_memory_editor(store, user_id="sync_test")

        # WHEN using sync method
        ids = editor.add_items_sync([MemoryItem(content="Sync test")])

        # THEN it should work
        assert len(ids) == 1

    def test_search_sync(self, store: ResourceStore) -> None:
        """Test sync search wrapper."""
        # GIVEN a memory editor with data
        editor = create_nat_memory_editor(store, user_id="sync_search_test")
        editor.add_items_sync([MemoryItem(content="Searchable content")])

        # WHEN using sync search
        results = editor.search_sync("Searchable")

        # THEN it should work
        assert len(results) >= 1

    def test_list_items_sync(self, store: ResourceStore) -> None:
        """Test sync list_items wrapper."""
        # GIVEN a memory editor with data
        editor = create_nat_memory_editor(store, user_id="sync_list_test")
        editor.add_items_sync([MemoryItem(content="Item 1")])

        # WHEN using sync list
        items = editor.list_items_sync()

        # THEN it should work
        assert len(items) == 1


@pytest.mark.asyncio
class TestCreateNatMemoryEditor:
    """Tests for create_nat_memory_editor convenience function."""

    async def test_create_editor(self, store: ResourceStore) -> None:
        """Test creating editor with convenience function."""
        # WHEN creating with convenience function
        editor = create_nat_memory_editor(store, user_id="convenience_test")

        # THEN it should work
        assert editor is not None
        assert editor._user_id == "convenience_test"

        # AND be functional
        ids = await editor.add_items([MemoryItem(content="Test")])
        assert len(ids) == 1
