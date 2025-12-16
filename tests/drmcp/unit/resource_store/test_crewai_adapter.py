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

"""Tests for CrewAI Memory Adapter."""

import pytest

from datarobot_genai.drmcp.core.resource_store.adapters.crewai_adapter import (
    ResourceStoreLongTermStorage,
)
from datarobot_genai.drmcp.core.resource_store.adapters.crewai_adapter import ResourceStoreStorage
from datarobot_genai.drmcp.core.resource_store.store import ResourceStore


@pytest.mark.asyncio
class TestResourceStoreStorage:
    """Tests for ResourceStoreStorage (CrewAI External Memory adapter)."""

    def test_save_and_search(self, store: ResourceStore) -> None:
        """Test saving and searching memories."""
        # GIVEN a ResourceStoreStorage instance
        storage = ResourceStoreStorage(store, user_id="test_user")

        # WHEN saving a memory
        storage.save(
            value="Remember that the user prefers dark mode",
            metadata={"category": "preference"},
            agent="assistant",
        )

        # THEN it should be searchable
        results = storage.search("dark mode")
        assert len(results) >= 1
        assert "dark mode" in results[0]["value"]
        assert results[0]["metadata"].get("agent") == "assistant"

    def test_search_no_results(self, store: ResourceStore) -> None:
        """Test search with no matching results."""
        # GIVEN a ResourceStoreStorage instance with no data
        storage = ResourceStoreStorage(store, user_id="empty_user")

        # WHEN searching for something
        results = storage.search("nonexistent query")

        # THEN no results should be returned
        assert len(results) == 0

    def test_search_with_limit(self, store: ResourceStore) -> None:
        """Test search respects limit parameter."""
        # GIVEN a ResourceStoreStorage with multiple memories
        storage = ResourceStoreStorage(store, user_id="limit_test_user")

        # WHEN saving multiple memories with "test" in content
        for i in range(5):
            storage.save(value=f"Test memory number {i}", metadata={"index": i})

        # THEN search with limit should respect the limit
        results = storage.search("test", limit=3)
        assert len(results) <= 3

    def test_reset(self, store: ResourceStore) -> None:
        """Test resetting (clearing) all memories."""
        # GIVEN a ResourceStoreStorage with some memories
        storage = ResourceStoreStorage(store, user_id="reset_test_user")
        storage.save(value="Memory 1")
        storage.save(value="Memory 2")

        # Verify memories exist
        results = storage.search("Memory")
        assert len(results) >= 2

        # WHEN resetting
        storage.reset()

        # THEN all memories should be cleared
        results = storage.search("Memory")
        assert len(results) == 0

    def test_user_isolation(self, store: ResourceStore) -> None:
        """Test that different users have isolated memories."""
        # GIVEN two storage instances with different user_ids
        storage_user1 = ResourceStoreStorage(store, user_id="user_1")
        storage_user2 = ResourceStoreStorage(store, user_id="user_2")

        # WHEN saving memories to each user
        storage_user1.save(value="User 1 secret data")
        storage_user2.save(value="User 2 private info")

        # THEN each user should only see their own memories
        user1_results = storage_user1.search("secret")
        user2_results = storage_user2.search("secret")

        assert len(user1_results) == 1
        assert len(user2_results) == 0


@pytest.mark.asyncio
class TestResourceStoreLongTermStorage:
    """Tests for ResourceStoreLongTermStorage (CrewAI Long-Term Memory adapter)."""

    def test_save_and_search_task(self, store: ResourceStore) -> None:
        """Test saving and searching task results."""
        # GIVEN a ResourceStoreLongTermStorage instance
        storage = ResourceStoreLongTermStorage(store, user_id="ltm_test_user")

        # WHEN saving a task result
        storage.save(
            task_description="Analyze customer feedback",
            result="Found 85% positive sentiment with main concerns about pricing",
            metadata={"task_type": "analysis"},
        )

        # THEN it should be searchable by task description
        results = storage.search("Analyze customer")
        assert len(results) >= 1
        assert "sentiment" in results[0]["result"]

    def test_reset_long_term_memory(self, store: ResourceStore) -> None:
        """Test resetting long-term memory."""
        # GIVEN a storage with some task results
        storage = ResourceStoreLongTermStorage(store, user_id="ltm_reset_user")
        storage.save(
            task_description="Task 1",
            result="Result 1",
        )

        # WHEN resetting
        storage.reset()

        # THEN no results should be found
        results = storage.search("Task")
        assert len(results) == 0
