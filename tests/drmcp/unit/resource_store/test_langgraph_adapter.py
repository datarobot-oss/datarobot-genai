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

"""Tests for LangGraph Checkpoint Adapter."""

import pytest

from datarobot_genai.drmcp.core.resource_store.adapters.langgraph_adapter import (
    ResourceStoreCheckpointSaver,
)
from datarobot_genai.drmcp.core.resource_store.store import ResourceStore


@pytest.mark.asyncio
class TestResourceStoreCheckpointSaver:
    """Tests for ResourceStoreCheckpointSaver (LangGraph Checkpointer adapter)."""

    def test_put_and_get_tuple(self, store: ResourceStore) -> None:
        """Test storing and retrieving a checkpoint."""
        # GIVEN a ResourceStoreCheckpointSaver instance
        checkpointer = ResourceStoreCheckpointSaver(store)

        config = {"configurable": {"thread_id": "test_thread_1"}}
        checkpoint = {
            "v": 1,
            "id": "checkpoint_1",
            "channel_values": {"messages": ["Hello", "World"]},
        }
        metadata = {"source": "input", "step": 1}
        new_versions = {"messages": 1}

        # WHEN storing a checkpoint
        result_config = checkpointer.put(config, checkpoint, metadata, new_versions)

        # THEN it should return a config with checkpoint_id
        assert "configurable" in result_config
        assert "checkpoint_id" in result_config["configurable"]
        assert result_config["configurable"]["thread_id"] == "test_thread_1"

        # AND the checkpoint should be retrievable
        tuple_result = checkpointer.get_tuple(result_config)
        assert tuple_result is not None
        assert tuple_result.checkpoint == checkpoint
        assert tuple_result.metadata == metadata

    def test_get_tuple_not_found(self, store: ResourceStore) -> None:
        """Test get_tuple returns None when checkpoint doesn't exist."""
        # GIVEN a checkpointer with no data
        checkpointer = ResourceStoreCheckpointSaver(store)

        config = {"configurable": {"thread_id": "nonexistent_thread"}}

        # WHEN getting a non-existent checkpoint
        result = checkpointer.get_tuple(config)

        # THEN it should return None
        assert result is None

    def test_list_checkpoints(self, store: ResourceStore) -> None:
        """Test listing checkpoints for a thread."""
        # GIVEN a checkpointer with multiple checkpoints
        checkpointer = ResourceStoreCheckpointSaver(store)

        config = {"configurable": {"thread_id": "list_test_thread"}}

        # Create multiple checkpoints
        for i in range(3):
            checkpointer.put(
                config,
                checkpoint={"step": i, "data": f"checkpoint_{i}"},
                metadata={"step": i},
                new_versions={"channel": i},
            )

        # WHEN listing checkpoints
        checkpoints = list(checkpointer.list(config))

        # THEN all checkpoints should be returned
        assert len(checkpoints) == 3

    def test_list_with_limit(self, store: ResourceStore) -> None:
        """Test listing checkpoints with a limit."""
        # GIVEN a checkpointer with multiple checkpoints
        checkpointer = ResourceStoreCheckpointSaver(store)

        config = {"configurable": {"thread_id": "limit_test_thread"}}

        # Create multiple checkpoints
        for i in range(5):
            checkpointer.put(
                config,
                checkpoint={"step": i},
                metadata={"step": i},
                new_versions={},
            )

        # WHEN listing with limit
        checkpoints = list(checkpointer.list(config, limit=2))

        # THEN only limited number should be returned
        assert len(checkpoints) == 2

    def test_thread_isolation(self, store: ResourceStore) -> None:
        """Test that different threads have isolated checkpoints."""
        # GIVEN a checkpointer
        checkpointer = ResourceStoreCheckpointSaver(store)

        config1 = {"configurable": {"thread_id": "thread_a"}}
        config2 = {"configurable": {"thread_id": "thread_b"}}

        # WHEN storing checkpoints in different threads
        checkpointer.put(
            config1,
            checkpoint={"data": "thread_a_data"},
            metadata={},
            new_versions={},
        )
        checkpointer.put(
            config2,
            checkpoint={"data": "thread_b_data"},
            metadata={},
            new_versions={},
        )

        # THEN each thread should only see its own checkpoints
        thread_a_checkpoints = list(checkpointer.list(config1))
        thread_b_checkpoints = list(checkpointer.list(config2))

        assert len(thread_a_checkpoints) == 1
        assert len(thread_b_checkpoints) == 1
        assert thread_a_checkpoints[0].checkpoint["data"] == "thread_a_data"
        assert thread_b_checkpoints[0].checkpoint["data"] == "thread_b_data"

    def test_parent_config_tracking(self, store: ResourceStore) -> None:
        """Test that parent checkpoint is tracked correctly."""
        # GIVEN a checkpointer
        checkpointer = ResourceStoreCheckpointSaver(store)

        config = {"configurable": {"thread_id": "parent_test_thread"}}

        # WHEN storing first checkpoint
        first_config = checkpointer.put(
            config,
            checkpoint={"step": 0},
            metadata={"step": 0},
            new_versions={},
        )

        # AND storing second checkpoint with first as parent
        second_config = checkpointer.put(
            first_config,
            checkpoint={"step": 1},
            metadata={"step": 1},
            new_versions={},
        )

        # THEN the second checkpoint should have parent_config
        result = checkpointer.get_tuple(second_config)
        assert result is not None
        assert result.parent_config is not None
        assert (
            result.parent_config["configurable"]["checkpoint_id"]
            == first_config["configurable"]["checkpoint_id"]
        )

    def test_missing_thread_id_raises_error(self, store: ResourceStore) -> None:
        """Test that missing thread_id raises ValueError."""
        # GIVEN a checkpointer
        checkpointer = ResourceStoreCheckpointSaver(store)

        config = {"configurable": {}}  # No thread_id

        # WHEN/THEN putting without thread_id should raise ValueError
        with pytest.raises(ValueError, match="thread_id is required"):
            checkpointer.put(
                config,
                checkpoint={},
                metadata={},
                new_versions={},
            )

    def test_put_writes(self, store: ResourceStore) -> None:
        """Test storing pending writes."""
        # GIVEN a checkpointer with a checkpoint
        checkpointer = ResourceStoreCheckpointSaver(store)

        config = {"configurable": {"thread_id": "writes_test_thread"}}
        result_config = checkpointer.put(
            config,
            checkpoint={"step": 0},
            metadata={},
            new_versions={},
        )

        # WHEN storing pending writes
        writes = [("messages", "Hello"), ("state", {"key": "value"})]
        checkpointer.put_writes(result_config, writes, task_id="task_1")

        # THEN the writes should be retrievable with the checkpoint
        result = checkpointer.get_tuple(result_config)
        assert result is not None
        # pending_writes format is [(task_id, channel, value), ...]
        assert len(result.pending_writes) == 2


@pytest.mark.asyncio
class TestResourceStoreCheckpointSaverAsync:
    """Async tests for ResourceStoreCheckpointSaver."""

    async def test_aput_and_aget_tuple(self, store: ResourceStore) -> None:
        """Test async storing and retrieving a checkpoint."""
        # GIVEN a checkpointer
        checkpointer = ResourceStoreCheckpointSaver(store)

        config = {"configurable": {"thread_id": "async_test_thread"}}
        checkpoint = {"async": True, "data": "test"}
        metadata = {"source": "async_test"}

        # WHEN storing async
        result_config = await checkpointer.aput(config, checkpoint, metadata, {})

        # THEN it should be retrievable async
        result = await checkpointer.aget_tuple(result_config)
        assert result is not None
        assert result.checkpoint == checkpoint

    async def test_alist(self, store: ResourceStore) -> None:
        """Test async listing checkpoints."""
        # GIVEN a checkpointer with checkpoints
        checkpointer = ResourceStoreCheckpointSaver(store)

        config = {"configurable": {"thread_id": "async_list_thread"}}

        for i in range(3):
            await checkpointer.aput(
                config,
                checkpoint={"step": i},
                metadata={},
                new_versions={},
            )

        # WHEN listing async
        checkpoints = []
        async for cp in checkpointer.alist(config):
            checkpoints.append(cp)

        # THEN all should be returned
        assert len(checkpoints) == 3
