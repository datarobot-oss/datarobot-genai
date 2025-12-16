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

"""
LangGraph Checkpoint Adapter for ResourceStore.

This module provides a checkpointer implementation that uses ResourceStore
as the persistence backend for LangGraph's state management.

LangGraph uses checkpointers to persist graph state, enabling:
- Session memory across conversations
- Error recovery and resumption
- Human-in-the-loop workflows
- State history and time travel

Usage:
    ```python
    from langgraph.graph import StateGraph
    from datarobot_genai.drmcp.core.resource_store import ResourceStore
    from datarobot_genai.drmcp.core.resource_store.backends.filesystem import FilesystemBackend
    from datarobot_genai.drmcp.core.resource_store.adapters.langgraph_adapter import (
        ResourceStoreCheckpointSaver,
    )

    # Create ResourceStore
    backend = FilesystemBackend("/path/to/storage")
    store = ResourceStore(backend)

    # Create LangGraph-compatible checkpointer
    checkpointer = ResourceStoreCheckpointSaver(store)

    # Use with LangGraph
    graph = StateGraph(...)
    graph.add_node(...)
    app = graph.compile(checkpointer=checkpointer)

    # Run with thread_id for persistence
    config = {"configurable": {"thread_id": "conversation_123"}}
    result = app.invoke({"messages": [...]}, config)
    ```

See: https://langchain-ai.github.io/langgraph/concepts/persistence/
"""

import asyncio
import concurrent.futures
import json
import logging
import uuid
from collections.abc import AsyncIterator
from collections.abc import Iterator
from collections.abc import Sequence
from datetime import datetime
from datetime import timezone
from typing import Any

from ..models import Scope
from ..store import ResourceStore

logger = logging.getLogger(__name__)

# Type aliases for LangGraph compatibility
RunnableConfig = dict[str, Any]
Checkpoint = dict[str, Any]
CheckpointMetadata = dict[str, Any]
ChannelVersions = dict[str, Any]


class CheckpointTuple:
    """
    Container for checkpoint data.

    This mirrors LangGraph's CheckpointTuple structure.
    """

    def __init__(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        parent_config: RunnableConfig | None = None,
        pending_writes: list[tuple[str, str, Any]] | None = None,
    ):
        self.config = config
        self.checkpoint = checkpoint
        self.metadata = metadata
        self.parent_config = parent_config
        self.pending_writes = pending_writes or []


class ResourceStoreCheckpointSaver:
    """
    LangGraph CheckpointSaver backed by ResourceStore.

    This class implements the BaseCheckpointSaver interface from LangGraph,
    using ResourceStore for persistent storage of graph checkpoints.

    Key features:
    - Thread-based isolation (each thread_id has its own checkpoint history)
    - Full checkpoint history with parent tracking
    - Pending writes support for intermediate states
    - Both sync and async interfaces

    Attributes
    ----------
        store: ResourceStore instance for persistence
        serde: Optional serializer (defaults to JSON)
    """

    def __init__(
        self,
        store: ResourceStore,
        serde: Any = None,
    ) -> None:
        """
        Initialize ResourceStoreCheckpointSaver.

        Args:
            store: ResourceStore instance for checkpoint storage
            serde: Optional custom serializer (defaults to JSON)
        """
        self.store = store
        self.serde = serde

    def _get_thread_id(self, config: RunnableConfig) -> str:
        """Extract thread_id from config."""
        configurable = config.get("configurable", {})
        thread_id = configurable.get("thread_id")
        if not thread_id:
            raise ValueError("thread_id is required in config['configurable']")
        return str(thread_id)

    def _get_checkpoint_id(self, config: RunnableConfig) -> str | None:
        """Extract checkpoint_id from config."""
        configurable = config.get("configurable", {})
        return configurable.get("checkpoint_id")

    def _get_checkpoint_ns(self, config: RunnableConfig) -> str:
        """Extract checkpoint namespace from config."""
        configurable = config.get("configurable", {})
        return configurable.get("checkpoint_ns", "")

    def _serialize(self, data: Any) -> str:
        """Serialize data to string."""
        if self.serde:
            return self.serde.dumps(data)
        return json.dumps(data, default=str)

    def _deserialize(self, data: str) -> Any:
        """Deserialize string to data."""
        if self.serde:
            return self.serde.loads(data)
        return json.loads(data)

    async def _aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Async implementation of put."""
        thread_id = self._get_thread_id(config)
        checkpoint_ns = self._get_checkpoint_ns(config)
        parent_checkpoint_id = self._get_checkpoint_id(config)

        # Generate new checkpoint_id
        checkpoint_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        # Create scope for this thread (use 'custom' type with langgraph prefix in id)
        scope = Scope(type="custom", id=f"langgraph:{thread_id}")

        # Prepare checkpoint data
        checkpoint_data = {
            "checkpoint": checkpoint,
            "metadata": metadata,
            "channel_versions": new_versions,
            "parent_checkpoint_id": parent_checkpoint_id,
            "checkpoint_ns": checkpoint_ns,
            "timestamp": timestamp,
        }

        # Store checkpoint
        await self.store.put(
            scope=scope,
            kind="checkpoint",
            data=self._serialize(checkpoint_data),
            lifetime="persistent",
            contentType="application/json",
            metadata={
                "checkpoint_id": checkpoint_id,
                "checkpoint_ns": checkpoint_ns,
                "parent_checkpoint_id": parent_checkpoint_id or "",
                "timestamp": timestamp,
            },
            resource_id=f"checkpoint:{thread_id}:{checkpoint_id}",
        )

        # Return updated config with new checkpoint_id
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id,
                "checkpoint_ns": checkpoint_ns,
            }
        }

    async def _aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Async implementation of put_writes."""
        thread_id = self._get_thread_id(config)
        checkpoint_id = self._get_checkpoint_id(config)
        checkpoint_ns = self._get_checkpoint_ns(config)

        if not checkpoint_id:
            return

        scope = Scope(type="custom", id=f"langgraph:{thread_id}")

        # Store pending writes
        writes_data = {
            "writes": [(channel, self._serialize(value)) for channel, value in writes],
            "task_id": task_id,
            "checkpoint_id": checkpoint_id,
            "checkpoint_ns": checkpoint_ns,
        }

        write_id = str(uuid.uuid4())
        await self.store.put(
            scope=scope,
            kind="pending_write",
            data=self._serialize(writes_data),
            lifetime="ephemeral",
            contentType="application/json",
            metadata={
                "checkpoint_id": checkpoint_id,
                "task_id": task_id,
            },
            resource_id=f"write:{thread_id}:{checkpoint_id}:{write_id}",
        )

    async def _aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Async implementation of get_tuple."""
        thread_id = self._get_thread_id(config)
        checkpoint_id = self._get_checkpoint_id(config)
        checkpoint_ns = self._get_checkpoint_ns(config)

        scope = Scope(type="custom", id=f"langgraph:{thread_id}")

        # Query checkpoints for this thread
        resources = await self.store.query(scope=scope, kind="checkpoint")

        if not resources:
            return None

        # Find the requested checkpoint (or latest if not specified)
        target_resource = None
        if checkpoint_id:
            for r in resources:
                if r.metadata.get("checkpoint_id") == checkpoint_id:
                    # Check namespace matches
                    if r.metadata.get("checkpoint_ns", "") == checkpoint_ns:
                        target_resource = r
                        break
        else:
            # Get latest checkpoint for this namespace
            ns_resources = [
                r for r in resources if r.metadata.get("checkpoint_ns", "") == checkpoint_ns
            ]
            if ns_resources:
                # Sort by timestamp descending
                ns_resources.sort(key=lambda x: x.metadata.get("timestamp", ""), reverse=True)
                target_resource = ns_resources[0]

        if not target_resource:
            return None

        # Get checkpoint data
        result = await self.store.get(target_resource.id)
        if not result:
            return None

        _, data = result
        content = data if isinstance(data, str) else data.decode("utf-8")
        checkpoint_data = self._deserialize(content)

        # Get pending writes for this checkpoint
        pending_writes: list[tuple[str, str, Any]] = []
        write_resources = await self.store.query(scope=scope, kind="pending_write")
        for wr in write_resources:
            if wr.metadata.get("checkpoint_id") == target_resource.metadata.get("checkpoint_id"):
                write_result = await self.store.get(wr.id)
                if write_result:
                    _, write_data = write_result
                    write_content = (
                        write_data if isinstance(write_data, str) else write_data.decode("utf-8")
                    )
                    writes_info = self._deserialize(write_content)
                    task_id = writes_info.get("task_id", "")
                    for channel, value_str in writes_info.get("writes", []):
                        pending_writes.append((task_id, channel, self._deserialize(value_str)))

        # Build parent config
        parent_config = None
        parent_id = checkpoint_data.get("parent_checkpoint_id")
        if parent_id:
            parent_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_id": parent_id,
                    "checkpoint_ns": checkpoint_ns,
                }
            }

        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_id": target_resource.metadata.get("checkpoint_id"),
                    "checkpoint_ns": checkpoint_ns,
                }
            },
            checkpoint=checkpoint_data.get("checkpoint", {}),
            metadata=checkpoint_data.get("metadata", {}),
            parent_config=parent_config,
            pending_writes=pending_writes,
        )

    async def _alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """Async implementation of list."""
        if not config:
            return

        thread_id = self._get_thread_id(config)
        checkpoint_ns = self._get_checkpoint_ns(config)
        scope = Scope(type="custom", id=f"langgraph:{thread_id}")

        # Query all checkpoints for this thread
        resources = await self.store.query(scope=scope, kind="checkpoint")

        # Filter by namespace
        ns_resources = [
            r for r in resources if r.metadata.get("checkpoint_ns", "") == checkpoint_ns
        ]

        # Sort by timestamp descending (newest first)
        ns_resources.sort(key=lambda x: x.metadata.get("timestamp", ""), reverse=True)

        # Apply before filter
        if before:
            before_id = self._get_checkpoint_id(before)
            if before_id:
                filtered = []
                found_before = False
                for r in ns_resources:
                    if r.metadata.get("checkpoint_id") == before_id:
                        found_before = True
                        continue
                    if found_before:
                        filtered.append(r)
                ns_resources = filtered

        # Apply limit
        if limit:
            ns_resources = ns_resources[:limit]

        # Yield checkpoint tuples
        for resource in ns_resources:
            result = await self.store.get(resource.id)
            if not result:
                continue

            _, data = result
            content = data if isinstance(data, str) else data.decode("utf-8")
            checkpoint_data = self._deserialize(content)

            # Build parent config
            parent_config = None
            parent_id = checkpoint_data.get("parent_checkpoint_id")
            if parent_id:
                parent_config = {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_id": parent_id,
                        "checkpoint_ns": checkpoint_ns,
                    }
                }

            yield CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_id": resource.metadata.get("checkpoint_id"),
                        "checkpoint_ns": checkpoint_ns,
                    }
                },
                checkpoint=checkpoint_data.get("checkpoint", {}),
                metadata=checkpoint_data.get("metadata", {}),
                parent_config=parent_config,
                pending_writes=[],
            )

    # Sync methods (run async in thread pool)

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """
        Store a checkpoint.

        Args:
            config: Configuration with thread_id
            checkpoint: The checkpoint state to store
            metadata: Checkpoint metadata
            new_versions: Channel version information

        Returns
        -------
            Updated config with new checkpoint_id
        """
        return _run_async(self._aput(config, checkpoint, metadata, new_versions))

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
    ) -> None:
        """
        Store pending writes for a checkpoint.

        Args:
            config: Configuration with thread_id and checkpoint_id
            writes: List of (channel, value) tuples
            task_id: Task identifier
        """
        _run_async(self._aput_writes(config, writes, task_id))

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """
        Get a checkpoint tuple.

        Args:
            config: Configuration with thread_id and optional checkpoint_id

        Returns
        -------
            CheckpointTuple if found, None otherwise
        """
        return _run_async(self._aget_tuple(config))

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """
        List checkpoints for a thread.

        Args:
            config: Configuration with thread_id
            filter: Optional filter criteria
            before: Return checkpoints before this one
            limit: Maximum number of checkpoints to return

        Yields
        ------
            CheckpointTuple instances
        """

        async def collect():
            results = []
            async for item in self._alist(config, filter=filter, before=before, limit=limit):
                results.append(item)
            return results

        yield from _run_async(collect())

    # Async methods

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Async version of put."""
        return await self._aput(config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Async version of put_writes."""
        await self._aput_writes(config, writes, task_id)

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Async version of get_tuple."""
        return await self._aget_tuple(config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """Async version of list."""
        async for item in self._alist(config, filter=filter, before=before, limit=limit):
            yield item


def _run_async(coro):
    """Run an async coroutine synchronously."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # We're inside an async context, create a new thread
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)
