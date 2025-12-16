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
NVIDIA NeMo Agent Toolkit (NAT) Memory Adapter for ResourceStore.

This module provides a memory provider implementation that uses ResourceStore
as the backend for NVIDIA's NeMo Agent Toolkit memory system.

NAT uses memory providers to:
- Store and retrieve memory items for agents
- Enable persistent memory across agent sessions
- Support semantic search over stored memories

Usage:
    ```python
    from datarobot_genai.drmcp.core.resource_store import ResourceStore
    from datarobot_genai.drmcp.core.resource_store.backends.filesystem import FilesystemBackend
    from datarobot_genai.drmcp.core.resource_store.adapters.nat_adapter import (
        ResourceStoreMemoryEditor,
        ResourceStoreMemoryConfig,
    )

    # Create ResourceStore
    backend = FilesystemBackend("/path/to/storage")
    store = ResourceStore(backend)

    # Create NAT-compatible memory editor
    config = ResourceStoreMemoryConfig(storage_path="/path/to/storage")
    memory_editor = ResourceStoreMemoryEditor(config, store=store)

    # Use with NAT
    items = [MemoryItem(content="Remember this", metadata={"key": "value"})]
    await memory_editor.add_items(items)

    results = await memory_editor.search("remember", top_k=5)
    ```

See: https://docs.nvidia.com/nemo/agent-toolkit/latest/extend/memory.html
"""

import asyncio
import concurrent.futures
import json
import logging
import uuid
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timezone
from typing import Any

from ..backends.filesystem import FilesystemBackend
from ..memory import MemoryAPI
from ..models import Scope
from ..store import ResourceStore

logger = logging.getLogger(__name__)


def _run_async(coro):
    """Run an async coroutine synchronously."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)


@dataclass
class MemoryItem:
    """
    Memory item data model compatible with NAT.

    This mirrors the MemoryItem structure used in NVIDIA's NeMo Agent Toolkit.

    Attributes
    ----------
        id: Unique identifier for the memory item
        content: The text content of the memory
        metadata: Additional metadata dictionary
        score: Relevance score (set during search results)
        created_at: Creation timestamp
    """

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str | None = None
    score: float | None = None
    created_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "score": self.score,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryItem":
        """Create from dictionary."""
        return cls(
            id=data.get("id"),
            content=data.get("content", ""),
            metadata=data.get("metadata", {}),
            score=data.get("score"),
            created_at=data.get("created_at"),
        )


@dataclass
class ResourceStoreMemoryConfig:
    """
    Configuration for ResourceStore memory provider.

    This mirrors NAT's MemoryBaseConfig structure for compatibility.

    Attributes
    ----------
        storage_path: Path to the storage directory (optional if store provided)
        user_id: User/scope identifier for memory isolation
        name: Provider name for NAT registry
    """

    storage_path: str | None = None
    user_id: str = "default"
    name: str = "resource_store_memory"


class ResourceStoreMemoryEditor:
    """
    NAT MemoryEditor implementation backed by ResourceStore.

    This class implements the MemoryEditor interface from NVIDIA's NeMo Agent
    Toolkit, using ResourceStore for persistent storage of memory items.

    The MemoryEditor interface methods:
        - add_items(items): Add memory items to storage
        - search(query, top_k): Search for relevant memories
        - remove_items(**kwargs): Remove memories by criteria

    Attributes
    ----------
        config: ResourceStoreMemoryConfig instance
        store: ResourceStore instance
        memory_api: MemoryAPI for storage operations
    """

    def __init__(
        self,
        config: ResourceStoreMemoryConfig,
        store: ResourceStore | None = None,
    ) -> None:
        """
        Initialize ResourceStoreMemoryEditor.

        Args:
            config: Configuration for the memory provider
            store: Optional ResourceStore instance. If not provided and
                   config.storage_path is set, a new store will be created.
        """
        self.config = config
        self._user_id = config.user_id

        if store is not None:
            self.store = store
        elif config.storage_path:
            backend = FilesystemBackend(config.storage_path)
            self.store = ResourceStore(backend)
        else:
            raise ValueError("Either 'store' or 'config.storage_path' must be provided")

        self.memory_api = MemoryAPI(self.store)

    def _get_scope(self) -> Scope:
        """Get scope for memory storage."""
        return Scope(type="memory", id=f"nat:{self._user_id}")

    async def add_items(self, items: list[MemoryItem]) -> list[str]:
        """
        Add memory items to storage.

        Args:
            items: List of MemoryItem objects to store

        Returns
        -------
            List of IDs for the stored items
        """
        ids = []
        for item in items:
            item_id = item.id or str(uuid.uuid4())
            timestamp = item.created_at or datetime.now(timezone.utc).isoformat()

            # Store in ResourceStore
            await self.store.put(
                scope=self._get_scope(),
                kind="nat_memory",
                data=json.dumps(
                    {
                        "content": item.content,
                        "metadata": item.metadata,
                        "created_at": timestamp,
                    }
                ),
                lifetime="persistent",
                contentType="application/json",
                metadata={
                    "item_id": item_id,
                    "created_at": timestamp,
                    **item.metadata,
                },
                resource_id=f"nat_memory:{self._user_id}:{item_id}",
            )
            ids.append(item_id)

        logger.debug(f"Added {len(ids)} memory items for user {self._user_id}")
        return ids

    async def search(
        self,
        query: str,
        top_k: int = 5,
        **kwargs: Any,
    ) -> list[MemoryItem]:
        """
        Search for relevant memory items.

        Note: This implementation uses simple text matching. For semantic
        search, integrate with an embedding model and vector store.

        Args:
            query: Search query string
            top_k: Maximum number of results to return
            **kwargs: Additional search parameters (filters, etc.)

        Returns
        -------
            List of MemoryItem objects matching the query
        """
        scope = self._get_scope()
        resources = await self.store.query(scope=scope, kind="nat_memory")

        # Simple text-based search
        results = []
        query_lower = query.lower()

        for resource in resources:
            result = await self.store.get(resource.id)
            if not result:
                continue

            _, data = result
            content_str = data if isinstance(data, str) else data.decode("utf-8")

            try:
                content_data = json.loads(content_str)
            except json.JSONDecodeError:
                content_data = {"content": content_str}

            text_content = content_data.get("content", "")

            # Simple relevance scoring based on term presence
            if query_lower in text_content.lower():
                # Calculate simple score based on match position
                pos = text_content.lower().find(query_lower)
                score = 1.0 - (pos / max(len(text_content), 1))

                results.append(
                    MemoryItem(
                        id=resource.metadata.get("item_id"),
                        content=text_content,
                        metadata=content_data.get("metadata", {}),
                        score=score,
                        created_at=content_data.get("created_at"),
                    )
                )

        # Sort by score descending and limit
        results.sort(key=lambda x: x.score or 0, reverse=True)
        return results[:top_k]

    async def remove_items(
        self,
        item_ids: list[str] | None = None,
        **kwargs: Any,
    ) -> int:
        """
        Remove memory items.

        Args:
            item_ids: Optional list of specific item IDs to remove.
                     If None, removes all items for the user.
            **kwargs: Additional filter criteria

        Returns
        -------
            Number of items removed
        """
        scope = self._get_scope()
        resources = await self.store.query(scope=scope, kind="nat_memory")

        removed = 0
        for resource in resources:
            if item_ids is None:
                # Remove all
                await self.store.delete(resource.id)
                removed += 1
            elif resource.metadata.get("item_id") in item_ids:
                # Remove specific items
                await self.store.delete(resource.id)
                removed += 1

        logger.debug(f"Removed {removed} memory items for user {self._user_id}")
        return removed

    async def get_item(self, item_id: str) -> MemoryItem | None:
        """
        Get a specific memory item by ID.

        Args:
            item_id: The ID of the item to retrieve

        Returns
        -------
            MemoryItem if found, None otherwise
        """
        resource_id = f"nat_memory:{self._user_id}:{item_id}"
        result = await self.store.get(resource_id)

        if not result:
            return None

        resource, data = result
        content_str = data if isinstance(data, str) else data.decode("utf-8")

        try:
            content_data = json.loads(content_str)
        except json.JSONDecodeError:
            content_data = {"content": content_str}

        return MemoryItem(
            id=item_id,
            content=content_data.get("content", ""),
            metadata=content_data.get("metadata", {}),
            created_at=content_data.get("created_at"),
        )

    async def update_item(
        self,
        item_id: str,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Update an existing memory item.

        Args:
            item_id: The ID of the item to update
            content: New content (if provided)
            metadata: New metadata (if provided, replaces existing)

        Returns
        -------
            True if updated, False if item not found
        """
        existing = await self.get_item(item_id)
        if not existing:
            return False

        # Update fields
        new_content = content if content is not None else existing.content
        new_metadata = metadata if metadata is not None else existing.metadata

        # Delete and re-create (simpler than update in place)
        await self.remove_items(item_ids=[item_id])
        await self.add_items(
            [
                MemoryItem(
                    id=item_id,
                    content=new_content,
                    metadata=new_metadata,
                    created_at=existing.created_at,
                )
            ]
        )

        return True

    async def list_items(
        self,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[MemoryItem]:
        """
        List all memory items.

        Args:
            limit: Maximum number of items to return
            offset: Number of items to skip

        Returns
        -------
            List of MemoryItem objects
        """
        scope = self._get_scope()
        resources = await self.store.query(scope=scope, kind="nat_memory")

        # Sort by creation time
        resources.sort(key=lambda r: r.metadata.get("created_at", ""), reverse=True)

        # Apply pagination
        if offset:
            resources = resources[offset:]
        if limit:
            resources = resources[:limit]

        items = []
        for resource in resources:
            result = await self.store.get(resource.id)
            if not result:
                continue

            _, data = result
            content_str = data if isinstance(data, str) else data.decode("utf-8")

            try:
                content_data = json.loads(content_str)
            except json.JSONDecodeError:
                content_data = {"content": content_str}

            items.append(
                MemoryItem(
                    id=resource.metadata.get("item_id"),
                    content=content_data.get("content", ""),
                    metadata=content_data.get("metadata", {}),
                    created_at=content_data.get("created_at"),
                )
            )

        return items

    async def clear(self) -> int:
        """
        Clear all memory items for this user.

        Returns
        -------
            Number of items removed
        """
        return await self.remove_items()

    # Sync wrapper methods for convenience

    def add_items_sync(self, items: list[MemoryItem]) -> list[str]:
        """Sync version of add_items."""
        return _run_async(self.add_items(items))

    def search_sync(self, query: str, top_k: int = 5, **kwargs: Any) -> list[MemoryItem]:
        """Sync version of search."""
        return _run_async(self.search(query, top_k, **kwargs))

    def remove_items_sync(
        self, item_ids: list[str] | None = None, **kwargs: Any
    ) -> int:
        """Sync version of remove_items."""
        return _run_async(self.remove_items(item_ids, **kwargs))

    def get_item_sync(self, item_id: str) -> MemoryItem | None:
        """Sync version of get_item."""
        return _run_async(self.get_item(item_id))

    def update_item_sync(
        self,
        item_id: str,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Sync version of update_item."""
        return _run_async(self.update_item(item_id, content, metadata))

    def list_items_sync(
        self, limit: int | None = None, offset: int = 0
    ) -> list[MemoryItem]:
        """Sync version of list_items."""
        return _run_async(self.list_items(limit, offset))

    def clear_sync(self) -> int:
        """Sync version of clear."""
        return _run_async(self.clear())


def create_nat_memory_editor(
    store: ResourceStore,
    user_id: str = "default",
) -> ResourceStoreMemoryEditor:
    """
    Create a NAT-compatible memory editor backed by ResourceStore.

    This is a convenience function for quick integration with NAT.

    Args:
        store: ResourceStore instance
        user_id: User identifier for memory isolation

    Returns
    -------
        ResourceStoreMemoryEditor instance

    Example:
        ```python
        from datarobot_genai.drmcp.core.resource_store import ResourceStore
        from datarobot_genai.drmcp.core.resource_store.backends.filesystem import FilesystemBackend
        from datarobot_genai.drmcp.core.resource_store.adapters.nat_adapter import (
            create_nat_memory_editor,
            MemoryItem,
        )

        backend = FilesystemBackend("/path/to/storage")
        store = ResourceStore(backend)
        memory = create_nat_memory_editor(store, user_id="user_123")

        # Add memories
        await memory.add_items([
            MemoryItem(content="User prefers dark mode"),
            MemoryItem(content="User is a Python developer"),
        ])

        # Search memories
        results = await memory.search("python", top_k=5)
        ```
    """
    config = ResourceStoreMemoryConfig(user_id=user_id)
    return ResourceStoreMemoryEditor(config, store=store)
