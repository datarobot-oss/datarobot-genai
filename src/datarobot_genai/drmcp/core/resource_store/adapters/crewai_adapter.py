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
CrewAI Memory Adapter for ResourceStore.

This module provides adapters to use ResourceStore/MemoryAPI as a backend
for CrewAI's memory system.

Usage:
    ```python
    from crewai import Crew
    from crewai.memory.external.external_memory import ExternalMemory
    from datarobot_genai.drmcp.core.resource_store.adapters.crewai_adapter import (
        ResourceStoreStorage,
    )
    from datarobot_genai.drmcp.core.resource_store import ResourceStore
    from datarobot_genai.drmcp.core.resource_store.backends.filesystem import FilesystemBackend

    # Create ResourceStore
    backend = FilesystemBackend("/path/to/storage")
    store = ResourceStore(backend)

    # Create CrewAI-compatible storage
    storage = ResourceStoreStorage(store, user_id="user_123")

    # Use with CrewAI ExternalMemory
    external_memory = ExternalMemory(storage=storage)

    crew = Crew(
        agents=[...],
        tasks=[...],
        external_memory=external_memory
    )
    ```

See: https://docs.crewai.com/en/concepts/memory
"""

import asyncio
import concurrent.futures
import json
import logging
from typing import Any

from ..memory import MemoryAPI
from ..store import ResourceStore

logger = logging.getLogger(__name__)


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


class ResourceStoreStorage:
    """
    CrewAI Storage implementation backed by ResourceStore.

    This class implements the Storage interface expected by CrewAI's
    memory system, using ResourceStore/MemoryAPI as the backend.

    CrewAI Storage Interface:
        - save(value, metadata=None, agent=None): Save a memory
        - search(query, limit=10, score_threshold=0.5): Search memories
        - reset(): Clear all memories

    Attributes
    ----------
        store: ResourceStore instance
        user_id: User/scope identifier for memory isolation
        memory_api: MemoryAPI instance for storage operations
    """

    def __init__(
        self,
        store: ResourceStore,
        user_id: str = "default",
    ) -> None:
        """
        Initialize ResourceStoreStorage.

        Args:
            store: ResourceStore instance for persistent storage
            user_id: User/scope identifier for memory isolation (default: "default")
        """
        self.store = store
        self.user_id = user_id
        self.memory_api = MemoryAPI(store)

    def save(
        self,
        value: str,
        metadata: dict[str, Any] | None = None,
        agent: str | None = None,
    ) -> None:
        """
        Save a memory entry.

        Args:
            value: The memory content to store
            metadata: Optional metadata dictionary
            agent: Optional agent role/name that created this memory
        """
        meta = metadata.copy() if metadata else {}
        if agent:
            meta["agent"] = agent

        # Determine kind based on metadata or default to "memory"
        kind = meta.pop("kind", "memory")

        _run_async(
            self.memory_api.write(
                scope_id=self.user_id,
                kind=kind,
                content=value,
                metadata=meta,
            )
        )

    def search(
        self,
        query: str,
        limit: int = 10,
        score_threshold: float = 0.5,
    ) -> list[dict[str, Any]]:
        """
        Search memory entries.

        Note: This implementation does simple text matching since ResourceStore
        doesn't have built-in semantic search. For semantic search, consider
        using an embedder with ResourceStore or a dedicated vector store.

        Args:
            query: Search query string
            limit: Maximum number of results to return
            score_threshold: Minimum relevance score (not used in basic implementation)

        Returns
        -------
            List of memory entries matching the query
        """
        # Get all memories for this user
        results = _run_async(self.memory_api.search(scope_id=self.user_id))

        # Simple text-based filtering
        matched = []
        query_lower = query.lower()
        for memory in results:
            content = memory.get("content", "")
            if query_lower in content.lower():
                matched.append(
                    {
                        "value": content,
                        "metadata": memory.get("metadata", {}),
                        "score": 1.0,  # Simple match, no scoring
                    }
                )

        # Sort by recency (created_at) and limit
        matched.sort(key=lambda x: x.get("metadata", {}).get("created_at", ""), reverse=True)
        return matched[:limit]

    def reset(self) -> None:
        """
        Clear all memories for this user.

        Warning: This permanently deletes all memory entries for the configured user_id.
        """
        results = _run_async(self.memory_api.search(scope_id=self.user_id))

        for memory in results:
            memory_id = memory.get("id")
            if memory_id:
                _run_async(self.memory_api.delete(memory_id))

        logger.info(f"Reset {len(results)} memories for user {self.user_id}")


class ResourceStoreLongTermStorage:
    """
    CrewAI Long-Term Memory Storage backed by ResourceStore.

    This provides persistent storage for CrewAI's long-term memory,
    storing task results and learnings across sessions.

    Usage:
        ```python
        from crewai import Crew
        from crewai.memory import LongTermMemory
        from datarobot_genai.drmcp.core.resource_store.adapters.crewai_adapter import (
            ResourceStoreLongTermStorage,
        )

        storage = ResourceStoreLongTermStorage(store, user_id="project_123")
        long_term_memory = LongTermMemory(storage=storage)

        crew = Crew(
            agents=[...],
            tasks=[...],
            memory=True,
            long_term_memory=long_term_memory
        )
        ```
    """

    def __init__(
        self,
        store: ResourceStore,
        user_id: str = "default",
    ) -> None:
        """
        Initialize ResourceStoreLongTermStorage.

        Args:
            store: ResourceStore instance
            user_id: User/scope identifier
        """
        self.store = store
        self.user_id = user_id
        self.memory_api = MemoryAPI(store)

    def save(
        self,
        task_description: str,
        result: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Save a long-term memory entry (task result).

        Args:
            task_description: Description of the task
            result: The task result/output
            metadata: Optional additional metadata
        """
        meta = metadata.copy() if metadata else {}
        meta["task_description"] = task_description

        content = json.dumps(
            {
                "task": task_description,
                "result": result,
            }
        )

        _run_async(
            self.memory_api.write(
                scope_id=self.user_id,
                kind="long_term_memory",
                content=content,
                metadata=meta,
            )
        )

    def search(
        self,
        task_description: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Search for relevant past task results.

        Args:
            task_description: Current task to find related memories for
            limit: Maximum results to return

        Returns
        -------
            List of relevant past task results
        """
        results = _run_async(
            self.memory_api.search(scope_id=self.user_id, kind="long_term_memory")
        )

        # Simple text matching on task description
        matched = []
        query_lower = task_description.lower()

        for memory in results:
            content_str = memory.get("content", "")
            try:
                content = json.loads(content_str)
            except json.JSONDecodeError:
                content = {"task": "", "result": content_str}

            task = content.get("task", "")
            if query_lower in task.lower() or any(
                word in task.lower() for word in query_lower.split()[:5]
            ):
                matched.append(
                    {
                        "task": task,
                        "result": content.get("result", ""),
                        "metadata": memory.get("metadata", {}),
                    }
                )

        return matched[:limit]

    def reset(self) -> None:
        """Clear all long-term memories."""
        results = _run_async(
            self.memory_api.search(scope_id=self.user_id, kind="long_term_memory")
        )

        for memory in results:
            memory_id = memory.get("id")
            if memory_id:
                _run_async(self.memory_api.delete(memory_id))


def create_crewai_external_memory(
    store: ResourceStore,
    user_id: str = "default",
):
    """
    Create a CrewAI ExternalMemory instance backed by ResourceStore.

    This is a convenience function for quick integration with CrewAI.

    Args:
        store: ResourceStore instance
        user_id: User/scope identifier for memory isolation

    Returns
    -------
        ExternalMemory instance configured to use ResourceStore

    Example:
        ```python
        from crewai import Crew
        from datarobot_genai.drmcp.core.resource_store import ResourceStore
        from datarobot_genai.drmcp.core.resource_store.backends.filesystem import FilesystemBackend
        from datarobot_genai.drmcp.core.resource_store.adapters.crewai_adapter import (
            create_crewai_external_memory,
        )

        backend = FilesystemBackend("/path/to/storage")
        store = ResourceStore(backend)
        external_memory = create_crewai_external_memory(store, user_id="user_123")

        crew = Crew(
            agents=[...],
            tasks=[...],
            external_memory=external_memory
        )
        ```
    """
    try:
        from crewai.memory.external.external_memory import ExternalMemory  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            "crewai is required for this adapter. Install with: pip install crewai"
        ) from e

    storage = ResourceStoreStorage(store, user_id=user_id)
    return ExternalMemory(storage=storage)
