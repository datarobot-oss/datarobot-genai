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

"""Memory API front door for ResourceStore.

This module provides MCP tools for managing long-term memory
using the ResourceStore with scope.type='memory' and lifetime='persistent'.
"""

import json
import logging
from typing import Any

from datarobot_genai.drmcp.core.mcp_instance import dr_core_mcp_tool

from .models import Scope
from .store import ResourceStore

logger = logging.getLogger(__name__)


class MemoryAPI:
    """
    Memory API for persistent storage.

    Uses ResourceStore with scope.type='memory' and lifetime='persistent'.
    """

    def __init__(self, store: ResourceStore) -> None:
        """
        Initialize MemoryAPI.

        Args:
            store: ResourceStore instance
        """
        self.store = store

    async def write(
        self,
        scope_id: str,
        kind: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Write a memory entry.

        Args:
            scope_id: Scope identifier (user id, agent id, project id)
            kind: Memory kind ('note', 'preference', 'profile', etc.)
            content: Content to store (text or JSON string)
            metadata: Optional metadata dictionary

        Returns
        -------
            Resource ID of the stored memory
        """
        scope = Scope(type="memory", id=scope_id)
        resource = await self.store.put(
            scope=scope,
            kind=kind,
            data=content,
            lifetime="persistent",
            contentType="text/markdown",
            metadata=metadata or {},
        )

        return resource.id

    async def read(self, resource_id: str) -> dict[str, Any] | None:
        """
        Read a memory entry.

        Args:
            resource_id: Resource identifier

        Returns
        -------
            Dictionary with resource metadata and content, or None if not found
        """
        result = await self.store.get(resource_id)
        if not result:
            return None

        resource, data = result
        return {
            "id": resource.id,
            "scope_id": resource.scope.id,
            "kind": resource.kind,
            "content": data if isinstance(data, str) else data.decode("utf-8") if data else "",
            "metadata": resource.metadata,
            "created_at": resource.createdAt.isoformat(),
        }

    async def search(
        self,
        scope_id: str,
        kind: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search memory entries.

        Args:
            scope_id: Scope identifier
            kind: Optional kind filter
            metadata: Optional metadata filters

        Returns
        -------
            List of memory entries matching the criteria
        """
        scope = Scope(type="memory", id=scope_id)
        resources = await self.store.query(scope=scope, kind=kind, metadata=metadata)

        results = []
        for resource in resources:
            result = await self.store.get(resource.id)
            if result:
                _, data = result
                results.append(
                    {
                        "id": resource.id,
                        "kind": resource.kind,
                        "content": (
                            data if isinstance(data, str) else data.decode("utf-8") if data else ""
                        ),
                        "metadata": resource.metadata,
                        "created_at": resource.createdAt.isoformat(),
                    }
                )

        return results

    async def delete(self, resource_id: str) -> bool:
        """
        Delete a memory entry.

        Args:
            resource_id: Resource identifier

        Returns
        -------
            True if deleted, False if not found
        """
        result = await self.store.get(resource_id)
        if not result:
            return False

        resource, _ = result
        if resource.scope.type != "memory":
            return False

        await self.store.delete(resource_id)
        return True


# Global MemoryAPI instance (will be initialized by drmcp server)
_memory_api: MemoryAPI | None = None


def set_memory_api(api: MemoryAPI) -> None:
    """Set the global MemoryAPI instance."""
    global _memory_api  # noqa: PLW0603
    _memory_api = api


@dr_core_mcp_tool()
async def memory_write(
    scope_id: str,
    kind: str,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    """
    Write a memory entry.

    Args:
        scope_id: Scope identifier (user id, agent id, project id)
        kind: Memory kind ('note', 'preference', 'profile', etc.)
        content: Content to store (text or JSON string)
        metadata: Optional metadata dictionary

    Returns
    -------
        Resource ID of the stored memory
    """
    if not _memory_api:
        return "Memory API not initialized"

    resource_id = await _memory_api.write(scope_id, kind, content, metadata)
    return f"Memory stored with ID: {resource_id}"


@dr_core_mcp_tool()
async def memory_read(resource_id: str) -> str:
    """
    Read a memory entry.

    Args:
        resource_id: Resource identifier

    Returns
    -------
        JSON string with memory entry data
    """
    if not _memory_api:
        return "Memory API not initialized"

    result = await _memory_api.read(resource_id)
    if not result:
        return "Memory not found"

    return json.dumps(result, default=str)


@dr_core_mcp_tool()
async def memory_search(
    scope_id: str,
    kind: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    """
    Search memory entries.

    Args:
        scope_id: Scope identifier
        kind: Optional kind filter
        metadata: Optional metadata filters

    Returns
    -------
        JSON string with list of matching memory entries
    """
    if not _memory_api:
        return "Memory API not initialized"

    results = await _memory_api.search(scope_id, kind, metadata)
    return json.dumps(results, default=str)


@dr_core_mcp_tool()
async def memory_delete(resource_id: str) -> str:
    """
    Delete a memory entry.

    Args:
        resource_id: Resource identifier

    Returns
    -------
        Success or error message
    """
    if not _memory_api:
        return "Memory API not initialized"

    success = await _memory_api.delete(resource_id)
    if success:
        return f"Memory {resource_id} deleted successfully"
    return f"Memory {resource_id} not found or not a memory resource"
