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
from datarobot_genai.drmcp.core.mcp_instance import mcp

from mem0 import MemoryClient

from .models import Scope
from .store import ResourceStore

logger = logging.getLogger(__name__)


class MemoryAPI:
    """
    Memory API for persistent storage.

    Uses ResourceStore with scope.type='memory' and lifetime='persistent'.
    """

    def __init__(self, client: Mem0Client) -> None:
        """
        Initialize MemoryAPI.

        Args:
            client: Mem0Client
        """
        self.client = client

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
            Mem0 memory ID of the stored entry
        """
        scope = Scope(type="memory", id=scope_id)
        meta = metadata.copy() if metadata else {}
        meta.setdefault("kind", kind)
        meta.setdefault("scope_id", scope_id)

        memory_id = await self.client.add_memory(
            user_id=scope_id,
            text=content,
            metadata=meta,
        )

        return memory_id


    async def read(self, resource_id: str) -> dict[str, Any] | None:
        """
        Read a memory entry.

        Args:
            resource_id: Resource identifier

        Returns
        -------
            Dictionary with resource metadata and content, or None if not found
        """
        mem = await self.client.get_memory(resource_id)
        if not mem:
            return None

        return {
            "id": mem.get("id", resource_id),
            "scope_id": mem.get("user_id") or mem.get("scope_id"),
            "kind": mem.get("metadata", {}).get("kind"),
            "content": mem.get("text") or mem.get("content", ""),
            "metadata": mem.get("metadata", {}),
            "created_at": mem.get("created_at"),
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
        meta_filters = metadata.copy() if metadata else {}
        if kind:
            meta_filters.setdefault("kind", kind)

        memories = await self.client.search_memories(
            user_id=scope_id,
            metadata=meta_filters or None,
        )

        results: list[dict[str, Any]] = []
        for mem in memories:
            results.append(
                {
                    "id": mem.get("id"),
                    "kind": mem.get("metadata", {}).get("kind"),
                    "content": mem.get("text") or mem.get("content", ""),
                    "metadata": mem.get("metadata", {}),
                    "created_at": mem.get("created_at"),
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
        deleted = await self.client.delete_memory(resource_id)
        return bool(deleted)

def _get_memory_api() -> MemoryAPI:
    """
    Factory for MemoryAPI using Mem0 as backend.

    Reads configuration from environment variables such as:
    - MEM0_BASE_URL
    - MEM0_API_KEY
    """
    base_url = os.getenv("MEM0_BASE_URL")
    api_key = os.getenv("MEM0_API_KEY")

    client = Mem0Client(
        base_url=base_url,
        api_key=api_key,
    )

    return MemoryAPI(client)


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
    memory_api = _get_memory_api()
    resource_id = await memory_api.write(scope_id, kind, content, metadata)
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
    memory_api = _get_memory_api()
    result = await memory_api.read(resource_id)
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
    memory_api = _get_memory_api()
    results = await memory_api.search(scope_id, kind, metadata)
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
    memory_api = _get_memory_api()
    success = await memory_api.delete(resource_id)
    if success:
        return f"Memory {resource_id} deleted successfully"
    return f"Memory {resource_id} not found or not a memory resource"
