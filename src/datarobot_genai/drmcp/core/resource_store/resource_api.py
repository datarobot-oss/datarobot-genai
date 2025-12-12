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

"""Resource API front door for ResourceStore.

This module provides MCP resources interface for managing large outputs
using the ResourceStore with scope.type='resource'.
"""

import logging
from typing import Any

from fastmcp.resources import HttpResource

from .models import Lifetime
from .models import Scope
from .store import ResourceStore

logger = logging.getLogger(__name__)


class ResourceAPI:
    """
    Resource API for MCP resources.

    Uses ResourceStore with scope.type='resource'.
    """

    def __init__(self, store: ResourceStore) -> None:
        """
        Initialize ResourceAPI.

        Args:
            store: ResourceStore instance
        """
        self.store = store

    async def store_resource(
        self,
        scope_id: str,
        data: bytes | str,
        content_type: str = "application/octet-stream",
        name: str | None = None,
        lifetime: Lifetime = "ephemeral",
        ttl_seconds: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Store a resource and return its ID.

        Args:
            scope_id: Scope identifier (conversation id, agent id)
            data: Resource content
            content_type: MIME type
            name: Optional resource name
            lifetime: 'ephemeral' or 'persistent'
            ttl_seconds: TTL for ephemeral resources
            metadata: Optional metadata

        Returns
        -------
            Resource ID
        """
        scope = Scope(type="resource", id=scope_id)
        metadata = metadata or {}
        if name:
            metadata["name"] = name

        resource = await self.store.put(
            scope=scope,
            kind="blob",
            data=data,
            lifetime=lifetime,
            contentType=content_type,
            ttlSeconds=ttl_seconds,
            metadata=metadata,
        )

        return resource.id

    async def get_resource(self, resource_id: str) -> tuple[bytes | str | None, str] | None:
        """
        Get resource data.

        Args:
            resource_id: Resource identifier

        Returns
        -------
            Tuple of (data, content_type) or None if not found
        """
        result = await self.store.get(resource_id)
        if not result:
            return None

        resource, data = result
        if resource.scope.type != "resource":
            return None

        return (data, resource.contentType)

    async def list_resources(self, scope_id: str) -> list[dict[str, Any]]:
        """
        List resources for a scope.

        Args:
            scope_id: Scope identifier

        Returns
        -------
            List of resource metadata dictionaries
        """
        scope = Scope(type="resource", id=scope_id)
        resources = await self.store.query(scope=scope)

        return [
            {
                "id": r.id,
                "name": r.metadata.get("name", ""),
                "content_type": r.contentType,
                "created_at": r.createdAt.isoformat(),
                "lifetime": r.lifetime,
            }
            for r in resources
        ]

    def create_mcp_resource(self, resource_id: str, url: str | None = None) -> HttpResource:
        """
        Create an MCP HttpResource for a stored resource.

        Args:
            resource_id: Resource identifier
            url: Optional URL (if None, uses mcp://resources/{resource_id})

        Returns
        -------
            HttpResource instance
        """
        if url is None:
            url = f"mcp://resources/{resource_id}"

        return HttpResource(
            uri=url,  # type: ignore[arg-type]
            url=url,
            name=f"Resource {resource_id}",
            mime_type="application/octet-stream",
        )
