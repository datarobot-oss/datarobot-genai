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

"""Main ResourceStore class providing unified API."""

import logging
import uuid
from typing import Any

from .backend import ResourceBackend
from .models import Lifetime
from .models import Resource
from .models import Scope

logger = logging.getLogger(__name__)


class ResourceStore:
    """
    Unified ResourceStore for conversation state, memory, and MCP resources.

    This class provides a high-level API over a ResourceBackend implementation.
    """

    def __init__(self, backend: ResourceBackend) -> None:
        """
        Initialize ResourceStore with a backend.

        Args:
            backend: Backend implementation (e.g., FilesystemBackend)
        """
        self.backend = backend

    async def put(
        self,
        scope: Scope,
        kind: str,
        data: bytes | str | None,
        lifetime: Lifetime = "persistent",
        contentType: str = "application/json",  # noqa: N803
        ttlSeconds: int | None = None,  # noqa: N803
        metadata: dict[str, Any] | None = None,
        resource_id: str | None = None,
    ) -> Resource:
        """
        Store a resource.

        Args:
            scope: Resource scope
            kind: Resource kind (e.g., 'message', 'tool-call', 'note', 'blob')
            data: Resource content (bytes, string, or None)
            lifetime: Resource lifetime ('ephemeral' or 'persistent')
            contentType: MIME type
            ttlSeconds: Time-to-live in seconds for ephemeral resources
            metadata: Additional metadata dictionary
            resource_id: Optional resource ID (auto-generated if not provided)

        Returns
        -------
            Stored Resource object
        """
        if resource_id is None:
            resource_id = str(uuid.uuid4())

        resource = Resource(
            id=resource_id,
            scope=scope,
            kind=kind,
            lifetime=lifetime,
            contentType=contentType,
            ttlSeconds=ttlSeconds,
            metadata=metadata or {},
            contentRef="",  # Will be set by backend
        )

        return await self.backend.put(resource, data)

    async def get(self, resource_id: str) -> tuple[Resource, bytes | str | None] | None:
        """
        Retrieve a resource and its data.

        Args:
            resource_id: Unique resource identifier

        Returns
        -------
            Tuple of (resource, data) if found, None otherwise
        """
        return await self.backend.get(resource_id)

    async def query(
        self,
        scope: Scope | None = None,
        kind: str | None = None,
        lifetime: Lifetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[Resource]:
        """
        Query resources by filters.

        Args:
            scope: Filter by scope
            kind: Filter by kind
            lifetime: Filter by lifetime
            metadata: Filter by metadata key-value pairs

        Returns
        -------
            List of matching resources
        """
        filters: dict[str, Any] = {}
        if scope:
            filters["scope"] = scope
        if kind:
            filters["kind"] = kind
        if lifetime:
            filters["lifetime"] = lifetime
        if metadata:
            filters["metadata"] = metadata

        return await self.backend.query(filters if filters else None)

    async def delete(self, resource_id: str) -> None:
        """
        Delete a resource.

        Args:
            resource_id: Unique resource identifier
        """
        await self.backend.delete(resource_id)

    async def cleanup_expired(self) -> int:
        """
        Clean up expired ephemeral resources.

        Returns
        -------
            Number of resources cleaned up
        """
        return await self.backend.cleanup_expired()
