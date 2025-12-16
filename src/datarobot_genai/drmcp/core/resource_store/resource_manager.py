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

"""ResourceStore-backed ResourceManager extending FastMCP's ResourceManager.

This class extends FastMCP's ResourceManager to use ResourceStore as the backend,
providing persistent storage with rich metadata and scoping capabilities.
"""

import asyncio
import concurrent.futures
import logging
from typing import Any

from fastmcp.resources import HttpResource
from fastmcp.resources import ResourceManager

from .models import Lifetime
from .models import Scope
from .resource_api import ResourceAPI
from .store import ResourceStore

logger = logging.getLogger(__name__)


class ResourceStoreBackedResourceManager(ResourceManager):
    """
    ResourceManager that uses ResourceStore as the backend.

    This extends FastMCP's ResourceManager to:
    1. Store resources in ResourceStore (unified backend)
    2. Maintain compatibility with FastMCP's resource protocol
    3. Provide rich metadata and scoping capabilities
    4. Support ephemeral and persistent resources

    Usage:
        ```python
        store = ResourceStore(backend)
        resource_manager = ResourceStoreBackedResourceManager(
            store, default_scope_id="conversation_123"
        )

        # Use like FastMCP's ResourceManager
        resource = HttpResource(uri="...", url="...", name="...", mime_type="...")
        resource_manager.add_resource(resource, data=b"content", lifetime="ephemeral")
        ```
    """

    def __init__(
        self,
        store: ResourceStore,
        default_scope_id: str | None = None,
        default_lifetime: Lifetime = "ephemeral",
        default_ttl_seconds: int | None = None,
    ) -> None:
        """
        Initialize ResourceStore-backed ResourceManager.

        Args:
            store: ResourceStore instance
            default_scope_id: Default scope ID for resources (e.g., conversation_id)
            default_lifetime: Default lifetime ('ephemeral' or 'persistent')
            default_ttl_seconds: Default TTL for ephemeral resources
        """
        super().__init__()
        self.store = store
        self.resource_api = ResourceAPI(store)
        self.default_scope_id = default_scope_id or "default"
        self.default_lifetime = default_lifetime
        self.default_ttl_seconds = default_ttl_seconds

    def add_resource(  # type: ignore[override]
        self,
        resource: HttpResource,
        data: bytes | str | None = None,
        scope_id: str | None = None,
        lifetime: Lifetime | None = None,
        ttl_seconds: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Add a resource, storing it in ResourceStore.

        This extends FastMCP's add_resource to also store data in ResourceStore.

        Args:
            resource: HttpResource instance
            data: Resource content (if None, resource.url is used as reference)
            scope_id: Scope identifier (defaults to self.default_scope_id)
            lifetime: Resource lifetime (defaults to self.default_lifetime)
            ttl_seconds: TTL for ephemeral resources (defaults to self.default_ttl_seconds)
            metadata: Additional metadata
        """
        # Call parent to register with FastMCP
        super().add_resource(resource)

        # Extract resource ID from URI (e.g., "mcp://resources/abc123" -> "abc123")
        resource_id = self._extract_resource_id_from_uri(resource.uri)

        # Determine scope
        scope_id = scope_id or self.default_scope_id
        lifetime = lifetime or self.default_lifetime
        ttl_seconds = ttl_seconds or self.default_ttl_seconds

        # Prepare metadata
        metadata = metadata or {}
        metadata.update(
            {
                "name": resource.name,
                "uri": str(resource.uri),
                "url": str(resource.url) if resource.url else "",
            }
        )

        # Store in ResourceStore
        # Use store.put directly to support resource_id parameter
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is running, we need to run in a thread-safe way
                # Use run_coroutine_threadsafe to execute in the running loop

                def run_in_thread() -> Any:
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(
                            self.store.put(
                                scope=Scope(type="resource", id=scope_id),
                                kind="blob",
                                data=data,
                                lifetime=lifetime,
                                contentType=resource.mime_type or "application/octet-stream",
                                ttlSeconds=ttl_seconds,
                                metadata=metadata,
                                resource_id=resource_id,
                            )
                        )
                    finally:
                        new_loop.close()

                # Run in a separate thread to avoid blocking
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_in_thread)
                    future.result()  # Wait for completion
            else:
                # Loop exists but not running
                loop.run_until_complete(
                    self.store.put(
                        scope=Scope(type="resource", id=scope_id),
                        kind="blob",
                        data=data,
                        lifetime=lifetime,
                        contentType=resource.mime_type or "application/octet-stream",
                        ttlSeconds=ttl_seconds,
                        metadata=metadata,
                        resource_id=resource_id,
                    )
                )
        except RuntimeError:
            # No event loop, create one
            asyncio.run(
                self.store.put(
                    scope=Scope(type="resource", id=scope_id),
                    kind="blob",
                    data=data,
                    lifetime=lifetime,
                    contentType=resource.mime_type or "application/octet-stream",
                    ttlSeconds=ttl_seconds,
                    metadata=metadata,
                    resource_id=resource_id,
                )
            )

        logger.debug(f"Stored resource {resource_id} in ResourceStore")

    def _extract_resource_id_from_uri(self, uri: str | Any) -> str:
        """Extract resource ID from URI."""
        # Convert AnyUrl to string if needed
        if hasattr(uri, "__str__"):
            uri = str(uri)

        # Handle various URI formats:
        # - "mcp://resources/abc123" -> "abc123"
        # - "predictions://abc123" -> "abc123"
        # - "http://example.com/resource/abc123" -> use last segment
        # - If no clear pattern, use URI as-is
        if "://" in uri:
            parts = uri.split("://", 1)
            if len(parts) > 1:
                path = parts[1]
                # Get last segment
                if "/" in path:
                    return path.split("/")[-1]
                return path
        return uri

    async def get_resource_data(self, resource_id: str) -> tuple[bytes | str | None, str] | None:
        """
        Get resource data from ResourceStore.

        Args:
            resource_id: Resource identifier

        Returns
        -------
            Tuple of (data, content_type) or None if not found
        """
        return await self.resource_api.get_resource(resource_id)

    async def list_resources_for_scope(self, scope_id: str | None = None) -> list[HttpResource]:
        """
        List resources for a scope.

        Args:
            scope_id: Scope identifier (defaults to default_scope_id)

        Returns
        -------
            List of HttpResource instances
        """
        scope_id = scope_id or self.default_scope_id
        resources = await self.resource_api.list_resources(scope_id)

        http_resources = []
        for resource_info in resources:
            resource_id = resource_info["id"]
            # Reconstruct URI from stored metadata or generate new one
            metadata = await self._get_resource_metadata(resource_id)
            uri = metadata.get("uri") if metadata else f"mcp://resources/{resource_id}"
            url = metadata.get("url") if metadata else f"mcp://resources/{resource_id}"

            http_resource = HttpResource(
                uri=str(uri),  # type: ignore[arg-type]
                url=str(url) if url else "",
                name=resource_info.get("name", f"Resource {resource_id}"),
                mime_type=resource_info.get("content_type", "application/octet-stream"),
            )
            http_resources.append(http_resource)

        return http_resources

    async def _get_resource_metadata(self, resource_id: str) -> dict[str, Any] | None:
        """Get resource metadata from ResourceStore."""
        result = await self.store.get(resource_id)
        if result:
            resource, _ = result
            return resource.metadata
        return None
