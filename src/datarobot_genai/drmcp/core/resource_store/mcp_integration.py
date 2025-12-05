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

"""Integration layer between ResourceStore and FastMCP Resources.

This module bridges the unified ResourceStore system with FastMCP's Resource classes,
allowing tools to store data in ResourceStore while exposing it via MCP's resource protocol.
"""

import logging
from typing import Any

from fastmcp import FastMCP
from fastmcp.resources import HttpResource

from .models import Lifetime
from .resource_api import ResourceAPI
from .store import ResourceStore

logger = logging.getLogger(__name__)


class MCPResourceIntegration:
    """
    Integration layer between ResourceStore and FastMCP Resources.

    This class:
    1. Stores data in ResourceStore (unified backend)
    2. Creates FastMCP HttpResource instances pointing to stored resources
    3. Registers resources with FastMCP's MCP server
    4. Implements MCP protocol handlers (list_resources, read_resource) that query ResourceStore
    """

    def __init__(self, store: ResourceStore, mcp: FastMCP) -> None:
        """
        Initialize MCP resource integration.

        Args:
            store: ResourceStore instance
            mcp: FastMCP instance
        """
        self.store = store
        self.mcp = mcp
        self.resource_api = ResourceAPI(store)

    async def store_and_register_resource(
        self,
        scope_id: str,
        data: bytes | str,
        content_type: str = "application/octet-stream",
        name: str | None = None,
        lifetime: Lifetime = "ephemeral",
        ttl_seconds: int | None = None,
        metadata: dict[str, Any] | None = None,
        url: str | None = None,
    ) -> tuple[str, HttpResource]:
        """
        Store data in ResourceStore and register as MCP resource.

        This is the main integration point: tools call this to store large outputs,
        and it handles both backend storage (ResourceStore) and MCP protocol registration.

        Args:
            scope_id: Scope identifier (conversation id, agent id)
            data: Resource content
            content_type: MIME type
            name: Optional resource name
            lifetime: 'ephemeral' or 'persistent'
            ttl_seconds: TTL for ephemeral resources
            metadata: Optional metadata
            url: Optional external URL (if None, uses mcp://resources/{resource_id})

        Returns
        -------
            Tuple of (resource_id, HttpResource)
        """
        # Store in ResourceStore
        resource_id = await self.resource_api.store_resource(
            scope_id=scope_id,
            data=data,
            content_type=content_type,
            name=name,
            lifetime=lifetime,
            ttl_seconds=ttl_seconds,
            metadata=metadata,
        )

        # Create HttpResource pointing to the stored resource
        if url is None:
            # Use a URI scheme that indicates this is a ResourceStore resource
            url = f"mcp://resources/{resource_id}"

        http_resource = HttpResource(
            uri=url,  # type: ignore[arg-type]
            url=url,
            name=name or f"Resource {resource_id}",
            mime_type=content_type,
        )

        # Register with FastMCP
        self.mcp.add_resource(http_resource)

        logger.debug(f"Stored and registered resource {resource_id} as MCP resource")
        return (resource_id, http_resource)

    async def list_resources_for_scope(self, scope_id: str) -> list[HttpResource]:
        """
        List MCP resources for a scope by querying ResourceStore.

        This implements the MCP list_resources handler using ResourceStore as the backend.

        Args:
            scope_id: Scope identifier

        Returns
        -------
            List of HttpResource instances
        """
        # Query ResourceStore
        resources = await self.resource_api.list_resources(scope_id)

        # Convert to HttpResource instances
        http_resources = []
        for resource_info in resources:
            resource_id = resource_info["id"]
            url = f"mcp://resources/{resource_id}"
            http_resource = HttpResource(
                uri=url,  # type: ignore[arg-type]
                url=url,
                name=resource_info.get("name", f"Resource {resource_id}"),
                mime_type=resource_info.get("content_type", "application/octet-stream"),
            )
            http_resources.append(http_resource)

        return http_resources

    async def read_resource_from_store(
        self, resource_id: str
    ) -> tuple[bytes | str | None, str] | None:
        """
        Read resource data from ResourceStore.

        This implements the MCP read_resource handler using ResourceStore as the backend.

        Args:
            resource_id: Resource identifier (extracted from URI like mcp://resources/{id})

        Returns
        -------
            Tuple of (data, content_type) or None if not found
        """
        return await self.resource_api.get_resource(resource_id)

    def register_mcp_handlers(self) -> None:
        """
        Register MCP protocol handlers with FastMCP.

        This hooks ResourceStore into FastMCP's resource handling, so when clients
        call list_resources or read_resource, they query ResourceStore.

        Note: FastMCP may handle this automatically via add_resource(), but this
        allows custom handling if needed.
        """
        # FastMCP automatically handles list_resources and read_resource for resources
        # added via add_resource(). However, if we want dynamic querying from ResourceStore,
        # we could override handlers here.

        # For now, resources are registered via add_resource() and FastMCP handles them.
        # If we need dynamic querying, we'd need to check FastMCP's extension points.
        logger.debug("MCP handlers registered (FastMCP handles resources automatically)")

