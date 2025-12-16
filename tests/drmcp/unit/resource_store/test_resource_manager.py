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

"""Tests for ResourceStoreBackedResourceManager."""

import pytest
from fastmcp.resources import HttpResource
from fastmcp.resources import ResourceManager

from datarobot_genai.drmcp.core.resource_store.models import Scope
from datarobot_genai.drmcp.core.resource_store.resource_manager import (
    ResourceStoreBackedResourceManager,
)
from datarobot_genai.drmcp.core.resource_store.store import ResourceStore


@pytest.mark.asyncio
class TestResourceStoreBackedResourceManager:
    """Tests for ResourceStoreBackedResourceManager."""

    async def test_initialization(self, store: ResourceStore) -> None:
        """Test ResourceManager initialization."""
        manager = ResourceStoreBackedResourceManager(
            store=store,
            default_scope_id="test_scope",
            default_lifetime="ephemeral",
            default_ttl_seconds=3600,
        )
        assert manager.store == store
        assert manager.default_scope_id == "test_scope"
        assert manager.default_lifetime == "ephemeral"
        assert manager.default_ttl_seconds == 3600

    async def test_add_resource_with_data(self, store: ResourceStore) -> None:
        """Test adding resource with data."""
        manager = ResourceStoreBackedResourceManager(store=store, default_scope_id="test_scope")

        resource = HttpResource(
            uri="mcp://resources/test_123",  # type: ignore[arg-type]
            url="mcp://resources/test_123",
            name="Test Resource",
            mime_type="text/plain",
        )

        manager.add_resource(
            resource,
            data="test content",
            scope_id="custom_scope",
            lifetime="ephemeral",
            ttl_seconds=1800,
        )

        # Verify stored in ResourceStore
        result = await store.get("test_123")
        assert result is not None
        retrieved_resource, data = result
        assert data == "test content"
        assert retrieved_resource.scope.id == "custom_scope"
        assert retrieved_resource.lifetime == "ephemeral"
        assert retrieved_resource.ttlSeconds == 1800

    async def test_add_resource_without_data(self, store: ResourceStore) -> None:
        """Test adding resource without data (metadata only)."""
        manager = ResourceStoreBackedResourceManager(store=store, default_scope_id="test_scope")

        resource = HttpResource(
            uri="mcp://resources/test_456",  # type: ignore[arg-type]
            url="https://external.com/resource",
            name="External Resource",
            mime_type="application/json",
        )

        manager.add_resource(resource, data=None)

        # Verify metadata stored
        result = await store.get("test_456")
        assert result is not None
        retrieved_resource, data = result
        assert retrieved_resource.metadata["name"] == "External Resource"
        assert retrieved_resource.metadata["url"] == "https://external.com/resource"

    async def test_add_resource_defaults(self, store: ResourceStore) -> None:
        """Test add_resource uses defaults."""
        manager = ResourceStoreBackedResourceManager(
            store=store,
            default_scope_id="default_scope",
            default_lifetime="persistent",
            default_ttl_seconds=7200,
        )

        resource = HttpResource(
            uri="mcp://resources/test_789",  # type: ignore[arg-type]
            url="mcp://resources/test_789",
            name="Default Resource",
            mime_type="text/plain",
        )

        manager.add_resource(resource, data="default content")

        result = await store.get("test_789")
        assert result is not None
        retrieved_resource, _ = result
        assert retrieved_resource.scope.id == "default_scope"
        assert retrieved_resource.lifetime == "persistent"
        assert retrieved_resource.ttlSeconds == 7200

    async def test_extract_resource_id_from_uri(self, store: ResourceStore) -> None:
        """Test URI parsing for resource ID extraction."""
        manager = ResourceStoreBackedResourceManager(store=store)

        # Test various URI formats
        test_cases = [
            ("mcp://resources/abc123", "abc123"),
            ("predictions://xyz789", "xyz789"),
            ("http://example.com/resource/def456", "def456"),
            ("simple_id", "simple_id"),
        ]

        for uri, expected_id in test_cases:
            resource_id = manager._extract_resource_id_from_uri(uri)
            assert resource_id == expected_id

    async def test_get_resource_data(self, store: ResourceStore) -> None:
        """Test getting resource data."""
        manager = ResourceStoreBackedResourceManager(store=store, default_scope_id="test_scope")

        # Store a resource
        await store.put(
            scope=Scope(type="resource", id="test_scope"),
            kind="blob",
            data="test data",
            lifetime="ephemeral",
            contentType="text/plain",
            resource_id="test_resource",
        )

        result = await manager.get_resource_data("test_resource")
        assert result is not None
        data, content_type = result
        assert data == "test data"
        assert content_type == "text/plain"

    async def test_get_resource_data_nonexistent(self, store: ResourceStore) -> None:
        """Test getting non-existent resource data."""
        manager = ResourceStoreBackedResourceManager(store=store)
        result = await manager.get_resource_data("nonexistent")
        assert result is None

    async def test_list_resources_for_scope(self, store: ResourceStore) -> None:
        """Test listing resources for a scope."""
        manager = ResourceStoreBackedResourceManager(store=store, default_scope_id="test_scope")

        # Store multiple resources
        scope = Scope(type="resource", id="test_scope")
        await store.put(
            scope=scope,
            kind="blob",
            data="data1",
            lifetime="ephemeral",
            contentType="text/plain",
            resource_id="res1",
            metadata={
                "name": "Resource 1",
                "uri": "mcp://resources/res1",
                "url": "mcp://resources/res1",
            },
        )
        await store.put(
            scope=scope,
            kind="blob",
            data="data2",
            lifetime="ephemeral",
            contentType="text/csv",
            resource_id="res2",
            metadata={
                "name": "Resource 2",
                "uri": "mcp://resources/res2",
                "url": "mcp://resources/res2",
            },
        )

        resources = await manager.list_resources_for_scope("test_scope")
        assert len(resources) == 2
        assert {r.name for r in resources} == {"Resource 1", "Resource 2"}

    async def test_list_resources_empty_scope(self, store: ResourceStore) -> None:
        """Test listing resources for empty scope."""
        manager = ResourceStoreBackedResourceManager(store=store)
        resources = await manager.list_resources_for_scope("empty_scope")
        assert len(resources) == 0

    async def test_inherits_from_resource_manager(self, store: ResourceStore) -> None:
        """Test that ResourceStoreBackedResourceManager inherits from ResourceManager."""
        manager = ResourceStoreBackedResourceManager(store=store)
        assert isinstance(manager, ResourceManager)
