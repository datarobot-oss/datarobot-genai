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

"""Tests for ResourceAPI."""

import pytest
from fastmcp.resources import HttpResource

from datarobot_genai.drmcp.core.resource_store.models import Scope
from datarobot_genai.drmcp.core.resource_store.resource_api import ResourceAPI
from datarobot_genai.drmcp.core.resource_store.store import ResourceStore


@pytest.mark.asyncio
class TestResourceAPI:
    """Tests for ResourceAPI."""

    async def test_store_resource(self, store: ResourceStore) -> None:
        """Test storing a resource."""
        api = ResourceAPI(store)
        resource_id = await api.store_resource(
            scope_id="conv_123",
            data="large data content",
            content_type="text/plain",
            name="Large Output",
            lifetime="ephemeral",
            ttl_seconds=3600,
        )
        assert resource_id is not None

        # Verify stored
        result = await store.get(resource_id)
        assert result is not None
        resource, data = result
        assert resource.scope.type == "resource"
        assert resource.scope.id == "conv_123"
        assert resource.kind == "blob"
        assert resource.lifetime == "ephemeral"
        assert resource.contentType == "text/plain"
        assert resource.metadata["name"] == "Large Output"
        assert data == "large data content"

    async def test_store_resource_bytes(self, store: ResourceStore) -> None:
        """Test storing binary resource."""
        api = ResourceAPI(store)
        binary_data = b"\x00\x01\x02\x03"
        resource_id = await api.store_resource(
            scope_id="conv_123",
            data=binary_data,
            content_type="application/octet-stream",
        )

        result = await api.get_resource(resource_id)
        assert result is not None
        data, content_type = result
        assert data == binary_data
        assert content_type == "application/octet-stream"

    async def test_get_resource(self, store: ResourceStore) -> None:
        """Test getting a resource."""
        api = ResourceAPI(store)
        resource_id = await api.store_resource(
            scope_id="conv_123",
            data="test content",
            content_type="text/csv",
        )

        result = await api.get_resource(resource_id)
        assert result is not None
        data, content_type = result
        assert data == "test content"
        assert content_type == "text/csv"

    async def test_get_resource_nonexistent(self, store: ResourceStore) -> None:
        """Test getting non-existent resource."""
        api = ResourceAPI(store)
        result = await api.get_resource("nonexistent")
        assert result is None

    async def test_get_resource_non_resource_scope(self, store: ResourceStore) -> None:
        """Test that get_resource only works for resource scope."""
        api = ResourceAPI(store)

        # Create a non-resource scope resource
        await store.put(
            scope=Scope(type="conversation", id="conv_123"),
            kind="message",
            data="test",
            lifetime="ephemeral",
            contentType="text/plain",
            resource_id="non_resource",
        )

        result = await api.get_resource("non_resource")
        assert result is None

    async def test_list_resources(self, store: ResourceStore) -> None:
        """Test listing resources."""
        api = ResourceAPI(store)

        await api.store_resource("conv_123", "data1", "text/plain", name="Resource 1")
        await api.store_resource("conv_123", "data2", "text/csv", name="Resource 2")
        await api.store_resource("conv_456", "data3", "text/plain", name="Resource 3")

        resources = await api.list_resources("conv_123")
        assert len(resources) == 2
        assert {r["name"] for r in resources} == {"Resource 1", "Resource 2"}

    async def test_list_resources_empty(self, store: ResourceStore) -> None:
        """Test listing resources for empty scope."""
        api = ResourceAPI(store)
        resources = await api.list_resources("empty_scope")
        assert len(resources) == 0

    async def test_create_mcp_resource(self, store: ResourceStore) -> None:
        """Test creating MCP HttpResource."""
        api = ResourceAPI(store)
        resource_id = "test_123"

        http_resource = api.create_mcp_resource(resource_id)
        assert isinstance(http_resource, HttpResource)
        assert str(http_resource.uri) == f"mcp://resources/{resource_id}"
        assert str(http_resource.url) == f"mcp://resources/{resource_id}"

    async def test_create_mcp_resource_custom_url(self, store: ResourceStore) -> None:
        """Test creating MCP HttpResource with custom URL."""
        api = ResourceAPI(store)
        resource_id = "test_123"
        custom_url = "https://example.com/resource"

        http_resource = api.create_mcp_resource(resource_id, url=custom_url)
        assert str(http_resource.uri) == custom_url
        assert str(http_resource.url) == custom_url
