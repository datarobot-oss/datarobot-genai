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

"""Tests for FilesystemBackend."""

from datetime import datetime
from datetime import timedelta
from datetime import timezone

import pytest

from datarobot_genai.drmcp.core.resource_store.backends.filesystem import FilesystemBackend
from datarobot_genai.drmcp.core.resource_store.models import Resource
from datarobot_genai.drmcp.core.resource_store.models import Scope


@pytest.mark.asyncio
class TestFilesystemBackend:
    """Tests for FilesystemBackend."""

    async def test_put_and_get_text(self, backend: FilesystemBackend) -> None:
        """Test storing and retrieving text data."""
        scope = Scope(type="conversation", id="conv_123")
        resource = Resource(
            id="res_123",
            scope=scope,
            kind="message",
            lifetime="ephemeral",
            contentType="text/plain",
            contentRef="",
        )

        stored = await backend.put(resource, data="Hello, World!")
        assert stored.id == "res_123"
        assert stored.contentRef != ""

        result = await backend.get("res_123")
        assert result is not None
        retrieved_resource, data = result
        assert retrieved_resource.id == "res_123"
        assert data == "Hello, World!"

    async def test_put_and_get_bytes(self, backend: FilesystemBackend) -> None:
        """Test storing and retrieving binary data."""
        scope = Scope(type="resource", id="res_scope")
        resource = Resource(
            id="res_456",
            scope=scope,
            kind="blob",
            lifetime="ephemeral",
            contentType="application/octet-stream",
            contentRef="",
        )

        binary_data = b"\x00\x01\x02\x03"
        stored = await backend.put(resource, data=binary_data)
        assert stored.id == "res_456"

        result = await backend.get("res_456")
        assert result is not None
        retrieved_resource, data = result
        assert data == binary_data

    async def test_put_and_get_json(self, backend: FilesystemBackend) -> None:
        """Test storing and retrieving JSON data."""
        scope = Scope(type="memory", id="user_123")
        resource = Resource(
            id="res_789",
            scope=scope,
            kind="note",
            lifetime="persistent",
            contentType="application/json",
            contentRef="",
        )

        json_data = '{"key": "value", "number": 42}'
        stored = await backend.put(resource, data=json_data)
        assert stored.id == "res_789"

        result = await backend.get("res_789")
        assert result is not None
        retrieved_resource, data = result
        assert data == json_data

    async def test_put_without_data(self, backend: FilesystemBackend) -> None:
        """Test storing resource metadata without data."""
        scope = Scope(type="resource", id="res_scope")
        resource = Resource(
            id="res_no_data",
            scope=scope,
            kind="blob",
            lifetime="ephemeral",
            contentType="application/octet-stream",
            contentRef="external://url",
        )

        stored = await backend.put(resource, data=None)
        assert stored.id == "res_no_data"
        assert stored.contentRef != ""

        result = await backend.get("res_no_data")
        assert result is not None
        retrieved_resource, data = result
        assert data is None or data == ""

    async def test_get_nonexistent(self, backend: FilesystemBackend) -> None:
        """Test retrieving non-existent resource."""
        result = await backend.get("nonexistent")
        assert result is None

    async def test_delete(self, backend: FilesystemBackend) -> None:
        """Test deleting a resource."""
        scope = Scope(type="conversation", id="conv_123")
        resource = Resource(
            id="res_to_delete",
            scope=scope,
            kind="message",
            lifetime="ephemeral",
            contentType="text/plain",
            contentRef="",
        )

        await backend.put(resource, data="test data")
        result = await backend.get("res_to_delete")
        assert result is not None

        await backend.delete("res_to_delete")
        result = await backend.get("res_to_delete")
        assert result is None

    async def test_delete_nonexistent(self, backend: FilesystemBackend) -> None:
        """Test deleting non-existent resource (should not raise)."""
        await backend.delete("nonexistent")  # Should not raise

    async def test_query_by_scope(self, backend: FilesystemBackend) -> None:
        """Test querying resources by scope."""
        scope1 = Scope(type="conversation", id="conv_123")
        scope2 = Scope(type="conversation", id="conv_456")

        resource1 = Resource(
            id="res_1",
            scope=scope1,
            kind="message",
            lifetime="ephemeral",
            contentType="text/plain",
            contentRef="",
        )
        resource2 = Resource(
            id="res_2",
            scope=scope1,
            kind="message",
            lifetime="ephemeral",
            contentType="text/plain",
            contentRef="",
        )
        resource3 = Resource(
            id="res_3",
            scope=scope2,
            kind="message",
            lifetime="ephemeral",
            contentType="text/plain",
            contentRef="",
        )

        await backend.put(resource1, data="data1")
        await backend.put(resource2, data="data2")
        await backend.put(resource3, data="data3")

        results = await backend.query(filters={"scope": scope1})
        assert len(results) == 2
        assert {r.id for r in results} == {"res_1", "res_2"}

    async def test_query_by_kind(self, backend: FilesystemBackend) -> None:
        """Test querying resources by kind."""
        scope = Scope(type="conversation", id="conv_123")

        resource1 = Resource(
            id="res_1",
            scope=scope,
            kind="message",
            lifetime="ephemeral",
            contentType="text/plain",
            contentRef="",
        )
        resource2 = Resource(
            id="res_2",
            scope=scope,
            kind="tool-call",
            lifetime="ephemeral",
            contentType="application/json",
            contentRef="",
        )
        resource3 = Resource(
            id="res_3",
            scope=scope,
            kind="message",
            lifetime="ephemeral",
            contentType="text/plain",
            contentRef="",
        )

        await backend.put(resource1, data="data1")
        await backend.put(resource2, data="data2")
        await backend.put(resource3, data="data3")

        results = await backend.query(filters={"kind": "message"})
        assert len(results) == 2
        assert {r.id for r in results} == {"res_1", "res_3"}

    async def test_query_by_lifetime(self, backend: FilesystemBackend) -> None:
        """Test querying resources by lifetime."""
        scope = Scope(type="memory", id="user_123")

        resource1 = Resource(
            id="res_1",
            scope=scope,
            kind="note",
            lifetime="ephemeral",
            contentType="text/markdown",
            contentRef="",
        )
        resource2 = Resource(
            id="res_2",
            scope=scope,
            kind="note",
            lifetime="persistent",
            contentType="text/markdown",
            contentRef="",
        )
        resource3 = Resource(
            id="res_3",
            scope=scope,
            kind="note",
            lifetime="persistent",
            contentType="text/markdown",
            contentRef="",
        )

        await backend.put(resource1, data="data1")
        await backend.put(resource2, data="data2")
        await backend.put(resource3, data="data3")

        results = await backend.query(filters={"lifetime": "persistent"})
        assert len(results) == 2
        assert {r.id for r in results} == {"res_2", "res_3"}

    async def test_query_by_metadata(self, backend: FilesystemBackend) -> None:
        """Test querying resources by metadata."""
        scope = Scope(type="memory", id="user_123")

        resource1 = Resource(
            id="res_1",
            scope=scope,
            kind="note",
            lifetime="persistent",
            contentType="text/markdown",
            metadata={"tag": "important", "category": "work"},
            contentRef="",
        )
        resource2 = Resource(
            id="res_2",
            scope=scope,
            kind="note",
            lifetime="persistent",
            contentType="text/markdown",
            metadata={"tag": "important", "category": "personal"},
            contentRef="",
        )
        resource3 = Resource(
            id="res_3",
            scope=scope,
            kind="note",
            lifetime="persistent",
            contentType="text/markdown",
            metadata={"tag": "unimportant"},
            contentRef="",
        )

        await backend.put(resource1, data="data1")
        await backend.put(resource2, data="data2")
        await backend.put(resource3, data="data3")

        results = await backend.query(filters={"metadata": {"tag": "important"}})
        assert len(results) == 2
        assert {r.id for r in results} == {"res_1", "res_2"}

    async def test_query_multiple_filters(self, backend: FilesystemBackend) -> None:
        """Test querying with multiple filters."""
        scope = Scope(type="conversation", id="conv_123")

        resource1 = Resource(
            id="res_1",
            scope=scope,
            kind="message",
            lifetime="ephemeral",
            contentType="text/plain",
            contentRef="",
        )
        resource2 = Resource(
            id="res_2",
            scope=scope,
            kind="message",
            lifetime="ephemeral",
            contentType="text/plain",
            contentRef="",
        )
        resource3 = Resource(
            id="res_3",
            scope=scope,
            kind="tool-call",
            lifetime="ephemeral",
            contentType="application/json",
            contentRef="",
        )

        await backend.put(resource1, data="data1")
        await backend.put(resource2, data="data2")
        await backend.put(resource3, data="data3")

        results = await backend.query(
            filters={"scope": scope, "kind": "message", "lifetime": "ephemeral"}
        )
        assert len(results) == 2
        assert {r.id for r in results} == {"res_1", "res_2"}

    async def test_query_empty(self, backend: FilesystemBackend) -> None:
        """Test querying with no filters returns all resources."""
        scope = Scope(type="conversation", id="conv_123")

        resource1 = Resource(
            id="res_1",
            scope=scope,
            kind="message",
            lifetime="ephemeral",
            contentType="text/plain",
            contentRef="",
        )
        resource2 = Resource(
            id="res_2",
            scope=scope,
            kind="message",
            lifetime="ephemeral",
            contentType="text/plain",
            contentRef="",
        )

        await backend.put(resource1, data="data1")
        await backend.put(resource2, data="data2")

        results = await backend.query(filters=None)
        assert len(results) == 2

    async def test_cleanup_expired(self, backend: FilesystemBackend) -> None:
        """Test cleaning up expired ephemeral resources."""
        scope = Scope(type="conversation", id="conv_123")

        # Create expired resource (TTL of 1 second, created 2 seconds ago)
        expired_resource = Resource(
            id="expired_res",
            scope=scope,
            kind="message",
            lifetime="ephemeral",
            contentType="text/plain",
            createdAt=datetime.now(timezone.utc) - timedelta(seconds=2),
            ttlSeconds=1,
            contentRef="",
        )

        # Create non-expired resource
        valid_resource = Resource(
            id="valid_res",
            scope=scope,
            kind="message",
            lifetime="ephemeral",
            contentType="text/plain",
            createdAt=datetime.now(timezone.utc),
            ttlSeconds=3600,
            contentRef="",
        )

        # Create persistent resource (should not be cleaned up)
        persistent_resource = Resource(
            id="persistent_res",
            scope=scope,
            kind="note",
            lifetime="persistent",
            contentType="text/markdown",
            contentRef="",
        )

        await backend.put(expired_resource, data="expired")
        await backend.put(valid_resource, data="valid")
        await backend.put(persistent_resource, data="persistent")

        # Cleanup should remove expired resource
        count = await backend.cleanup_expired()
        assert count >= 1  # At least the expired one

        # Verify expired is gone, others remain
        assert await backend.get("expired_res") is None
        assert await backend.get("valid_res") is not None
        assert await backend.get("persistent_res") is not None

    async def test_put_updates_existing(self, backend: FilesystemBackend) -> None:
        """Test that put updates existing resource."""
        scope = Scope(type="conversation", id="conv_123")
        resource = Resource(
            id="res_update",
            scope=scope,
            kind="message",
            lifetime="ephemeral",
            contentType="text/plain",
            contentRef="",
        )

        await backend.put(resource, data="original")
        result = await backend.get("res_update")
        assert result is not None
        _, data = result
        assert data == "original"

        # Update with new data
        await backend.put(resource, data="updated")
        result = await backend.get("res_update")
        assert result is not None
        _, data = result
        assert data == "updated"
