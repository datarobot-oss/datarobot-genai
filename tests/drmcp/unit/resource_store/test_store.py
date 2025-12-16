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

"""Tests for ResourceStore high-level API."""

import pytest

from datarobot_genai.drmcp.core.resource_store.models import Scope
from datarobot_genai.drmcp.core.resource_store.store import ResourceStore


@pytest.mark.asyncio
class TestResourceStore:
    """Tests for ResourceStore."""

    async def test_put_with_auto_id(self, store: ResourceStore) -> None:
        """Test put with auto-generated ID."""
        scope = Scope(type="conversation", id="conv_123")
        resource = await store.put(
            scope=scope,
            kind="message",
            data="Hello, World!",
            lifetime="ephemeral",
            contentType="text/plain",
        )
        assert resource.id is not None
        assert len(resource.id) > 0

    async def test_put_with_custom_id(self, store: ResourceStore) -> None:
        """Test put with custom ID."""
        scope = Scope(type="conversation", id="conv_123")
        resource = await store.put(
            scope=scope,
            kind="message",
            data="Hello, World!",
            lifetime="ephemeral",
            contentType="text/plain",
            resource_id="custom_id_123",
        )
        assert resource.id == "custom_id_123"

    async def test_get(self, store: ResourceStore) -> None:
        """Test get operation."""
        scope = Scope(type="conversation", id="conv_123")
        resource = await store.put(
            scope=scope,
            kind="message",
            data="test data",
            lifetime="ephemeral",
            contentType="text/plain",
        )

        result = await store.get(resource.id)
        assert result is not None
        retrieved_resource, data = result
        assert retrieved_resource.id == resource.id
        assert data == "test data"

    async def test_get_bytes(self, store: ResourceStore) -> None:
        """Test storing and retrieving bytes."""
        scope = Scope(type="resource", id="res_scope")
        resource = await store.put(
            scope=scope,
            kind="blob",
            data=b"\x00\x01\x02\x03",
            lifetime="ephemeral",
            contentType="application/octet-stream",
        )

        result = await store.get(resource.id)
        assert result is not None
        _, data = result
        assert data == b"\x00\x01\x02\x03"

    async def test_query_by_scope(self, store: ResourceStore) -> None:
        """Test querying by scope."""
        scope1 = Scope(type="conversation", id="conv_123")
        scope2 = Scope(type="conversation", id="conv_456")

        await store.put(
            scope=scope1,
            kind="message",
            data="msg1",
            lifetime="ephemeral",
            contentType="text/plain",
        )
        await store.put(
            scope=scope1,
            kind="message",
            data="msg2",
            lifetime="ephemeral",
            contentType="text/plain",
        )
        await store.put(
            scope=scope2,
            kind="message",
            data="msg3",
            lifetime="ephemeral",
            contentType="text/plain",
        )

        results = await store.query(scope=scope1)
        assert len(results) == 2

    async def test_query_by_kind(self, store: ResourceStore) -> None:
        """Test querying by kind."""
        scope = Scope(type="conversation", id="conv_123")

        await store.put(
            scope=scope, kind="message", data="msg1", lifetime="ephemeral", contentType="text/plain"
        )
        await store.put(
            scope=scope,
            kind="tool-call",
            data="call1",
            lifetime="ephemeral",
            contentType="application/json",
        )
        await store.put(
            scope=scope, kind="message", data="msg2", lifetime="ephemeral", contentType="text/plain"
        )

        results = await store.query(scope=scope, kind="message")
        assert len(results) == 2

    async def test_query_by_lifetime(self, store: ResourceStore) -> None:
        """Test querying by lifetime."""
        scope = Scope(type="memory", id="user_123")

        await store.put(
            scope=scope,
            kind="note",
            data="note1",
            lifetime="ephemeral",
            contentType="text/markdown",
        )
        await store.put(
            scope=scope,
            kind="note",
            data="note2",
            lifetime="persistent",
            contentType="text/markdown",
        )
        await store.put(
            scope=scope,
            kind="note",
            data="note3",
            lifetime="persistent",
            contentType="text/markdown",
        )

        results = await store.query(scope=scope, lifetime="persistent")
        assert len(results) == 2

    async def test_query_by_metadata(self, store: ResourceStore) -> None:
        """Test querying by metadata."""
        scope = Scope(type="memory", id="user_123")

        await store.put(
            scope=scope,
            kind="note",
            data="note1",
            lifetime="persistent",
            contentType="text/markdown",
            metadata={"tag": "important"},
        )
        await store.put(
            scope=scope,
            kind="note",
            data="note2",
            lifetime="persistent",
            contentType="text/markdown",
            metadata={"tag": "unimportant"},
        )
        await store.put(
            scope=scope,
            kind="note",
            data="note3",
            lifetime="persistent",
            contentType="text/markdown",
            metadata={"tag": "important"},
        )

        results = await store.query(scope=scope, metadata={"tag": "important"})
        assert len(results) == 2

    async def test_delete(self, store: ResourceStore) -> None:
        """Test delete operation."""
        scope = Scope(type="conversation", id="conv_123")
        resource = await store.put(
            scope=scope,
            kind="message",
            data="test data",
            lifetime="ephemeral",
            contentType="text/plain",
        )

        assert await store.get(resource.id) is not None
        await store.delete(resource.id)
        assert await store.get(resource.id) is None

    async def test_cleanup_expired(self, store: ResourceStore) -> None:
        """Test cleanup_expired operation."""
        scope = Scope(type="conversation", id="conv_123")

        # Create resource with short TTL
        await store.put(
            scope=scope,
            kind="message",
            data="expires soon",
            lifetime="ephemeral",
            contentType="text/plain",
            ttlSeconds=1,
        )

        # Wait a bit and cleanup
        import asyncio  # noqa: PLC0415

        await asyncio.sleep(1.1)

        count = await store.cleanup_expired()
        assert count >= 1
