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

"""Tests for ResourceStore models."""

from datetime import datetime

from datarobot_genai.drmcp.core.resource_store.models import Resource
from datarobot_genai.drmcp.core.resource_store.models import Scope


class TestScope:
    """Tests for Scope model."""

    def test_scope_creation(self) -> None:
        """Test creating a Scope."""
        scope = Scope(type="conversation", id="conv_123")
        assert scope.type == "conversation"
        assert scope.id == "conv_123"

    def test_scope_all_types(self) -> None:
        """Test all scope types."""
        for scope_type in ["conversation", "memory", "resource", "custom"]:
            scope = Scope(type=scope_type, id="test_id")
            assert scope.type == scope_type

    def test_scope_equality(self) -> None:
        """Test scope equality."""
        scope1 = Scope(type="conversation", id="conv_123")
        scope2 = Scope(type="conversation", id="conv_123")
        assert scope1 == scope2

    def test_scope_inequality(self) -> None:
        """Test scope inequality."""
        scope1 = Scope(type="conversation", id="conv_123")
        scope2 = Scope(type="memory", id="conv_123")
        assert scope1 != scope2


class TestResource:
    """Tests for Resource model."""

    def test_resource_creation_minimal(self) -> None:
        """Test creating a Resource with minimal fields."""
        scope = Scope(type="conversation", id="conv_123")
        resource = Resource(
            id="res_123",
            scope=scope,
            kind="message",
            lifetime="ephemeral",
            contentType="application/json",
            contentRef="/path/to/content",
        )
        assert resource.id == "res_123"
        assert resource.scope == scope
        assert resource.kind == "message"
        assert resource.lifetime == "ephemeral"
        assert resource.contentType == "application/json"
        assert resource.contentRef == "/path/to/content"
        assert isinstance(resource.createdAt, datetime)
        assert resource.metadata == {}

    def test_resource_creation_full(self) -> None:
        """Test creating a Resource with all fields."""
        scope = Scope(type="memory", id="user_123")
        resource = Resource(
            id="res_456",
            scope=scope,
            kind="note",
            lifetime="persistent",
            createdAt=datetime(2024, 1, 1, 12, 0, 0),
            ttlSeconds=3600,
            contentType="text/markdown",
            metadata={"tag": "important", "version": 1},
            contentRef="/path/to/note.md",
        )
        assert resource.id == "res_456"
        assert resource.scope == scope
        assert resource.kind == "note"
        assert resource.lifetime == "persistent"
        assert resource.ttlSeconds == 3600
        assert resource.contentType == "text/markdown"
        assert resource.metadata == {"tag": "important", "version": 1}
        assert resource.contentRef == "/path/to/note.md"

    def test_resource_default_created_at(self) -> None:
        """Test that createdAt is auto-generated if not provided."""
        scope = Scope(type="conversation", id="conv_123")
        resource = Resource(
            id="res_123",
            scope=scope,
            kind="message",
            lifetime="ephemeral",
            contentType="application/json",
            contentRef="/path/to/content",
        )
        assert isinstance(resource.createdAt, datetime)
        assert resource.createdAt <= datetime.utcnow()

    def test_resource_default_metadata(self) -> None:
        """Test that metadata defaults to empty dict."""
        scope = Scope(type="conversation", id="conv_123")
        resource = Resource(
            id="res_123",
            scope=scope,
            kind="message",
            lifetime="ephemeral",
            contentType="application/json",
            contentRef="/path/to/content",
        )
        assert resource.metadata == {}

    def test_resource_all_lifetimes(self) -> None:
        """Test all lifetime values."""
        scope = Scope(type="conversation", id="conv_123")
        for lifetime in ["ephemeral", "persistent"]:
            resource = Resource(
                id="res_123",
                scope=scope,
                kind="message",
                lifetime=lifetime,
                contentType="application/json",
                contentRef="/path/to/content",
            )
            assert resource.lifetime == lifetime

    def test_resource_json_serialization(self) -> None:
        """Test Resource JSON serialization."""
        scope = Scope(type="conversation", id="conv_123")
        resource = Resource(
            id="res_123",
            scope=scope,
            kind="message",
            lifetime="ephemeral",
            contentType="application/json",
            contentRef="/path/to/content",
        )
        # Should be able to serialize to JSON
        json_str = resource.model_dump_json()
        assert isinstance(json_str, str)
        assert "res_123" in json_str
        assert "conversation" in json_str
