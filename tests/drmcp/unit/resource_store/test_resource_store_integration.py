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

"""Integration tests for ResourceStore system."""

import tempfile
from pathlib import Path

import pytest

from datarobot_genai.drmcp.core.resource_store.backends.filesystem import FilesystemBackend
from datarobot_genai.drmcp.core.resource_store.conversation import ConversationState
from datarobot_genai.drmcp.core.resource_store.memory import MemoryAPI
from datarobot_genai.drmcp.core.resource_store.registration import get_resource_manager
from datarobot_genai.drmcp.core.resource_store.registration import initialize_resource_store
from datarobot_genai.drmcp.core.resource_store.resource_api import ResourceAPI
from datarobot_genai.drmcp.core.resource_store.resource_manager import ResourceStoreBackedResourceManager
from datarobot_genai.drmcp.core.resource_store.store import ResourceStore
from fastmcp import FastMCP
from fastmcp.resources import HttpResource


@pytest.mark.asyncio
class TestResourceStoreIntegration:
    """Integration tests for ResourceStore system."""

    async def test_full_workflow(self) -> None:
        """Test a complete workflow using all components."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = FilesystemBackend(Path(tmpdir))
            store = ResourceStore(backend)

            # Test conversation state
            conversation = ConversationState(store)
            await conversation.add_message("conv_123", "user", "Hello")
            await conversation.add_message("conv_123", "assistant", "Hi!")
            history = await conversation.get_history("conv_123")
            assert len(history) == 2

            # Test memory API
            memory = MemoryAPI(store)
            note_id = await memory.write("user_123", "note", "Remember this")
            note = await memory.read(note_id)
            assert note is not None
            assert note["content"] == "Remember this"

            # Test resource API
            resource_api = ResourceAPI(store)
            resource_id = await resource_api.store_resource(
                "conv_123",
                data="large data content",
                content_type="text/plain",
                name="Large Output",
            )
            data, content_type = await resource_api.get_resource(resource_id)
            assert data == "large data content"
            assert content_type == "text/plain"

    async def test_resource_manager_integration(self) -> None:
        """Test ResourceStoreBackedResourceManager integration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = FilesystemBackend(Path(tmpdir))
            store = ResourceStore(backend)

            manager = ResourceStoreBackedResourceManager(
                store=store,
                default_scope_id="test_scope",
            )

            resource = HttpResource(
                uri="mcp://resources/test_res",  # type: ignore[arg-type]
                url="mcp://resources/test_res",
                name="Test Resource",
                mime_type="text/plain",
            )

            manager.add_resource(resource, data="test content")

            # Verify stored
            result = await store.get("test_res")
            assert result is not None
            _, data = result
            assert data == "test content"

    async def test_initialization_and_registration(self) -> None:
        """Test initialize_resource_store and get_resource_manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mcp = FastMCP("test_server")

            # Initialize
            manager = initialize_resource_store(
                mcp=mcp,
                storage_path=tmpdir,
                default_scope_id="default_scope",
            )

            assert isinstance(manager, ResourceStoreBackedResourceManager)
            assert manager.default_scope_id == "default_scope"

            # Get global instance
            retrieved_manager = get_resource_manager()
            assert retrieved_manager is not None
            assert retrieved_manager == manager

    async def test_multiple_scopes_isolation(self) -> None:
        """Test that different scopes are isolated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = FilesystemBackend(Path(tmpdir))
            store = ResourceStore(backend)

            # Create resources in different scopes
            conversation = ConversationState(store)
            await conversation.add_message("conv_1", "user", "Message 1")
            await conversation.add_message("conv_2", "user", "Message 2")

            memory = MemoryAPI(store)
            await memory.write("user_1", "note", "Note 1")
            await memory.write("user_2", "note", "Note 2")

            # Verify isolation
            history_1 = await conversation.get_history("conv_1")
            history_2 = await conversation.get_history("conv_2")
            assert len(history_1) == 1
            assert len(history_2) == 1
            assert history_1[0]["content"] == "Message 1"
            assert history_2[0]["content"] == "Message 2"

            notes_1 = await memory.search("user_1")
            notes_2 = await memory.search("user_2")
            assert len(notes_1) == 1
            assert len(notes_2) == 1

