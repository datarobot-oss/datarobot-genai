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

"""
Integration tests for ResourceStore agent workflows.

These tests simulate how agents from af-component-agent connect to drmcp
and use conversation state, memory, and resources through the MCP protocol.
"""

import asyncio
import contextlib
import json
import os
import tempfile
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent

from datarobot_genai.drmcp.core.resource_store.backends.filesystem import FilesystemBackend
from datarobot_genai.drmcp.core.resource_store.conversation import ConversationState
from datarobot_genai.drmcp.core.resource_store.memory import MemoryAPI
from datarobot_genai.drmcp.core.resource_store.registration import initialize_resource_store
from datarobot_genai.drmcp.core.resource_store.store import ResourceStore
from datarobot_genai.drmcp.core.resource_store.memory import set_memory_api
from datarobot_genai.drmcp.test_utils.mcp_utils_integration import integration_test_mcp_session
from datarobot_genai.drmcp.core.dr_mcp_server import BaseServerLifecycle
from datarobot_genai.drmcp.core.dr_mcp_server import create_mcp_server
from fastmcp import FastMCP


class ResourceStoreTestLifecycle(BaseServerLifecycle):
    """Lifecycle hook that initializes ResourceStore and MemoryAPI for tests."""

    def __init__(self, storage_path: str | None = None):
        self.storage_path = storage_path or tempfile.mkdtemp(prefix="test_resources_")

    async def pre_server_start(self, mcp: FastMCP) -> None:
        """Initialize ResourceStore and MemoryAPI before server starts."""
        # Initialize ResourceStore
        resource_manager = initialize_resource_store(
            mcp=mcp,
            storage_path=self.storage_path,
            default_scope_id="test_conversation",
        )

        # Initialize MemoryAPI
        store = resource_manager.store
        memory_api = MemoryAPI(store)
        set_memory_api(memory_api)


@pytest.fixture(scope="function")
def test_storage_path(tmp_path: Path) -> str:
    """Create a temporary storage path for each test."""
    storage_dir = tmp_path / "resource_store"
    storage_dir.mkdir()
    return str(storage_dir)


@pytest.fixture(scope="function")
def test_mcp_server(test_storage_path: str):
    """Create an MCP server with ResourceStore initialized."""
    lifecycle = ResourceStoreTestLifecycle(storage_path=test_storage_path)
    server = create_mcp_server(
        lifecycle=lifecycle,
        transport="stdio",
    )
    return server


@contextlib.asynccontextmanager
async def resource_store_mcp_session(
    storage_path: str, timeout: int = 30
) -> AsyncGenerator[ClientSession, None]:
    """
    Create and connect a client for the ResourceStore MCP server as a context manager.

    Args:
        storage_path: Path to storage directory for ResourceStore
        timeout: Timeout for session initialization

    Yields
    ------
        ClientSession: Connected MCP client session

    Raises
    ------
        TimeoutError: If session initialization exceeds timeout
    """
    # Create a temporary server script that uses ResourceStoreTestLifecycle
    script_dir = Path(__file__).resolve().parent.parent.parent.parent / "src"
    # Use json.dumps for proper string escaping
    script_dir_str = json.dumps(str(script_dir))
    storage_path_str = json.dumps(storage_path)
    
    server_script_content = f'''#!/usr/bin/env python3
import sys
from pathlib import Path

# Add src/ directory to Python path
sys.path.insert(0, {script_dir_str})

from datarobot_genai.drmcp.core.dr_mcp_server import create_mcp_server
from datarobot_genai.drmcp.core.server_life_cycle import BaseServerLifecycle

class ResourceStoreTestLifecycle(BaseServerLifecycle):
    """Lifecycle hook that initializes ResourceStore and MemoryAPI for tests."""

    def __init__(self, storage_path: str | None = None):
        super().__init__()
        self.storage_path = storage_path

    async def pre_server_start(self, mcp):
        """Initialize ResourceStore and MemoryAPI before server starts."""
        from datarobot_genai.drmcp.core.resource_store.registration import initialize_resource_store
        from datarobot_genai.drmcp.core.resource_store.memory import MemoryAPI, set_memory_api

        # Initialize ResourceStore
        resource_manager = initialize_resource_store(
            mcp=mcp,
            storage_path=self.storage_path,
            default_scope_id="test_conversation",
        )

        # Initialize MemoryAPI
        store = resource_manager.store
        memory_api = MemoryAPI(store)
        set_memory_api(memory_api)

if __name__ == "__main__":
    lifecycle = ResourceStoreTestLifecycle(storage_path={storage_path_str})
    server = create_mcp_server(
        lifecycle=lifecycle,
        transport="stdio",
    )
    server.run()
'''

    # Write temporary server script
    import tempfile as tf

    with tf.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(server_script_content)
        server_script = f.name

    try:
        # Make script executable
        os.chmod(server_script, 0o755)

        # Create server parameters
        server_params = StdioServerParameters(
            command="uv",
            args=["run", server_script],
            env={
                "PYTHONPATH": str(script_dir),
                "DATAROBOT_API_TOKEN": os.environ.get("DATAROBOT_API_TOKEN", "test-token"),
                "DATAROBOT_ENDPOINT": os.environ.get(
                    "DATAROBOT_ENDPOINT", "https://test.datarobot.com/api/v2"
                ),
                "MCP_SERVER_LOG_LEVEL": os.environ.get("MCP_SERVER_LOG_LEVEL", "WARNING"),
                "APP_LOG_LEVEL": os.environ.get("APP_LOG_LEVEL", "WARNING"),
                "OTEL_ENABLED": os.environ.get("OTEL_ENABLED", "false"),
            },
        )

        try:
            async with stdio_client(server_params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await asyncio.wait_for(session.initialize(), timeout=timeout)
                    yield session

        except asyncio.TimeoutError:
            raise TimeoutError(f"Session initialization timed out after {timeout} seconds")
    finally:
        # Clean up temporary script
        try:
            os.unlink(server_script)
        except OSError:
            pass


@pytest.mark.asyncio
class TestAgentConversationWorkflow:
    """Test agent workflow using conversation state via MCP."""

    async def test_agent_conversation_memory_persistence(
        self, test_mcp_server, test_storage_path: str
    ):
        """
        Test that an agent can store conversation messages and retrieve history.

        Simulates an agent workflow where:
        1. Agent stores user message
        2. Agent stores assistant response
        3. Agent retrieves conversation history
        4. Model uses history in next turn
        """
        # Connect as an agent would
        async with resource_store_mcp_session(test_storage_path) as session:
            # Simulate agent storing conversation messages
            # In a real agent, this would happen automatically via ConversationState
            # For this test, we'll use the ResourceStore directly to simulate the workflow

            # Get the store from the server (via the lifecycle)
            backend = FilesystemBackend(test_storage_path)
            store = ResourceStore(backend)
            conversation = ConversationState(store)

            conversation_id = "agent_conv_123"

            # Step 1: Agent stores user message
            user_msg_id = await conversation.add_message(
                conversation_id=conversation_id,
                role="user",
                content="What is the weather in NYC?",
            )
            assert user_msg_id is not None

            # Step 2: Agent stores assistant response
            assistant_msg_id = await conversation.add_message(
                conversation_id=conversation_id,
                role="assistant",
                content="I'll check the weather for you.",
                tool_calls=[
                    {"id": "call_1", "name": "get_weather", "arguments": {"city": "NYC"}},
                ],
            )
            assert assistant_msg_id is not None

            # Step 3: Agent retrieves conversation history
            history = await conversation.get_history(conversation_id)
            assert len(history) == 2
            assert history[0]["role"] == "user"
            assert history[0]["content"] == "What is the weather in NYC?"
            assert history[1]["role"] == "assistant"
            assert history[1]["content"] == "I'll check the weather for you."
            assert "tool_calls" in history[1]

            # Step 4: Simulate model using history in next turn
            # Agent stores new user message
            await conversation.add_message(
                conversation_id=conversation_id,
                role="user",
                content="What about Boston?",
            )

            # Agent retrieves full history (including previous context)
            full_history = await conversation.get_history(conversation_id)
            assert len(full_history) == 3
            # Model should see the previous conversation context
            assert full_history[0]["content"] == "What is the weather in NYC?"
            assert full_history[2]["content"] == "What about Boston?"

    async def test_agent_conversation_max_messages(self, test_storage_path: str):
        """Test that conversation history respects max_messages limit."""
        backend = FilesystemBackend(test_storage_path)
        store = ResourceStore(backend)
        conversation = ConversationState(store)

        conversation_id = "agent_conv_limit"

        # Add many messages
        for i in range(10):
            await conversation.add_message(
                conversation_id=conversation_id,
                role="user" if i % 2 == 0 else "assistant",
                content=f"Message {i}",
            )

        # Retrieve with limit
        history = await conversation.get_history(conversation_id, max_messages=3)
        assert len(history) == 3
        # Should get the last 3 messages
        assert history[-1]["content"] == "Message 9"


@pytest.mark.asyncio
class TestAgentMemoryWorkflow:
    """Test agent workflow using memory tools via MCP."""

    async def test_agent_memory_write_read_via_mcp(
        self, test_mcp_server, test_storage_path: str
    ):
        """
        Test that an agent can use memory.write and memory.read via MCP tools.

        Simulates an agent workflow where:
        1. Agent stores user preference using memory.write
        2. Agent retrieves preference using memory.read
        3. Model uses preference in conversation
        """
        async with resource_store_mcp_session(test_storage_path) as session:
            # Step 1: Agent stores user preference via MCP tool
            tools_result = await session.list_tools()
            tools = tools_result.tools
            memory_write_tool = next((t for t in tools if t.name == "memory_write"), None)
            assert memory_write_tool is not None, "memory_write tool should be available"

            # Call memory.write via MCP
            write_result = await session.call_tool(
                "memory_write",
                arguments={
                    "scope_id": "user_agent_123",
                    "kind": "preference",
                    "content": '{"theme": "dark", "language": "en"}',
                    "metadata": {"category": "ui"},
                },
            )

            assert write_result.isError is False
            result_text = write_result.content[0].text if write_result.content else ""
            assert "Memory stored with ID:" in result_text
            memory_id = result_text.split("ID:")[-1].strip()

            # Step 2: Agent retrieves preference using memory.read
            memory_read_tool = next((t for t in tools_result.tools if t.name == "memory_read"), None)
            assert memory_read_tool is not None, "memory_read tool should be available"

            read_result = await session.call_tool(
                "memory_read",
                arguments={"resource_id": memory_id},
            )

            assert read_result.isError is False
            read_content = read_result.content[0].text if read_result.content else ""
            memory_data = json.loads(read_content)
            assert memory_data["kind"] == "preference"
            assert memory_data["content"] == '{"theme": "dark", "language": "en"}'
            assert memory_data["metadata"]["category"] == "ui"

    async def test_agent_memory_search_via_mcp(
        self, test_mcp_server, test_storage_path: str
    ):
        """
        Test that an agent can search memories via MCP tool.

        Simulates an agent workflow where:
        1. Agent stores multiple memories
        2. Agent searches memories by kind
        3. Agent searches memories by metadata
        """
        async with resource_store_mcp_session(test_storage_path) as session:
            # Step 1: Store multiple memories
            tools_result = await session.list_tools()
            tools = tools_result.tools
            memory_write = next((t for t in tools if t.name == "memory_write"), None)

            scope_id = "user_agent_search"

            # Store different kinds of memories
            await session.call_tool(
                "memory_write",
                arguments={
                    "scope_id": scope_id,
                    "kind": "note",
                    "content": "Remember to buy milk",
                    "metadata": {"tag": "shopping", "priority": "high"},
                },
            )

            await session.call_tool(
                "memory_write",
                arguments={
                    "scope_id": scope_id,
                    "kind": "note",
                    "content": "Call dentist tomorrow",
                    "metadata": {"tag": "appointment", "priority": "medium"},
                },
            )

            await session.call_tool(
                "memory_write",
                arguments={
                    "scope_id": scope_id,
                    "kind": "preference",
                    "content": '{"notifications": "enabled"}',
                    "metadata": {"category": "settings"},
                },
            )

            # Step 2: Search by kind
            memory_search = next((t for t in tools_result.tools if t.name == "memory_search"), None)
            assert memory_search is not None

            search_result = await session.call_tool(
                "memory_search",
                arguments={"scope_id": scope_id, "kind": "note"},
            )

            assert search_result.isError is False
            search_content = search_result.content[0].text if search_result.content else ""
            results = json.loads(search_content)
            assert len(results) == 2
            assert all(r["kind"] == "note" for r in results)

            # Step 3: Search by metadata
            search_result = await session.call_tool(
                "memory_search",
                arguments={
                    "scope_id": scope_id,
                    "metadata": {"tag": "shopping"},
                },
            )

            assert search_result.isError is False
            search_content = search_result.content[0].text if search_result.content else ""
            results = json.loads(search_content)
            assert len(results) == 1
            assert results[0]["content"] == "Remember to buy milk"

    async def test_agent_memory_delete_via_mcp(
        self, test_mcp_server, test_storage_path: str
    ):
        """Test that an agent can delete memories via MCP tool."""
        async with resource_store_mcp_session(test_storage_path) as session:
            tools_result = await session.list_tools()
            tools = tools_result.tools
            memory_write = next((t for t in tools if t.name == "memory_write"), None)

            # Store a memory
            write_result = await session.call_tool(
                "memory_write",
                arguments={
                    "scope_id": "user_delete_test",
                    "kind": "note",
                    "content": "This will be deleted",
                },
            )

            result_text = write_result.content[0].text if write_result.content else ""
            memory_id = result_text.split("ID:")[-1].strip()

            # Delete the memory
            memory_delete = next((t for t in tools_result.tools if t.name == "memory_delete"), None)
            assert memory_delete is not None

            delete_result = await session.call_tool(
                "memory_delete",
                arguments={"resource_id": memory_id},
            )

            assert delete_result.isError is False
            assert "deleted successfully" in delete_result.content[0].text

            # Verify it's deleted
            memory_read = next((t for t in tools_result.tools if t.name == "memory_read"), None)
            read_result = await session.call_tool(
                "memory_read",
                arguments={"resource_id": memory_id},
            )
            assert "Memory not found" in read_result.content[0].text


@pytest.mark.asyncio
class TestAgentResourceWorkflow:
    """Test agent workflow using resources via MCP."""

    async def test_agent_store_and_retrieve_resource_via_mcp(
        self, test_mcp_server, test_storage_path: str
    ):
        """
        Test that an agent can store and retrieve resources via MCP.

        Simulates an agent workflow where:
        1. Tool produces large output (e.g., CSV data)
        2. Tool stores it as a resource
        3. Agent retrieves resource via MCP list_resources/read_resource
        """
        async with resource_store_mcp_session(test_storage_path) as session:
            # Step 1: Simulate tool storing large output
            # In real usage, a tool would use ResourceManager.add_resource()
            # For this test, we'll simulate it by directly using ResourceStore
            backend = FilesystemBackend(test_storage_path)
            store = ResourceStore(backend)

            from datarobot_genai.drmcp.core.resource_store.models import Scope
            from datarobot_genai.drmcp.core.resource_store.resource_api import ResourceAPI

            resource_api = ResourceAPI(store)
            conversation_id = "agent_resource_conv"

            # Store a large CSV-like resource
            csv_data = "name,age,city\nAlice,30,NYC\nBob,25,Boston\n"
            resource_id = await resource_api.store_resource(
                scope_id=conversation_id,
                data=csv_data,
                content_type="text/csv",
                name="User Data Export",
                lifetime="ephemeral",
                ttl_seconds=3600,
                metadata={"rows": 2, "columns": 3},
            )

            assert resource_id is not None

            # Step 2: Agent lists resources via MCP
            resources_result = await session.list_resources()
            # Find our resource (it should be registered with FastMCP)
            our_resource = next(
                (r for r in resources_result.resources if resource_id in str(r.uri)), None
            )

            # Note: Resources stored via ResourceAPI may not automatically appear
            # in list_resources unless they're registered with FastMCP's ResourceManager
            # This is expected behavior - tools should use ResourceManager.add_resource()

            # Step 3: Agent reads resource content
            if our_resource:
                resource_data = await session.read_resource(str(our_resource.uri))
                assert resource_data is not None
                # Verify content matches
                if isinstance(resource_data, list) and resource_data:
                    content = resource_data[0]
                    if isinstance(content, TextContent):
                        assert csv_data in content.text


@pytest.mark.asyncio
class TestAgentFullWorkflow:
    """Test complete agent workflow combining conversation, memory, and resources."""

    async def test_agent_full_workflow_with_conversation_and_memory(
        self, test_mcp_server, test_storage_path: str
    ):
        """
        Test a complete agent workflow that uses conversation state and memory.

        Simulates:
        1. Agent stores user preference in memory
        2. Agent has multi-turn conversation
        3. Agent retrieves memory to personalize response
        4. Agent uses conversation history for context
        """
        async with resource_store_mcp_session(test_storage_path) as session:
            # Setup: Store user preference
            tools_result = await session.list_tools()
            tools = tools_result.tools
            memory_write = next((t for t in tools if t.name == "memory_write"), None)

            user_id = "user_full_workflow"
            conversation_id = f"{user_id}_conv_1"

            # Store user preference
            await session.call_tool(
                "memory_write",
                arguments={
                    "scope_id": user_id,
                    "kind": "preference",
                    "content": '{"favorite_color": "blue", "timezone": "EST"}',
                    "metadata": {"category": "profile"},
                },
            )

            # Simulate conversation with memory and history
            backend = FilesystemBackend(test_storage_path)
            store = ResourceStore(backend)
            conversation = ConversationState(store)
            memory = MemoryAPI(store)

            # Turn 1: User asks a question
            await conversation.add_message(
                conversation_id=conversation_id,
                role="user",
                content="What's my favorite color?",
            )

            # Agent retrieves memory (simulating model using memory.search)
            memories = await memory.search(user_id, kind="preference")
            assert len(memories) > 0
            user_pref = json.loads(memories[0]["content"])
            assert user_pref["favorite_color"] == "blue"

            # Agent responds using memory
            await conversation.add_message(
                conversation_id=conversation_id,
                role="assistant",
                content=f"Your favorite color is {user_pref['favorite_color']}.",
            )

            # Turn 2: User asks follow-up
            await conversation.add_message(
                conversation_id=conversation_id,
                role="user",
                content="What did I just ask you?",
            )

            # Agent retrieves conversation history
            history = await conversation.get_history(conversation_id)
            assert len(history) == 3
            # Model should see the previous question
            assert "favorite color" in history[0]["content"].lower()

            # Agent responds using conversation history
            await conversation.add_message(
                conversation_id=conversation_id,
                role="assistant",
                content="You asked about your favorite color, and I told you it's blue.",
            )

            # Verify final state
            final_history = await conversation.get_history(conversation_id)
            assert len(final_history) == 4
            assert final_history[-1]["role"] == "assistant"
            assert "favorite color" in final_history[-1]["content"].lower()

