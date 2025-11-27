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

"""Tests for ConversationState."""

import json

import pytest

from datarobot_genai.drmcp.core.resource_store.conversation import ConversationState
from datarobot_genai.drmcp.core.resource_store.store import ResourceStore


@pytest.mark.asyncio
class TestConversationState:
    """Tests for ConversationState."""

    async def test_add_message(self, store: ResourceStore) -> None:
        """Test adding a message."""
        conversation = ConversationState(store)
        message_id = await conversation.add_message(
            conversation_id="conv_123",
            role="user",
            content="Hello, world!",
        )
        assert message_id is not None

        # Verify stored
        result = await store.get(message_id)
        assert result is not None
        resource, data = result
        assert resource.kind == "message"
        assert resource.scope.id == "conv_123"
        assert resource.scope.type == "conversation"
        assert resource.lifetime == "ephemeral"

        message_data = json.loads(data) if isinstance(data, str) else data
        assert message_data["role"] == "user"
        assert message_data["content"] == "Hello, world!"

    async def test_add_message_with_tool_calls(self, store: ResourceStore) -> None:
        """Test adding a message with tool calls."""
        conversation = ConversationState(store)
        tool_calls = [
            {"id": "call_1", "name": "get_weather", "arguments": {"city": "NYC"}},
            {"id": "call_2", "name": "get_time", "arguments": {}},
        ]

        message_id = await conversation.add_message(
            conversation_id="conv_123",
            role="assistant",
            content="I'll check the weather for you.",
            tool_calls=tool_calls,
        )

        result = await store.get(message_id)
        assert result is not None
        _, data = result
        message_data = json.loads(data) if isinstance(data, str) else data
        assert "tool_calls" in message_data
        assert len(message_data["tool_calls"]) == 2

    async def test_get_history(self, store: ResourceStore) -> None:
        """Test getting conversation history."""
        conversation = ConversationState(store)

        await conversation.add_message("conv_123", "user", "Hello")
        await conversation.add_message("conv_123", "assistant", "Hi there!")
        await conversation.add_message("conv_123", "user", "How are you?")

        history = await conversation.get_history("conv_123")
        assert len(history) == 3
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello"
        assert history[1]["role"] == "assistant"
        assert history[1]["content"] == "Hi there!"
        assert history[2]["role"] == "user"
        assert history[2]["content"] == "How are you?"

    async def test_get_history_max_messages(self, store: ResourceStore) -> None:
        """Test getting history with max_messages limit."""
        conversation = ConversationState(store)

        await conversation.add_message("conv_123", "user", "Message 1")
        await conversation.add_message("conv_123", "assistant", "Response 1")
        await conversation.add_message("conv_123", "user", "Message 2")
        await conversation.add_message("conv_123", "assistant", "Response 2")

        history = await conversation.get_history("conv_123", max_messages=2)
        assert len(history) == 2
        # Last 2 messages should be Message 2 and Response 2
        assert history[-2]["content"] == "Message 2"
        assert history[-1]["content"] == "Response 2"

    async def test_get_history_empty(self, store: ResourceStore) -> None:
        """Test getting history for empty conversation."""
        conversation = ConversationState(store)
        history = await conversation.get_history("empty_conv")
        assert len(history) == 0

    async def test_get_history_filters_messages(self, store: ResourceStore) -> None:
        """Test that get_history only returns messages, not other resource types."""
        conversation = ConversationState(store)

        await conversation.add_message("conv_123", "user", "Message")
        # Add a non-message resource
        from datarobot_genai.drmcp.core.resource_store.models import Scope

        await store.put(
            scope=Scope(type="conversation", id="conv_123"),
            kind="tool-call",
            data='{"tool": "test"}',
            lifetime="ephemeral",
            contentType="application/json",
        )

        history = await conversation.get_history("conv_123")
        assert len(history) == 1
        # History should only contain messages, not tool-calls
        assert history[0]["role"] in ["user", "assistant"]

    async def test_clear_history(self, store: ResourceStore) -> None:
        """Test clearing conversation history."""
        conversation = ConversationState(store)

        await conversation.add_message("conv_123", "user", "Message 1")
        await conversation.add_message("conv_123", "assistant", "Response 1")

        history = await conversation.get_history("conv_123")
        assert len(history) == 2

        await conversation.clear_history("conv_123")

        history = await conversation.get_history("conv_123")
        assert len(history) == 0

    async def test_clear_history_empty(self, store: ResourceStore) -> None:
        """Test clearing empty history."""
        conversation = ConversationState(store)
        await conversation.clear_history("empty_conv")  # Should not raise

