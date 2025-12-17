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
LlamaIndex Memory Adapter for ResourceStore.

This module provides adapters to use ResourceStore as a backend for
LlamaIndex's memory and chat storage systems.

LlamaIndex uses:
- ChatStore: For storing chat message history
- MemoryBlock: For long-term memory processing and retrieval

Usage:
    ```python
    from llama_index.core.memory import Memory
    from datarobot_genai.drmcp.core.resource_store import ResourceStore
    from datarobot_genai.drmcp.core.resource_store.backends.filesystem import FilesystemBackend
    from datarobot_genai.drmcp.core.resource_store.adapters.llamaindex_adapter import (
        ResourceStoreChatStore,
        ResourceStoreMemoryBlock,
    )

    # Create ResourceStore
    backend = FilesystemBackend("/path/to/storage")
    store = ResourceStore(backend)

    # Create LlamaIndex-compatible chat store
    chat_store = ResourceStoreChatStore(store)

    # Or create a memory block for long-term memory
    memory_block = ResourceStoreMemoryBlock(store, name="my_memory")

    # Use with LlamaIndex Memory
    memory = Memory.from_defaults(
        session_id="my_session",
        chat_store=chat_store,
        memory_blocks=[memory_block],
    )
    ```

See: https://developers.llamaindex.ai/python/examples/memory/memory/
"""

import asyncio
import concurrent.futures
import json
import logging
from typing import Any

from ..memory import MemoryAPI
from ..models import Scope
from ..store import ResourceStore

logger = logging.getLogger(__name__)


def _run_async(coro):
    """Run an async coroutine synchronously."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)


# Type for ChatMessage - we define our own to avoid requiring llama_index as a dependency
class ChatMessageDict:
    """Simple chat message representation compatible with LlamaIndex."""

    def __init__(self, role: str, content: str, **kwargs: Any):
        self.role = role
        self.content = content
        self.additional_kwargs = kwargs

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "role": self.role,
            "content": self.content,
            **self.additional_kwargs,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChatMessageDict":
        """Create from dictionary."""
        role = data.pop("role", "user")
        content = data.pop("content", "")
        return cls(role=role, content=content, **data)


class ResourceStoreChatStore:
    """
    LlamaIndex ChatStore backed by ResourceStore.

    This class implements the BaseChatStore interface from LlamaIndex,
    using ResourceStore for persistent storage of chat messages.

    The ChatStore interface methods:
        - set_messages(key, messages): Set all messages for a key
        - get_messages(key): Get all messages for a key
        - add_message(key, message): Add a single message
        - delete_messages(key): Delete all messages for a key
        - delete_message(key, idx): Delete a specific message by index
        - delete_last_message(key): Delete the last message
        - get_keys(): Get all session keys

    Attributes
    ----------
        store: ResourceStore instance
    """

    def __init__(self, store: ResourceStore) -> None:
        """
        Initialize ResourceStoreChatStore.

        Args:
            store: ResourceStore instance for message storage
        """
        self.store = store

    def _get_scope(self, key: str) -> Scope:
        """Get scope for a session key."""
        return Scope(type="conversation", id=f"llamaindex:{key}")

    async def _aget_messages_internal(self, key: str) -> list[dict[str, Any]]:
        """Get all messages for a key (internal async)."""
        scope = self._get_scope(key)
        resources = await self.store.query(scope=scope, kind="chat_message")

        messages = []
        for resource in sorted(resources, key=lambda r: r.metadata.get("index", 0)):
            result = await self.store.get(resource.id)
            if result:
                _, data = result
                content = data if isinstance(data, str) else data.decode("utf-8")
                try:
                    msg_data = json.loads(content)
                    messages.append(msg_data)
                except json.JSONDecodeError:
                    messages.append({"role": "user", "content": content})

        return messages

    async def _aset_messages_internal(
        self, key: str, messages: list[dict[str, Any]]
    ) -> None:
        """Set all messages for a key (internal async)."""
        # First, delete existing messages
        await self._adelete_messages_internal(key)

        # Then add new messages
        scope = self._get_scope(key)
        for idx, msg in enumerate(messages):
            await self.store.put(
                scope=scope,
                kind="chat_message",
                data=json.dumps(msg),
                lifetime="persistent",
                contentType="application/json",
                metadata={"index": idx},
                resource_id=f"chat:{key}:{idx}",
            )

    async def _aadd_message_internal(self, key: str, message: dict[str, Any]) -> None:
        """Add a message (internal async)."""
        scope = self._get_scope(key)

        # Get current message count
        resources = await self.store.query(scope=scope, kind="chat_message")
        idx = len(resources)

        await self.store.put(
            scope=scope,
            kind="chat_message",
            data=json.dumps(message),
            lifetime="persistent",
            contentType="application/json",
            metadata={"index": idx},
            resource_id=f"chat:{key}:{idx}",
        )

    async def _adelete_messages_internal(self, key: str) -> list[dict[str, Any]]:
        """Delete all messages for a key (internal async)."""
        messages = await self._aget_messages_internal(key)

        scope = self._get_scope(key)
        resources = await self.store.query(scope=scope, kind="chat_message")
        for resource in resources:
            await self.store.delete(resource.id)

        return messages

    async def _adelete_message_internal(
        self, key: str, idx: int
    ) -> dict[str, Any] | None:
        """Delete a specific message by index (internal async)."""
        messages = await self._aget_messages_internal(key)

        if 0 <= idx < len(messages):
            deleted = messages.pop(idx)
            await self._aset_messages_internal(key, messages)
            return deleted

        return None

    async def _aget_keys_internal(self) -> list[str]:
        """Get all session keys (internal async)."""
        # Query all conversation scopes with llamaindex prefix
        # This is a simplified implementation - in production you might
        # want a more efficient way to track keys
        all_resources = await self.store.query(kind="chat_message")
        keys = set()
        for resource in all_resources:
            if resource.scope.id.startswith("llamaindex:"):
                key = resource.scope.id.replace("llamaindex:", "", 1)
                keys.add(key)
        return list(keys)

    # Sync interface (required by BaseChatStore)

    def set_messages(self, key: str, messages: list[Any]) -> None:
        """
        Set all messages for a key.

        Args:
            key: Session/conversation key
            messages: List of ChatMessage objects
        """
        msg_dicts = []
        for msg in messages:
            if hasattr(msg, "to_dict"):
                msg_dicts.append(msg.to_dict())
            elif hasattr(msg, "dict"):
                msg_dicts.append(msg.dict())
            elif isinstance(msg, dict):
                msg_dicts.append(msg)
            else:
                msg_dicts.append({"role": getattr(msg, "role", "user"), "content": str(msg)})

        _run_async(self._aset_messages_internal(key, msg_dicts))

    def get_messages(self, key: str) -> list[Any]:
        """
        Get all messages for a key.

        Args:
            key: Session/conversation key

        Returns
        -------
            List of message dictionaries
        """
        return _run_async(self._aget_messages_internal(key))

    def add_message(self, key: str, message: Any) -> None:
        """
        Add a message for a key.

        Args:
            key: Session/conversation key
            message: ChatMessage object to add
        """
        if hasattr(message, "to_dict"):
            msg_dict = message.to_dict()
        elif hasattr(message, "dict"):
            msg_dict = message.dict()
        elif isinstance(message, dict):
            msg_dict = message
        else:
            msg_dict = {"role": getattr(message, "role", "user"), "content": str(message)}

        _run_async(self._aadd_message_internal(key, msg_dict))

    def delete_messages(self, key: str) -> list[Any] | None:
        """
        Delete all messages for a key.

        Args:
            key: Session/conversation key

        Returns
        -------
            List of deleted messages, or None if key didn't exist
        """
        messages = _run_async(self._adelete_messages_internal(key))
        return messages if messages else None

    def delete_message(self, key: str, idx: int) -> Any | None:
        """
        Delete a specific message by index.

        Args:
            key: Session/conversation key
            idx: Message index to delete

        Returns
        -------
            Deleted message, or None if not found
        """
        return _run_async(self._adelete_message_internal(key, idx))

    def delete_last_message(self, key: str) -> Any | None:
        """
        Delete the last message for a key.

        Args:
            key: Session/conversation key

        Returns
        -------
            Deleted message, or None if no messages
        """
        messages = _run_async(self._aget_messages_internal(key))
        if messages:
            return self.delete_message(key, len(messages) - 1)
        return None

    def get_keys(self) -> list[str]:
        """
        Get all session keys.

        Returns
        -------
            List of session keys
        """
        return _run_async(self._aget_keys_internal())

    # Async interface

    async def aset_messages(self, key: str, messages: list[Any]) -> None:
        """Async version of set_messages."""
        msg_dicts = []
        for msg in messages:
            if hasattr(msg, "to_dict"):
                msg_dicts.append(msg.to_dict())
            elif hasattr(msg, "dict"):
                msg_dicts.append(msg.dict())
            elif isinstance(msg, dict):
                msg_dicts.append(msg)
            else:
                msg_dicts.append({"role": getattr(msg, "role", "user"), "content": str(msg)})

        await self._aset_messages_internal(key, msg_dicts)

    async def aget_messages(self, key: str) -> list[Any]:
        """Async version of get_messages."""
        return await self._aget_messages_internal(key)

    async def aadd_message(self, key: str, message: Any) -> None:
        """Async version of add_message."""
        if hasattr(message, "to_dict"):
            msg_dict = message.to_dict()
        elif hasattr(message, "dict"):
            msg_dict = message.dict()
        elif isinstance(message, dict):
            msg_dict = message
        else:
            msg_dict = {"role": getattr(message, "role", "user"), "content": str(message)}

        await self._aadd_message_internal(key, msg_dict)

    async def adelete_messages(self, key: str) -> list[Any] | None:
        """Async version of delete_messages."""
        messages = await self._adelete_messages_internal(key)
        return messages if messages else None

    async def adelete_message(self, key: str, idx: int) -> Any | None:
        """Async version of delete_message."""
        return await self._adelete_message_internal(key, idx)

    async def adelete_last_message(self, key: str) -> Any | None:
        """Async version of delete_last_message."""
        messages = await self._aget_messages_internal(key)
        if messages:
            return await self._adelete_message_internal(key, len(messages) - 1)
        return None

    async def aget_keys(self) -> list[str]:
        """Async version of get_keys."""
        return await self._aget_keys_internal()


class ResourceStoreMemoryBlock:
    """
    LlamaIndex MemoryBlock backed by ResourceStore.

    This class implements a memory block that stores and retrieves
    information using ResourceStore. It can be used with LlamaIndex's
    Memory class for long-term memory persistence.

    Memory blocks in LlamaIndex:
        - Receive flushed messages from short-term memory
        - Process and store information
        - Provide retrieved content when memory is queried

    Attributes
    ----------
        store: ResourceStore instance
        name: Block name identifier
        priority: Block priority (0 = always kept, higher = can be disabled)
        memory_api: MemoryAPI for storage operations
    """

    def __init__(
        self,
        store: ResourceStore,
        name: str = "resource_store_memory",
        priority: int = 1,
        max_entries: int = 100,
    ) -> None:
        """
        Initialize ResourceStoreMemoryBlock.

        Args:
            store: ResourceStore instance
            name: Block name for identification
            priority: Block priority (0=highest, always kept in memory)
            max_entries: Maximum number of memory entries to store
        """
        self.store = store
        self.name = name
        self.priority = priority
        self.max_entries = max_entries
        self.memory_api = MemoryAPI(store)

    async def _aget(
        self,
        messages: list[Any] | None = None,
        **block_kwargs: Any,
    ) -> str:
        """
        Get memory content for insertion into prompts.

        This method retrieves stored memories and formats them for
        inclusion in the conversation context.

        Args:
            messages: Recent messages for context (used for retrieval)
            **block_kwargs: Additional block-specific arguments

        Returns
        -------
            Formatted string of memory content
        """
        # Get recent memories
        scope_id = block_kwargs.get("session_id", "default")
        memories = await self.memory_api.search(scope_id=scope_id, kind="memory_block")

        if not memories:
            return ""

        # Format memories for output
        memory_lines = []
        for mem in memories[-self.max_entries :]:
            content = mem.get("content", "")
            if content:
                memory_lines.append(f"- {content}")

        if not memory_lines:
            return ""

        return f"<{self.name}>\n" + "\n".join(memory_lines) + f"\n</{self.name}>"

    async def _aput(self, messages: list[Any]) -> None:
        """
        Store messages that were flushed from short-term memory.

        Args:
            messages: List of ChatMessage objects to process and store
        """
        for msg in messages:
            # Extract content from message
            if hasattr(msg, "content"):
                content = msg.content
            elif isinstance(msg, dict):
                content = msg.get("content", str(msg))
            else:
                content = str(msg)

            # Extract role
            if hasattr(msg, "role"):
                role = str(msg.role)
            elif isinstance(msg, dict):
                role = msg.get("role", "user")
            else:
                role = "user"

            # Store the message content
            await self.memory_api.write(
                scope_id="default",
                kind="memory_block",
                content=content,
                metadata={"role": role, "block_name": self.name},
            )

    async def atruncate(
        self,
        content: str,
        tokens_to_truncate: int,
    ) -> str | None:
        """
        Truncate memory content when over token limit.

        Args:
            content: Current memory content
            tokens_to_truncate: Number of tokens to remove

        Returns
        -------
            Truncated content, or None/empty string to disable block
        """
        # Simple truncation: return empty to disable this block
        # when token budget is exceeded
        if tokens_to_truncate > 0:
            return ""
        return content

    # Sync versions for compatibility

    def get(
        self,
        messages: list[Any] | None = None,
        **block_kwargs: Any,
    ) -> str:
        """Sync version of _aget."""
        return _run_async(self._aget(messages, **block_kwargs))

    def put(self, messages: list[Any]) -> None:
        """Sync version of _aput."""
        _run_async(self._aput(messages))

    def truncate(self, content: str, tokens_to_truncate: int) -> str | None:
        """Sync version of atruncate."""
        return _run_async(self.atruncate(content, tokens_to_truncate))


class ResourceStoreVectorMemoryBlock:
    """
    A vector-based memory block using ResourceStore.

    This memory block stores batches of chat messages and retrieves
    them based on semantic similarity (when used with embeddings).

    For full vector search, you would typically combine this with
    an embedding model. This implementation provides the storage
    layer that can be extended with vector search capabilities.

    Attributes
    ----------
        store: ResourceStore instance
        name: Block name identifier
        priority: Block priority
    """

    def __init__(
        self,
        store: ResourceStore,
        name: str = "vector_memory",
        priority: int = 2,
        top_k: int = 5,
    ) -> None:
        """
        Initialize ResourceStoreVectorMemoryBlock.

        Args:
            store: ResourceStore instance
            name: Block name
            priority: Block priority (higher = can be disabled first)
            top_k: Number of message batches to retrieve
        """
        self.store = store
        self.name = name
        self.priority = priority
        self.top_k = top_k
        self.memory_api = MemoryAPI(store)

    async def _aget(
        self,
        messages: list[Any] | None = None,
        **block_kwargs: Any,
    ) -> str:
        """
        Get relevant memories based on recent messages.

        Args:
            messages: Recent messages for context-based retrieval
            **block_kwargs: Additional arguments

        Returns
        -------
            Formatted string of relevant memories
        """
        scope_id = block_kwargs.get("session_id", "default")

        # Get stored message batches
        memories = await self.memory_api.search(
            scope_id=scope_id, kind="vector_memory_batch"
        )

        if not memories:
            return ""

        # Get the most recent batches (in production, use vector similarity)
        recent_memories = memories[-self.top_k :]

        # Format output
        output_lines = []
        for mem in recent_memories:
            content = mem.get("content", "")
            if content:
                try:
                    batch = json.loads(content)
                    for msg in batch:
                        role = msg.get("role", "user")
                        text = msg.get("content", "")
                        output_lines.append(f"<message role='{role}'>{text}</message>")
                except json.JSONDecodeError:
                    output_lines.append(content)

        if not output_lines:
            return ""

        return f"<{self.name}>\n" + "\n".join(output_lines) + f"\n</{self.name}>"

    async def _aput(self, messages: list[Any]) -> None:
        """
        Store a batch of messages.

        Args:
            messages: List of messages to store as a batch
        """
        # Convert messages to serializable format
        batch = []
        for msg in messages:
            if hasattr(msg, "content") and hasattr(msg, "role"):
                batch.append({"role": str(msg.role), "content": msg.content})
            elif isinstance(msg, dict):
                batch.append(msg)
            else:
                batch.append({"role": "user", "content": str(msg)})

        if batch:
            await self.memory_api.write(
                scope_id="default",
                kind="vector_memory_batch",
                content=json.dumps(batch),
                metadata={"batch_size": len(batch), "block_name": self.name},
            )

    async def atruncate(
        self,
        content: str,
        tokens_to_truncate: int,
    ) -> str | None:
        """Truncate content when over token limit."""
        if tokens_to_truncate > 0:
            return ""
        return content

    # Sync versions

    def get(self, messages: list[Any] | None = None, **block_kwargs: Any) -> str:
        """Sync version of _aget."""
        return _run_async(self._aget(messages, **block_kwargs))

    def put(self, messages: list[Any]) -> None:
        """Sync version of _aput."""
        _run_async(self._aput(messages))

    def truncate(self, content: str, tokens_to_truncate: int) -> str | None:
        """Sync version of atruncate."""
        return _run_async(self.atruncate(content, tokens_to_truncate))
