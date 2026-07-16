# Copyright 2026 DataRobot, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dependency-free in-memory repositories used to drive the AG-UI storage tests.

They satisfy :class:`ChatRepositoryLike` / :class:`MessageRepositoryLike`
structurally (no ORM, no HTTP), so the AG-UI storage agent runs against them
unchanged — this is the loose-coupling extensibility point (requirement 6.4).
The message repository builds messages via an overridable :meth:`build_message`
so a subclass can persist extra fields declared on a ``Message`` subclass.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from collections.abc import Sequence
from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID
from uuid import uuid4

from datarobot_genai.application_utils.chat_history.constants import participant_id
from datarobot_genai.application_utils.chat_history.models import Chat
from datarobot_genai.application_utils.chat_history.models import ChatCreate
from datarobot_genai.application_utils.chat_history.models import Message
from datarobot_genai.application_utils.chat_history.models import MessageCreate
from datarobot_genai.application_utils.chat_history.models import MessageReasoningCreate
from datarobot_genai.application_utils.chat_history.models import MessageReasoningUpdate
from datarobot_genai.application_utils.chat_history.models import MessageToolCallCreate
from datarobot_genai.application_utils.chat_history.models import MessageToolCallUpdate
from datarobot_genai.application_utils.chat_history.models import MessageUpdate
from datarobot_genai.application_utils.chat_history.models import Reasoning
from datarobot_genai.application_utils.chat_history.models import ToolCall


class InMemoryChatRepository:
    """An in-process chat store that satisfies ``ChatRepositoryLike``."""

    def __init__(self) -> None:
        self.chats: dict[UUID, Chat] = {}

    async def create_chat(self, chat_data: ChatCreate) -> Chat:
        """Create (idempotent by ``(user, thread)``) or return the existing chat."""
        assert chat_data.thread_id is not None
        assert chat_data.user_uuid is not None
        existing = await self.get_chat_by_thread_id(chat_data.user_uuid, chat_data.thread_id)
        if existing is not None:
            return existing
        chat = Chat(
            thread_id=chat_data.thread_id,
            dedup_key=f"dedup-{chat_data.user_uuid}-{chat_data.thread_id}",
            name=chat_data.name,
            user_uuid=chat_data.user_uuid,
            participants=[participant_id(chat_data.user_uuid)],
        )
        self.chats[chat.chat_uuid] = chat
        return chat

    async def get_chat_by_thread_id(self, user_uuid: UUID, thread_id: str) -> Chat | None:
        """Return the chat for a ``(user, thread_id)`` pair, or ``None``."""
        for chat in self.chats.values():
            if chat.thread_id == thread_id and chat.user_uuid == user_uuid:
                return chat
        return None

    async def get_all_chats(self, user_uuid: UUID | None) -> Sequence[Chat]:
        """Return every chat, optionally scoped to a single user."""
        if user_uuid is None:
            return list(self.chats.values())
        return [c for c in self.chats.values() if c.user_uuid == user_uuid]

    async def update_chat_name(self, chat_uuid: UUID, name: str) -> Chat | None:
        """Rename a chat, returning it or ``None`` when unknown."""
        chat = self.chats.get(chat_uuid)
        if chat is None:
            return None
        chat.name = name
        return chat

    async def delete_chat(self, chat_uuid: UUID) -> Chat | None:
        """Delete a chat, returning it or ``None`` when unknown."""
        return self.chats.pop(chat_uuid, None)


class InMemoryMessageRepository:
    """An in-process message store (one message record per logical message)."""

    #: The concrete ``Message`` class this repository builds.
    message_cls: type[Message] = Message

    def __init__(self) -> None:
        self.messages: dict[UUID, Message] = {}
        self.order: list[UUID] = []

    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[None, None]:
        """No-op batching scope for interface parity."""
        yield

    def build_message(self, data: MessageCreate) -> Message:
        """Build a message record from a create DTO (overridable for extra fields)."""
        return self.message_cls(
            content=data.content,
            emitter_type="user" if data.role == "user" else "agent",
            chat_id=data.chat_id,
            agui_id=data.agui_id,
            role=data.role,
            name=data.name,
            step=data.step,
            in_progress=data.in_progress,
            status=data.status,
            error=data.error,
            timestamp=data.timestamp,
        )

    async def create_message(self, message_data: MessageCreate) -> Message:
        """Persist a new message and return it."""
        message = self.build_message(message_data)
        self.messages[message.message_uuid] = message
        self.order.append(message.message_uuid)
        return message

    async def update_message(self, message_uuid: UUID, update: MessageUpdate) -> Message | None:
        """Patch a message's own fields in place."""
        message = self.messages.get(message_uuid)
        if message is None:
            return None
        for name, value in update.model_dump(exclude_unset=True).items():
            if value is not None:
                setattr(message, name, value)
        return message

    async def create_message_tool_call(self, data: MessageToolCallCreate) -> ToolCall:
        """Append a tool call to its parent message's body."""
        message = self.messages[data.message_uuid]
        tool_call = ToolCall(**data.model_dump(exclude={"message_uuid"}), uuid=uuid4())
        message.tool_calls = [*message.tool_calls, tool_call]
        return tool_call

    async def update_message_tool_call(
        self, uuid: UUID, update: MessageToolCallUpdate
    ) -> ToolCall | None:
        """Patch a nested tool call in place."""
        for message in self.messages.values():
            for tool_call in message.tool_calls:
                if tool_call.uuid == uuid:
                    for name, value in update.model_dump(exclude_unset=True).items():
                        if value is not None:
                            setattr(tool_call, name, value)
                    return tool_call
        return None

    async def create_message_reasoning(self, data: MessageReasoningCreate) -> Reasoning:
        """Append a reasoning step to its parent message's body."""
        message = self.messages[data.message_uuid]
        reasoning = Reasoning(**data.model_dump(exclude={"message_uuid"}), uuid=uuid4())
        message.reasonings = [*message.reasonings, reasoning]
        return reasoning

    async def update_message_reasoning(
        self, uuid: UUID, update: MessageReasoningUpdate
    ) -> Reasoning | None:
        """Patch a nested reasoning step in place."""
        for message in self.messages.values():
            for reasoning in message.reasonings:
                if reasoning.uuid == uuid:
                    for name, value in update.model_dump(exclude_unset=True).items():
                        if value is not None:
                            setattr(reasoning, name, value)
                    return reasoning
        return None

    async def get_message(self, message_uuid: UUID) -> Message | None:
        """Return a message by its application UUID, or ``None``."""
        return self.messages.get(message_uuid)

    async def get_message_by_agui_id(self, chat_id: UUID, agui_id: str) -> Message | None:
        """Return a message by its AG-UI id within a chat, or ``None``."""
        for uuid in self.order:
            message = self.messages[uuid]
            if message.chat_id == chat_id and message.agui_id == agui_id:
                return message
        return None

    async def get_tool_call_by_agui_id(self, message_uuid: UUID, agui_id: str) -> ToolCall | None:
        """Return a tool call by its AG-UI id within a message, or ``None``."""
        message = self.messages.get(message_uuid)
        if message is None:
            return None
        for tool_call in message.tool_calls:
            if tool_call.agui_id == agui_id:
                return tool_call
        return None

    async def get_chat_messages(self, chat_id: UUID) -> Sequence[Message]:
        """Return every message in a chat, oldest first (insertion order)."""
        return [self.messages[u] for u in self.order if self.messages[u].chat_id == chat_id]

    async def get_last_messages(self, chat_ids: list[UUID]) -> dict[UUID, Message]:
        """Return the most recently created message for each of the given chats."""
        result: dict[UUID, Message] = {}
        for chat_id in chat_ids:
            matches = [self.messages[u] for u in self.order if self.messages[u].chat_id == chat_id]
            if matches:
                result[chat_id] = matches[-1]
        return result


class RecordingMessageRepository(InMemoryMessageRepository):
    """An ``InMemoryMessageRepository`` that records how often each write is called."""

    def __init__(self) -> None:
        super().__init__()
        self.update_message_calls = 0

    async def update_message(self, message_uuid: UUID, update: MessageUpdate) -> Message | None:
        """Count and delegate to the base ``update_message``."""
        self.update_message_calls += 1
        return await super().update_message(message_uuid, update)


def dump_extra(data: Any, *names: str) -> dict[str, Any]:
    """Return the subset of *names* present as attributes on *data*."""
    return {name: getattr(data, name) for name in names if hasattr(data, name)}
