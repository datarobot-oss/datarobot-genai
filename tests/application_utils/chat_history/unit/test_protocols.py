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

"""Loose-coupling tests for the repository :class:`~typing.Protocol` types.

Proves the ``*Like`` protocols are structural: a self-contained, dependency-free
in-memory repository satisfies them (so a SQL / test backend is a drop-in), the
concrete ORM-backed repositories satisfy them too, and a class missing a method
does not.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from collections.abc import Sequence
from contextlib import asynccontextmanager
from uuid import UUID
from uuid import uuid4

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
from datarobot_genai.application_utils.chat_history.repositories import ChatRepository
from datarobot_genai.application_utils.chat_history.repositories import ChatRepositoryLike
from datarobot_genai.application_utils.chat_history.repositories import MessageRepository
from datarobot_genai.application_utils.chat_history.repositories import MessageRepositoryLike


class InMemoryChatRepository:
    """A dependency-free chat store — no ORM, no HTTP — that satisfies the Protocol."""

    def __init__(self) -> None:
        self._chats: dict[UUID, Chat] = {}

    async def create_chat(self, chat_data: ChatCreate) -> Chat:
        assert chat_data.thread_id is not None
        existing = await self.get_chat_by_thread_id(
            chat_data.user_uuid or uuid4(), chat_data.thread_id
        )
        if existing is not None:
            return existing
        chat = Chat(
            thread_id=chat_data.thread_id,
            dedup_key=f"dedup-{chat_data.thread_id}",
            name=chat_data.name,
            user_uuid=chat_data.user_uuid,
        )
        self._chats[chat.chat_uuid] = chat
        return chat

    async def get_chat_by_thread_id(self, user_uuid: UUID, thread_id: str) -> Chat | None:
        for chat in self._chats.values():
            if chat.thread_id == thread_id and chat.user_uuid == user_uuid:
                return chat
        return None

    async def get_all_chats(self, user_uuid: UUID | None) -> Sequence[Chat]:
        if user_uuid is None:
            return list(self._chats.values())
        return [c for c in self._chats.values() if c.user_uuid == user_uuid]

    async def update_chat_name(self, chat_uuid: UUID, name: str) -> Chat | None:
        chat = self._chats.get(chat_uuid)
        if chat is None:
            return None
        chat.name = name
        return chat

    async def delete_chat(self, chat_uuid: UUID) -> Chat | None:
        return self._chats.pop(chat_uuid, None)


class InMemoryMessageRepository:
    """A dependency-free message store nesting tool calls / reasonings on the message."""

    def __init__(self) -> None:
        self._messages: dict[UUID, Message] = {}

    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[None, None]:
        yield

    async def create_message(self, message_data: MessageCreate) -> Message:
        message = Message(
            content=message_data.content,
            emitter_type="user" if message_data.role == "user" else "agent",
            chat_id=message_data.chat_id,
            agui_id=message_data.agui_id,
            role=message_data.role,
        )
        self._messages[message.message_uuid] = message
        return message

    async def update_message(self, message_uuid: UUID, update: MessageUpdate) -> Message | None:
        message = self._messages.get(message_uuid)
        if message is None:
            return None
        for name, value in update.model_dump(exclude_unset=True).items():
            if value is not None:
                setattr(message, name, value)
        return message

    async def create_message_tool_call(self, data: MessageToolCallCreate) -> ToolCall:
        message = self._messages[data.message_uuid]
        tool_call = ToolCall(**data.model_dump(exclude={"message_uuid"}), uuid=uuid4())
        message.tool_calls = [*message.tool_calls, tool_call]
        return tool_call

    async def update_message_tool_call(
        self, uuid: UUID, update: MessageToolCallUpdate
    ) -> ToolCall | None:
        for message in self._messages.values():
            for tool_call in message.tool_calls:
                if tool_call.uuid == uuid:
                    for name, value in update.model_dump(exclude_unset=True).items():
                        if value is not None:
                            setattr(tool_call, name, value)
                    return tool_call
        return None

    async def create_message_reasoning(self, data: MessageReasoningCreate) -> Reasoning:
        message = self._messages[data.message_uuid]
        reasoning = Reasoning(**data.model_dump(exclude={"message_uuid"}), uuid=uuid4())
        message.reasonings = [*message.reasonings, reasoning]
        return reasoning

    async def update_message_reasoning(
        self, uuid: UUID, update: MessageReasoningUpdate
    ) -> Reasoning | None:
        for message in self._messages.values():
            for reasoning in message.reasonings:
                if reasoning.uuid == uuid:
                    for name, value in update.model_dump(exclude_unset=True).items():
                        if value is not None:
                            setattr(reasoning, name, value)
                    return reasoning
        return None

    async def get_message(self, message_uuid: UUID) -> Message | None:
        return self._messages.get(message_uuid)

    async def get_message_by_agui_id(self, chat_id: UUID, agui_id: str) -> Message | None:
        for message in self._messages.values():
            if message.chat_id == chat_id and message.agui_id == agui_id:
                return message
        return None

    async def get_tool_call_by_agui_id(self, message_uuid: UUID, agui_id: str) -> ToolCall | None:
        message = self._messages.get(message_uuid)
        if message is None:
            return None
        for tool_call in message.tool_calls:
            if tool_call.agui_id == agui_id:
                return tool_call
        return None

    async def get_chat_messages(self, chat_id: UUID) -> Sequence[Message]:
        return [m for m in self._messages.values() if m.chat_id == chat_id]

    async def get_last_messages(self, chat_ids: list[UUID]) -> dict[UUID, Message]:
        result: dict[UUID, Message] = {}
        for chat_id in chat_ids:
            matches = [m for m in self._messages.values() if m.chat_id == chat_id]
            if matches:
                result[chat_id] = matches[-1]
        return result


class NotAChatRepository:
    """Missing ``delete_chat`` — must therefore fail the structural check."""

    async def create_chat(self, chat_data: ChatCreate) -> Chat: ...
    async def get_chat_by_thread_id(self, user_uuid: UUID, thread_id: str) -> Chat | None: ...
    async def get_all_chats(self, user_uuid: UUID | None) -> Sequence[Chat]: ...
    async def update_chat_name(self, chat_uuid: UUID, name: str) -> Chat | None: ...


# ── Structural conformance ────────────────────────────────────────────────────


def test_in_memory_repos_satisfy_the_protocols() -> None:
    """GIVEN a dependency-free fake WHEN checked THEN it satisfies the repository protocols."""
    assert isinstance(InMemoryChatRepository(), ChatRepositoryLike)
    assert isinstance(InMemoryMessageRepository(), MessageRepositoryLike)


def test_concrete_orm_repos_satisfy_the_protocols() -> None:
    """GIVEN the ORM-backed repos WHEN checked THEN they satisfy the same protocols."""
    # __new__ bypasses __init__ (no space needed) — structural checks read class methods.
    assert isinstance(ChatRepository.__new__(ChatRepository), ChatRepositoryLike)
    assert isinstance(MessageRepository.__new__(MessageRepository), MessageRepositoryLike)


def test_missing_method_fails_the_structural_check() -> None:
    """GIVEN a class missing delete_chat WHEN checked THEN it does not satisfy the protocol."""
    assert not isinstance(NotAChatRepository(), ChatRepositoryLike)


# ── The fake is a functional drop-in ───────────────────────────────────────────


async def test_in_memory_repository_round_trips_a_full_turn() -> None:
    """GIVEN the in-memory repos WHEN driving a full turn THEN nested state round-trips."""
    chat_repo: ChatRepositoryLike = InMemoryChatRepository()
    message_repo: MessageRepositoryLike = InMemoryMessageRepository()

    user = uuid4()
    chat = await chat_repo.create_chat(ChatCreate(name="C", thread_id="t", user_uuid=user))
    assert (await chat_repo.get_all_chats(user))[0].chat_uuid == chat.chat_uuid

    message = await message_repo.create_message(
        MessageCreate(role="assistant", content="hi", chat_id=chat.chat_uuid, agui_id="m1")
    )
    tool_call = await message_repo.create_message_tool_call(
        MessageToolCallCreate(message_uuid=message.message_uuid, name="search", agui_id="tc1")
    )
    await message_repo.create_message_reasoning(
        MessageReasoningCreate(message_uuid=message.message_uuid, content="plan")
    )

    async with message_repo.transaction():
        await message_repo.update_message_tool_call(
            tool_call.uuid, MessageToolCallUpdate(status="complete")
        )

    fetched = await message_repo.get_message(message.message_uuid)
    assert fetched is not None
    assert fetched.tool_calls[0].status == "complete"
    assert fetched.reasonings[0].content == "plan"
    assert (await message_repo.get_tool_call_by_agui_id(message.message_uuid, "tc1")) is not None
    assert (await message_repo.get_message_by_agui_id(chat.chat_uuid, "m1")) is not None
    last = await message_repo.get_last_messages([chat.chat_uuid])
    assert last[chat.chat_uuid].message_uuid == message.message_uuid
