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

"""Chat / message repositories over the Memory Service persistence ORM.

Storage layout
--------------
* Each :class:`~datarobot_genai.application_utils.chat_history.models.Chat` is one
  Memory Service session (range-keyed on ``thread_id``, deduplicated on
  ``(user, thread)``, participant-scoped to the owning user).
* Each logical message is **one** event under that session.  A message's tool
  calls and reasoning steps are **not** separate events: they live as typed
  nested models in the same event body and are re-serialised on every update.

Loose coupling
--------------
The AG-UI storage layer (Phase 3) depends only on the :class:`ChatRepositoryLike`
and :class:`MessageRepositoryLike` :class:`~typing.Protocol` types, never on the
concrete classes below.  Any backend (a SQL store, an in-memory fake) that
implements the same method set is a drop-in replacement.

Content placeholder boundary
-----------------------------
The event's own ``content`` is a base ORM field and is not routed through the
nested-model serializer, so this layer encodes empty content to the zero-width
placeholder before ``post``/``patch`` and decodes it on read.  Nested
``arguments`` / ``content`` on :class:`ToolCall` / :class:`Reasoning` carry the
codec on the model itself and round-trip through the ORM automatically.

Concurrency
-----------
Event patches are guarded by the event ``createdAt`` token and session patches by
the ``If-Match`` version; both can raise
:class:`~datarobot_genai.application_utils.persistence.DRMemoryVersionConflictError`
under concurrent writers.  The repositories re-read and retry a bounded number of
times before giving up.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from collections.abc import Callable
from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager
from contextlib import asynccontextmanager
from typing import Protocol
from typing import cast
from typing import runtime_checkable
from uuid import UUID
from uuid import uuid4

from datarobot_genai.application_utils.chat_history.constants import MEMORY_CHAT_MESSAGE_EVENT_TYPE
from datarobot_genai.application_utils.chat_history.constants import PAYLOAD_VERSION
from datarobot_genai.application_utils.chat_history.constants import app_str
from datarobot_genai.application_utils.chat_history.constants import chat_deduplication_key
from datarobot_genai.application_utils.chat_history.constants import emitter_for_role
from datarobot_genai.application_utils.chat_history.constants import participant_id
from datarobot_genai.application_utils.chat_history.constants import wire_non_empty_str
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
from datarobot_genai.application_utils.persistence import DRMemorySpace
from datarobot_genai.application_utils.persistence import DRMemoryVersionConflictError

logger = logging.getLogger(__name__)

#: Bounded read-modify-write retries on an optimistic-concurrency conflict.
EVENT_PATCH_RETRIES: int = 3
#: Bounded retries for a session (chat metadata) patch conflict.
SESSION_PATCH_RETRIES: int = 3
#: Page size for event pagination (the service caps a single page at 100).
_EVENT_PAGE_SIZE: int = 100
#: Tail window scanned for the last message of a chat.
_LAST_MESSAGE_WINDOW: int = 100


# ── Protocols (structural; a SQL backend satisfies them without importing us) ──


@runtime_checkable
class ChatRepositoryLike(Protocol):
    """Structural interface for chat persistence.

    Any object exposing this method set (a SQL store, an in-memory fake) is an
    accepted chat repository; consumers depend on this ``Protocol`` rather than
    on :class:`ChatRepository`.
    """

    async def create_chat(self, chat_data: ChatCreate) -> Chat:
        """Create a chat (idempotent by ``(user, thread)``) or return the existing one."""
        ...

    async def get_chat_by_thread_id(self, user_uuid: UUID, thread_id: str) -> Chat | None:
        """Return the chat for a ``(user, thread_id)`` pair, or ``None``."""
        ...

    async def get_all_chats(self, user_uuid: UUID | None) -> Sequence[Chat]:
        """Return every chat, optionally scoped to a single user."""
        ...

    async def update_chat_name(self, chat_uuid: UUID, name: str) -> Chat | None:
        """Rename a chat, returning the updated chat, or ``None`` when it is unknown."""
        ...

    async def delete_chat(self, chat_uuid: UUID) -> Chat | None:
        """Delete a chat, returning the deleted chat, or ``None`` when it is unknown."""
        ...


@runtime_checkable
class MessageRepositoryLike(Protocol):
    """Structural interface for message persistence (one event per message)."""

    def transaction(self) -> AbstractAsyncContextManager[None]:
        """Return an async context manager scoping a batch of writes."""
        ...

    async def create_message(self, message_data: MessageCreate) -> Message:
        """Persist a new message as one event and return it."""
        ...

    async def update_message(self, message_uuid: UUID, update: MessageUpdate) -> Message | None:
        """Patch a message in place, or return ``None`` when it is unknown."""
        ...

    async def create_message_tool_call(self, data: MessageToolCallCreate) -> ToolCall:
        """Append a tool call to its parent message's body."""
        ...

    async def update_message_tool_call(
        self, uuid: UUID, update: MessageToolCallUpdate
    ) -> ToolCall | None:
        """Patch a nested tool call, or return ``None`` when it is unknown."""
        ...

    async def create_message_reasoning(self, data: MessageReasoningCreate) -> Reasoning:
        """Append a reasoning step to its parent message's body."""
        ...

    async def update_message_reasoning(
        self, uuid: UUID, update: MessageReasoningUpdate
    ) -> Reasoning | None:
        """Patch a nested reasoning step, or return ``None`` when it is unknown."""
        ...

    async def get_message(self, message_uuid: UUID) -> Message | None:
        """Return a message by its application UUID, or ``None``."""
        ...

    async def get_message_by_agui_id(self, chat_id: UUID, agui_id: str) -> Message | None:
        """Return a message by its AG-UI id within a chat, or ``None``."""
        ...

    async def get_tool_call_by_agui_id(self, message_uuid: UUID, agui_id: str) -> ToolCall | None:
        """Return a tool call by its AG-UI id within a message, or ``None``."""
        ...

    async def get_chat_messages(self, chat_id: UUID) -> Sequence[Message]:
        """Return every message in a chat, ordered oldest first."""
        ...

    async def get_last_messages(self, chat_ids: list[UUID]) -> dict[UUID, Message]:
        """Return the most recent message for each of the given chats."""
        ...


# ── Session registry ──────────────────────────────────────────────────────────


class ChatSessionRegistry:
    """Map an app chat UUID to a Memory Service session id.

    A local cache covers hot paths.  Because a :class:`Chat`'s indexed
    ``description`` is keyed by ``thread_id`` (not by ``chat_uuid``), a cold-cache
    resolve — e.g. after a process restart — falls back to a full metadata scan
    matching ``chat_uuid``.  Fast, indexed ``(user, thread_id)`` lookups live on
    :meth:`ChatRepository.get_chat_by_thread_id` instead.
    """

    def __init__(self, space: DRMemorySpace, chat_cls: type[Chat] = Chat) -> None:
        self._space = space
        self._chat_cls = chat_cls
        self._chat_to_session: dict[UUID, str] = {}

    def register(self, chat_uuid: UUID, session_id: str) -> None:
        """Cache the ``chat_uuid`` → ``session_id`` mapping."""
        self._chat_to_session[chat_uuid] = session_id

    def unregister(self, chat_uuid: UUID) -> None:
        """Drop a cached mapping (e.g. after the chat is deleted)."""
        self._chat_to_session.pop(chat_uuid, None)

    def get_session_id(self, chat_uuid: UUID) -> str | None:
        """Return the cached session id for a chat, without hitting the service."""
        return self._chat_to_session.get(chat_uuid)

    async def resolve(self, chat_uuid: UUID) -> str | None:
        """Resolve a chat UUID to its session id, scanning the space on a cache miss.

        Parameters
        ----------
        chat_uuid : UUID
            The application chat identifier.

        Returns
        -------
        str | None
            The Memory Service session id, or ``None`` when no session carries
            this ``chat_uuid``.
        """
        if sid := self.get_session_id(chat_uuid):
            return sid
        for session in await self._chat_cls.list(self._space):
            chat = cast(Chat, session)
            if chat.chat_uuid == chat_uuid and chat.id:
                self.register(chat_uuid, chat.id)
                return chat.id
        logger.debug("No memory session found for chat_uuid=%s", chat_uuid)
        return None


# ── Chat repository ─────────────────────────────────────────────────────────────


class ChatRepository:
    """Chat persistence backed by Memory Service sessions."""

    def __init__(
        self,
        space: DRMemorySpace,
        registry: ChatSessionRegistry,
        chat_cls: type[Chat] = Chat,
    ) -> None:
        self._space = space
        self._registry = registry
        self._chat_cls = chat_cls

    async def create_chat(self, chat_data: ChatCreate) -> Chat:
        """Create a chat, short-circuiting to the existing one for a known ``(user, thread)``.

        Parameters
        ----------
        chat_data : ChatCreate
            Must carry both ``user_uuid`` and ``thread_id``; they derive the
            deduplication key and the single session participant.

        Returns
        -------
        Chat
            The created (or adopted) chat.

        Raises
        ------
        ValueError
            If ``user_uuid`` or ``thread_id`` is missing.
        """
        if chat_data.user_uuid is None:
            raise ValueError("user_uuid is required to store a chat in the memory service")
        if chat_data.thread_id is None:
            raise ValueError("thread_id is required to store a chat in the memory service")

        existing = await self.get_chat_by_thread_id(chat_data.user_uuid, chat_data.thread_id)
        if existing is not None:
            return existing

        dedup_key = chat_deduplication_key(chat_data.user_uuid, chat_data.thread_id)
        pid = participant_id(chat_data.user_uuid)
        chat = cast(
            Chat,
            await self._chat_cls.post(
                self._space,
                thread_id=chat_data.thread_id,
                dedup_key=dedup_key,
                name=chat_data.name,
                chat_uuid=uuid4(),
                user_uuid=chat_data.user_uuid,
                participants=[pid],
            ),
        )
        if chat.id:
            self._registry.register(chat.chat_uuid, chat.id)
        return chat

    async def get_chat_by_thread_id(self, user_uuid: UUID, thread_id: str) -> Chat | None:
        """Return the chat for a ``(user, thread_id)`` pair, or ``None``.

        Tries the indexed ``description`` filter first (participant + thread id),
        then falls back to a participant-scoped scan for robustness.

        Parameters
        ----------
        user_uuid : UUID
            Owning user.
        thread_id : str
            AG-UI thread identifier.

        Returns
        -------
        Chat | None
        """
        pid = participant_id(user_uuid)

        indexed = await self._chat_cls.list(self._space, participant=pid, thread_id=thread_id)
        if match := self._match_thread(indexed, user_uuid, thread_id):
            return match

        scan = await self._chat_cls.list(self._space, participant=pid)
        return self._match_thread(scan, user_uuid, thread_id)

    def _match_thread(
        self, sessions: Sequence[object], user_uuid: UUID, thread_id: str
    ) -> Chat | None:
        """Return the first chat matching ``(user_uuid, thread_id)``, registering it."""
        for session in sessions:
            chat = cast(Chat, session)
            if chat.thread_id == thread_id and chat.user_uuid == user_uuid:
                if chat.id:
                    self._registry.register(chat.chat_uuid, chat.id)
                return chat
        return None

    async def get_all_chats(self, user_uuid: UUID | None) -> Sequence[Chat]:
        """Return every chat, optionally scoped to a single user.

        Parameters
        ----------
        user_uuid : UUID | None
            When given, only chats participant-scoped to this user are returned.

        Returns
        -------
        Sequence[Chat]
        """
        if user_uuid is not None:
            sessions = await self._chat_cls.list(self._space, participant=participant_id(user_uuid))
        else:
            sessions = await self._chat_cls.list(self._space)

        result: list[Chat] = []
        for session in sessions:
            chat = cast(Chat, session)
            if chat.id:
                self._registry.register(chat.chat_uuid, chat.id)
            result.append(chat)
        return result

    async def update_chat_name(self, chat_uuid: UUID, name: str) -> Chat | None:
        """Rename a chat, retrying on a version conflict.

        Parameters
        ----------
        chat_uuid : UUID
            The chat to rename.
        name : str
            The new display name.

        Returns
        -------
        Chat | None
            The updated chat, or ``None`` when no session carries the chat UUID.
        """
        sid = await self._registry.resolve(chat_uuid)
        if sid is None:
            return None

        last_exc: DRMemoryVersionConflictError | None = None
        for _ in range(SESSION_PATCH_RETRIES):
            chat = cast(Chat, await self._chat_cls.get(self._space, id=sid))
            try:
                await chat.patch(name=name)
                return chat
            except DRMemoryVersionConflictError as exc:
                last_exc = exc
        assert last_exc is not None
        raise last_exc

    async def delete_chat(self, chat_uuid: UUID) -> Chat | None:
        """Delete a chat and drop its registry entry.

        Parameters
        ----------
        chat_uuid : UUID
            The chat to delete.

        Returns
        -------
        Chat | None
            The deleted chat, or ``None`` when no session carries the chat UUID.
        """
        sid = await self._registry.resolve(chat_uuid)
        if sid is None:
            return None
        chat = cast(Chat, await self._chat_cls.get(self._space, id=sid))
        await chat.delete()
        self._registry.unregister(chat_uuid)
        return chat


# ── Message repository ───────────────────────────────────────────────────────────


class MessageRepository:
    """Message persistence backed by session events — one event per logical message.

    Tool calls and reasoning steps are stored as typed nested models inside the
    parent message's event body; a mutation re-serialises and patches the whole
    event body.  Small in-process caches short-circuit the ``uuid → chat`` and
    ``child → parent`` lookups; a cold cache falls back to a full-space scan.
    """

    def __init__(
        self,
        space: DRMemorySpace,
        registry: ChatSessionRegistry,
        chat_cls: type[Chat] = Chat,
        message_cls: type[Message] = Message,
    ) -> None:
        self._space = space
        self._registry = registry
        self._chat_cls = chat_cls
        self._message_cls = message_cls
        self._msg_chat: dict[UUID, UUID] = {}
        self._tc_chat: dict[UUID, UUID] = {}
        self._rs_chat: dict[UUID, UUID] = {}
        self._tool_call_message: dict[UUID, UUID] = {}
        self._reasoning_message: dict[UUID, UUID] = {}

    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[None, None]:
        """No-op batching scope; the Memory Service has no cross-document transaction."""
        yield

    # ── Cache bookkeeping ──────────────────────────────────────────────────

    def _remember_maps(self, msg: Message, chat_id: UUID) -> None:
        """Record the ``uuid → chat`` and ``child → parent`` mappings for *msg*."""
        self._msg_chat[msg.message_uuid] = chat_id
        for tc in msg.tool_calls:
            self._tc_chat[tc.uuid] = chat_id
            self._tool_call_message[tc.uuid] = msg.message_uuid
        for r in msg.reasonings:
            self._rs_chat[r.uuid] = chat_id
            self._reasoning_message[r.uuid] = msg.message_uuid

    # ── Session / event helpers ────────────────────────────────────────────

    async def _chat_for(self, chat_id: UUID) -> Chat:
        """Return the session-backed :class:`Chat` for a chat UUID.

        Raises
        ------
        ValueError
            When the chat UUID resolves to no Memory Service session.
        """
        sid = await self._registry.resolve(chat_id)
        if sid is None:
            raise ValueError(f"No memory session for chat_id={chat_id}")
        return cast(Chat, await self._chat_cls.get(self._space, id=sid))

    @staticmethod
    def _hydrate(message: Message) -> Message:
        """Decode the base ``content`` placeholder in place and return *message*.

        Nested ``tool_calls`` / ``reasonings`` decode their own placeholders via the
        model field validators as the ORM hydrates them.
        """
        message.content = app_str(message.content)
        return message

    async def _list_message_events(self, chat: Chat) -> list[Message]:
        """Return every chat-message event under *chat* (paginated, unordered)."""
        out: list[Message] = []
        offset = 0
        while True:
            batch = await self._message_cls.list(
                chat, type=MEMORY_CHAT_MESSAGE_EVENT_TYPE, offset=offset, limit=_EVENT_PAGE_SIZE
            )
            if not batch:
                break
            for event in batch:
                message = cast(Message, event)
                if message.v == PAYLOAD_VERSION:
                    out.append(message)
            offset += len(batch)
            if len(batch) < _EVENT_PAGE_SIZE:
                break
        return out

    async def _find_event_for_message_uuid(self, chat: Chat, message_uuid: UUID) -> Message | None:
        """Return the event carrying *message_uuid* under *chat*, or ``None``."""
        for message in await self._list_message_events(chat):
            if message.message_uuid == message_uuid:
                return message
        return None

    async def _read_modify_write(
        self,
        chat: Chat,
        message_uuid: UUID,
        mutate: Callable[[Message], bool],
    ) -> Message | None:
        """Re-read an event, apply *mutate*, and patch it back with a bounded retry.

        Re-reading on every attempt (including the first) guarantees a fresh
        ``createdAt`` concurrency token and lets *mutate* rebuild its change on top
        of the latest server state after a conflict.

        Parameters
        ----------
        chat : Chat
            The session holding the event.
        message_uuid : UUID
            Application UUID of the message event to patch.
        mutate : Callable[[Message], bool]
            Applies the change in place; returns ``True`` to patch, ``False`` to
            skip (e.g. the targeted child was not found).

        Returns
        -------
        Message | None
            The patched (and re-decoded) event, or ``None`` when the event is
            missing or *mutate* declined.

        Raises
        ------
        DRMemoryVersionConflictError
            If every attempt loses the optimistic-concurrency race.
        """
        last_exc: DRMemoryVersionConflictError | None = None
        for _ in range(EVENT_PATCH_RETRIES):
            event = await self._find_event_for_message_uuid(chat, message_uuid)
            if event is None:
                return None
            self._hydrate(event)
            if not mutate(event):
                return None
            try:
                await event.patch(content=wire_non_empty_str(event.content))
            except DRMemoryVersionConflictError as exc:
                last_exc = exc
                continue
            return self._hydrate(event)
        assert last_exc is not None
        raise last_exc

    # ── Chat resolution for a message / child ──────────────────────────────

    async def _resolve_chat_for_message(self, message_uuid: UUID) -> UUID | None:
        """Return the chat UUID owning *message_uuid* (cache, then full-space scan)."""
        if (chat_id := self._msg_chat.get(message_uuid)) is not None:
            return chat_id
        return await self._discover_chat_for_message(message_uuid)

    async def _discover_chat_for_message(self, message_uuid: UUID) -> UUID | None:
        """Scan every session for the event carrying *message_uuid*; register on a hit."""
        for session in await self._chat_cls.list(self._space):
            chat = cast(Chat, session)
            event = await self._find_event_for_message_uuid(chat, message_uuid)
            if event is not None and event.chat_id is not None and chat.id:
                self._registry.register(event.chat_id, chat.id)
                return event.chat_id
        return None

    async def _discover_child(
        self, child_uuid: UUID, *, in_tool_calls: bool
    ) -> tuple[UUID, UUID] | None:
        """Scan every session for the message owning a tool call / reasoning child.

        Returns ``(message_uuid, chat_id)`` on a hit (also registering the chat),
        else ``None``.
        """
        for session in await self._chat_cls.list(self._space):
            chat = cast(Chat, session)
            for message in await self._list_message_events(chat):
                children = message.tool_calls if in_tool_calls else message.reasonings
                if any(child.uuid == child_uuid for child in children):
                    if message.chat_id is not None and chat.id:
                        self._registry.register(message.chat_id, chat.id)
                        return message.message_uuid, message.chat_id
        return None

    # ── Writes ──────────────────────────────────────────────────────────────

    async def create_message(self, message_data: MessageCreate) -> Message:
        """Persist a new message as a single event.

        Parameters
        ----------
        message_data : MessageCreate
            Message fields; ``chat_id`` is required.

        Returns
        -------
        Message
            The persisted message (base ``content`` decoded).

        Raises
        ------
        ValueError
            If ``chat_id`` is missing.
        """
        if message_data.chat_id is None:
            raise ValueError("chat_id is required to create a message")

        chat = await self._chat_for(message_data.chat_id)
        user_pid = chat.participants[0] if chat.participants else None
        emitter_type, emitter_id = emitter_for_role(message_data.role, user_pid)

        event = cast(
            Message,
            await self._message_cls.post(
                chat,
                content=wire_non_empty_str(message_data.content),
                emitter_type=emitter_type,
                emitter_id=emitter_id,
                v=PAYLOAD_VERSION,
                message_uuid=uuid4(),
                chat_id=message_data.chat_id,
                agui_id=message_data.agui_id,
                role=message_data.role,
                name=message_data.name,
                step=message_data.step,
                in_progress=message_data.in_progress,
                error=message_data.error,
                status=message_data.status,
                timestamp=message_data.timestamp,
                tool_calls=[],
                reasonings=[],
            ),
        )
        self._hydrate(event)
        self._remember_maps(event, message_data.chat_id)
        return event

    async def update_message(self, message_uuid: UUID, update: MessageUpdate) -> Message | None:
        """Patch a message's own fields in place.

        Parameters
        ----------
        message_uuid : UUID
            Application UUID of the message.
        update : MessageUpdate
            Only the explicitly-set, non-``None`` fields are applied.

        Returns
        -------
        Message | None
            The updated message, or ``None`` when it does not exist.
        """
        chat_id = await self._resolve_chat_for_message(message_uuid)
        if chat_id is None:
            return None
        chat = await self._chat_for(chat_id)

        fields = {k: v for k, v in update.model_dump(exclude_unset=True).items() if v is not None}

        def mutate(message: Message) -> bool:
            for name, value in fields.items():
                setattr(message, name, value)
            return True

        event = await self._read_modify_write(chat, message_uuid, mutate)
        if event is None:
            return None
        self._remember_maps(event, chat_id)
        return event

    async def create_message_tool_call(self, data: MessageToolCallCreate) -> ToolCall:
        """Append a tool call to its parent message's body.

        Parameters
        ----------
        data : MessageToolCallCreate
            Tool-call fields; ``message_uuid`` names the parent message.

        Returns
        -------
        ToolCall
            The newly appended tool call.

        Raises
        ------
        ValueError
            If the parent message does not exist.
        """
        chat_id = await self._resolve_chat_for_message(data.message_uuid)
        if chat_id is None:
            raise ValueError(f"Message {data.message_uuid} does not exist")
        chat = await self._chat_for(chat_id)

        tool_call = ToolCall(**data.model_dump(exclude={"message_uuid"}), uuid=uuid4())

        def mutate(message: Message) -> bool:
            message.tool_calls = [*message.tool_calls, tool_call]
            return True

        event = await self._read_modify_write(chat, data.message_uuid, mutate)
        if event is None:
            raise ValueError(f"Message {data.message_uuid} does not exist")
        self._remember_maps(event, chat_id)
        return tool_call

    async def update_message_tool_call(
        self, uuid: UUID, update: MessageToolCallUpdate
    ) -> ToolCall | None:
        """Patch a nested tool call in place.

        Parameters
        ----------
        uuid : UUID
            The tool call UUID.
        update : MessageToolCallUpdate
            Only explicitly-set, non-``None`` fields are applied.

        Returns
        -------
        ToolCall | None
            The updated tool call, or ``None`` when it does not exist.
        """
        located = await self._resolve_child(uuid, in_tool_calls=True)
        if located is None:
            return None
        message_uuid, chat_id = located
        chat = await self._chat_for(chat_id)

        fields = {k: v for k, v in update.model_dump(exclude_unset=True).items() if v is not None}
        updated: list[ToolCall] = []

        def mutate(message: Message) -> bool:
            for tool_call in message.tool_calls:
                if tool_call.uuid == uuid:
                    for name, value in fields.items():
                        setattr(tool_call, name, value)
                    updated.append(tool_call)
                    return True
            return False

        event = await self._read_modify_write(chat, message_uuid, mutate)
        if event is None or not updated:
            return None
        self._remember_maps(event, chat_id)
        return updated[-1]

    async def create_message_reasoning(self, data: MessageReasoningCreate) -> Reasoning:
        """Append a reasoning step to its parent message's body.

        Parameters
        ----------
        data : MessageReasoningCreate
            Reasoning fields; ``message_uuid`` names the parent message.

        Returns
        -------
        Reasoning
            The newly appended reasoning step.

        Raises
        ------
        ValueError
            If the parent message does not exist.
        """
        chat_id = await self._resolve_chat_for_message(data.message_uuid)
        if chat_id is None:
            raise ValueError(f"Message {data.message_uuid} does not exist")
        chat = await self._chat_for(chat_id)

        reasoning = Reasoning(**data.model_dump(exclude={"message_uuid"}), uuid=uuid4())

        def mutate(message: Message) -> bool:
            message.reasonings = [*message.reasonings, reasoning]
            return True

        event = await self._read_modify_write(chat, data.message_uuid, mutate)
        if event is None:
            raise ValueError(f"Message {data.message_uuid} does not exist")
        self._remember_maps(event, chat_id)
        return reasoning

    async def update_message_reasoning(
        self, uuid: UUID, update: MessageReasoningUpdate
    ) -> Reasoning | None:
        """Patch a nested reasoning step in place.

        Parameters
        ----------
        uuid : UUID
            The reasoning UUID.
        update : MessageReasoningUpdate
            Only explicitly-set, non-``None`` fields are applied.

        Returns
        -------
        Reasoning | None
            The updated reasoning step, or ``None`` when it does not exist.
        """
        located = await self._resolve_child(uuid, in_tool_calls=False)
        if located is None:
            return None
        message_uuid, chat_id = located
        chat = await self._chat_for(chat_id)

        fields = {k: v for k, v in update.model_dump(exclude_unset=True).items() if v is not None}
        updated: list[Reasoning] = []

        def mutate(message: Message) -> bool:
            for reasoning in message.reasonings:
                if reasoning.uuid == uuid:
                    for name, value in fields.items():
                        setattr(reasoning, name, value)
                    updated.append(reasoning)
                    return True
            return False

        event = await self._read_modify_write(chat, message_uuid, mutate)
        if event is None or not updated:
            return None
        self._remember_maps(event, chat_id)
        return updated[-1]

    async def _resolve_child(
        self, child_uuid: UUID, *, in_tool_calls: bool
    ) -> tuple[UUID, UUID] | None:
        """Return ``(message_uuid, chat_id)`` for a tool call / reasoning child."""
        message_map = self._tool_call_message if in_tool_calls else self._reasoning_message
        chat_map = self._tc_chat if in_tool_calls else self._rs_chat
        message_uuid = message_map.get(child_uuid)
        chat_id = chat_map.get(child_uuid)
        if message_uuid is not None and chat_id is not None:
            return message_uuid, chat_id
        return await self._discover_child(child_uuid, in_tool_calls=in_tool_calls)

    # ── Reads ─────────────────────────────────────────────────────────────────

    async def get_message(self, message_uuid: UUID) -> Message | None:
        """Return a message by its application UUID, or ``None``."""
        chat_id = await self._resolve_chat_for_message(message_uuid)
        if chat_id is None:
            return None
        chat = await self._chat_for(chat_id)
        event = await self._find_event_for_message_uuid(chat, message_uuid)
        if event is None:
            return None
        self._hydrate(event)
        self._remember_maps(event, chat_id)
        return event

    async def get_message_by_agui_id(self, chat_id: UUID, agui_id: str) -> Message | None:
        """Return a message by its AG-UI id within a chat, or ``None``."""
        chat = await self._chat_for(chat_id)
        for message in await self._list_message_events(chat):
            if message.agui_id == agui_id:
                self._hydrate(message)
                self._remember_maps(message, chat_id)
                return message
        return None

    async def get_tool_call_by_agui_id(self, message_uuid: UUID, agui_id: str) -> ToolCall | None:
        """Return a tool call by its AG-UI id within a message, or ``None``."""
        chat_id = await self._resolve_chat_for_message(message_uuid)
        if chat_id is None:
            return None
        chat = await self._chat_for(chat_id)
        event = await self._find_event_for_message_uuid(chat, message_uuid)
        if event is None:
            return None
        self._hydrate(event)
        for tool_call in event.tool_calls:
            if tool_call.agui_id == agui_id:
                self._remember_maps(event, chat_id)
                return tool_call
        return None

    async def get_chat_messages(self, chat_id: UUID) -> Sequence[Message]:
        """Return every message in a chat, ordered oldest first (by sequence id)."""
        chat = await self._chat_for(chat_id)
        events = await self._list_message_events(chat)
        events.sort(key=lambda message: message.sequence_id)
        for message in events:
            self._hydrate(message)
            self._remember_maps(message, chat_id)
        return events

    async def get_last_messages(self, chat_ids: list[UUID]) -> dict[UUID, Message]:
        """Return the most recent message for each of the given chats.

        Parameters
        ----------
        chat_ids : list[UUID]
            Chats to fetch the tail message for.

        Returns
        -------
        dict[UUID, Message]
            Maps each chat UUID that has at least one message to its latest one.
        """
        result: dict[UUID, Message] = {}
        for chat_id in chat_ids:
            sid = await self._registry.resolve(chat_id)
            if sid is None:
                continue
            chat = cast(Chat, await self._chat_cls.get(self._space, id=sid))
            recent = await self._message_cls.last(
                chat, n=_LAST_MESSAGE_WINDOW, type=MEMORY_CHAT_MESSAGE_EVENT_TYPE
            )
            messages = [
                cast(Message, event)
                for event in recent
                if cast(Message, event).v == PAYLOAD_VERSION
            ]
            if not messages:
                continue
            latest = max(messages, key=lambda message: message.sequence_id)
            self._hydrate(latest)
            self._remember_maps(latest, chat_id)
            result[chat_id] = latest
        return result
