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

Secondary index & consistency model
------------------------------------
The Memory Service exposes no cross-session event query, so every cold
``uuid → location`` lookup (chat → session, message → chat, tool-call/reasoning →
parent message) is served by a dedup-keyed
:class:`~datarobot_genai.application_utils.chat_history.models.EntityLocator`
secondary index — an indexed O(1) point ``get`` — rather than a full-space scan.
Locators share the chats' Memory Space; their ``//loc/`` description prefix keeps
them out of every ``Chat.list`` result.

Writes are ordered **event/session (truth) → locator (index)**, with no
transaction, and the locator write is *best-effort* (a failure is logged and
swallowed).  The trade-off, deliberately accepted here: if a locator write is
lost, a *different* replica (or a post-restart lookup) that addresses the entity
**by uuid** may see it as "not found", even though the event exists and
``get_chat_messages`` (which lists the chat's events directly) still returns it.
This backend therefore offers no strong cross-replica read-after-write
consistency for uuid-addressed lookups; a caller that cannot tolerate that needs
a transactional backend.  :meth:`ChatRepository.delete_chat` soft-deletes the
session (and its events) in one shot and **leaves the chat's locators as
orphans** — cheap, bounded by the soft-delete TTL, and harmless (a stale locator
resolves to a dead session → treated as not-found; uuids never collide).

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
from collections import OrderedDict
from collections.abc import AsyncGenerator
from collections.abc import Callable
from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager
from contextlib import asynccontextmanager
from typing import Generic
from typing import Protocol
from typing import TypeVar
from typing import cast
from typing import runtime_checkable
from uuid import UUID
from uuid import uuid4

from datarobot_genai.application_utils.chat_history.constants import LOCATOR_KIND_CHAT
from datarobot_genai.application_utils.chat_history.constants import LOCATOR_KIND_MESSAGE
from datarobot_genai.application_utils.chat_history.constants import LOCATOR_KIND_REASONING
from datarobot_genai.application_utils.chat_history.constants import LOCATOR_KIND_TOOL_CALL
from datarobot_genai.application_utils.chat_history.constants import MEMORY_CHAT_MESSAGE_EVENT_TYPE
from datarobot_genai.application_utils.chat_history.constants import PAYLOAD_VERSION
from datarobot_genai.application_utils.chat_history.constants import app_str
from datarobot_genai.application_utils.chat_history.constants import chat_deduplication_key
from datarobot_genai.application_utils.chat_history.constants import emitter_for_role
from datarobot_genai.application_utils.chat_history.constants import locator_key
from datarobot_genai.application_utils.chat_history.constants import participant_id
from datarobot_genai.application_utils.chat_history.constants import wire_non_empty_str
from datarobot_genai.application_utils.chat_history.models import Chat
from datarobot_genai.application_utils.chat_history.models import ChatCreate
from datarobot_genai.application_utils.chat_history.models import EntityLocator
from datarobot_genai.application_utils.chat_history.models import Message
from datarobot_genai.application_utils.chat_history.models import MessageCreate
from datarobot_genai.application_utils.chat_history.models import MessageReasoningCreate
from datarobot_genai.application_utils.chat_history.models import MessageReasoningUpdate
from datarobot_genai.application_utils.chat_history.models import MessageToolCallCreate
from datarobot_genai.application_utils.chat_history.models import MessageToolCallUpdate
from datarobot_genai.application_utils.chat_history.models import MessageUpdate
from datarobot_genai.application_utils.chat_history.models import Reasoning
from datarobot_genai.application_utils.chat_history.models import ToolCall
from datarobot_genai.application_utils.persistence import DRMemoryNotFoundError
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
#: Hard cap on each in-process id→parent / chat→session cache (see :class:`_BoundedLRU`).
_CACHE_MAXSIZE: int = 4096

_K = TypeVar("_K")
_V = TypeVar("_V")


class _BoundedLRU(Generic[_K, _V]):
    """A size-capped, access-ordered mapping used for the repositories' hot caches.

    A near-drop-in for the small ``dict`` caches that short-circuit the
    ``uuid → parent`` and ``chat → session`` lookups.  Every mapping it holds is
    **immutable** (a message never changes chat, a tool call never changes
    message, a chat never changes session), so evicting the least-recently-used
    entry can only cost a recomputable indexed locator lookup on the next
    access — never correctness.  Bounding the caches keeps a long-lived
    multi-replica worker from growing them without limit.
    """

    def __init__(self, maxsize: int = _CACHE_MAXSIZE) -> None:
        self._maxsize = maxsize
        self._data: OrderedDict[_K, _V] = OrderedDict()

    def __getitem__(self, key: _K) -> _V:
        self._data.move_to_end(key)
        return self._data[key]

    def __setitem__(self, key: _K, value: _V) -> None:
        self._data[key] = value
        self._data.move_to_end(key)
        while len(self._data) > self._maxsize:
            self._data.popitem(last=False)

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)

    def get(self, key: _K, default: _V | None = None) -> _V | None:
        """Return the cached value for *key* (bumping its recency), or *default*."""
        if key not in self._data:
            return default
        self._data.move_to_end(key)
        return self._data[key]

    def pop(self, key: _K, default: _V | None = None) -> _V | None:
        """Discard and return *key*'s value, or *default* when absent."""
        return self._data.pop(key, default)


# ── Secondary index (dedup-keyed uuid → location locators) ─────────────────────


class LocatorIndex:
    """Best-effort, dedup-keyed secondary index over :class:`EntityLocator` sessions.

    Reads are indexed O(1) point ``get``s (by deduplication key); writes are
    idempotent (a duplicate ``post`` adopts the existing locator) and
    **best-effort** — a failed write is logged at ``WARNING`` and never propagates,
    because the event / session the locator points at is the source of truth.
    """

    def __init__(
        self,
        space: DRMemorySpace,
        locator_cls: type[EntityLocator] = EntityLocator,
    ) -> None:
        self._space = space
        self._locator_cls = locator_cls

    async def get(self, kind: str, entity_uuid: UUID) -> EntityLocator | None:
        """Return the locator for ``(kind, entity_uuid)``, or ``None`` when absent."""
        try:
            return cast(
                EntityLocator,
                await self._locator_cls.get(
                    self._space, locator_key=locator_key(kind, entity_uuid)
                ),
            )
        except DRMemoryNotFoundError:
            return None

    async def put(
        self,
        kind: str,
        entity_uuid: UUID,
        *,
        session_id: str,
        chat_uuid: UUID,
        message_uuid: UUID | None = None,
    ) -> None:
        """Write (idempotently, best-effort) the ``(kind, entity_uuid)`` locator."""
        try:
            await self._locator_cls.post(
                self._space,
                locator_key=locator_key(kind, entity_uuid),
                kind=kind,
                session_id=session_id,
                chat_uuid=chat_uuid,
                message_uuid=message_uuid,
            )
        except Exception as exc:  # index write is best-effort; it must never fail the operation
            logger.warning(
                "Failed to write %s locator for %s (best-effort index; ignoring): %s",
                kind,
                entity_uuid,
                exc,
            )


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

    A bounded in-process cache covers hot paths.  Because a :class:`Chat`'s
    indexed ``description`` is keyed by ``thread_id`` (not by ``chat_uuid``), a
    cold-cache resolve — e.g. on a replica that did not create the chat, or after
    a process restart — reads the dedup-keyed ``chat:<uuid>`` :class:`EntityLocator`
    (an indexed O(1) point lookup), not a full-space scan.  Fast, indexed
    ``(user, thread_id)`` lookups live on
    :meth:`ChatRepository.get_chat_by_thread_id` instead.
    """

    def __init__(
        self,
        space: DRMemorySpace,
        chat_cls: type[Chat] = Chat,
        locator_cls: type[EntityLocator] = EntityLocator,
    ) -> None:
        self._space = space
        self._chat_cls = chat_cls
        self._locators = LocatorIndex(space, locator_cls)
        self._chat_to_session: _BoundedLRU[UUID, str] = _BoundedLRU()

    @property
    def locators(self) -> LocatorIndex:
        """The shared :class:`LocatorIndex` for uuid → location lookups."""
        return self._locators

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
        """Resolve a chat UUID to its session id via the locator index on a cache miss.

        Parameters
        ----------
        chat_uuid : UUID
            The application chat identifier.

        Returns
        -------
        str | None
            The Memory Service session id, or ``None`` when no ``chat:<uuid>``
            locator exists (e.g. a lost best-effort index write, or an unknown
            chat).
        """
        if sid := self.get_session_id(chat_uuid):
            return sid
        locator = await self._locators.get(LOCATOR_KIND_CHAT, chat_uuid)
        if locator is not None and locator.session_id:
            self.register(chat_uuid, locator.session_id)
            return locator.session_id
        logger.debug("No memory session located for chat_uuid=%s", chat_uuid)
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
            # Post the chat:<uuid> locator so other replicas can resolve this chat
            # to its session without a full-space scan (best-effort index write).
            await self._registry.locators.put(
                LOCATOR_KIND_CHAT,
                chat.chat_uuid,
                session_id=chat.id,
                chat_uuid=chat.chat_uuid,
            )
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
    event body.  Bounded in-process caches short-circuit the ``uuid → chat`` and
    ``child → parent`` lookups; on a cold cache these resolve via the dedup-keyed
    :class:`EntityLocator` index (an O(1) point lookup), never a full-space scan.
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
        self._msg_chat: _BoundedLRU[UUID, UUID] = _BoundedLRU()
        self._tc_chat: _BoundedLRU[UUID, UUID] = _BoundedLRU()
        self._rs_chat: _BoundedLRU[UUID, UUID] = _BoundedLRU()
        self._tool_call_message: _BoundedLRU[UUID, UUID] = _BoundedLRU()
        self._reasoning_message: _BoundedLRU[UUID, UUID] = _BoundedLRU()

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
        """Return the chat UUID owning *message_uuid* (cache, then locator index).

        On a cache miss this reads the ``message:<uuid>`` locator — an indexed
        point lookup — and registers the resolved ``chat_uuid → session_id`` so the
        subsequent :meth:`_chat_for` is a warm-cache resolve (no extra hop).
        Returns ``None`` when no locator exists (unknown message, or a lost
        best-effort index write).
        """
        if (chat_id := self._msg_chat.get(message_uuid)) is not None:
            return chat_id
        locator = await self._registry.locators.get(LOCATOR_KIND_MESSAGE, message_uuid)
        if locator is None:
            return None
        if locator.session_id:
            self._registry.register(locator.chat_uuid, locator.session_id)
        self._msg_chat[message_uuid] = locator.chat_uuid
        return locator.chat_uuid

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
        # Post the message:<uuid> locator so a cold replica can resolve this
        # message to its chat/session without a full-space scan (best-effort).
        await self._registry.locators.put(
            LOCATOR_KIND_MESSAGE,
            event.message_uuid,
            session_id=chat.id,
            chat_uuid=message_data.chat_id,
        )
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
        if not fields:
            # Nothing to change: skip the read-modify-write cycle so a no-op
            # update doesn't cost an unnecessary API patch.
            event = await self._find_event_for_message_uuid(chat, message_uuid)
            if event is None:
                return None
            self._remember_maps(event, chat_id)
            return self._hydrate(event)

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
        # Post the tool_call:<uuid> locator (best-effort) → (message, chat, session).
        await self._registry.locators.put(
            LOCATOR_KIND_TOOL_CALL,
            tool_call.uuid,
            session_id=chat.id,
            chat_uuid=chat_id,
            message_uuid=data.message_uuid,
        )
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
        # Post the reasoning:<uuid> locator (best-effort) → (message, chat, session).
        await self._registry.locators.put(
            LOCATOR_KIND_REASONING,
            reasoning.uuid,
            session_id=chat.id,
            chat_uuid=chat_id,
            message_uuid=data.message_uuid,
        )
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
        """Return ``(message_uuid, chat_id)`` for a tool call / reasoning child.

        On a cache miss this reads the ``tool_call:<uuid>`` / ``reasoning:<uuid>``
        locator (an indexed point lookup) and registers the chat → session mapping,
        so the following :meth:`_chat_for` is a warm resolve.  Returns ``None`` when
        no locator exists (unknown child, or a lost best-effort index write).
        """
        message_map = self._tool_call_message if in_tool_calls else self._reasoning_message
        chat_map = self._tc_chat if in_tool_calls else self._rs_chat
        message_uuid = message_map.get(child_uuid)
        chat_id = chat_map.get(child_uuid)
        if message_uuid is not None and chat_id is not None:
            return message_uuid, chat_id
        kind = LOCATOR_KIND_TOOL_CALL if in_tool_calls else LOCATOR_KIND_REASONING
        locator = await self._registry.locators.get(kind, child_uuid)
        if locator is None or locator.message_uuid is None:
            return None
        if locator.session_id:
            self._registry.register(locator.chat_uuid, locator.session_id)
        message_map[child_uuid] = locator.message_uuid
        chat_map[child_uuid] = locator.chat_uuid
        return locator.message_uuid, locator.chat_uuid

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
