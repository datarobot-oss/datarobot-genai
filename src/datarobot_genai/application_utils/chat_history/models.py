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

"""Chat-history ORM models and data-transfer objects.

The chat layer maps onto the Memory Service persistence ORM:

* :class:`Chat` is a :class:`~datarobot_genai.application_utils.persistence.DRSession`
  — one session per AG-UI thread.  ``thread_id`` is an indexed range key,
  ``dedup_key`` is the idempotent-create key, and ``name`` / ``chat_uuid`` /
  ``user_uuid`` are metadata.
* :class:`Message` is a :class:`~datarobot_genai.application_utils.persistence.DREvent`
  bound to :class:`Chat` — one event per logical message.  A message's tool
  calls and reasoning steps are **not** separate events; they are typed nested
  models (:class:`ToolCall` / :class:`Reasoning`) stored inside the same event
  body and re-serialised on every update.

Because the persistence ORM (de)serialises declared body/metadata fields through
Pydantic, the nested models round-trip natively, and a consumer can add fields
by subclassing :class:`Chat` / :class:`Message` / :class:`ToolCall` /
:class:`Reasoning` — the repositories persist and retrieve the extra fields with
no further wiring.

Empty ``arguments`` / ``content`` on the nested models are transparently encoded
as the zero-width placeholder on the wire (Memory Service ``min_length=1``) and
stripped back to ``""`` on read via per-field (de)serialisers.  ``Message``'s own
``content`` is a base ORM field and is handled at the repository boundary
instead.

Notes
-----
:class:`Message` stores its application-assigned creation time under the
``timestamp`` field rather than ``created_at``: the ORM base
(:class:`~datarobot_genai.application_utils.persistence.DREvent`) already exposes
the server-assigned ``created_at`` as a read-only property, and a same-named
field would be shadowed by it.  The nested :class:`ToolCall` / :class:`Reasoning`
models are plain ``BaseModel`` subclasses and keep the ``created_at`` name.
"""

from __future__ import annotations

from datetime import UTC
from datetime import datetime
from enum import StrEnum
from typing import Annotated
from typing import Any
from uuid import UUID
from uuid import uuid4

from pydantic import BaseModel
from pydantic import Field
from pydantic import field_serializer
from pydantic import field_validator

from datarobot_genai.application_utils.chat_history.constants import MEMORY_CHAT_MESSAGE_EVENT_TYPE
from datarobot_genai.application_utils.chat_history.constants import PAYLOAD_VERSION
from datarobot_genai.application_utils.chat_history.constants import app_str
from datarobot_genai.application_utils.chat_history.constants import wire_non_empty_str
from datarobot_genai.application_utils.persistence import DRDeduplicationKey
from datarobot_genai.application_utils.persistence import DREvent
from datarobot_genai.application_utils.persistence import DRRangeKey
from datarobot_genai.application_utils.persistence import DRSession


def _utcnow() -> datetime:
    """Return the current timezone-aware UTC time (default factory for timestamps)."""
    return datetime.now(UTC)


# ── Enums ─────────────────────────────────────────────────────────────────────


class Role(StrEnum):
    """Message source role, mirroring the AG-UI message roles.

    See https://docs.ag-ui.com/concepts/messages.
    """

    DEVELOPER = "developer"
    SYSTEM = "system"
    ASSISTANT = "assistant"
    USER = "user"
    TOOL = "tool"
    REASONING = "reasoning"


class MessageStatus(StrEnum):
    """Lifecycle status of a message, tool call or reasoning step."""

    ACTIVE = "active"
    COMPLETE = "complete"
    INTERRUPTED = "interrupted"
    ERRORED = "errored"


# ── Nested body models ──────────────────────────────────────────────────────────


class ToolCall(BaseModel):
    """A tool call nested inside a :class:`Message` body.

    ``arguments`` and ``content`` carry the zero-width placeholder codec so an
    empty string round-trips through the Memory Service ``min_length=1``
    constraint transparently.
    """

    uuid: UUID = Field(default_factory=uuid4)
    agui_id: str | None = None
    tool_call_id: str | None = None
    role: str = Role.TOOL.value
    name: str = ""
    arguments: str = ""
    content: str = ""
    in_progress: bool = True
    error: str | None = None
    status: str = MessageStatus.ACTIVE.value
    created_at: datetime = Field(default_factory=_utcnow)

    @field_validator("arguments", "content", mode="before")
    @classmethod
    def _decode_placeholder(cls, value: Any) -> Any:
        """Strip the wire placeholder from ``arguments`` / ``content`` on input."""
        return app_str(value) if isinstance(value, str) else value

    @field_serializer("arguments", "content", when_used="json")
    def _encode_placeholder(self, value: str) -> str:
        """Substitute the wire placeholder for an empty ``arguments`` / ``content``."""
        return wire_non_empty_str(value)


class Reasoning(BaseModel):
    """A reasoning step nested inside a :class:`Message` body.

    ``content`` carries the zero-width placeholder codec so an empty string
    round-trips through the Memory Service ``min_length=1`` constraint
    transparently.
    """

    uuid: UUID = Field(default_factory=uuid4)
    agui_id: str | None = None
    role: str = Role.REASONING.value
    name: str = ""
    content: str = ""
    in_progress: bool = True
    error: str | None = None
    status: str = MessageStatus.ACTIVE.value
    created_at: datetime = Field(default_factory=_utcnow)

    @field_validator("content", mode="before")
    @classmethod
    def _decode_placeholder(cls, value: Any) -> Any:
        """Strip the wire placeholder from ``content`` on input."""
        return app_str(value) if isinstance(value, str) else value

    @field_serializer("content", when_used="json")
    def _encode_placeholder(self, value: str) -> str:
        """Substitute the wire placeholder for an empty ``content``."""
        return wire_non_empty_str(value)


# ── Session / event ORM models ────────────────────────────────────────────────


class Chat(DRSession):
    """A chat thread, persisted as one Memory Service session.

    Field mapping
    -------------
    thread_id : Annotated[str, DRRangeKey]
        Indexed ``description`` segment (``//thread/{thread_id}/``); powers the
        fast ``get_chat_by_thread_id`` lookup.
    dedup_key : Annotated[str, DRDeduplicationKey]
        Idempotent-create key (see
        :func:`~datarobot_genai.application_utils.chat_history.constants.chat_deduplication_key`).
    name, chat_uuid, user_uuid
        Session ``metadata``.

    The session's single ``participants`` entry is the user's participant id
    (see :func:`~datarobot_genai.application_utils.chat_history.constants.participant_id`);
    the agent is not a session participant.
    """

    __description_prefix__ = "thread"

    thread_id: Annotated[str, DRRangeKey]
    dedup_key: Annotated[str, DRDeduplicationKey]
    name: str = "New Chat"
    chat_uuid: UUID = Field(default_factory=uuid4)
    user_uuid: UUID | None = None


class EntityLocator(DRSession):
    """A dedup-keyed secondary index mapping an entity UUID to its storage location.

    Every cold ``uuid → location`` lookup in the repositories (chat → session,
    message → chat, tool-call/reasoning → parent message) resolves to an indexed
    O(1) deduplication-key point ``get`` on one of these locators, instead of a
    full-space session scan.  One model serves all four entity kinds; the
    ``"<kind>:"`` dedup-key namespace (see
    :func:`~datarobot_genai.application_utils.chat_history.constants.locator_key`)
    keeps the lookup unambiguous even though UUIDs are already globally unique.

    Locators share the same Memory Space as the :class:`Chat` sessions they point
    at; the distinct ``"loc"`` description prefix keeps them out of every
    ``Chat.list`` result (trailing-slash anchoring makes ``//loc/`` and
    ``//thread/`` disjoint).  A locator is a **best-effort** index: the event /
    session it points at is the source of truth, so a lost or stale locator only
    costs cross-replica addressability, never data (see the repository module
    docstring).

    Field mapping
    -------------
    locator_key : Annotated[str, DRDeduplicationKey]
        ``"<kind>:<uuid>"`` point-lookup key.
    kind, session_id, chat_uuid, message_uuid
        Session ``metadata``.  ``session_id`` lets a message/child lookup resolve
        straight to the Memory session without a second registry hop;
        ``message_uuid`` is set only for ``tool_call`` / ``reasoning`` locators.
    """

    __description_prefix__ = "loc"

    locator_key: Annotated[str, DRDeduplicationKey]
    kind: str
    session_id: str
    chat_uuid: UUID
    message_uuid: UUID | None = None


class Message(DREvent, session=Chat):
    """A chat message, persisted as one Memory Service event under a :class:`Chat`.

    Tool calls and reasoning steps live in ``tool_calls`` / ``reasonings`` inside
    this event's body (not as separate events).  ``v`` is the body payload schema
    version and gates reads.

    See the module docstring for why the application timestamp is named
    ``timestamp`` rather than ``created_at``.
    """

    __event_type__ = MEMORY_CHAT_MESSAGE_EVENT_TYPE

    v: int = PAYLOAD_VERSION
    message_uuid: UUID = Field(default_factory=uuid4)
    chat_id: UUID | None = None
    agui_id: str | None = None
    role: str = Role.USER.value
    name: str = ""
    step: str | None = None
    in_progress: bool = True
    error: str | None = None
    status: str = MessageStatus.ACTIVE.value
    timestamp: datetime = Field(default_factory=_utcnow)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    reasonings: list[Reasoning] = Field(default_factory=list)


# ── Data-transfer objects ────────────────────────────────────────────────────


class ChatCreate(BaseModel):
    """Input DTO for creating a chat.

    ``thread_id`` and ``user_uuid`` are required by the repository at create time
    (both are needed to derive the deduplication key and participant); they are
    optional here to mirror the agent-application surface.
    """

    name: str = "New Chat"
    thread_id: str | None = None
    user_uuid: UUID | None = None


class MessageCreate(BaseModel):
    """Input DTO for creating a message.

    The application-assigned creation time is named ``timestamp`` (matching
    :class:`Message` / :class:`MessagePublic`); ``created_at`` is reserved by the
    ORM base for the server-assigned timestamp.
    """

    agui_id: str | None = None
    role: str = Role.USER.value
    name: str = ""
    content: str = ""
    step: str | None = None
    chat_id: UUID | None = None
    in_progress: bool = True
    status: str = MessageStatus.ACTIVE.value
    error: str | None = None
    timestamp: datetime = Field(default_factory=_utcnow)


class MessageUpdate(BaseModel):
    """Input DTO for a partial message update; every field is optional."""

    agui_id: str | None = None
    content: str | None = None
    error: str | None = None
    in_progress: bool | None = None
    status: str | None = None


class MessageToolCallCreate(BaseModel):
    """Input DTO for creating a tool call on a message."""

    message_uuid: UUID
    agui_id: str | None = None
    tool_call_id: str | None = None
    role: str = Role.TOOL.value
    name: str = ""
    arguments: str = ""
    content: str = ""
    in_progress: bool = True
    status: str = MessageStatus.ACTIVE.value
    error: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)


class MessageToolCallUpdate(BaseModel):
    """Input DTO for a partial tool-call update; every field is optional."""

    arguments: str | None = None
    content: str | None = None
    error: str | None = None
    in_progress: bool | None = None
    status: str | None = None


class MessageReasoningCreate(BaseModel):
    """Input DTO for creating a reasoning step on a message."""

    message_uuid: UUID
    agui_id: str | None = None
    role: str = Role.REASONING.value
    name: str = ""
    content: str = ""
    in_progress: bool = True
    status: str = MessageStatus.ACTIVE.value
    error: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)


class MessageReasoningUpdate(BaseModel):
    """Input DTO for a partial reasoning update; every field is optional."""

    content: str | None = None
    error: str | None = None
    in_progress: bool | None = None
    status: str | None = None


class MessagePublic(BaseModel):
    """Non-recursive read view of a message with its nested tool calls / reasonings.

    Suitable for serialising to an API response: it carries the nested
    :class:`ToolCall` / :class:`Reasoning` models directly rather than any table
    relationship.
    """

    message_uuid: UUID = Field(default_factory=uuid4)
    chat_id: UUID | None = None
    agui_id: str | None = None
    role: str = Role.USER.value
    name: str = ""
    content: str = ""
    step: str | None = None
    in_progress: bool = True
    status: str = MessageStatus.ACTIVE.value
    error: str | None = None
    timestamp: datetime = Field(default_factory=_utcnow)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    reasonings: list[Reasoning] = Field(default_factory=list)

    @classmethod
    def from_message(cls, message: Message) -> MessagePublic:
        """Build a :class:`MessagePublic` from a :class:`Message` ORM instance.

        Parameters
        ----------
        message : Message
            A hydrated message (e.g. read back from the repository).

        Returns
        -------
        MessagePublic
            A flat, non-recursive view carrying the message's nested tool calls
            and reasoning steps.
        """
        return cls(
            message_uuid=message.message_uuid,
            chat_id=message.chat_id,
            agui_id=message.agui_id,
            role=message.role,
            name=message.name,
            content=message.content,
            step=message.step,
            in_progress=message.in_progress,
            status=message.status,
            error=message.error,
            timestamp=message.timestamp,
            tool_calls=list(message.tool_calls),
            reasonings=list(message.reasonings),
        )
