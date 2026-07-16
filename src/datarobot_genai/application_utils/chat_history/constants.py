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

"""Constants and small pure helpers for the chat-history layer.

This module is deliberately transport-agnostic: it holds no request/middleware
state and imports nothing from ``ag_ui`` or the persistence ORM.  The helpers
here are the low-level building blocks used by :mod:`.models` and (later) the
repository / AG-UI storage layers:

* the zero-width placeholder codec (:func:`wire_non_empty_str` /
  :func:`app_str`) that works around the Memory Service ``min_length=1``
  constraint on some string body fields;
* deterministic identifier derivation (:func:`participant_id`,
  :func:`chat_deduplication_key`);
* the event-emitter rule (:func:`emitter_for_role`).
"""

from __future__ import annotations

import hashlib
from typing import Literal
from uuid import UUID

#: Event ``type`` tag used for every chat-message event.  The Memory Service
#: list API only accepts ``message | tool_output | status`` as ``eventType``
#: filter values; chat messages are always stored as ``"message"``.
MEMORY_CHAT_MESSAGE_EVENT_TYPE: str = "message"

#: Body payload schema version.  Written as the ``v`` body field and used as a
#: gate on read (a body whose ``v`` differs is not a chat message).
PAYLOAD_VERSION: int = 1

#: Zero-width space used as an on-the-wire placeholder for empty strings on body
#: fields the Memory Service rejects when empty (``min_length=1``).
_ZW_PLACEHOLDER: str = "\u200b"

#: Memory Service deduplication keys may be up to 72 characters; we truncate the
#: hex digest to a stable 64.
DEDUPLICATION_KEY_LENGTH: int = 64

#: Participant ids are 24-hex-character values (BSON ObjectId length).
_PARTICIPANT_ID_LENGTH: int = 24

#: EntityLocator dedup-key namespaces — one per locatable entity kind.  The
#: ``"<kind>:<uuid>"`` key stays well under the Memory Service 72-char dedup-key
#: limit (longest kind ``"tool_call"`` + ``":"`` + a 36-char UUID = 46 chars), so
#: no hashing is needed and the key is human-legible.
LOCATOR_KIND_CHAT: str = "chat"
LOCATOR_KIND_MESSAGE: str = "message"
LOCATOR_KIND_TOOL_CALL: str = "tool_call"
LOCATOR_KIND_REASONING: str = "reasoning"


def wire_non_empty_str(value: str | None) -> str:
    """Encode a possibly-empty string for the wire, substituting the placeholder.

    Parameters
    ----------
    value : str | None
        The application-side string (``None`` is treated as empty).

    Returns
    -------
    str
        *value* when it is non-empty, otherwise the zero-width placeholder so the
        Memory Service ``min_length=1`` constraint is satisfied.
    """
    s = value or ""
    return s if s else _ZW_PLACEHOLDER


def app_str(value: str | None) -> str:
    """Decode a wire string back to its application value, stripping the placeholder.

    Parameters
    ----------
    value : str | None
        The raw wire string (``None`` is treated as empty).

    Returns
    -------
    str
        The empty string when *value* is the placeholder (or falsy), otherwise
        *value* unchanged.
    """
    s = value or ""
    return "" if s == _ZW_PLACEHOLDER else s


def session_deduplication_key(namespace: str, *parts: str) -> str:
    """Build a stable, namespaced deduplication key for idempotent session create.

    Parameters
    ----------
    namespace : str
        Logical document namespace (e.g. ``"chat"``).
    *parts : str
        Ordered key components; combined with NUL separators before hashing.

    Returns
    -------
    str
        The lowercase hex SHA-256 digest, truncated to
        :data:`DEDUPLICATION_KEY_LENGTH` characters.
    """
    raw = namespace + "\0" + "\0".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:DEDUPLICATION_KEY_LENGTH]


def chat_deduplication_key(user_uuid: UUID, thread_id: str) -> str:
    """Return the deduplication key for a chat, keyed by user and AG-UI thread id.

    Parameters
    ----------
    user_uuid : UUID
        Owning user's UUID.
    thread_id : str
        AG-UI thread identifier.

    Returns
    -------
    str
        The SHA-256 of ``"chat"``, the user UUID and the thread id (NUL-separated),
        truncated to :data:`DEDUPLICATION_KEY_LENGTH` characters.  Idempotent: a
        retried create for the same ``(user, thread)`` adopts the existing session.
    """
    return session_deduplication_key("chat", str(user_uuid), thread_id)


def locator_key(kind: str, entity_uuid: UUID) -> str:
    """Build the dedup key for an :class:`.models.EntityLocator`: ``"<kind>:<uuid>"``.

    Parameters
    ----------
    kind : str
        One of the ``LOCATOR_KIND_*`` namespaces (``"chat"``, ``"message"``,
        ``"tool_call"``, ``"reasoning"``).
    entity_uuid : UUID
        The entity's application UUID (globally unique across kinds).

    Returns
    -------
    str
        The namespaced point-lookup key, e.g. ``"message:0f9c…"``.
    """
    return f"{kind}:{entity_uuid}"


def normalize_participant_id(raw: str | None) -> str | None:
    """Normalise a caller-supplied participant id to 24-char lowercase hex, or ``None``.

    Parameters
    ----------
    raw : str | None
        A candidate participant id (e.g. a ``X-DataRobot-User-Id`` header value).

    Returns
    -------
    str | None
        The normalised 24-hex id, or ``None`` when *raw* is missing or not a
        valid 24-character hexadecimal string.
    """
    if not raw:
        return None
    candidate = raw.strip().lower()
    if len(candidate) != _PARTICIPANT_ID_LENGTH:
        return None
    try:
        int(candidate, 16)
    except ValueError:
        return None
    return candidate


def participant_id(user_uuid: UUID, *, override: str | None = None) -> str:
    """Return a stable 24-hex participant id for a user.

    Unlike the agent-application helper this is transport-agnostic: it takes an
    explicit *override* argument instead of reading request/middleware context.

    Parameters
    ----------
    user_uuid : UUID
        The user's UUID; hashed to derive a deterministic ObjectId-shaped id.
    override : str | None
        An explicit participant id (e.g. a DataRobot user id).  When it
        normalises to a valid 24-hex value it is used verbatim; otherwise the
        derived value is returned.

    Returns
    -------
    str
        A 24-character lowercase hex participant id.
    """
    if normalized := normalize_participant_id(override):
        return normalized
    return hashlib.sha256(user_uuid.bytes).hexdigest()[:_PARTICIPANT_ID_LENGTH]


def emitter_for_role(
    role: str,
    user_participant_id: str | None,
) -> tuple[Literal["user", "agent"], str | None]:
    """Derive the Memory Service event emitter ``(type, id)`` for a message role.

    A session carries exactly one participant (the user); the agent is not a real
    participant, so only user messages carry an emitter id.

    Parameters
    ----------
    role : str
        The message role (see :class:`.models.Role`).
    user_participant_id : str | None
        The user's participant id (from :func:`participant_id`).

    Returns
    -------
    tuple[Literal["user", "agent"], str | None]
        ``("user", user_participant_id)`` for user messages; ``("agent", None)``
        for assistant, tool and reasoning messages.
    """
    if role == "user":
        return "user", user_participant_id
    return "agent", None
