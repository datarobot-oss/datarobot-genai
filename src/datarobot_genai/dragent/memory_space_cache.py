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

"""DataRobot MemorySpace-backed KV cache for dragent shared caches.

Uses the agentic memory **Session API** (``datarobot.models.memory.Session``) — the same
surface the agent-application recipe uses for chat history when
``USE_APPLICATION_MEMORY_SPACE`` is enabled — not the mem0-compatible sub-route.

Each provisioned memory space has a unique ``memory_space_id`` and platform-level
access control scoped to the deploying user or workload API token. Unlike shared
Redis, no per-deployment namespace or HMAC signing is required for this backend.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from typing import Any
from typing import Literal

import datarobot as dr
from datarobot.core.config import DataRobotAppFrameworkBaseSettings
from datarobot.errors import MemorySessionDeduplicationError
from datarobot.models.memory import Session
from pydantic import Field

logger = logging.getLogger(__name__)

# Stable 24-hex participant id (BSON ObjectId length) for cache sessions.
DRAGENT_CACHE_PARTICIPANT_ID = hashlib.sha256(b"datarobot-genai:dragent-cache").hexdigest()[:24]

CACHE_METADATA_VERSION = 1
CACHE_EVENT_TYPE = "status"
DEDUPLICATION_KEY_LENGTH = 64

CacheKind = Literal["agent_card", "xaa_token"]

_MEMORY_SPACE_REQUIRED_MSG = (
    "Memory space cache backends require a provisioned DataRobot MemorySpace ID. "
    "Set AGENT_CARD_REGISTRY_MEMORY_SPACE_ID."
)


class MemorySpaceCacheConfig(DataRobotAppFrameworkBaseSettings):
    """Connection settings for DataRobot MemorySpace cache backends."""

    agent_card_registry_memory_space_id: str | None = Field(
        default=None,
        description="DataRobot MemorySpace ID for the agent card registry L2 cache.",
    )

    datarobot_endpoint: str | None = Field(
        default=None,
        description="DataRobot API base URL (DATAROBOT_ENDPOINT).",
    )

    datarobot_api_token: str | None = Field(
        default=None,
        description="DataRobot API token (DATAROBOT_API_TOKEN).",
    )


def try_resolve_memory_space_id(explicit: str | None = None) -> str | None:
    """Return the agent card registry MemorySpace ID, or ``None`` when unset."""
    cfg = MemorySpaceCacheConfig()
    space_id = explicit or cfg.agent_card_registry_memory_space_id
    if not space_id or not space_id.strip():
        return None
    return space_id.strip()


def resolve_memory_space_id(explicit: str | None = None) -> str:
    """Return the MemorySpace ID for cache backends."""
    space_id = try_resolve_memory_space_id(explicit)
    if space_id is None:
        raise ValueError(_MEMORY_SPACE_REQUIRED_MSG)
    return space_id


def try_configure_datarobot_memory_client(
    *,
    endpoint: str | None = None,
    api_token: str | None = None,
) -> bool:
    """Configure the DataRobot client for memory Session API calls when possible."""
    try:
        configure_datarobot_memory_client(endpoint=endpoint, api_token=api_token)
    except Exception as exc:
        logger.debug("MemorySpace client unavailable: %s", exc)
        return False
    return True


def configure_datarobot_memory_client(
    *,
    endpoint: str | None = None,
    api_token: str | None = None,
) -> None:
    """Configure the process-global DataRobot client for memory Session API calls."""
    cfg = MemorySpaceCacheConfig()
    token = api_token or cfg.datarobot_api_token or os.getenv("DATAROBOT_API_TOKEN")
    base = endpoint or cfg.datarobot_endpoint or os.getenv("DATAROBOT_ENDPOINT")
    if not token:
        raise ValueError("DATAROBOT_API_TOKEN is required when using memory_space cache backends.")
    if not base:
        raise ValueError("DATAROBOT_ENDPOINT is required when using memory_space cache backends.")
    dr.Client(token=token, endpoint=base)


def _cache_deduplication_key(logical_key: str) -> str:
    raw = "dragent-cache\0" + logical_key
    return hashlib.sha256(raw.encode()).hexdigest()[:DEDUPLICATION_KEY_LENGTH]


def _cache_session_description(logical_key: str) -> str:
    return f"/dragent/cache/{logical_key}"


def _cache_session_metadata(logical_key: str, *, kind: CacheKind) -> dict[str, Any]:
    return {
        "v": CACHE_METADATA_VERSION,
        "dragent_cache": True,
        "cache_key": logical_key,
        "cache_kind": kind,
    }


def _payload_event_body(payload: str) -> dict[str, Any]:
    return {"v": CACHE_METADATA_VERSION, "payload": payload}


def _payload_from_event_body(body: dict[str, Any] | None) -> str | None:
    if not body or body.get("v") != CACHE_METADATA_VERSION:
        return None
    value = body.get("payload")
    return str(value) if value is not None else None


def _create_cache_session(
    memory_space_id: str,
    *,
    logical_key: str,
    kind: CacheKind,
) -> Session:
    """Create a cache session, adopting an existing one on deduplication collision."""
    try:
        return Session.create(
            memory_space_id,
            [DRAGENT_CACHE_PARTICIPANT_ID],
            metadata=_cache_session_metadata(logical_key, kind=kind),
            description=_cache_session_description(logical_key),
            deduplication_key=_cache_deduplication_key(logical_key),
        )
    except MemorySessionDeduplicationError as exc:
        if exc.existing_session_id is None:
            raise
        return Session.get(memory_space_id, exc.existing_session_id)


def _find_cache_session(memory_space_id: str, logical_key: str) -> Session | None:
    description = _cache_session_description(logical_key)
    sessions = Session.list(
        memory_space_id,
        participants=[DRAGENT_CACHE_PARTICIPANT_ID],
        description=description,
        limit=1,
    )
    return sessions[0] if sessions else None


def _read_payload(session: Session) -> str | None:
    events = session.events(last_n=1)
    if not events:
        return None
    return _payload_from_event_body(events[0].body)


def _write_payload(session: Session, payload: str) -> None:
    body = _payload_event_body(payload)
    events = session.events(last_n=1)
    if events and events[0].sequence_id is not None:
        session.update_event(events[0].sequence_id, body=body)
        return
    session.post_event(
        body=body,
        emitter={"type": "agent"},
        event_type=CACHE_EVENT_TYPE,
    )


class MemorySpaceKVCache:
    """Store opaque JSON payloads in a DataRobot MemorySpace by logical key."""

    def __init__(self, *, memory_space_id: str, key_prefix: str = "dragent:") -> None:
        self._memory_space_id = memory_space_id
        normalized = key_prefix if key_prefix.endswith(":") else f"{key_prefix}:"
        self._key_prefix = normalized

    def _logical_key(self, key: str, *, kind: CacheKind) -> str:
        return f"{self._key_prefix}{kind}:{key}"

    async def get_value(self, key: str, *, kind: CacheKind) -> str | None:
        """Return the stored JSON payload for *key*, or ``None`` when missing."""
        logical_key = self._logical_key(key, kind=kind)

        def _get() -> str | None:
            session = _find_cache_session(self._memory_space_id, logical_key)
            if session is None:
                return None
            return _read_payload(session)

        try:
            return await asyncio.to_thread(_get)
        except Exception:
            logger.exception("MemorySpace cache read failed for %s", logical_key)
            return None

    async def set_value(self, key: str, payload: str, *, kind: CacheKind) -> None:
        """Upsert a JSON payload for *key*."""
        logical_key = self._logical_key(key, kind=kind)

        def _set() -> None:
            session = _find_cache_session(self._memory_space_id, logical_key)
            if session is None:
                session = _create_cache_session(
                    self._memory_space_id,
                    logical_key=logical_key,
                    kind=kind,
                )
            _write_payload(session, payload)

        try:
            await asyncio.to_thread(_set)
        except Exception:
            logger.exception("MemorySpace cache write failed for %s", logical_key)

    async def delete_value(self, key: str, *, kind: CacheKind) -> None:
        """Remove a cached payload for *key* when present."""
        logical_key = self._logical_key(key, kind=kind)

        def _delete() -> None:
            session = _find_cache_session(self._memory_space_id, logical_key)
            if session is not None:
                session.delete()

        try:
            await asyncio.to_thread(_delete)
        except Exception:
            logger.exception("MemorySpace cache delete failed for %s", logical_key)
