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

Built on the Memory Service light ORM (``datarobot.application_utils.persistence``,
the ``datarobot[application-utils]`` extra) — the same surface
``datarobot.application_utils.chat_history`` uses for AG-UI chat history — instead
of hand-rolling session management against ``datarobot.models.memory.Session``.

Each provisioned memory space has a unique ``memory_space_id`` and platform-level
access control scoped to the deploying user or workload API token. Unlike shared
Redis, no per-deployment namespace or HMAC signing is required for this backend.

Each cache entry is one Memory Service session — found by an exact-match
``DRDeduplicationKey`` lookup on the logical cache key — carrying a single event
whose ``content`` is the opaque JSON payload. ``set_value`` patches that event in
place instead of appending; the cache only ever needs the current value, never a
history.
"""

from __future__ import annotations

import logging
import os
from typing import Annotated

from datarobot.application_utils.persistence import DRDeduplicationKey
from datarobot.application_utils.persistence import DREvent
from datarobot.application_utils.persistence import DRMemoryNotFoundError
from datarobot.application_utils.persistence import DRMemoryServiceClient
from datarobot.application_utils.persistence import DRMemorySpace
from datarobot.application_utils.persistence import DRSession
from datarobot.core.config import DataRobotAppFrameworkBaseSettings
from pydantic import Field

from datarobot_genai.dragent.deployment_urls import resolve_datarobot_endpoint

logger = logging.getLogger(__name__)

CACHE_KIND = "agent_card"

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


def try_build_memory_service_client(
    *,
    endpoint: str | None = None,
    api_token: str | None = None,
) -> DRMemoryServiceClient | None:
    """Build a Memory Service ORM client, or ``None`` when unconfigured.

    Does no I/O: only resolves ``DATAROBOT_ENDPOINT``/``DATAROBOT_API_TOKEN``
    (explicit argument > settings > environment) and constructs the client —
    building a :class:`DRMemoryServiceClient` merely opens an ``httpx.AsyncClient``
    and stores headers; the first network call happens lazily on first use. Safe
    to call from a synchronous, I/O-free constructor such as
    :class:`~datarobot_genai.dragent.agent_card_registry.AgentCardRegistry`'s.
    """
    cfg = MemorySpaceCacheConfig()
    token = api_token or cfg.datarobot_api_token or os.getenv("DATAROBOT_API_TOKEN")
    if not token:
        logger.debug("MemorySpace client unavailable: no DATAROBOT_API_TOKEN configured.")
        return None
    try:
        base = endpoint or resolve_datarobot_endpoint(require=True)
    except ValueError as exc:
        logger.debug("MemorySpace client unavailable: %s", exc)
        return None
    return DRMemoryServiceClient(endpoint=base, api_token=token)


class _CacheEntrySession(DRSession):
    """One Memory Service session per logical cache key."""

    __description_prefix__ = "dragent-cache"

    key: Annotated[str, DRDeduplicationKey]


class _CacheEntryEvent(DREvent, session=_CacheEntrySession):  # type: ignore[call-arg]
    """The single event carrying a cache entry's opaque JSON payload."""

    __event_type__ = "status"


class MemorySpaceKVCache:
    """Store opaque JSON payloads in a DataRobot MemorySpace by logical key.

    Parameters
    ----------
    memory_space_id : str
        Provisioned MemorySpace ID to store cache entries in.
    client : DRMemoryServiceClient
        Memory Service ORM client (see :func:`try_build_memory_service_client`).
        Required explicitly rather than resolved internally, since the ORM has no
        global client to fall back on.
    key_prefix : str
        Prefix namespacing every logical key this cache writes.
    """

    def __init__(
        self,
        *,
        memory_space_id: str,
        client: DRMemoryServiceClient,
        key_prefix: str = "dragent:",
    ) -> None:
        self._client = client
        # Wrap the space ID directly rather than fetching it via `DRMemorySpace.get()`:
        # DRSession/DREvent calls only ever read `space.id` and `space._client`, so a
        # round trip for metadata (user_id/tenant_id) this cache never uses would be
        # pure overhead — and, more importantly, would make construction perform I/O.
        self._space = DRMemorySpace(client, id=memory_space_id, user_id="", tenant_id="")
        normalized = key_prefix if key_prefix.endswith(":") else f"{key_prefix}:"
        self._key_prefix = normalized
        self._session_ids: dict[str, str] = {}

    def _logical_key(self, key: str) -> str:
        return f"{self._key_prefix}{CACHE_KIND}:{key}"

    def _cache_session_id(self, logical_key: str, session: _CacheEntrySession) -> None:
        self._session_ids[logical_key] = session.id

    def _invalidate_session_id(self, logical_key: str) -> None:
        self._session_ids.pop(logical_key, None)

    async def _resolve_session(self, logical_key: str) -> _CacheEntrySession | None:
        """Return the cache session, using a process-local session-id cache when possible."""
        if session_id := self._session_ids.get(logical_key):
            try:
                return await _CacheEntrySession.get(self._space, id=session_id)
            except DRMemoryNotFoundError:
                logger.debug(
                    "MemorySpace session cache miss for %s (session_id=%s)",
                    logical_key,
                    session_id,
                )
                self._invalidate_session_id(logical_key)

        try:
            session = await _CacheEntrySession.get(self._space, key=logical_key)
        except DRMemoryNotFoundError:
            return None
        self._cache_session_id(logical_key, session)
        return session

    async def get_value(self, key: str) -> str | None:
        """Return the stored JSON payload for *key*, or ``None`` when missing."""
        logical_key = self._logical_key(key)

        try:
            session = await self._resolve_session(logical_key)
            if session is None:
                return None
            events = await _CacheEntryEvent.last(session, n=1)
        except Exception:
            logger.exception("MemorySpace cache read failed for %s", logical_key)
            return None
        return events[0].content if events else None

    async def set_value(self, key: str, payload: str) -> None:
        """Upsert a JSON payload for *key*."""
        logical_key = self._logical_key(key)

        try:
            session = await self._resolve_session(logical_key)
            if session is None:
                session = await _CacheEntrySession.post(self._space, key=logical_key)
                self._cache_session_id(logical_key, session)
            events = await _CacheEntryEvent.last(session, n=1)
            if events:
                await events[0].patch(content=payload)
            else:
                await _CacheEntryEvent.post(session=session, content=payload, emitter_type="agent")
        except Exception:
            logger.exception("MemorySpace cache write failed for %s", logical_key)

    async def delete_value(self, key: str) -> None:
        """Remove a cached payload for *key* when present."""
        logical_key = self._logical_key(key)

        try:
            session = await self._resolve_session(logical_key)
            if session is not None:
                await session.delete()
        except Exception:
            logger.exception("MemorySpace cache delete failed for %s", logical_key)
        finally:
            self._invalidate_session_id(logical_key)
