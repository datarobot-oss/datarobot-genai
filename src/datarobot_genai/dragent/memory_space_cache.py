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

"""DataRobot MemorySpace-backed KV cache for the agent card registry L2 layer.

Used only on enclave workloads (``DR_WORKLOAD_EXTERNAL_URL_HOST`` +
``DR_WORKLOAD_EXTERNAL_URL_PREFIX`` with ``WORKLOAD_ID``). Other runtimes use
in-process L1 caching only.

Uses the agentic memory **Session API** (``datarobot.models.memory.Session``) — not the
mem0-compatible sub-route used by agent memory on the control hub.

Each provisioned memory space has a unique ``memory_space_id`` and platform-level
access control scoped to the deploying user or workload API token. Unlike shared
Redis, no per-deployment namespace or HMAC signing is required for this backend.

Each cache entry is one Memory Service session — found by a ``description``-filtered
lookup on the logical cache key — carrying a single event whose ``body["content"]``
is the opaque JSON payload. ``set_value`` patches that event in place instead of
appending; the cache only ever needs the current value, never a history.

Deliberately built on stable ``datarobot[core]`` rather than the Memory Service
light ORM (``DRMemorySpace`` / ``DRSession`` / ``DREvent`` /
``DRDeduplicationKey``) that BUZZOK-32180 standardizes this cache on: that ORM
ships only as ``application_utils.persistence`` in the pre-release
``datarobot-early-access`` distribution today, which we can't take as a
production dependency. Two things fall out of that constraint that a future
migration to the ORM should pick back up:

* **Session lookup by logical key** goes through ``Session.list(description=...)``
  (see ``_find_cache_session``) rather than an exact-match ``deduplicationKey``
  point lookup — the stable SDK's ``Session.list`` has no such filter, so a
  ``deduplication_key`` here only dedupes concurrent *creates*
  (``MemorySessionDeduplicationError``), not reads.
* **The DataRobot client is process-global** (configured once by
  ``configure_datarobot_memory_client``), not an object explicitly threaded into
  ``MemorySpaceKVCache`` the way ``DRMemoryServiceClient`` is. Stable
  ``datarobot.models.memory.Session`` always resolves credentials through
  ``datarobot.client.get_client()``; there's no per-instance client to inject
  without giving up the pooled, keep-alive ``requests.Session`` this module
  relies on (see the note on ``_STALE_CONNECTION_RETRIES`` below). Enclave
  gateways expose the memory Session API but not ``GET /version/``, so the
  client is built with ``RESTClientObject.from_config`` instead of ``dr.Client()``.

Once ``application_utils.persistence`` ships in a stable ``datarobot`` release,
this module should be replaced with that ORM the same way BUZZOK-32180 did.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from collections.abc import Callable
from typing import Any
from typing import TypeVar

import requests
from datarobot.errors import MemorySessionDeduplicationError
from datarobot.errors import MemorySpaceDeduplicationError
from datarobot.models.memory import MemorySpace
from datarobot.models.memory import Session

from datarobot_genai.core.runtime import get_workload_id
from datarobot_genai.core.runtime import is_workload_mode
from datarobot_genai.dragent.deployment_urls import resolve_external_workload_api_endpoint

logger = logging.getLogger(__name__)

T = TypeVar("T")

# `dr.Client()` keeps a single process-lifetime `requests.Session` (and pooled
# `HTTPAdapter`) shared by every DataRobot API call -- see
# `datarobot.rest.RESTClientObject`. Memory-space cache calls are infrequent
# (on-demand L1-cache misses, plus the agent card registry's 30-minute
# background refresh), so a pooled keep-alive connection can sit idle longer
# than the server side's (or an intervening proxy's) idle-connection timeout.
# The next reuse then fails with a `ConnectionError` wrapping
# `RemoteDisconnected`/`ProtocolError` ("Remote end closed connection without
# response") -- not the `ConnectionResetError` that the DataRobot client's own
# `handle_connection_reset` retry wrapper looks for, so it is never retried
# there and surfaces on every call that lands on a stale connection.
#
# Retrying here is a cheap, safe mitigation: the failed attempt evicts the
# dead connection from the pool, so the retry opens a fresh one.
_STALE_CONNECTION_RETRIES = 2


def _call_with_stale_connection_retry(func: Callable[[], T], *, op: str) -> T:
    """Call *func*, retrying on a stale pooled-connection ``ConnectionError``.

    Only ``requests.exceptions.ConnectionError`` (e.g. a stale keep-alive
    connection closed by the remote end) is retried; any other exception --
    including a real API error -- propagates immediately.
    """
    for attempt in range(_STALE_CONNECTION_RETRIES + 1):
        try:
            return func()
        except requests.exceptions.ConnectionError as exc:
            if attempt >= _STALE_CONNECTION_RETRIES:
                raise
            logger.debug(
                "MemorySpace %s hit a connection error (attempt %d/%d), retrying: %s",
                op,
                attempt + 1,
                _STALE_CONNECTION_RETRIES + 1,
                exc,
            )
    raise AssertionError("unreachable")  # pragma: no cover


# Stable 24-hex participant id (BSON ObjectId length) for cache sessions.
DRAGENT_CACHE_PARTICIPANT_ID = hashlib.sha256(b"datarobot-genai:dragent-cache").hexdigest()[:24]

CACHE_EVENT_TYPE = "status"
DEDUPLICATION_KEY_LENGTH = 64
CACHE_KIND = "agent_card"

_REGISTRY_CACHE_SPACE_DEDUP_PREFIX = "dragent:agent-card-registry"


class _ProvisionedRegistryCacheSpaceState:
    """Mutable container for the provisioned registry L2 MemorySpace ID."""

    space_id: str | None = None


def is_enclave_l2_workload() -> bool:
    """Return True when this process can use the registry L2 MemorySpace cache."""
    return resolve_external_workload_api_endpoint() is not None and is_workload_mode()


def registry_cache_deduplication_key(workload_id: str) -> str:
    """Return the stable deduplication key for an enclave workload's registry L2 space."""
    return f"{_REGISTRY_CACHE_SPACE_DEDUP_PREFIX}:workload:{workload_id}"


def try_resolve_memory_space_id() -> str | None:
    """Return the registry L2 MemorySpace ID on enclave workloads, else ``None``.

    Creates or adopts the space on first call. Uses ``WORKLOAD_ID`` as a
    ``deduplication_key`` so replicas share one space. No-op when not on an enclave
    workload, when credentials are unavailable, or after the first successful
    provision in this process.

    This is the agent card registry L2 cache, not agent memory
    (``AGENT_MEMORY_SPACE_ID``). Agent memory is provisioned on the control hub
    (Pulumi / ``task deploy-dev``) and the Mem0 client talks to that same host.
    Other runtimes use in-process L1 caching only.
    """
    if _ProvisionedRegistryCacheSpaceState.space_id is not None:
        return _ProvisionedRegistryCacheSpaceState.space_id

    if not is_enclave_l2_workload():
        return None

    workload_id = get_workload_id()
    assert workload_id is not None  # guaranteed by is_enclave_l2_workload()
    deduplication_key = registry_cache_deduplication_key(workload_id)

    if not try_configure_datarobot_memory_client():
        return None

    description = "Agent card registry L2 cache"

    def _create() -> MemorySpace:
        try:
            return _call_with_stale_connection_retry(
                lambda: MemorySpace.create(
                    description=description,
                    deduplication_key=deduplication_key,
                ),
                op="provision_registry_cache_memory_space",
            )
        except MemorySpaceDeduplicationError as exc:
            if exc.existing_memory_space_id is None:
                raise
            existing_space_id = exc.existing_memory_space_id
            return _call_with_stale_connection_retry(
                lambda: MemorySpace.get(existing_space_id),
                op="get_registry_cache_memory_space",
            )

    try:
        space = _create()
    except Exception:
        logger.exception(
            "Failed to provision agent card registry L2 MemorySpace (dedup_key=%s)",
            deduplication_key,
        )
        return None

    _ProvisionedRegistryCacheSpaceState.space_id = space.id
    logger.info(
        "Provisioned agent card registry L2 MemorySpace %s (dedup_key=%s)",
        space.id,
        deduplication_key,
    )
    return space.id


def try_configure_datarobot_memory_client(
    *,
    api_token: str | None = None,
) -> bool:
    """Configure the enclave memory client when possible; return ``False`` on failure."""
    try:
        configure_datarobot_memory_client(api_token=api_token)
    except Exception as exc:
        logger.debug("MemorySpace client unavailable: %s", exc)
        return False
    return True


def configure_datarobot_memory_client(
    *,
    api_token: str | None = None,
) -> None:
    """Configure the process-global DataRobot client for enclave memory Session API calls.

    Uses the enclave API gateway (``DR_WORKLOAD_EXTERNAL_URL_HOST`` +
    ``DR_WORKLOAD_EXTERNAL_URL_PREFIX`` → ``{host}/api/v2``). Skips ``dr.Client()``'s
    ``GET /version/`` probe because enclave gateways expose the memory API only.
    """
    enclave_endpoint = resolve_external_workload_api_endpoint()
    if enclave_endpoint is None:
        raise ValueError(
            "MemorySpace cache backends require an enclave API gateway "
            "(DR_WORKLOAD_EXTERNAL_URL_HOST and DR_WORKLOAD_EXTERNAL_URL_PREFIX)."
        )
    token = api_token or os.getenv("DATAROBOT_API_TOKEN")
    if not token:
        raise ValueError("DATAROBOT_API_TOKEN is required when using memory_space cache backends.")
    logger.info(
        "Configuring DataRobot memory client for enclave gateway %s "
        "(skipping dr.Client /version/ compatibility check).",
        enclave_endpoint,
    )
    from datarobot.client import set_client
    from datarobot.config import create_drconfig
    from datarobot.rest import RESTClientObject

    drconfig = create_drconfig(token=token, endpoint=enclave_endpoint.rstrip("/"))
    set_client(RESTClientObject.from_config(drconfig))


def _cache_deduplication_key(logical_key: str) -> str:
    raw = "dragent-cache\0" + logical_key
    return hashlib.sha256(raw.encode()).hexdigest()[:DEDUPLICATION_KEY_LENGTH]


def _cache_session_description(logical_key: str) -> str:
    return f"/dragent/cache/{logical_key}"


def _cache_session_metadata(logical_key: str) -> dict[str, Any]:
    return {
        "dragent_cache": True,
        "cache_key": logical_key,
        "cache_kind": CACHE_KIND,
    }


def _create_cache_session(
    memory_space_id: str,
    *,
    logical_key: str,
) -> Session:
    """Create a cache session, adopting an existing one on deduplication collision."""
    try:
        return _call_with_stale_connection_retry(
            lambda: Session.create(
                memory_space_id,
                [DRAGENT_CACHE_PARTICIPANT_ID],
                metadata=_cache_session_metadata(logical_key),
                description=_cache_session_description(logical_key),
                deduplication_key=_cache_deduplication_key(logical_key),
            ),
            op="create_cache_session",
        )
    except MemorySessionDeduplicationError as exc:
        if exc.existing_session_id is None:
            raise
        existing_session_id = exc.existing_session_id
        return _call_with_stale_connection_retry(
            lambda: Session.get(memory_space_id, existing_session_id),
            op="get_cache_session",
        )


def _find_cache_session(memory_space_id: str, logical_key: str) -> Session | None:
    description = _cache_session_description(logical_key)
    sessions = _call_with_stale_connection_retry(
        lambda: Session.list(
            memory_space_id,
            participants=[DRAGENT_CACHE_PARTICIPANT_ID],
            description=description,
            limit=1,
        ),
        op="find_cache_session",
    )
    return sessions[0] if sessions else None


def _read_payload(session: Session) -> str | None:
    """Return the cache entry's payload, or ``None`` when the session has no event yet.

    The payload is the event's ``content`` directly -- no cache-specific envelope
    or schema version -- matching how BUZZOK-32180's ``DREvent.content`` is read.
    """
    events = _call_with_stale_connection_retry(lambda: session.events(last_n=1), op="read_payload")
    if not events:
        return None
    body = events[0].body
    if not body:
        return None
    value = body.get("content")
    return str(value) if value is not None else None


def _write_payload(session: Session, payload: str) -> None:
    # The Memory Sessions Events API requires a top-level "content" field on every
    # event body (schema validation: `body.content` is required), so the payload
    # is stored there directly -- no extra wrapper field is needed.
    body = {"content": payload}
    events = _call_with_stale_connection_retry(
        lambda: session.events(last_n=1), op="write_payload_read"
    )
    if events and events[0].sequence_id is not None:
        sequence_id = events[0].sequence_id
        _call_with_stale_connection_retry(
            lambda: session.update_event(sequence_id, body=body),
            op="write_payload_update",
        )
        return
    _call_with_stale_connection_retry(
        lambda: session.post_event(
            body=body,
            emitter={"type": "agent"},
            event_type=CACHE_EVENT_TYPE,
        ),
        op="write_payload_post",
    )


class MemorySpaceKVCache:
    """Store opaque JSON payloads in a DataRobot MemorySpace by logical key."""

    def __init__(self, *, memory_space_id: str, key_prefix: str = "dragent:") -> None:
        self._memory_space_id = memory_space_id
        normalized = key_prefix if key_prefix.endswith(":") else f"{key_prefix}:"
        self._key_prefix = normalized
        self._session_ids: dict[str, str] = {}

    def _logical_key(self, key: str) -> str:
        return f"{self._key_prefix}{CACHE_KIND}:{key}"

    def _cache_session_id(self, logical_key: str, session: Session) -> None:
        self._session_ids[logical_key] = session.id

    def _invalidate_session_id(self, logical_key: str) -> None:
        self._session_ids.pop(logical_key, None)

    def _resolve_session(self, logical_key: str) -> Session | None:
        """Return the cache session, using a process-local session-id cache when possible."""
        if session_id := self._session_ids.get(logical_key):
            try:
                return _call_with_stale_connection_retry(
                    lambda: Session.get(self._memory_space_id, session_id),
                    op="resolve_session_get",
                )
            except Exception:
                logger.debug(
                    "MemorySpace session cache miss for %s (session_id=%s)",
                    logical_key,
                    session_id,
                )
                self._invalidate_session_id(logical_key)

        session = _find_cache_session(self._memory_space_id, logical_key)
        if session is not None:
            self._cache_session_id(logical_key, session)
        return session

    async def get_value(self, key: str) -> str | None:
        """Return the stored JSON payload for *key*, or ``None`` when missing."""
        logical_key = self._logical_key(key)

        def _get() -> str | None:
            session = self._resolve_session(logical_key)
            if session is None:
                return None
            return _read_payload(session)

        try:
            return await asyncio.to_thread(_get)
        except Exception:
            logger.exception("MemorySpace cache read failed for %s", logical_key)
            return None

    async def set_value(self, key: str, payload: str) -> None:
        """Upsert a JSON payload for *key*."""
        logical_key = self._logical_key(key)

        def _set() -> None:
            session = self._resolve_session(logical_key)
            if session is None:
                session = _create_cache_session(
                    self._memory_space_id,
                    logical_key=logical_key,
                )
                self._cache_session_id(logical_key, session)
            _write_payload(session, payload)

        try:
            await asyncio.to_thread(_set)
        except Exception:
            logger.exception("MemorySpace cache write failed for %s", logical_key)

    async def delete_value(self, key: str) -> None:
        """Remove a cached payload for *key* when present."""
        logical_key = self._logical_key(key)

        def _delete() -> None:
            session = self._resolve_session(logical_key)
            if session is not None:
                _call_with_stale_connection_retry(session.delete, op="delete_session")
            self._invalidate_session_id(logical_key)

        try:
            await asyncio.to_thread(_delete)
        except Exception:
            logger.exception("MemorySpace cache delete failed for %s", logical_key)
