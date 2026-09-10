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

Uses the Memory Service light ORM from ``datarobot.application_utils.persistence``
(``DRMemorySpace`` / ``DRSession`` / ``DREvent`` / ``DRMemoryServiceClient``).
Each cache entry is one session located by a stable ``DRDeduplicationKey`` hash
and carrying a single ``status`` event whose ``content`` is the opaque JSON
payload. ``set_value`` patches that event in place instead of appending.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Coroutine
from typing import Annotated
from typing import Any
from typing import ClassVar
from typing import TypeVar
from typing import cast

from datarobot.application_utils.persistence import DRDeduplicationKey
from datarobot.application_utils.persistence import DREvent
from datarobot.application_utils.persistence import DRMemoryServiceClient
from datarobot.application_utils.persistence import DRMemorySpace
from datarobot.application_utils.persistence import DRRangeKey
from datarobot.application_utils.persistence import DRSession
from datarobot.application_utils.persistence.exceptions import DRMemoryNotFoundError
from datarobot.application_utils.persistence.exceptions import DRMemoryUnavailableError

from datarobot_genai.core.runtime import get_workload_id
from datarobot_genai.core.runtime import is_workload_mode
from datarobot_genai.dragent.deployment_urls import resolve_external_workload_api_endpoint

logger = logging.getLogger(__name__)

T = TypeVar("T")

_TRANSPORT_RETRIES = 2

# Stable 24-hex participant id (BSON ObjectId length) for cache sessions.
DRAGENT_CACHE_PARTICIPANT_ID = hashlib.sha256(b"datarobot-genai:dragent-cache").hexdigest()[:24]

CACHE_EVENT_TYPE = "status"
DEDUPLICATION_KEY_LENGTH = 64
CACHE_KIND = "agent_card"

_REGISTRY_CACHE_SPACE_DEDUP_PREFIX = "dragent:agent-card-registry"


class AgentCardCacheSession(DRSession):
    """Memory Service session model for one agent-card registry L2 cache entry."""

    __description_prefix__ = "dragent_cache"
    __lifecycle_strategies__: ClassVar[list[dict[str, Any]]] = []

    cache_kind: Annotated[str, DRRangeKey]
    dedup_key: Annotated[str, DRDeduplicationKey]
    logical_key: str
    dragent_cache: bool = True


class AgentCardCacheEvent(DREvent, session=AgentCardCacheSession):  # type: ignore[call-arg]
    """Single-value status event carrying the opaque JSON cache payload."""

    __event_type__ = CACHE_EVENT_TYPE


class _ProvisionedRegistryCacheSpaceState:
    """Mutable container for the provisioned registry L2 MemorySpace ID."""

    space_id: str | None = None


class _MemoryClientState:
    """Process-global Memory Service ORM client."""

    client: DRMemoryServiceClient | None = None


def is_enclave_l2_workload() -> bool:
    """Return True when this process can use the registry L2 MemorySpace cache."""
    return resolve_external_workload_api_endpoint() is not None and is_workload_mode()


def registry_cache_deduplication_key(workload_id: str) -> str:
    """Return the stable deduplication key for an enclave workload's registry L2 space."""
    return f"{_REGISTRY_CACHE_SPACE_DEDUP_PREFIX}:workload:{workload_id}"


def get_memory_service_client() -> DRMemoryServiceClient | None:
    """Return the configured Memory Service client, if any."""
    return _MemoryClientState.client


def _run_async(coro_factory: Callable[[], Coroutine[Any, Any, T]]) -> T:
    """Run *coro_factory* from a synchronous caller when no event loop is running.

    The factory is invoked only after confirming no loop is running, so a
    coroutine is never left unawaited.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro_factory())
    raise RuntimeError("memory_space_cache sync helper called from a running event loop")


async def _call_with_transport_retry(
    coro_factory: Callable[[], Awaitable[T]],
    *,
    op: str,
) -> T:
    """Await *coro_factory*, retrying on transient transport failures."""
    for attempt in range(_TRANSPORT_RETRIES + 1):
        try:
            return await coro_factory()
        except DRMemoryUnavailableError as exc:
            if attempt >= _TRANSPORT_RETRIES:
                raise
            logger.debug(
                "MemorySpace %s hit a transport error (attempt %d/%d), retrying: %s",
                op,
                attempt + 1,
                _TRANSPORT_RETRIES + 1,
                exc,
            )
    raise AssertionError("unreachable")  # pragma: no cover


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

    Import-time bootstrap uses this synchronous helper. Lifespan warmup and
    :func:`~datarobot_genai.dragent.agent_card_registry.get_default_registry`
    must call :func:`try_resolve_memory_space_id_async` instead — ``asyncio.run``
    cannot nest inside an already running event loop.
    """
    try:
        return _run_async(_try_resolve_memory_space_id)
    except RuntimeError as exc:
        if "running event loop" not in str(exc):
            raise
        logger.debug(
            "try_resolve_memory_space_id called from a running event loop; "
            "use try_resolve_memory_space_id_async"
        )
        return None


async def try_resolve_memory_space_id_async() -> str | None:
    """Async counterpart of :func:`try_resolve_memory_space_id`.

    Use this from a running event loop (lifespan warmup, ``get_default_registry``).
    """
    return await _try_resolve_memory_space_id()


async def _try_resolve_memory_space_id() -> str | None:
    if _ProvisionedRegistryCacheSpaceState.space_id is not None:
        return _ProvisionedRegistryCacheSpaceState.space_id

    if not is_enclave_l2_workload():
        return None

    workload_id = get_workload_id()
    assert workload_id is not None  # guaranteed by is_enclave_l2_workload()
    deduplication_key = registry_cache_deduplication_key(workload_id)

    if not try_configure_datarobot_memory_client():
        return None

    try:
        space_id = await _provision_registry_cache_memory_space(deduplication_key)
    except Exception:
        logger.exception(
            "Failed to provision agent card registry L2 MemorySpace (dedup_key=%s)",
            deduplication_key,
        )
        return None

    _ProvisionedRegistryCacheSpaceState.space_id = space_id
    logger.info(
        "Provisioned agent card registry L2 MemorySpace %s (dedup_key=%s)",
        space_id,
        deduplication_key,
    )
    return space_id


async def _provision_registry_cache_memory_space(deduplication_key: str) -> str:
    client = _require_memory_client()
    space = await _call_with_transport_retry(
        lambda: DRMemorySpace.post(
            client,
            description="Agent card registry L2 cache",
            deduplication_key=deduplication_key,
        ),
        op="provision_registry_cache_memory_space",
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
    """Configure the process-global Memory Service ORM client for enclave workloads.

    Uses the enclave API gateway (``DR_WORKLOAD_EXTERNAL_URL_HOST`` +
    ``DR_WORKLOAD_EXTERNAL_URL_PREFIX`` → ``{host}/api/v2``). The ORM client talks
    directly to ``{endpoint}/memory`` and does not require ``dr.Client()``'s
    ``GET /version/`` probe.
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
        "Configuring DataRobot memory client for enclave gateway %s.",
        enclave_endpoint,
    )
    _MemoryClientState.client = DRMemoryServiceClient(
        endpoint=enclave_endpoint.rstrip("/"),
        api_token=token,
    )


def _require_memory_client() -> DRMemoryServiceClient:
    client = _MemoryClientState.client
    if client is None:
        raise RuntimeError(
            "MemorySpace client is not configured; call configure_datarobot_memory_client first."
        )
    return client


def _cache_deduplication_key(logical_key: str) -> str:
    raw = "dragent-cache\0" + logical_key
    return hashlib.sha256(raw.encode()).hexdigest()[:DEDUPLICATION_KEY_LENGTH]


class MemorySpaceKVCache:
    """Store opaque JSON payloads in a DataRobot MemorySpace by logical key."""

    def __init__(
        self,
        *,
        memory_space_id: str,
        key_prefix: str = "dragent:",
        client: DRMemoryServiceClient | None = None,
    ) -> None:
        self._memory_space_id = memory_space_id
        normalized = key_prefix if key_prefix.endswith(":") else f"{key_prefix}:"
        self._key_prefix = normalized
        self._client = client
        self._space: DRMemorySpace | None = None
        self._sessions: dict[str, AgentCardCacheSession] = {}

    def _logical_key(self, key: str) -> str:
        return f"{self._key_prefix}{CACHE_KIND}:{key}"

    def _client_or_global(self) -> DRMemoryServiceClient:
        return self._client or _require_memory_client()

    async def _resolve_space(self) -> DRMemorySpace:
        if self._space is None:
            client = self._client_or_global()
            self._space = await _call_with_transport_retry(
                lambda: DRMemorySpace.get(client, self._memory_space_id),
                op="resolve_space",
            )
        return self._space

    def _cache_session(self, logical_key: str, session: AgentCardCacheSession) -> None:
        self._sessions[logical_key] = session

    def _invalidate_session(self, logical_key: str) -> None:
        self._sessions.pop(logical_key, None)

    async def _resolve_session(self, logical_key: str) -> AgentCardCacheSession | None:
        if session := self._sessions.get(logical_key):
            return session

        space = await self._resolve_space()
        dedup_key = _cache_deduplication_key(logical_key)
        try:
            resolved = cast(
                AgentCardCacheSession,
                await _call_with_transport_retry(
                    lambda: AgentCardCacheSession.get(space, dedup_key=dedup_key),
                    op="resolve_session",
                ),
            )
        except DRMemoryNotFoundError:
            return None
        except DRMemoryUnavailableError:
            raise
        except Exception:
            logger.debug("MemorySpace session lookup failed for %s", logical_key)
            return None

        self._cache_session(logical_key, resolved)
        return resolved

    async def _read_payload(self, session: AgentCardCacheSession) -> str | None:
        events = await _call_with_transport_retry(
            lambda: AgentCardCacheEvent.last(session, n=1, type=CACHE_EVENT_TYPE),
            op="read_payload",
        )
        if not events:
            return None
        return events[0].content

    async def _write_payload(self, session: AgentCardCacheSession, payload: str) -> None:
        events = await _call_with_transport_retry(
            lambda: AgentCardCacheEvent.last(session, n=1, type=CACHE_EVENT_TYPE),
            op="write_payload_read",
        )
        if events:
            await _call_with_transport_retry(
                lambda: events[0].patch(content=payload),
                op="write_payload_update",
            )
            return
        await _call_with_transport_retry(
            lambda: AgentCardCacheEvent.post(
                session,
                content=payload,
                emitter_type="agent",
            ),
            op="write_payload_post",
        )

    async def get_value(self, key: str) -> str | None:
        """Return the stored JSON payload for *key*, or ``None`` when missing."""
        logical_key = self._logical_key(key)
        try:
            session = await self._resolve_session(logical_key)
            if session is None:
                return None
            return await self._read_payload(session)
        except DRMemoryUnavailableError:
            logger.exception("MemorySpace cache read failed for %s", logical_key)
            return None
        except Exception:
            logger.exception("MemorySpace cache read failed for %s", logical_key)
            return None

    async def set_value(self, key: str, payload: str) -> None:
        """Upsert a JSON payload for *key*."""
        logical_key = self._logical_key(key)
        dedup_key = _cache_deduplication_key(logical_key)

        try:
            space = await self._resolve_space()
            session = await self._resolve_session(logical_key)
            if session is None:
                session = cast(
                    AgentCardCacheSession,
                    await _call_with_transport_retry(
                        lambda: AgentCardCacheSession.post(
                            space,
                            cache_kind=CACHE_KIND,
                            dedup_key=dedup_key,
                            logical_key=logical_key,
                            participants=[DRAGENT_CACHE_PARTICIPANT_ID],
                        ),
                        op="create_cache_session",
                    ),
                )
                self._cache_session(logical_key, session)
            await self._write_payload(session, payload)
        except Exception:
            logger.exception("MemorySpace cache write failed for %s", logical_key)

    async def delete_value(self, key: str) -> None:
        """Remove a cached payload for *key* when present."""
        logical_key = self._logical_key(key)
        try:
            session = await self._resolve_session(logical_key)
            if session is not None:
                await _call_with_transport_retry(session.delete, op="delete_session")
            self._invalidate_session(logical_key)
        except Exception:
            logger.exception("MemorySpace cache delete failed for %s", logical_key)
