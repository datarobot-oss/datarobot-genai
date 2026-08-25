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

"""Pluggable cache backends for the central agent card registry."""

from __future__ import annotations

import logging
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from typing import TYPE_CHECKING
from typing import Literal
from typing import Protocol

from a2a.types import AgentCard
from pydantic import BaseModel
from pydantic import Field

from datarobot_genai.dragent.memory_space_cache import MemorySpaceKVCache
from datarobot_genai.dragent.memory_space_cache import try_configure_datarobot_memory_client
from datarobot_genai.dragent.memory_space_cache import try_resolve_memory_space_id

if TYPE_CHECKING:
    from datarobot_genai.dragent.agent_card_registry import AgentCardRegistryConfig

logger = logging.getLogger(__name__)

LookupKeyType = Literal["deployment", "external"]
_DEFAULT_KEY_PREFIX = "dragent:"


class AgentCardCacheRecord(BaseModel):
    """Serialized agent card cache entry shared across backends."""

    version: int = 1
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    card: AgentCard
    source: str = "registry"
    deployment_id: str | None = None
    external_id: str | None = None

    def age_seconds(self) -> float:
        """Return the entry age in seconds (wall clock, safe across processes)."""
        return max(0.0, (datetime.now(UTC) - self.fetched_at).total_seconds())

    def is_fresh(self, cache_ttl: int) -> bool:
        """Return *True* if this entry is within the soft TTL."""
        if cache_ttl == 0:
            return False
        return self.age_seconds() < cache_ttl

    def is_within_staleness(self, max_staleness_seconds: int) -> bool:
        """Return *True* if this entry may be served under stale-if-error."""
        if max_staleness_seconds == 0:
            return False
        return self.age_seconds() <= max_staleness_seconds


def build_cache_record(
    card: AgentCard,
    *,
    lookup_key: str,
    key_type: LookupKeyType,
    deployment_id: str | None = None,
    external_id: str | None = None,
) -> AgentCardCacheRecord:
    """Build a cache record for *lookup_key* with optional registry ID metadata."""
    resolved_dep = (
        deployment_id
        if deployment_id is not None
        else (lookup_key if key_type == "deployment" else None)
    )
    resolved_ext = (
        external_id if external_id is not None else (lookup_key if key_type == "external" else None)
    )
    return AgentCardCacheRecord(
        card=card,
        deployment_id=resolved_dep,
        external_id=resolved_ext,
    )


class AgentCardCacheBackend(Protocol):
    """Async cache backend for agent card registry entries."""

    async def get_fresh(self, lookup_key: str, *, cache_ttl: int) -> AgentCardCacheRecord | None:
        """Return a cached record when within the soft TTL."""

    async def get_stale(
        self,
        lookup_key: str,
        *,
        max_staleness_seconds: int,
    ) -> AgentCardCacheRecord | None:
        """Return a cached record within the hard staleness bound."""

    async def store(
        self,
        cards: dict[str, AgentCard],
        *,
        key_types: dict[str, LookupKeyType],
    ) -> None:
        """Persist one or more cards keyed by lookup ID."""

    async def evict(
        self,
        lookup_key: str,
        *,
        key_type: LookupKeyType | None = None,
    ) -> None:
        """Remove a cached entry so stale-if-error cannot resurrect it."""


class MemoryAgentCardCacheBackend:
    """In-process dict cache (L1)."""

    def __init__(self) -> None:
        self._entries: dict[str, AgentCardCacheRecord] = {}

    async def get_fresh(self, lookup_key: str, *, cache_ttl: int) -> AgentCardCacheRecord | None:
        record = self._entries.get(lookup_key)
        if record is None or not record.is_fresh(cache_ttl):
            return None
        return record

    async def get_stale(
        self,
        lookup_key: str,
        *,
        max_staleness_seconds: int,
    ) -> AgentCardCacheRecord | None:
        record = self._entries.get(lookup_key)
        if record is None or not record.is_within_staleness(max_staleness_seconds):
            return None
        return record

    async def store(
        self,
        cards: dict[str, AgentCard],
        *,
        key_types: dict[str, LookupKeyType],
    ) -> None:
        for lookup_key, card in cards.items():
            key_type = key_types.get(lookup_key, "deployment")
            self._entries[lookup_key] = build_cache_record(
                card, lookup_key=lookup_key, key_type=key_type
            )

    async def evict(
        self,
        lookup_key: str,
        *,
        key_type: LookupKeyType | None = None,
    ) -> None:
        self._entries.pop(lookup_key, None)

    def age_entry_for_test(self, lookup_key: str, seconds: float) -> None:
        """Shift *lookup_key* fetch time backward (tests only)."""
        record = self._entries.get(lookup_key)
        if record is None:
            return
        record.fetched_at -= timedelta(seconds=seconds)


class MemorySpaceAgentCardCacheBackend:
    """Shared DataRobot MemorySpace L2 cache using JSON payloads."""

    def __init__(
        self,
        *,
        kv_cache: MemorySpaceKVCache,
    ) -> None:
        self._kv = kv_cache

    def _storage_keys_for_record(self, record: AgentCardCacheRecord) -> list[str]:
        keys: list[str] = []
        if record.deployment_id:
            keys.append(f"deployment:{record.deployment_id}")
        if record.external_id:
            keys.append(f"external:{record.external_id}")
        return keys

    def _storage_keys_for_lookup(
        self,
        lookup_key: str,
        *,
        key_type: LookupKeyType | None = None,
    ) -> list[str]:
        if key_type == "deployment":
            return [f"deployment:{lookup_key}"]
        if key_type == "external":
            return [f"external:{lookup_key}"]
        return [f"deployment:{lookup_key}", f"external:{lookup_key}"]

    async def _get_record(self, storage_key: str) -> AgentCardCacheRecord | None:
        payload = await self._kv.get_value(storage_key, kind="agent_card")
        if payload is None:
            return None
        try:
            return AgentCardCacheRecord.model_validate_json(payload)
        except Exception:
            logger.warning("Ignoring invalid MemorySpace agent card payload for %s", storage_key)
            return None

    async def get_fresh(self, lookup_key: str, *, cache_ttl: int) -> AgentCardCacheRecord | None:
        for storage_key in self._storage_keys_for_lookup(lookup_key):
            record = await self._get_record(storage_key)
            if record is not None and record.is_fresh(cache_ttl):
                return record
        return None

    async def get_stale(
        self,
        lookup_key: str,
        *,
        max_staleness_seconds: int,
    ) -> AgentCardCacheRecord | None:
        for storage_key in self._storage_keys_for_lookup(lookup_key):
            record = await self._get_record(storage_key)
            if record is not None and record.is_within_staleness(max_staleness_seconds):
                return record
        return None

    async def store(
        self,
        cards: dict[str, AgentCard],
        *,
        key_types: dict[str, LookupKeyType],
    ) -> None:
        for lookup_key, card in cards.items():
            key_type = key_types.get(lookup_key, "deployment")
            record = build_cache_record(card, lookup_key=lookup_key, key_type=key_type)
            payload = record.model_dump_json()
            for storage_key in self._storage_keys_for_record(record):
                await self._kv.set_value(storage_key, payload, kind="agent_card")

    async def evict(
        self,
        lookup_key: str,
        *,
        key_type: LookupKeyType | None = None,
    ) -> None:
        for storage_key in self._storage_keys_for_lookup(lookup_key, key_type=key_type):
            await self._kv.delete_value(storage_key, kind="agent_card")


class LayeredAgentCardCacheBackend:
    """L1 memory read-through / write-through over an L2 backend."""

    def __init__(self, l1: MemoryAgentCardCacheBackend, l2: AgentCardCacheBackend) -> None:
        self._l1 = l1
        self._l2 = l2

    async def get_fresh(self, lookup_key: str, *, cache_ttl: int) -> AgentCardCacheRecord | None:
        if record := await self._l1.get_fresh(lookup_key, cache_ttl=cache_ttl):
            return record
        if record := await self._l2.get_fresh(lookup_key, cache_ttl=cache_ttl):
            await self._l1.store(
                {lookup_key: record.card},
                key_types={lookup_key: _infer_key_type(record, lookup_key)},
            )
            return record
        return None

    async def get_stale(
        self,
        lookup_key: str,
        *,
        max_staleness_seconds: int,
    ) -> AgentCardCacheRecord | None:
        if record := await self._l1.get_stale(
            lookup_key,
            max_staleness_seconds=max_staleness_seconds,
        ):
            return record
        if record := await self._l2.get_stale(
            lookup_key,
            max_staleness_seconds=max_staleness_seconds,
        ):
            await self._l1.store(
                {lookup_key: record.card},
                key_types={lookup_key: _infer_key_type(record, lookup_key)},
            )
            return record
        return None

    async def store(
        self,
        cards: dict[str, AgentCard],
        *,
        key_types: dict[str, LookupKeyType],
    ) -> None:
        await self._l1.store(cards, key_types=key_types)
        await self._l2.store(cards, key_types=key_types)

    async def evict(
        self,
        lookup_key: str,
        *,
        key_type: LookupKeyType | None = None,
    ) -> None:
        await self._l1.evict(lookup_key, key_type=key_type)
        await self._l2.evict(lookup_key, key_type=key_type)

    @property
    def memory(self) -> MemoryAgentCardCacheBackend:
        """Expose the L1 backend (tests)."""
        return self._l1


def _infer_key_type(record: AgentCardCacheRecord, lookup_key: str) -> LookupKeyType:
    if record.deployment_id == lookup_key:
        return "deployment"
    if record.external_id == lookup_key:
        return "external"
    return "deployment"


def create_agent_card_cache_backend(
    config: AgentCardRegistryConfig,
) -> AgentCardCacheBackend:
    """Instantiate the agent card cache backend (L1, plus MemorySpace L2 when available)."""
    l1 = MemoryAgentCardCacheBackend()
    memory_space_id = try_resolve_memory_space_id(config.agent_card_registry_memory_space_id)
    if memory_space_id is None:
        logger.debug("Agent card registry cache: L1 only (no MemorySpace ID configured)")
        return l1

    if not try_configure_datarobot_memory_client():
        logger.warning(
            "Agent card registry cache: L1 only — MemorySpace L2 unavailable "
            "(set DATAROBOT_API_TOKEN and DATAROBOT_ENDPOINT)"
        )
        return l1

    kv_cache = MemorySpaceKVCache(
        memory_space_id=memory_space_id,
        key_prefix=config.agent_card_registry_key_prefix,
    )
    logger.debug(
        "Agent card registry cache: L1 + MemorySpace L2 (space_id=%s)",
        memory_space_id,
    )
    return LayeredAgentCardCacheBackend(
        l1,
        MemorySpaceAgentCardCacheBackend(kv_cache=kv_cache),
    )
