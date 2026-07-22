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
import time
from datetime import UTC
from datetime import datetime
from typing import TYPE_CHECKING
from typing import Literal
from typing import Protocol

from a2a.types import AgentCard
from pydantic import BaseModel
from pydantic import Field

if TYPE_CHECKING:
    from datarobot_genai.dragent.agent_card_registry import AgentCardRegistryConfig

logger = logging.getLogger(__name__)

LookupKeyType = Literal["dep", "ext"]
_DEFAULT_REDIS_PREFIX = "dragent:"


class AgentCardCacheRecord(BaseModel):
    """Serialized agent card cache entry shared across backends."""

    version: int = 1
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    fetched_at_mono: float = Field(default_factory=time.monotonic)
    card: AgentCard
    source: str = "registry"
    deployment_id: str | None = None
    external_id: str | None = None

    def age_seconds(self) -> float:
        """Return the entry age in seconds (monotonic clock)."""
        return time.monotonic() - self.fetched_at_mono

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
        deployment_id if deployment_id is not None else (lookup_key if key_type == "dep" else None)
    )
    resolved_ext = (
        external_id if external_id is not None else (lookup_key if key_type == "ext" else None)
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
            key_type = key_types.get(lookup_key, "dep")
            self._entries[lookup_key] = build_cache_record(
                card, lookup_key=lookup_key, key_type=key_type
            )

    def age_entry_for_test(self, lookup_key: str, seconds: float) -> None:
        """Shift *lookup_key* fetch time backward (tests only)."""
        record = self._entries.get(lookup_key)
        if record is None:
            return
        record.fetched_at_mono -= seconds


class RedisAgentCardCacheBackend:
    """Shared Redis L2 cache using JSON STRING values."""

    def __init__(
        self,
        *,
        redis_url: str,
        key_prefix: str = _DEFAULT_REDIS_PREFIX,
        max_staleness_seconds: int,
    ) -> None:
        try:
            import redis.asyncio as redis
        except ImportError as exc:
            raise ImportError(
                "Redis agent card registry backend requires the 'redis' package. "
                "Install with: pip install 'datarobot-genai[dragent]'"
            ) from exc

        self._redis = redis.from_url(redis_url, decode_responses=True)
        self._key_prefix = key_prefix
        self._max_staleness_seconds = max_staleness_seconds

    def _redis_keys_for_record(self, record: AgentCardCacheRecord) -> list[str]:
        keys: list[str] = []
        if record.deployment_id:
            keys.append(f"{self._key_prefix}agent_card:dep:{record.deployment_id}")
        if record.external_id:
            keys.append(f"{self._key_prefix}agent_card:ext:{record.external_id}")
        return keys

    def _redis_keys_for_lookup(
        self,
        lookup_key: str,
        *,
        key_type: LookupKeyType | None = None,
    ) -> list[str]:
        if key_type == "dep":
            return [f"{self._key_prefix}agent_card:dep:{lookup_key}"]
        if key_type == "ext":
            return [f"{self._key_prefix}agent_card:ext:{lookup_key}"]
        return [
            f"{self._key_prefix}agent_card:dep:{lookup_key}",
            f"{self._key_prefix}agent_card:ext:{lookup_key}",
        ]

    async def _get_record(self, redis_key: str) -> AgentCardCacheRecord | None:
        payload = await self._redis.get(redis_key)
        if payload is None:
            return None
        return AgentCardCacheRecord.model_validate_json(payload)

    async def get_fresh(self, lookup_key: str, *, cache_ttl: int) -> AgentCardCacheRecord | None:
        for redis_key in self._redis_keys_for_lookup(lookup_key):
            record = await self._get_record(redis_key)
            if record is not None and record.is_fresh(cache_ttl):
                return record
        return None

    async def get_stale(
        self,
        lookup_key: str,
        *,
        max_staleness_seconds: int,
    ) -> AgentCardCacheRecord | None:
        for redis_key in self._redis_keys_for_lookup(lookup_key):
            record = await self._get_record(redis_key)
            if record is not None and record.is_within_staleness(max_staleness_seconds):
                return record
        return None

    async def store(
        self,
        cards: dict[str, AgentCard],
        *,
        key_types: dict[str, LookupKeyType],
    ) -> None:
        ttl = self._max_staleness_seconds
        if ttl <= 0:
            return

        for lookup_key, card in cards.items():
            key_type = key_types.get(lookup_key, "dep")
            record = build_cache_record(card, lookup_key=lookup_key, key_type=key_type)
            payload = record.model_dump_json()
            for redis_key in self._redis_keys_for_record(record):
                await self._redis.set(redis_key, payload, ex=ttl)


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

    @property
    def memory(self) -> MemoryAgentCardCacheBackend:
        """Expose the L1 backend (tests)."""
        return self._l1


def _infer_key_type(record: AgentCardCacheRecord, lookup_key: str) -> LookupKeyType:
    if record.deployment_id == lookup_key:
        return "dep"
    if record.external_id == lookup_key:
        return "ext"
    return "dep"


def create_agent_card_cache_backend(
    config: AgentCardRegistryConfig,
) -> AgentCardCacheBackend:
    """Instantiate the configured cache backend."""
    backend = config.agent_card_registry_backend
    if backend == "memory":
        return MemoryAgentCardCacheBackend()

    if backend == "redis":
        redis_url = config.agent_card_registry_redis_url
        if not redis_url:
            raise ValueError(
                "AGENT_CARD_REGISTRY_REDIS_URL is required when AGENT_CARD_REGISTRY_BACKEND=redis."
            )
        l1 = MemoryAgentCardCacheBackend()
        l2 = RedisAgentCardCacheBackend(
            redis_url=redis_url,
            key_prefix=config.agent_card_registry_redis_prefix,
            max_staleness_seconds=config.agent_card_registry_max_staleness_seconds,
        )
        return LayeredAgentCardCacheBackend(l1, l2)

    raise ValueError(
        f"Unsupported agent card registry cache backend: {backend!r}. Expected 'memory' or 'redis'."
    )
