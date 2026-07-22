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

"""Cache for Okta XAA exchanged access tokens."""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import UTC
from datetime import datetime
from typing import Literal
from typing import Protocol

import jwt
from datarobot.core.config import DataRobotAppFrameworkBaseSettings
from pydantic import BaseModel
from pydantic import Field

from datarobot_genai.dragent.agent_card_registry import AgentCardRegistryConfig

logger = logging.getLogger(__name__)

_DEFAULT_SKEW_SECONDS = 60
_DEFAULT_MAX_TTL_SECONDS = 3600


class XAATokenCacheConfig(DataRobotAppFrameworkBaseSettings):
    """Configuration for the XAA exchanged-token cache."""

    agent_card_xaa_token_cache_enabled: bool = Field(
        default=True,
        description=(
            "When true, cache exchanged XAA access tokens in-memory (or Redis when "
            "agent_card_xaa_token_cache_backend=redis). "
            "Set AGENT_CARD_XAA_TOKEN_CACHE_ENABLED=false to disable."
        ),
    )

    agent_card_xaa_token_cache_backend: Literal["memory", "redis"] = Field(
        default="memory",
        description=(
            "XAAToken cache backend. 'memory' uses in-process cache only; "
            "'redis' uses shared Redis (same URL/prefix as agent card registry)."
        ),
    )

    agent_card_xaa_token_skew_seconds: int = Field(
        default=_DEFAULT_SKEW_SECONDS,
        ge=0,
        description="Refresh cached tokens this many seconds before JWT exp. Default: 60.",
    )

    agent_card_xaa_token_max_ttl_seconds: int = Field(
        default=_DEFAULT_MAX_TTL_SECONDS,
        ge=0,
        description=("Cap cache TTL regardless of token exp. Default: 3600. Set to 0 for no cap."),
    )


class XAATokenCacheRecord(BaseModel):
    """Serialized exchanged-token cache entry."""

    version: int = 1
    access_token: str
    expires_at: datetime
    token_type: str = "Bearer"

    def is_valid(self, *, skew_seconds: int) -> bool:
        """Return *True* if the token is still within its usable window."""
        skew = datetime.fromtimestamp(
            self.expires_at.timestamp() - skew_seconds,
            tz=UTC,
        )
        return datetime.now(UTC) < skew


class XAATokenCache(Protocol):
    """Async cache for exchanged XAA bearer tokens."""

    async def get(self, cache_key: str) -> str | None:
        """Return a cached access token when still valid."""

    async def set(self, cache_key: str, access_token: str, ttl_seconds: int) -> None:
        """Store *access_token* under *cache_key* for *ttl_seconds*."""


def build_xaa_cache_key(
    *,
    subject_token: str,
    target_audience: str | None,
    token_url: str,
    scopes: list[str],
    exchange_audience: str,
) -> str:
    """Build deterministic cache-key material for an XAA exchange.

    The subject token is never stored verbatim — a SHA-256 fingerprint is used
    so cache entries are scoped per caller session.
    """
    token_fingerprint = hashlib.sha256(subject_token.encode()).hexdigest()
    scopes_joined = ",".join(sorted(scopes))
    target = target_audience or ""
    return f"{token_fingerprint}|{target}|{token_url}|{scopes_joined}|{exchange_audience}"


def _hash_cache_key(cache_key: str) -> str:
    return hashlib.sha256(cache_key.encode()).hexdigest()


def compute_token_ttl_seconds(
    access_token: str,
    *,
    skew_seconds: int,
    max_ttl_seconds: int,
) -> int:
    """Derive Redis/memory TTL from JWT ``exp`` with skew and optional cap."""
    now = int(time.time())
    exp = _jwt_exp_unverified(access_token)
    if exp is None:
        return max_ttl_seconds if max_ttl_seconds > 0 else _DEFAULT_MAX_TTL_SECONDS
    ttl = max(0, exp - now - skew_seconds)
    if max_ttl_seconds > 0:
        ttl = min(ttl, max_ttl_seconds)
    return ttl


def _jwt_exp_unverified(token: str) -> int | None:
    try:
        claims = jwt.decode(token, options={"verify_signature": False})
    except Exception:
        return None
    exp = claims.get("exp")
    return int(exp) if isinstance(exp, int | float) else None


class MemoryXAATokenCache:
    """In-process XAA token cache."""

    def __init__(self, *, skew_seconds: int) -> None:
        self._skew_seconds = skew_seconds
        self._entries: dict[str, XAATokenCacheRecord] = {}

    async def get(self, cache_key: str) -> str | None:
        record = self._entries.get(cache_key)
        if record is None or not record.is_valid(skew_seconds=self._skew_seconds):
            return None
        return record.access_token

    async def set(self, cache_key: str, access_token: str, ttl_seconds: int) -> None:
        if ttl_seconds <= 0:
            return
        expires_at = datetime.fromtimestamp(time.time() + ttl_seconds, tz=UTC)
        self._entries[cache_key] = XAATokenCacheRecord(
            access_token=access_token,
            expires_at=expires_at,
        )


class RedisXAATokenCache:
    """Shared Redis cache for exchanged XAA tokens."""

    def __init__(
        self,
        *,
        redis_url: str,
        key_prefix: str,
        skew_seconds: int,
    ) -> None:
        try:
            import redis.asyncio as redis
        except ImportError as exc:
            raise ImportError(
                "Redis XAA token cache requires the 'redis' package. "
                "Install with: pip install 'datarobot-genai[dragent]'"
            ) from exc

        self._redis = redis.from_url(redis_url, decode_responses=True)
        self._key_prefix = key_prefix
        self._skew_seconds = skew_seconds

    def _redis_key(self, cache_key: str) -> str:
        return f"{self._key_prefix}xaa_token:{_hash_cache_key(cache_key)}"

    async def get(self, cache_key: str) -> str | None:
        payload = await self._redis.get(self._redis_key(cache_key))
        if payload is None:
            return None
        record = XAATokenCacheRecord.model_validate_json(payload)
        if not record.is_valid(skew_seconds=self._skew_seconds):
            return None
        return record.access_token

    async def set(self, cache_key: str, access_token: str, ttl_seconds: int) -> None:
        if ttl_seconds <= 0:
            return
        expires_at = datetime.fromtimestamp(time.time() + ttl_seconds, tz=UTC)
        record = XAATokenCacheRecord(access_token=access_token, expires_at=expires_at)
        await self._redis.set(
            self._redis_key(cache_key),
            record.model_dump_json(),
            ex=ttl_seconds,
        )


class LayeredXAATokenCache:
    """L1 memory read-through / write-through over an L2 backend."""

    def __init__(
        self,
        l1: MemoryXAATokenCache,
        l2: XAATokenCache,
        *,
        max_ttl_seconds: int,
    ) -> None:
        self._l1 = l1
        self._l2 = l2
        self._max_ttl_seconds = max_ttl_seconds

    async def get(self, cache_key: str) -> str | None:
        if token := await self._l1.get(cache_key):
            return token
        if token := await self._l2.get(cache_key):
            ttl = compute_token_ttl_seconds(
                token,
                skew_seconds=0,
                max_ttl_seconds=self._max_ttl_seconds,
            )
            await self._l1.set(cache_key, token, max(ttl, 1))
            return token
        return None

    async def set(self, cache_key: str, access_token: str, ttl_seconds: int) -> None:
        await self._l1.set(cache_key, access_token, ttl_seconds)
        await self._l2.set(cache_key, access_token, ttl_seconds)


class _XAATokenCacheHolder:
    instance: XAATokenCache | None = None


def create_xaa_token_cache(config: XAATokenCacheConfig | None = None) -> XAATokenCache | None:
    """Instantiate the configured XAA token cache, or ``None`` when disabled."""
    cfg = config or XAATokenCacheConfig()
    if not cfg.agent_card_xaa_token_cache_enabled:
        return None

    memory = MemoryXAATokenCache(skew_seconds=cfg.agent_card_xaa_token_skew_seconds)
    if cfg.agent_card_xaa_token_cache_backend == "memory":
        return memory

    if cfg.agent_card_xaa_token_cache_backend == "redis":
        registry_cfg = AgentCardRegistryConfig()
        redis_url = registry_cfg.agent_card_registry_redis_url
        if not redis_url:
            raise ValueError(
                "AGENT_CARD_REGISTRY_REDIS_URL is required when "
                "AGENT_CARD_XAA_TOKEN_CACHE_BACKEND=redis."
            )
        redis_backend = RedisXAATokenCache(
            redis_url=redis_url,
            key_prefix=registry_cfg.agent_card_registry_redis_prefix,
            skew_seconds=cfg.agent_card_xaa_token_skew_seconds,
        )
        return LayeredXAATokenCache(
            memory,
            redis_backend,
            max_ttl_seconds=cfg.agent_card_xaa_token_max_ttl_seconds,
        )

    raise ValueError(
        f"Unsupported XAA token cache backend: {cfg.agent_card_xaa_token_cache_backend!r}. "
        "Expected 'memory' or 'redis'."
    )


def get_xaa_token_cache() -> XAATokenCache | None:
    """Return the module-level XAA token cache singleton."""
    if _XAATokenCacheHolder.instance is None:
        _XAATokenCacheHolder.instance = create_xaa_token_cache()
    return _XAATokenCacheHolder.instance


def reset_xaa_token_cache() -> None:
    """Reset the module-level singleton (for tests)."""
    _XAATokenCacheHolder.instance = None
