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

Each provisioned memory space has a unique ``memory_space_id`` and platform-level
access control scoped to the deploying user or workload API token. Unlike shared
Redis, no per-deployment namespace or HMAC signing is required for this backend.
"""

from __future__ import annotations

import logging
import os
from typing import Any
from typing import Literal

from datarobot.core.config import DataRobotAppFrameworkBaseSettings
from pydantic import Field

logger = logging.getLogger(__name__)

# Dedicated mem0 user_id so cache entries do not mix with conversational memories.
DRAGENT_CACHE_USER_ID = "__dragent_cache__"

METADATA_DRAGENT_CACHE = "dragent_cache"
METADATA_CACHE_KEY = "dragent_cache_key"
METADATA_CACHE_KIND = "dragent_cache_kind"

CacheKind = Literal["agent_card", "xaa_token"]

_MEMORY_SPACE_REQUIRED_MSG = (
    "Memory space cache backends require a provisioned DataRobot MemorySpace ID. "
    "Set AGENT_CARD_REGISTRY_MEMORY_SPACE_ID or AGENT_MEMORY_SPACE_ID."
)


class MemorySpaceCacheConfig(DataRobotAppFrameworkBaseSettings):
    """Connection settings for DataRobot MemorySpace cache backends."""

    agent_card_registry_memory_space_id: str | None = Field(
        default=None,
        description=(
            "DataRobot MemorySpace ID for dragent L2 caches when "
            "AGENT_CARD_REGISTRY_BACKEND=memory_space (or XAA backend=memory_space). "
            "Defaults to AGENT_MEMORY_SPACE_ID."
        ),
    )

    agent_memory_space_id: str | None = Field(
        default=None,
        description="Platform-injected MemorySpace ID (AGENT_MEMORY_SPACE_ID).",
    )

    datarobot_endpoint: str | None = Field(
        default=None,
        description="DataRobot API base URL (DATAROBOT_ENDPOINT).",
    )

    datarobot_api_token: str | None = Field(
        default=None,
        description="DataRobot API token (DATAROBOT_API_TOKEN).",
    )


def resolve_memory_space_id(explicit: str | None = None) -> str:
    """Return the MemorySpace ID for cache backends."""
    cfg = MemorySpaceCacheConfig()
    space_id = explicit or cfg.agent_card_registry_memory_space_id or cfg.agent_memory_space_id
    if not space_id or not space_id.strip():
        raise ValueError(_MEMORY_SPACE_REQUIRED_MSG)
    return space_id.strip()


def _memory_space_endpoint(space_id: str, endpoint: str | None) -> str:
    base = endpoint or os.getenv("DATAROBOT_ENDPOINT")
    if not base:
        raise ValueError("DATAROBOT_ENDPOINT is required when using memory_space cache backends.")
    return f"{base.rstrip('/')}/memory/{space_id}"


def create_memory_space_client(
    *,
    memory_space_id: str | None = None,
    endpoint: str | None = None,
    api_token: str | None = None,
) -> Any:
    """Instantiate a mem0 client pointed at a DataRobot MemorySpace."""
    cfg = MemorySpaceCacheConfig()
    resolved_id = resolve_memory_space_id(memory_space_id)
    resolved_token = api_token or cfg.datarobot_api_token or os.getenv("DATAROBOT_API_TOKEN")
    if not resolved_token:
        raise ValueError("DATAROBOT_API_TOKEN is required when using memory_space cache backends.")

    os.environ["MEM0_TELEMETRY"] = "False"
    try:
        from datarobot_genai.core.memory.mem0client import Mem0Client
    except ImportError as exc:
        raise ImportError(
            "Memory space cache backends require the dragent extra. "
            'Install with: pip install "datarobot-genai[dragent]"'
        ) from exc

    host = _memory_space_endpoint(
        resolved_id,
        endpoint or cfg.datarobot_endpoint,
    )
    return Mem0Client(api_key=resolved_token, host=host)


class MemorySpaceKVCache:
    """Store opaque JSON payloads in a DataRobot MemorySpace by logical key."""

    def __init__(self, client: Any, *, key_prefix: str = "dragent:") -> None:
        self._client = client._memory
        normalized = key_prefix if key_prefix.endswith(":") else f"{key_prefix}:"
        self._key_prefix = normalized

    def _logical_key(self, key: str, *, kind: CacheKind) -> str:
        return f"{self._key_prefix}{kind}:{key}"

    def _metadata(self, logical_key: str, *, kind: CacheKind) -> dict[str, Any]:
        return {
            METADATA_DRAGENT_CACHE: True,
            METADATA_CACHE_KEY: logical_key,
            METADATA_CACHE_KIND: kind,
        }

    def _filters(self, logical_key: str, *, kind: CacheKind) -> dict[str, Any]:
        return {
            "AND": [
                {"user_id": DRAGENT_CACHE_USER_ID},
                {
                    "metadata": {
                        METADATA_DRAGENT_CACHE: True,
                        METADATA_CACHE_KEY: logical_key,
                        METADATA_CACHE_KIND: kind,
                    }
                },
            ]
        }

    @staticmethod
    def _extract_payload(entry: dict[str, Any]) -> str | None:
        for field in ("memory", "text", "content"):
            value = entry.get(field)
            if value:
                return str(value)
        return None

    @staticmethod
    def _extract_memory_id(entry: dict[str, Any]) -> str | None:
        for field in ("id", "memory_id"):
            value = entry.get(field)
            if value:
                return str(value)
        return None

    async def _find_entry(self, logical_key: str, *, kind: CacheKind) -> dict[str, Any] | None:
        result = await self._client.get_all(
            user_id=DRAGENT_CACHE_USER_ID,
            filters=self._filters(logical_key, kind=kind),
            output_format="v1.1",
        )
        results = result.get("results", []) if isinstance(result, dict) else []
        if not results:
            return None
        first = results[0]
        return first if isinstance(first, dict) else None

    async def get_value(self, key: str, *, kind: CacheKind) -> str | None:
        """Return the stored JSON payload for *key*, or ``None`` when missing."""
        logical_key = self._logical_key(key, kind=kind)
        try:
            entry = await self._find_entry(logical_key, kind=kind)
        except Exception:
            logger.exception("MemorySpace cache read failed for %s", logical_key)
            return None
        if entry is None:
            return None
        return self._extract_payload(entry)

    async def set_value(self, key: str, payload: str, *, kind: CacheKind) -> None:
        """Upsert a JSON payload for *key*."""
        logical_key = self._logical_key(key, kind=kind)
        metadata = self._metadata(logical_key, kind=kind)
        try:
            existing = await self._find_entry(logical_key, kind=kind)
            if existing and (memory_id := self._extract_memory_id(existing)):
                await self._client.update(memory_id, text=payload, metadata=metadata)
                return

            await self._client.add(
                [{"role": "user", "content": payload}],
                user_id=DRAGENT_CACHE_USER_ID,
                metadata=metadata,
                output_format="v1.1",
                async_mode=False,
            )
        except Exception:
            logger.exception("MemorySpace cache write failed for %s", logical_key)
