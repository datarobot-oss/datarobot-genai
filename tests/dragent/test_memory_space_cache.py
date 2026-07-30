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

from __future__ import annotations

from typing import Any

import pytest

from datarobot_genai.dragent.memory_space_cache import DRAGENT_CACHE_USER_ID
from datarobot_genai.dragent.memory_space_cache import METADATA_CACHE_KEY
from datarobot_genai.dragent.memory_space_cache import METADATA_CACHE_KIND
from datarobot_genai.dragent.memory_space_cache import METADATA_DRAGENT_CACHE
from datarobot_genai.dragent.memory_space_cache import MemorySpaceKVCache
from datarobot_genai.dragent.memory_space_cache import resolve_memory_space_id


class _FakeMemoryBackend:
    def __init__(self) -> None:
        self._entries: dict[str, dict[str, Any]] = {}
        self._next_id = 1

    async def get_all(self, **kwargs: Any) -> dict[str, Any]:
        filters = kwargs.get("filters", {})
        metadata_filter = None
        for condition in filters.get("AND", []):
            if "metadata" in condition:
                metadata_filter = condition["metadata"]
                break

        results = []
        for entry in self._entries.values():
            meta = entry.get("metadata") or {}
            if metadata_filter and all(meta.get(k) == v for k, v in metadata_filter.items()):
                results.append(entry)
        return {"results": results}

    async def add(self, messages: Any, **kwargs: Any) -> dict[str, Any]:
        metadata = kwargs.get("metadata") or {}
        memory_id = f"mem-{self._next_id}"
        self._next_id += 1
        payload = messages[0]["content"] if messages else ""
        self._entries[memory_id] = {
            "id": memory_id,
            "memory": payload,
            "metadata": metadata,
            "user_id": kwargs.get("user_id"),
        }
        return {"results": [{"id": memory_id}]}

    async def update(
        self, memory_id: str, text: str | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        entry = self._entries.get(memory_id)
        if entry is None:
            raise KeyError(memory_id)
        if text is not None:
            entry["memory"] = text
        if kwargs.get("metadata"):
            entry["metadata"] = kwargs["metadata"]
        return {"id": memory_id}


class _FakeMem0Client:
    def __init__(self) -> None:
        self._memory = _FakeMemoryBackend()


@pytest.fixture
def kv_cache() -> MemorySpaceKVCache:
    return MemorySpaceKVCache(_FakeMem0Client(), key_prefix="dragent:")


class TestResolveMemorySpaceId:
    def test_explicit_id(self):
        assert resolve_memory_space_id("space-explicit") == "space-explicit"

    def test_missing_id_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("AGENT_MEMORY_SPACE_ID", raising=False)
        monkeypatch.delenv("AGENT_CARD_REGISTRY_MEMORY_SPACE_ID", raising=False)
        with pytest.raises(ValueError, match="MemorySpace ID"):
            resolve_memory_space_id(None)


class TestMemorySpaceKVCache:
    async def test_set_and_get_round_trip(self, kv_cache: MemorySpaceKVCache):
        await kv_cache.set_value("dep-1", '{"version": 1}', kind="agent_card")
        assert await kv_cache.get_value("dep-1", kind="agent_card") == '{"version": 1}'

    async def test_update_existing_entry(self, kv_cache: MemorySpaceKVCache):
        await kv_cache.set_value("dep-1", "v1", kind="agent_card")
        await kv_cache.set_value("dep-1", "v2", kind="agent_card")
        assert await kv_cache.get_value("dep-1", kind="agent_card") == "v2"
        backend: _FakeMemoryBackend = kv_cache._client
        assert len(backend._entries) == 1

    async def test_different_kinds_are_isolated(self, kv_cache: MemorySpaceKVCache):
        await kv_cache.set_value("same-key", "agent", kind="agent_card")
        await kv_cache.set_value("same-key", "xaa", kind="xaa_token")
        assert await kv_cache.get_value("same-key", kind="agent_card") == "agent"
        assert await kv_cache.get_value("same-key", kind="xaa_token") == "xaa"

    async def test_uses_dedicated_cache_user(self, kv_cache: MemorySpaceKVCache):
        await kv_cache.set_value("dep-1", "payload", kind="agent_card")
        backend: _FakeMemoryBackend = kv_cache._client
        entry = next(iter(backend._entries.values()))
        assert entry["user_id"] == DRAGENT_CACHE_USER_ID
        assert entry["metadata"][METADATA_DRAGENT_CACHE] is True
        assert entry["metadata"][METADATA_CACHE_KIND] == "agent_card"
        assert entry["metadata"][METADATA_CACHE_KEY] == "dragent:agent_card:dep-1"
