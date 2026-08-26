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

import asyncio
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest
from a2a.types import AgentCard

from datarobot_genai.dragent.agent_card_registry import AgentCardRegistryConfig
from datarobot_genai.dragent.agent_card_registry_backends import AgentCardCacheRecord
from datarobot_genai.dragent.agent_card_registry_backends import LayeredAgentCardCacheBackend
from datarobot_genai.dragent.agent_card_registry_backends import MemoryAgentCardCacheBackend
from datarobot_genai.dragent.agent_card_registry_backends import MemorySpaceAgentCardCacheBackend
from datarobot_genai.dragent.agent_card_registry_backends import build_cache_record
from datarobot_genai.dragent.agent_card_registry_backends import create_agent_card_cache_backend
from datarobot_genai.dragent.memory_space_cache import MemorySpaceKVCache

_SAMPLE_AGENT_CARD = AgentCard.model_validate(
    {
        "name": "MemorySpace Agent",
        "description": "Cached in MemorySpace",
        "url": "https://agent.example.com/a2a/",
        "version": "1.0.0",
        "skills": [],
        "defaultInputModes": ["text"],
        "defaultOutputModes": ["text"],
        "capabilities": {"streaming": False},
    }
)


@pytest.fixture
def memory_space_backend():
    kv = MemorySpaceKVCache(memory_space_id="space-1")
    return MemorySpaceAgentCardCacheBackend(kv_cache=kv)


class TestMemorySpaceAgentCardCacheBackend:
    async def test_store_and_get_fresh_by_deployment_id(self, memory_space_backend):
        with (
            patch.object(
                memory_space_backend._kv,
                "set_value",
                wraps=memory_space_backend._kv.set_value,
            ) as set_mock,
            patch.object(
                memory_space_backend._kv,
                "get_value",
                return_value=AgentCardCacheRecord(card=_SAMPLE_AGENT_CARD).model_dump_json(),
            ),
        ):
            await memory_space_backend.store(
                {"dep-1": _SAMPLE_AGENT_CARD},
                key_types={"dep-1": "deployment"},
            )
            record = await memory_space_backend.get_fresh("dep-1", cache_ttl=3600)

        set_mock.assert_awaited()
        assert record is not None
        assert record.card.name == "MemorySpace Agent"

    async def test_get_stale_after_soft_ttl(self, memory_space_backend):
        stale_record = AgentCardCacheRecord(card=_SAMPLE_AGENT_CARD)
        stale_record.fetched_at = datetime.now(UTC) - timedelta(seconds=120)
        stale_json = stale_record.model_dump_json()

        with (
            patch.object(memory_space_backend._kv, "set_value", new_callable=AsyncMock),
            patch.object(
                memory_space_backend._kv,
                "get_value",
                return_value=stale_json,
            ),
        ):
            await memory_space_backend.store(
                {"dep-1": _SAMPLE_AGENT_CARD},
                key_types={"dep-1": "deployment"},
            )
            assert await memory_space_backend.get_fresh("dep-1", cache_ttl=60) is None
            stale = await memory_space_backend.get_stale("dep-1", max_staleness_seconds=3600)

        assert stale is not None
        assert stale.card.name == "MemorySpace Agent"

    async def test_store_writes_both_storage_keys_when_ids_known(self, memory_space_backend):
        storage: dict[str, str] = {}

        async def capture_set(key: str, payload: str) -> None:
            storage[key] = payload

        with patch.object(memory_space_backend._kv, "set_value", side_effect=capture_set):
            await memory_space_backend.store(
                {"dep-1": _SAMPLE_AGENT_CARD, "ext-1": _SAMPLE_AGENT_CARD},
                key_types={"dep-1": "deployment", "ext-1": "external"},
                registry_ids={"dep-1": ("dep-1", "ext-1"), "ext-1": ("dep-1", "ext-1")},
            )

        assert "deployment:dep-1" in storage
        assert "external:ext-1" in storage
        record = AgentCardCacheRecord.model_validate_json(storage["deployment:dep-1"])
        assert record.deployment_id == "dep-1"
        assert record.external_id == "ext-1"

    async def test_get_fresh_uses_key_type_for_single_storage_probe(self, memory_space_backend):
        get_calls: list[str] = []

        async def track_get(key: str) -> str | None:
            get_calls.append(key)
            return None

        with patch.object(memory_space_backend._kv, "get_value", side_effect=track_get):
            assert (
                await memory_space_backend.get_fresh(
                    "dep-1",
                    cache_ttl=3600,
                    key_type="deployment",
                )
                is None
            )

        assert get_calls == ["deployment:dep-1"]

    async def test_get_fresh_without_key_type_probes_both_aliases(self, memory_space_backend):
        get_calls: list[str] = []

        async def track_get(key: str) -> str | None:
            get_calls.append(key)
            return None

        with patch.object(memory_space_backend._kv, "get_value", side_effect=track_get):
            assert await memory_space_backend.get_fresh("dep-1", cache_ttl=3600) is None

        assert get_calls == ["deployment:dep-1", "external:dep-1"]
        """Typed eviction must delete every storage alias for the card."""
        record = build_cache_record(
            _SAMPLE_AGENT_CARD,
            lookup_key="dep-1",
            key_type="deployment",
            deployment_id="dep-1",
            external_id="ext-1",
        )
        payload = record.model_dump_json()
        storage = {
            "deployment:dep-1": payload,
            "external:ext-1": payload,
        }
        deleted: list[str] = []

        async def mock_get(key: str) -> str | None:
            return storage.get(key)

        async def mock_delete(key: str) -> None:
            deleted.append(key)
            storage.pop(key, None)

        with (
            patch.object(memory_space_backend._kv, "get_value", side_effect=mock_get),
            patch.object(memory_space_backend._kv, "delete_value", side_effect=mock_delete),
        ):
            await memory_space_backend.evict("dep-1", key_type="deployment")

        assert sorted(deleted) == ["deployment:dep-1", "external:ext-1"]
        assert storage == {}


class TestCreateAgentCardCacheBackend:
    def test_l1_only_when_no_memory_space_id(self):
        config = AgentCardRegistryConfig()
        env = {
            "DATAROBOT_ENDPOINT": "https://app.datarobot.com/api/v2",
            "DATAROBOT_API_TOKEN": "token",
        }
        with patch.dict("os.environ", env, clear=True):
            from datarobot_genai.dragent.agent_card_registry_backends import (
                MemoryAgentCardCacheBackend,
            )

            backend = create_agent_card_cache_backend(config)

        assert type(backend) is MemoryAgentCardCacheBackend

    def test_l1_only_when_memory_client_unconfigured(self):
        config = AgentCardRegistryConfig(agent_card_registry_memory_space_id="space-123")
        with patch(
            "datarobot_genai.dragent.agent_card_registry_backends.try_configure_datarobot_memory_client",
            return_value=False,
        ):
            from datarobot_genai.dragent.agent_card_registry_backends import (
                MemoryAgentCardCacheBackend,
            )

            backend = create_agent_card_cache_backend(config)

        assert type(backend) is MemoryAgentCardCacheBackend

    def test_creates_layered_backend_when_memory_space_configured(self):
        config = AgentCardRegistryConfig(agent_card_registry_memory_space_id="space-123")
        env = {
            "DATAROBOT_ENDPOINT": "https://app.datarobot.com/api/v2",
            "DATAROBOT_API_TOKEN": "token",
        }
        with patch.dict("os.environ", env, clear=False):
            with patch(
                "datarobot_genai.dragent.agent_card_registry_backends.try_configure_datarobot_memory_client",
                return_value=True,
            ) as configure_mock:
                backend = create_agent_card_cache_backend(config)

        assert isinstance(backend, LayeredAgentCardCacheBackend)
        configure_mock.assert_called_once()


class TestLayeredAgentCardCacheBackend:
    async def test_get_fresh_read_through_preserves_fetched_at(self):
        l1 = MemoryAgentCardCacheBackend()
        l2 = MemoryAgentCardCacheBackend()
        layered = LayeredAgentCardCacheBackend(l1, l2)

        # GIVEN an L2 record that is already 30s old
        await l2.store({"dep-1": _SAMPLE_AGENT_CARD}, key_types={"dep-1": "deployment"})
        l2.age_entry_for_test("dep-1", 30)

        # WHEN L1 misses and L2 hits within the 60s soft TTL
        record = await layered.get_fresh("dep-1", cache_ttl=60)

        # THEN the promoted L1 entry keeps the original fetch time
        assert record is not None
        assert await l1.get_fresh("dep-1", cache_ttl=60) is not None
        assert await l1.get_fresh("dep-1", cache_ttl=20) is None

    async def test_get_stale_read_through_does_not_reset_soft_ttl(self):
        l1 = MemoryAgentCardCacheBackend()
        l2 = MemoryAgentCardCacheBackend()
        layered = LayeredAgentCardCacheBackend(l1, l2)

        # GIVEN an L2 record past the 60s soft TTL but within max staleness
        await l2.store({"dep-1": _SAMPLE_AGENT_CARD}, key_types={"dep-1": "deployment"})
        l2.age_entry_for_test("dep-1", 90)

        # WHEN L1 misses and L2 serves a stale-if-error hit
        record = await layered.get_stale("dep-1", max_staleness_seconds=120)

        # THEN L1 must not treat the promoted card as freshly fetched
        assert record is not None
        assert await l1.get_fresh("dep-1", cache_ttl=60) is None
        assert await l1.get_stale("dep-1", max_staleness_seconds=80) is None
        assert await l1.get_stale("dep-1", max_staleness_seconds=120) is not None

    async def test_get_fresh_skips_l2_when_l1_has_stale_entry(self):
        l1 = MemoryAgentCardCacheBackend()
        l2 = AsyncMock()
        l2.get_fresh = AsyncMock(return_value=None)
        layered = LayeredAgentCardCacheBackend(l1, l2)

        await l1.store({"dep-1": _SAMPLE_AGENT_CARD}, key_types={"dep-1": "deployment"})
        l1.age_entry_for_test("dep-1", 120)

        assert await layered.get_fresh("dep-1", cache_ttl=60) is None
        l2.get_fresh.assert_not_awaited()

    async def test_store_write_behind_does_not_block_on_l2(self):
        l1 = MemoryAgentCardCacheBackend()
        l2_blocked = asyncio.Event()
        l2_started = asyncio.Event()

        async def slow_store(*args, **kwargs):
            l2_started.set()
            await l2_blocked.wait()

        l2 = AsyncMock()
        l2.store = AsyncMock(side_effect=slow_store)
        layered = LayeredAgentCardCacheBackend(l1, l2)

        await layered.store({"dep-1": _SAMPLE_AGENT_CARD}, key_types={"dep-1": "deployment"})

        assert await l1.get_fresh("dep-1", cache_ttl=3600) is not None
        await asyncio.wait_for(l2_started.wait(), timeout=1.0)
        l2.store.assert_awaited_once()
        l2_blocked.set()
        await layered.flush_l2_tasks()

    async def test_evict_write_behind_clears_l1_before_l2(self):
        l1 = MemoryAgentCardCacheBackend()
        l2 = MemoryAgentCardCacheBackend()
        layered = LayeredAgentCardCacheBackend(l1, l2)

        await layered.store({"dep-1": _SAMPLE_AGENT_CARD}, key_types={"dep-1": "deployment"})
        await layered.flush_l2_tasks()
        assert await l2.get_fresh("dep-1", cache_ttl=3600) is not None

        await layered.evict("dep-1", key_type="deployment")
        assert not l1.has_entry("dep-1")
        assert await l2.get_fresh("dep-1", cache_ttl=3600) is not None

        await layered.flush_l2_tasks()
        assert await l2.get_fresh("dep-1", cache_ttl=3600) is None
