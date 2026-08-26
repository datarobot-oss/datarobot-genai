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
    kv = MemorySpaceKVCache(memory_space_id="space-1", key_prefix="dragent:")
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
