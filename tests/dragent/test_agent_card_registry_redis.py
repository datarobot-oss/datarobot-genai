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

from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest
from a2a.types import AgentCard

from datarobot_genai.dragent.agent_card_registry import AgentCardRegistry
from datarobot_genai.dragent.agent_card_registry import AgentCardRegistryConfig
from datarobot_genai.dragent.agent_card_registry import ParsedRegistryCards
from datarobot_genai.dragent.agent_card_registry_backends import AgentCardCacheRecord
from datarobot_genai.dragent.agent_card_registry_backends import LayeredAgentCardCacheBackend
from datarobot_genai.dragent.agent_card_registry_backends import MemoryAgentCardCacheBackend
from datarobot_genai.dragent.agent_card_registry_backends import RedisAgentCardCacheBackend
from datarobot_genai.dragent.agent_card_registry_backends import create_agent_card_cache_backend

_SAMPLE_AGENT_CARD = AgentCard.model_validate(
    {
        "name": "Redis Agent",
        "description": "Cached in Redis",
        "url": "https://agent.example.com/a2a/",
        "version": "1.0.0",
        "skills": [],
        "defaultInputModes": ["text"],
        "defaultOutputModes": ["text"],
        "capabilities": {"streaming": False},
    }
)


def _parsed(cards: dict) -> ParsedRegistryCards:
    return ParsedRegistryCards(cards=cards, key_types={key: "dep" for key in cards})


@pytest.fixture
def fake_redis_client():
    import fakeredis

    server = fakeredis.FakeServer()
    return fakeredis.FakeAsyncRedis(server=server, decode_responses=True)


@pytest.fixture
def redis_backend(fake_redis_client):
    backend = RedisAgentCardCacheBackend(
        redis_url="redis://fake",
        key_prefix="test:",
        max_staleness_seconds=3600,
    )
    backend._redis = fake_redis_client
    return backend


class TestRedisAgentCardCacheBackend:
    async def test_store_and_get_fresh_by_deployment_id(self, redis_backend):
        await redis_backend.store(
            {"dep-1": _SAMPLE_AGENT_CARD},
            key_types={"dep-1": "dep"},
        )

        record = await redis_backend.get_fresh("dep-1", cache_ttl=3600)
        assert record is not None
        assert record.card.name == "Redis Agent"
        assert record.deployment_id == "dep-1"

    async def test_store_indexes_external_id(self, redis_backend):
        await redis_backend.store(
            {"ext-1": _SAMPLE_AGENT_CARD},
            key_types={"ext-1": "ext"},
        )

        record = await redis_backend.get_fresh("ext-1", cache_ttl=3600)
        assert record is not None
        assert record.external_id == "ext-1"

    async def test_get_stale_after_soft_ttl(self, redis_backend):
        await redis_backend.store(
            {"dep-1": _SAMPLE_AGENT_CARD},
            key_types={"dep-1": "dep"},
        )

        redis_key = "test:agent_card:dep:dep-1"
        payload = await redis_backend._redis.get(redis_key)
        record = AgentCardCacheRecord.model_validate_json(payload)
        record.fetched_at_mono -= 120
        await redis_backend._redis.set(redis_key, record.model_dump_json(), ex=3600)

        assert await redis_backend.get_fresh("dep-1", cache_ttl=60) is None
        stale = await redis_backend.get_stale("dep-1", max_staleness_seconds=3600)
        assert stale is not None
        assert stale.card.name == "Redis Agent"


class TestLayeredAgentCardCacheBackend:
    async def test_l2_populates_l1_on_read(self, redis_backend):
        warm = LayeredAgentCardCacheBackend(MemoryAgentCardCacheBackend(), redis_backend)
        await warm.store({"dep-1": _SAMPLE_AGENT_CARD}, key_types={"dep-1": "dep"})

        cold_l1 = MemoryAgentCardCacheBackend()
        reader = LayeredAgentCardCacheBackend(cold_l1, redis_backend)
        record = await reader.get_fresh("dep-1", cache_ttl=3600)

        assert record is not None
        assert await cold_l1.get_fresh("dep-1", cache_ttl=3600) is not None


class TestAgentCardRegistryRedisBackend:
    @pytest.fixture
    def mock_fetch(self):
        with patch.object(AgentCardRegistry, "_fetch", new_callable=AsyncMock) as m:
            yield m

    def _make_redis_backend(self, fake_redis_client) -> LayeredAgentCardCacheBackend:
        config = AgentCardRegistryConfig(
            agent_card_registry_backend="redis",
            agent_card_registry_redis_url="redis://fake",
            agent_card_registry_redis_prefix="test:",
            agent_card_registry_max_staleness_seconds=3600,
        )
        backend = create_agent_card_cache_backend(config)
        assert isinstance(backend, LayeredAgentCardCacheBackend)
        backend._l2._redis = fake_redis_client
        return backend

    async def test_registry_with_redis_backend_shares_cache(self, mock_fetch, fake_redis_client):
        card_mock = _SAMPLE_AGENT_CARD.model_copy()
        mock_fetch.return_value = _parsed({"dep-1": card_mock})

        backend_a = self._make_redis_backend(fake_redis_client)
        registry_a = AgentCardRegistry(
            api_token="tok",
            endpoint="https://ep",
            cache_ttl=3600,
            cache_backend=backend_a,
        )
        card_a = await registry_a.get(deployment_id="dep-1")
        mock_fetch.assert_awaited_once()
        assert card_a is card_mock

        backend_b = self._make_redis_backend(fake_redis_client)
        registry_b = AgentCardRegistry(
            api_token="tok",
            endpoint="https://ep",
            cache_ttl=3600,
            cache_backend=backend_b,
        )
        mock_fetch.reset_mock()
        card_b = await registry_b.get(deployment_id="dep-1")
        mock_fetch.assert_not_awaited()
        assert card_b.name == card_mock.name

    def test_create_backend_requires_redis_url(self):
        config = AgentCardRegistryConfig(agent_card_registry_backend="redis")
        with pytest.raises(ValueError, match="AGENT_CARD_REGISTRY_REDIS_URL"):
            create_agent_card_cache_backend(config)
