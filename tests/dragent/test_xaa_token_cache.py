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

import time
from unittest.mock import patch

import jwt
import pytest

from datarobot_genai.dragent.redis_cache_signing import resolve_redis_signing_key
from datarobot_genai.dragent.xaa_token_cache import MemoryXAATokenCache
from datarobot_genai.dragent.xaa_token_cache import RedisXAATokenCache
from datarobot_genai.dragent.xaa_token_cache import XAATokenCacheConfig
from datarobot_genai.dragent.xaa_token_cache import build_xaa_cache_key
from datarobot_genai.dragent.xaa_token_cache import compute_token_ttl_seconds
from datarobot_genai.dragent.xaa_token_cache import create_xaa_token_cache
from datarobot_genai.dragent.xaa_token_cache import reset_xaa_token_cache


def _jwt_with_exp(exp: int) -> str:
    return jwt.encode({"exp": exp, "sub": "user-1"}, "secret", algorithm="HS256")


@pytest.fixture(autouse=True)
def _reset_cache_singleton():
    reset_xaa_token_cache()
    yield
    reset_xaa_token_cache()


class TestBuildXAACacheKey:
    def test_same_inputs_produce_same_key(self):
        key_a = build_xaa_cache_key(
            subject_token="user-token",
            principal_id="agent-principal-a",
            target_audience="https://api.example.com/",
            token_url="https://okta/token",
            scopes=["b", "a"],
            exchange_audience="https://okta/as",
        )
        key_b = build_xaa_cache_key(
            subject_token="user-token",
            principal_id="agent-principal-a",
            target_audience="https://api.example.com/",
            token_url="https://okta/token",
            scopes=["a", "b"],
            exchange_audience="https://okta/as",
        )
        assert key_a == key_b

    def test_different_subject_tokens_produce_different_keys(self):
        base = dict(
            principal_id="agent-principal",
            target_audience="https://api.example.com/",
            token_url="https://okta/token",
            scopes=["scope"],
            exchange_audience="https://okta/as",
        )
        assert build_xaa_cache_key(subject_token="a", **base) != build_xaa_cache_key(
            subject_token="b", **base
        )

    def test_different_principal_ids_produce_different_keys(self):
        base = dict(
            subject_token="user-token",
            target_audience="https://api.example.com/",
            token_url="https://okta/token",
            scopes=["scope"],
            exchange_audience="https://okta/as",
        )
        assert build_xaa_cache_key(principal_id="agent-a", **base) != build_xaa_cache_key(
            principal_id="agent-b", **base
        )


class TestComputeTokenTTL:
    def test_uses_jwt_exp_with_skew(self):
        exp = int(time.time()) + 600
        token = _jwt_with_exp(exp)
        ttl = compute_token_ttl_seconds(token, skew_seconds=60, max_ttl_seconds=3600)
        assert 500 <= ttl <= 540

    def test_caps_at_max_ttl(self):
        exp = int(time.time()) + 10_000
        token = _jwt_with_exp(exp)
        ttl = compute_token_ttl_seconds(token, skew_seconds=0, max_ttl_seconds=300)
        assert ttl == 300


class TestMemoryXAATokenCache:
    async def test_set_and_get(self):
        cache = MemoryXAATokenCache(skew_seconds=60)
        key = "cache-key"
        token = _jwt_with_exp(int(time.time()) + 600)
        await cache.set(key, token, ttl_seconds=600)
        assert await cache.get(key) == token

    async def test_expired_entry_returns_none(self):
        cache = MemoryXAATokenCache(skew_seconds=0)
        key = "cache-key"
        token = _jwt_with_exp(int(time.time()) - 10)
        await cache.set(key, token, ttl_seconds=1)
        assert await cache.get(key) is None


class TestXAATokenCacheFactory:
    def test_disabled_returns_none(self):
        config = XAATokenCacheConfig(agent_card_xaa_token_cache_enabled=False)
        assert create_xaa_token_cache(config) is None

    def test_memory_backend(self):
        config = XAATokenCacheConfig(agent_card_xaa_token_cache_backend="memory")
        cache = create_xaa_token_cache(config)
        assert isinstance(cache, MemoryXAATokenCache)

    def test_redis_backend_requires_registry_url(self):
        config = XAATokenCacheConfig(agent_card_xaa_token_cache_backend="redis")
        env = {
            "AGENT_CARD_REGISTRY_CACHE_NAMESPACE": "dep-1",
            "AGENT_CARD_REGISTRY_REDIS_SIGNING_KEY": "unit-test-signing-secret",
        }
        with patch.dict("os.environ", env, clear=False):
            with pytest.raises(ValueError, match="AGENT_CARD_REGISTRY_REDIS_URL"):
                create_xaa_token_cache(config)

    def test_redis_backend_requires_namespace(self):
        config = XAATokenCacheConfig(agent_card_xaa_token_cache_backend="redis")
        env = {
            "AGENT_CARD_REGISTRY_REDIS_URL": "redis://localhost:6379/0",
            "AGENT_CARD_REGISTRY_REDIS_SIGNING_KEY": "unit-test-signing-secret",
        }
        with patch.dict("os.environ", env, clear=True):
            with pytest.raises(ValueError, match="MLOPS_DEPLOYMENT_ID"):
                create_xaa_token_cache(config)

    def test_memory_space_backend_requires_space_id(self):
        config = XAATokenCacheConfig(agent_card_xaa_token_cache_backend="memory_space")
        env = {
            "DATAROBOT_ENDPOINT": "https://app.datarobot.com/api/v2",
            "DATAROBOT_API_TOKEN": "token",
        }
        with patch.dict("os.environ", env, clear=True):
            with pytest.raises(ValueError, match="MemorySpace ID"):
                create_xaa_token_cache(config)


class TestRedisXAATokenCache:
    async def test_store_and_get(self, fake_redis_client):
        backend = RedisXAATokenCache(
            redis_url="redis://fake",
            key_prefix="test:dev:dep-1:",
            skew_seconds=60,
            signing_key=resolve_redis_signing_key("unit-test-signing-secret"),
        )
        backend._redis = fake_redis_client
        token = _jwt_with_exp(int(time.time()) + 600)
        await backend.set("key-1", token, ttl_seconds=600)
        assert await backend.get("key-1") == token


@pytest.fixture
def fake_redis_client():
    import fakeredis

    server = fakeredis.FakeServer()
    return fakeredis.FakeAsyncRedis(server=server, decode_responses=True)
