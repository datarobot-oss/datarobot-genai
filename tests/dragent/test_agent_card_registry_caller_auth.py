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

"""Registry lookups with the caller's credentials (``auth_headers``) instead of the owner key."""

import asyncio
import logging
import traceback
from unittest.mock import AsyncMock
from unittest.mock import patch

import httpx
import pytest
import respx

from datarobot_genai.dragent.agent_card_registry import AgentCardRegistry
from datarobot_genai.dragent.agent_card_registry import AgentCardRegistryConfig
from datarobot_genai.dragent.agent_card_registry import AgentCardRegistryError
from datarobot_genai.dragent.agent_card_registry_backends import LayeredAgentCardCacheBackend
from datarobot_genai.dragent.agent_card_registry_backends import MemoryAgentCardCacheBackend
from datarobot_genai.dragent.agent_card_registry_backends import MemorySpaceAgentCardCacheBackend
from datarobot_genai.dragent.memory_space_cache import MemorySpaceKVCache

OWNER_KEY = "dummy-owner-key-O1"
USER_A = "dummy-user-token-A7f3"
USER_B = "dummy-user-token-B9c1"
DEP = "dep-1"


def _card_dict(name: str) -> dict:
    return {
        "name": name,
        "description": "d",
        "url": "https://agent.example.com/a2a/",
        "version": "1.0.0",
        "skills": [],
        "defaultInputModes": ["text"],
        "defaultOutputModes": ["text"],
        "capabilities": {"streaming": False},
    }


def _ok(name: str) -> httpx.Response:
    entry = {
        "id": f"doc-{DEP}",
        "deploymentId": DEP,
        "externalId": None,
        "workloadId": None,
        "agentCard": _card_dict(name),
    }
    return httpx.Response(200, json={"data": [entry]})


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _registry(backend=None) -> AgentCardRegistry:
    registry = AgentCardRegistry(
        config=AgentCardRegistryConfig(),
        api_token=OWNER_KEY,
        endpoint="https://ep",
    )
    registry._backend = backend or MemoryAgentCardCacheBackend()
    return registry


def _by_bearer(responses: dict[str, httpx.Response | Exception]):
    """Respx side effect choosing the response by the Bearer token on the request."""

    def side_effect(request: httpx.Request):
        token = request.headers["Authorization"].removeprefix("Bearer ")
        result = responses[token]
        if isinstance(result, Exception):
            raise result
        return result

    return side_effect


@pytest.fixture
def router():
    with respx.mock(assert_all_called=False) as mock:
        yield mock.get(host="ep")


def _tokens_sent(route) -> list[str]:
    return [c.request.headers["Authorization"].removeprefix("Bearer ") for c in route.calls]


class TestCallerAuthLookup:
    async def test_fresh_cache_hit_makes_no_request(self, router):
        """GIVEN a fresh cached card WHEN looked up with caller headers THEN no HTTP call."""
        registry = _registry()
        router.side_effect = _by_bearer({OWNER_KEY: _ok("cached")})
        await registry.get(deployment_id=DEP)
        calls_before = router.call_count

        card = await registry.get(deployment_id=DEP, auth_headers=_headers(USER_A))

        assert card.name == "cached"
        assert router.call_count == calls_before

    async def test_miss_uses_caller_bearer_not_owner_key(self, router):
        """GIVEN a cache miss WHEN auth_headers are given THEN only the caller token is sent."""
        registry = _registry()
        router.side_effect = _by_bearer({USER_A: _ok("for-a")})

        card = await registry.get(deployment_id=DEP, auth_headers=_headers(USER_A))

        assert card.name == "for-a"
        assert _tokens_sent(router) == [USER_A]

    async def test_no_auth_headers_uses_owner_key_and_caches(self, router):
        """GIVEN no auth_headers WHEN looked up THEN today's behaviour: owner key, L1 stored."""
        registry = _registry()
        router.side_effect = _by_bearer({OWNER_KEY: _ok("owner")})

        card = await registry.get(deployment_id=DEP)

        assert card.name == "owner"
        assert _tokens_sent(router) == [OWNER_KEY]
        assert DEP in registry._backend._entries

    @pytest.mark.parametrize(
        "failure",
        [httpx.Response(401), httpx.Response(403), httpx.ConnectError("refused")],
        ids=["401", "403", "connect-error"],
    )
    async def test_caller_failure_falls_back_to_owner_key(self, router, failure):
        """GIVEN the caller lookup fails THEN the owner key is tried once and its card returned."""
        registry = _registry()
        router.side_effect = _by_bearer({USER_A: failure, OWNER_KEY: _ok("owner")})

        card = await registry.get(deployment_id=DEP, auth_headers=_headers(USER_A))

        assert card.name == "owner"
        assert _tokens_sent(router) == [USER_A, OWNER_KEY]

    async def test_caller_not_found_falls_back_to_owner_key(self, router):
        """GIVEN the caller cannot see the card THEN the owner key is tried."""
        registry = _registry()
        router.side_effect = _by_bearer(
            {USER_A: httpx.Response(200, json={"data": []}), OWNER_KEY: _ok("owner")}
        )

        card = await registry.get(deployment_id=DEP, auth_headers=_headers(USER_A))

        assert card.name == "owner"
        assert _tokens_sent(router) == [USER_A, OWNER_KEY]

    async def test_non_ascii_token_falls_back_to_owner_key(self, router):
        """GIVEN a caller header httpx cannot encode THEN the owner key is used."""
        registry = _registry()
        router.side_effect = _by_bearer({OWNER_KEY: _ok("owner")})

        card = await registry.get(
            deployment_id=DEP, auth_headers=_headers("tok\u00e9-dummy-user-token-C3d5")
        )

        assert card.name == "owner"
        assert _tokens_sent(router) == [OWNER_KEY]

    async def test_both_fail_raises_owner_error(self, router):
        """GIVEN caller and owner both rejected THEN today's owner error, after two requests."""
        registry = _registry()
        router.side_effect = _by_bearer(
            {USER_A: httpx.Response(401), OWNER_KEY: httpx.Response(401)}
        )

        with pytest.raises(AgentCardRegistryError) as caller_first:
            await registry.get(deployment_id=DEP, auth_headers=_headers(USER_A))
        with pytest.raises(AgentCardRegistryError) as owner_only:
            await _registry().get(deployment_id=DEP)

        assert str(caller_first.value) == str(owner_only.value)
        assert _tokens_sent(router) == [USER_A, OWNER_KEY, OWNER_KEY]

    async def test_caller_lookup_does_not_wait_for_shared_lock(self, router):
        """GIVEN the shared lock is held THEN a caller-auth lookup still completes."""
        registry = _registry()
        router.side_effect = _by_bearer({USER_A: _ok("for-a")})

        async with registry._lock:
            card = await asyncio.wait_for(
                registry.get(deployment_id=DEP, auth_headers=_headers(USER_A)), timeout=1
            )

        assert card.name == "for-a"


class TestBackgroundPathsKeepOwnerKey:
    async def test_prefetch_and_refresh_use_owner_key(self, router):
        """GIVEN registered IDs WHEN prefetch and refresh run THEN they send the owner key."""
        registry = _registry()
        router.side_effect = _by_bearer({OWNER_KEY: _ok("owner")})
        registry.register(deployment_id=DEP)

        await registry.prefetch(deployment_ids=[DEP])
        registry._age_cache_entry_for_test(DEP, 10**9)
        await registry.refresh_all_registered()

        assert _tokens_sent(router) == [OWNER_KEY, OWNER_KEY]


class TestSharedCache:
    async def test_caller_fetched_card_is_shared(self, router):
        """GIVEN a card fetched as the caller THEN later lookups, by anyone, hit the cache."""
        l1, l2 = MemoryAgentCardCacheBackend(), MemoryAgentCardCacheBackend()
        backend = LayeredAgentCardCacheBackend(l1, l2)
        registry = _registry(backend)
        router.side_effect = _by_bearer({USER_A: _ok("card")})

        await registry.get(deployment_id=DEP, auth_headers=_headers(USER_A))
        await backend.flush_l2_tasks()
        by_b = await registry.get(deployment_id=DEP, auth_headers=_headers(USER_B))
        by_owner = await registry.get(deployment_id=DEP)

        assert by_b.name == by_owner.name == "card"
        assert DEP in l1._entries and DEP in l2._entries
        assert _tokens_sent(router) == [USER_A]


def _exception_chain(exc: BaseException):
    while exc is not None:
        yield exc
        exc = exc.__cause__ or exc.__context__


class TestNoSecretLeaks:
    @pytest.mark.parametrize(
        "failure",
        [
            httpx.Response(401),
            httpx.Response(500),
            httpx.ConnectError(f"refused for {USER_A}"),
            httpx.ReadTimeout(f"Bearer {USER_A}"),
        ],
        ids=["401", "500", "connect-error", "timeout"],
    )
    async def test_caller_token_absent_from_logs_and_errors(self, router, caplog, failure):
        """GIVEN caller and owner fail THEN the caller token is in no log, message or chain."""
        registry = _registry()
        router.side_effect = _by_bearer({USER_A: failure, OWNER_KEY: httpx.Response(401)})

        with caplog.at_level(logging.DEBUG), pytest.raises(AgentCardRegistryError) as info:
            await registry.get(deployment_id=DEP, auth_headers=_headers(USER_A))

        exc = info.value
        rendered = "".join(traceback.format_exception(exc))
        for text in (caplog.text, str(exc), rendered):
            assert USER_A not in text
            assert USER_A[:8] not in text
        for link in _exception_chain(exc):
            request = getattr(link, "request", None)
            assert request is None or USER_A not in request.headers["Authorization"]


class TestDeactivatedOwnerKeyMemoryService:
    """The L2 MemorySpace client is built from the owner key; a dead key must not crash lookups."""

    @pytest.fixture
    def dead_l2(self):
        kv = MemorySpaceKVCache(memory_space_id="space-1")
        unauthorized = RuntimeError("401 Unauthorized: API key is deactivated")
        with patch.object(kv, "_resolve_space", AsyncMock(side_effect=unauthorized)):
            yield LayeredAgentCardCacheBackend(
                MemoryAgentCardCacheBackend(), MemorySpaceAgentCardCacheBackend(kv_cache=kv)
            )

    async def test_kv_operations_swallow_memory_service_errors(self):
        """GIVEN the memory service rejects the key THEN L2 read/write/delete never raise."""
        kv = MemorySpaceKVCache(memory_space_id="space-1")
        unauthorized = RuntimeError("401 Unauthorized: API key is deactivated")

        with patch.object(kv, "_resolve_space", AsyncMock(side_effect=unauthorized)):
            assert await kv.get_value("deployment:dep-1") is None
            await kv.set_value("deployment:dep-1", "{}")
            await kv.delete_value("deployment:dep-1")

    async def test_caller_token_lookup_succeeds_with_dead_memory_service(self, router, dead_l2):
        """GIVEN L2 rejects the owner key WHEN an XAA caller misses L1 THEN the card is returned."""
        registry = _registry(dead_l2)
        router.side_effect = _by_bearer({USER_A: _ok("for-a")})

        card = await registry.get(deployment_id=DEP, auth_headers=_headers(USER_A))
        await dead_l2.flush_l2_tasks()

        assert card.name == "for-a"
        assert _tokens_sent(router) == [USER_A]

    async def test_owner_lookup_survives_failing_l2_write_behind_and_evict(self, router, dead_l2):
        """GIVEN L2 rejects the owner key THEN store (write-behind) and evict do not raise."""
        registry = _registry(dead_l2)
        router.side_effect = _by_bearer({OWNER_KEY: _ok("owner")})

        card = await registry.get(deployment_id=DEP)
        await dead_l2.flush_l2_tasks()
        await dead_l2.evict(DEP, key_type="deployment")

        assert card.name == "owner"
