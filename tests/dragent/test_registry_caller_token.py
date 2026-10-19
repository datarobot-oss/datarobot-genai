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

"""Caller-token-only registry and L2 cache access (``registry_caller_token_scope``)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import httpx
import pytest
import respx

from datarobot_genai.dragent import memory_space_cache
from datarobot_genai.dragent.agent_card_registry import AgentCardRegistry
from datarobot_genai.dragent.agent_card_registry import AgentCardRegistryConfig
from datarobot_genai.dragent.agent_card_registry import AgentCardRegistryError
from datarobot_genai.dragent.agent_card_registry_backends import MemoryAgentCardCacheBackend
from datarobot_genai.dragent.inbound_token import OAUTH_ACCESS_TOKEN_HEADER
from datarobot_genai.dragent.memory_space_cache import MemorySpaceKVCache
from datarobot_genai.dragent.plugins.auth_a2a_client import RegistryCallerTokenAware
from datarobot_genai.dragent.plugins.okta_registry_auth import OktaRegistryAPIKeyAuthProvider
from datarobot_genai.dragent.plugins.okta_registry_auth import OktaRegistryDRCallAuthProviderConfig
from datarobot_genai.dragent.registry_caller_token import current_registry_caller_token
from datarobot_genai.dragent.registry_caller_token import registry_caller_token_scope

OWNER_KEY = "dummy-owner-key-O1"
OKTA = "dummy-okta-token-K4"
DEP = "dep-1"
ENDPOINT = "https://enclave.example/api/v2"


def _ok(name: str) -> httpx.Response:
    card = {
        "name": name,
        "description": "d",
        "url": "https://agent.example.com/a2a/",
        "version": "1.0.0",
        "skills": [],
        "defaultInputModes": ["text"],
        "defaultOutputModes": ["text"],
        "capabilities": {"streaming": False},
    }
    entry = {"id": "doc", "deploymentId": DEP, "agentCard": card}
    return httpx.Response(200, json={"data": [entry]})


def _registry() -> AgentCardRegistry:
    registry = AgentCardRegistry(
        config=AgentCardRegistryConfig(), api_token=OWNER_KEY, endpoint="https://ep"
    )
    registry._backend = MemoryAgentCardCacheBackend()
    return registry


@pytest.fixture
def router():
    with respx.mock(assert_all_called=False) as mock:
        yield mock.get(host="ep")


def _tokens_sent(route) -> list[str]:
    return [c.request.headers["Authorization"].removeprefix("Bearer ") for c in route.calls]


def _provider() -> OktaRegistryAPIKeyAuthProvider:
    return OktaRegistryAPIKeyAuthProvider(OktaRegistryDRCallAuthProviderConfig())


def _request_headers(headers: dict[str, str]):
    context = SimpleNamespace(metadata=SimpleNamespace(headers=headers))
    return patch(
        "datarobot_genai.dragent.plugins.okta_registry_auth.Context.get",
        return_value=context,
    )


class TestScope:
    def test_scope_sets_and_resets(self):
        assert current_registry_caller_token() is None
        with registry_caller_token_scope(OKTA):
            assert current_registry_caller_token() == OKTA
        assert current_registry_caller_token() is None

    @pytest.mark.parametrize("token", [None, ""])
    def test_empty_scope_is_a_no_op(self, token):
        with registry_caller_token_scope(token):
            assert current_registry_caller_token() is None


class TestOktaRegistryProvider:
    def test_is_registry_caller_token_aware(self):
        assert isinstance(_provider(), RegistryCallerTokenAware)

    def test_class_name_is_bearer_auth_compatible(self):
        """NAT's A2A credential service matches bearerAuth by 'APIKey' in the class name."""
        assert "APIKey" in OktaRegistryAPIKeyAuthProvider.__name__

    def test_registry_caller_token_reads_the_inbound_header(self):
        with _request_headers({OAUTH_ACCESS_TOKEN_HEADER: OKTA}):
            assert _provider().registry_caller_token() == OKTA
        with _request_headers({}):
            assert _provider().registry_caller_token() is None

    async def test_discovery_forwards_okta_bearer(self):
        with _request_headers({OAUTH_ACCESS_TOKEN_HEADER: OKTA}):
            headers = await _provider().authenticate_for_discovery("user")
        assert headers == {"Authorization": f"Bearer {OKTA}"}

    async def test_discovery_without_header_raises(self):
        with _request_headers({}), pytest.raises(RuntimeError, match=OAUTH_ACCESS_TOKEN_HEADER):
            await _provider().authenticate_for_discovery("user")

    async def test_rpc_uses_datarobot_api_token(self):
        config = SimpleNamespace(resolve_datarobot_api_token=lambda: OWNER_KEY)
        with patch(
            "datarobot_genai.dragent.plugins.okta_registry_auth.resolve_config",
            return_value=config,
        ):
            result = await _provider().authenticate("user")
        assert result is not None
        assert result.credentials[0].token.get_secret_value() == OWNER_KEY


class TestRegistryInScope:
    async def test_miss_uses_caller_token_only(self, router):
        router.side_effect = lambda request: _ok("okta")
        with registry_caller_token_scope(OKTA):
            card = await _registry().get(deployment_id=DEP)
        assert card.name == "okta"
        assert _tokens_sent(router) == [OKTA]

    async def test_caller_failure_does_not_fall_back_to_owner(self, router):
        router.side_effect = lambda request: httpx.Response(403)
        with registry_caller_token_scope(OKTA), pytest.raises(AgentCardRegistryError):
            await _registry().get(deployment_id=DEP)
        assert _tokens_sent(router) == [OKTA]

    async def test_caller_not_found_does_not_fall_back_to_owner(self, router):
        router.side_effect = lambda request: httpx.Response(200, json={"data": []})
        with registry_caller_token_scope(OKTA), pytest.raises(AgentCardRegistryError):
            await _registry().get(deployment_id=DEP)
        assert _tokens_sent(router) == [OKTA]

    async def test_scope_overrides_explicit_auth_headers(self, router):
        router.side_effect = lambda request: _ok("okta")
        with registry_caller_token_scope(OKTA):
            await _registry().get(deployment_id=DEP, auth_headers={"Authorization": "Bearer x"})
        assert _tokens_sent(router) == [OKTA]

    async def test_owner_paths_use_caller_token(self, router):
        """Prefetch (an owner-key path) sends the caller token inside a scope."""
        router.side_effect = lambda request: _ok("okta")
        with registry_caller_token_scope(OKTA):
            await _registry().prefetch(deployment_ids=[DEP])
        assert _tokens_sent(router) == [OKTA]

    async def test_outside_scope_uses_owner_key(self, router):
        router.side_effect = lambda request: _ok("owner")
        await _registry().get(deployment_id=DEP)
        assert _tokens_sent(router) == [OWNER_KEY]


class _FakeEvent:
    def __init__(self, content: str) -> None:
        self.content = content


@pytest.fixture
def memory_state():
    state = memory_space_cache._MemoryClientState
    saved = (state.client, state.endpoint, state.api_token)
    state.client, state.endpoint, state.api_token = MagicMock(), ENDPOINT, OWNER_KEY
    yield state
    state.client, state.endpoint, state.api_token = saved


class TestL2InScope:
    async def test_read_opens_the_space_with_the_caller_token(self, memory_state):
        cache = MemorySpaceKVCache(memory_space_id="space-1")
        caller_client = MagicMock()
        caller_client.__aenter__ = AsyncMock(return_value=caller_client)
        caller_client.__aexit__ = AsyncMock(return_value=None)
        space, session = object(), object()
        with (
            patch.object(
                memory_space_cache, "DRMemoryServiceClient", return_value=caller_client
            ) as client_cls,
            patch.object(
                memory_space_cache.DRMemorySpace, "get", AsyncMock(return_value=space)
            ) as get_space,
            patch.object(
                memory_space_cache.AgentCardCacheSession, "get", AsyncMock(return_value=session)
            ),
            patch.object(
                memory_space_cache.AgentCardCacheEvent,
                "last",
                AsyncMock(return_value=[_FakeEvent("cached")]),
            ),
            registry_caller_token_scope(OKTA),
        ):
            assert await cache.get_value("deployment:dep-1") == "cached"

        client_cls.assert_called_once_with(endpoint=ENDPOINT, api_token=OKTA)
        get_space.assert_awaited_once_with(caller_client, "space-1")
        caller_client.__aexit__.assert_awaited_once()
        # Caller-bound objects never land in the slots the API-key path reuses.
        assert cache._space is None
        assert cache._sessions == {}

    async def test_outside_scope_uses_the_global_client(self, memory_state):
        cache = MemorySpaceKVCache(memory_space_id="space-1")
        space = object()
        with (
            patch.object(memory_space_cache, "DRMemoryServiceClient") as client_cls,
            patch.object(
                memory_space_cache.DRMemorySpace, "get", AsyncMock(return_value=space)
            ) as get_space,
            patch.object(
                memory_space_cache.AgentCardCacheSession,
                "get",
                AsyncMock(side_effect=memory_space_cache.DRMemoryNotFoundError("nope")),
            ),
        ):
            assert await cache.get_value("deployment:dep-1") is None

        client_cls.assert_not_called()
        get_space.assert_awaited_once_with(memory_state.client, "space-1")
        assert cache._space is space


def _share_body(org_id: str, role: str = "EDITOR") -> dict[str, Any]:
    return {
        "operation": "updateRoles",
        "roles": [{"shareRecipientType": "organization", "id": org_id, "role": role}],
    }


@pytest.fixture
def memory_client():
    client = MagicMock()
    client.request = AsyncMock()
    with patch.object(memory_space_cache, "_require_memory_client", return_value=client):
        yield client


@pytest.fixture
def posted_space():
    with patch.object(
        memory_space_cache.DRMemorySpace, "post", AsyncMock(return_value=MagicMock(id="space-1"))
    ):
        yield


class TestShareOnProvision:
    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        for name in (
            "REGISTRY_L2_SHARE_WITH_ORG",
            "REGISTRY_L2_SHARE_ORG_ID",
            "REGISTRY_L2_SHARE_ROLE",
        ):
            monkeypatch.delenv(name, raising=False)

    async def test_shares_with_the_agents_own_organization(self, memory_client, posted_space):
        with patch.object(memory_space_cache, "_own_org_id", AsyncMock(return_value="org-1")):
            space_id = await memory_space_cache._provision_registry_cache_memory_space("dedup")

        assert space_id == "space-1"
        memory_client.request.assert_awaited_once_with(
            "PATCH", "space-1/sharedRoles/", json=_share_body("org-1")
        )

    async def test_configured_organization_and_role_are_used(
        self, monkeypatch, memory_client, posted_space
    ):
        monkeypatch.setenv("REGISTRY_L2_SHARE_ORG_ID", "org-main")
        monkeypatch.setenv("REGISTRY_L2_SHARE_ROLE", "CONSUMER")
        own_org = AsyncMock()
        with patch.object(memory_space_cache, "_own_org_id", own_org):
            await memory_space_cache._provision_registry_cache_memory_space("dedup")

        own_org.assert_not_awaited()
        memory_client.request.assert_awaited_once_with(
            "PATCH", "space-1/sharedRoles/", json=_share_body("org-main", "CONSUMER")
        )

    async def test_sharing_can_be_turned_off(self, monkeypatch, memory_client, posted_space):
        monkeypatch.setenv("REGISTRY_L2_SHARE_WITH_ORG", "false")

        assert await memory_space_cache._provision_registry_cache_memory_space("dedup") == (
            "space-1"
        )
        memory_client.request.assert_not_awaited()

    async def test_failed_share_never_fails_provisioning(
        self, monkeypatch, memory_client, posted_space
    ):
        """A Memory Service without the sharing proxy answers 404/405; provisioning still works."""
        monkeypatch.setenv("REGISTRY_L2_SHARE_ORG_ID", "org-main")
        memory_client.request.side_effect = RuntimeError("405 Method Not Allowed")

        assert await memory_space_cache._provision_registry_cache_memory_space("dedup") == (
            "space-1"
        )
        assert await memory_space_cache.share_registry_cache_space("space-1") is False

    async def test_own_org_id_reads_account_info(self, memory_state):
        response = MagicMock()
        response.json.return_value = {"orgId": "org-7"}
        account = MagicMock()
        account.request = AsyncMock(return_value=response)
        account.__aenter__ = AsyncMock(return_value=account)
        account.__aexit__ = AsyncMock(return_value=None)
        with patch.object(
            memory_space_cache, "DRMemoryServiceClient", return_value=account
        ) as client_cls:
            assert await memory_space_cache._own_org_id() == "org-7"

        client_cls.assert_called_once_with(
            endpoint=ENDPOINT, api_token=OWNER_KEY, base_path="account"
        )
        account.request.assert_awaited_once_with("GET", "info/")
