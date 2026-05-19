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

from contextlib import contextmanager
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from nat.data_models.authentication import AuthResult
from nat.data_models.authentication import BearerTokenCred
from nat.data_models.authentication import HeaderCred
from nat.plugins.a2a.client.client_config import A2AClientConfig

from datarobot_genai.dragent.agent_card_registry import reset_default_registry
from datarobot_genai.dragent.plugins.auth_a2a_client import A2ADiscoveryAuthMixin
from datarobot_genai.dragent.plugins.auth_a2a_client import AgentCardRegistryLookup
from datarobot_genai.dragent.plugins.auth_a2a_client import AuthenticatedA2AClientConfig
from datarobot_genai.dragent.plugins.auth_a2a_client import AuthenticatedA2AClientFunctionGroup
from datarobot_genai.dragent.plugins.auth_a2a_client import _AuthenticatedA2ABaseClient
from datarobot_genai.dragent.plugins.auth_a2a_client import _extract_auth_headers

_AGENT_URL = "http://agent.example.com"

_MODULE = "datarobot_genai.dragent.plugins.auth_a2a_client"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@contextmanager
def _skip_agent_card_resolution(client):
    """Patch _resolve_agent_card to set a mock agent card without network access."""

    async def _set_mock_card():
        mock_card = MagicMock()
        # Ensure A2ACredentialService skips compatibility validation (no security schemes).
        mock_card.security_schemes = None
        client._agent_card = mock_card

    with patch.object(client, "_resolve_agent_card", side_effect=_set_mock_card):
        yield


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def a2a_config():
    return AuthenticatedA2AClientConfig(url=_AGENT_URL)


@pytest.fixture
def mock_builder():
    return AsyncMock()


@pytest.fixture
def bearer_auth_provider():
    provider = AsyncMock()
    provider.authenticate.return_value = AuthResult(
        credentials=[BearerTokenCred(token="test_token")]
    )
    return provider


@pytest.fixture
def mixin_auth_provider():
    """Auth provider that implements A2ADiscoveryAuthMixin."""

    class _MockDiscoveryProvider(A2ADiscoveryAuthMixin):
        async def authenticate_for_discovery(self, user_id=None):
            return {"Authorization": "Bearer discovery_token"}

        authenticate = AsyncMock(
            return_value=AuthResult(credentials=[BearerTokenCred(token="call_token")])
        )

    return _MockDiscoveryProvider()


@pytest.fixture
def patched_base_client_env():
    """Patch httpx, ClientFactory, and Context for _AuthenticatedA2ABaseClient tests."""
    with (
        patch(f"{_MODULE}.httpx") as mock_httpx,
        patch(f"{_MODULE}.ClientFactory") as mock_factory,
        patch(f"{_MODULE}.Context") as mock_ctx,
    ):
        mock_httpx.AsyncClient.return_value = MagicMock(aclose=AsyncMock())
        mock_factory.return_value.create.return_value = MagicMock(aclose=AsyncMock())
        mock_ctx.get.return_value.user_id = "test-user"
        yield mock_httpx, mock_factory


@pytest.fixture
def patched_fg_env():
    """Patch Context, _AuthenticatedA2ABaseClient, and _register_functions."""
    with (
        patch(f"{_MODULE}.Context") as mock_ctx,
        patch(f"{_MODULE}._AuthenticatedA2ABaseClient") as mock_cls,
        patch.object(AuthenticatedA2AClientFunctionGroup, "_register_functions"),
    ):
        mock_ctx.get.return_value.user_id = "test-user"
        yield mock_cls


@pytest.fixture(autouse=True)
def _reset_registry():
    """Reset the agent card registry singleton between tests."""
    reset_default_registry()
    yield
    reset_default_registry()


# ---------------------------------------------------------------------------
# Tests: _extract_auth_headers
# ---------------------------------------------------------------------------


class TestExtractAuthHeaders:
    def test_bearer_token(self):
        auth_result = AuthResult(credentials=[BearerTokenCred(token="test_token")])
        assert _extract_auth_headers(auth_result) == {"Authorization": "Bearer test_token"}

    def test_header_cred(self):
        auth_result = AuthResult(credentials=[HeaderCred(name="X-Api-Key", value="my_api_key")])
        assert _extract_auth_headers(auth_result) == {"X-Api-Key": "my_api_key"}

    def test_empty(self):
        assert _extract_auth_headers(AuthResult(credentials=[])) == {}


# ---------------------------------------------------------------------------
# Tests: A2ADiscoveryAuthMixin
# ---------------------------------------------------------------------------


class TestA2ADiscoveryAuthMixin:
    def test_is_abstract(self):
        """A2ADiscoveryAuthMixin cannot be instantiated without implementing the abstract method."""
        with pytest.raises(TypeError):
            A2ADiscoveryAuthMixin()  # type: ignore[abstract]

    def test_concrete_subclass_is_recognised(self, mixin_auth_provider):
        assert isinstance(mixin_auth_provider, A2ADiscoveryAuthMixin)


# ---------------------------------------------------------------------------
# Tests: AuthenticatedA2AClientConfig
# ---------------------------------------------------------------------------


class TestAgentCardRegistryLookup:
    def test_deployment_id_only(self):
        lookup = AgentCardRegistryLookup(deployment_id="dep-123")
        assert lookup.deployment_id == "dep-123"
        assert lookup.external_id is None

    def test_external_id_only(self):
        lookup = AgentCardRegistryLookup(external_id="ext-456")
        assert lookup.external_id == "ext-456"
        assert lookup.deployment_id is None

    def test_both_ids_raises(self):
        with pytest.raises(ValueError, match="not both"):
            AgentCardRegistryLookup(deployment_id="dep-1", external_id="ext-1")

    def test_neither_id_raises(self):
        with pytest.raises(ValueError, match="requires exactly one"):
            AgentCardRegistryLookup()


class TestAuthenticatedA2AClientConfig:
    def test_is_a2a_client_config(self, a2a_config):
        assert isinstance(a2a_config, A2AClientConfig)

    def test_url_only_is_valid(self):
        cfg = AuthenticatedA2AClientConfig(url=_AGENT_URL)
        assert str(cfg.url).rstrip("/") == _AGENT_URL

    def test_registry_only_is_valid(self):
        cfg = AuthenticatedA2AClientConfig(
            registry=AgentCardRegistryLookup(deployment_id="dep-123")
        )
        assert cfg.registry is not None
        assert cfg.registry.deployment_id == "dep-123"
        assert cfg.url is None

    def test_registry_with_external_id(self):
        cfg = AuthenticatedA2AClientConfig(registry=AgentCardRegistryLookup(external_id="ext-456"))
        assert cfg.registry is not None
        assert cfg.registry.external_id == "ext-456"

    def test_url_and_registry_raises(self):
        with pytest.raises(ValueError, match="not both"):
            AuthenticatedA2AClientConfig(
                url=_AGENT_URL,
                registry=AgentCardRegistryLookup(deployment_id="dep-1"),
            )

    def test_neither_url_nor_registry_raises(self):
        with pytest.raises(ValueError, match="Either 'url' or 'registry'"):
            AuthenticatedA2AClientConfig()


# ---------------------------------------------------------------------------
# Tests: _AuthenticatedA2ABaseClient — _resolve_agent_card dispatch
# ---------------------------------------------------------------------------


class TestAuthenticatedA2ABaseClientResolveCard:
    """Unit-tests for the two-branch discovery-auth dispatch in _resolve_agent_card."""

    @pytest.fixture
    def patched_resolver_env(self):
        """Patch A2ACardResolver and Context; yield (mock_resolver_cls, mock_ctx)."""
        with (
            patch(f"{_MODULE}.A2ACardResolver") as mock_resolver_cls,
            patch(f"{_MODULE}.Context") as mock_ctx,
            patch(f"{_MODULE}.httpx") as mock_httpx,
        ):
            mock_httpx.AsyncClient.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
            mock_httpx.AsyncClient.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_httpx.Timeout = MagicMock()
            mock_resolver_cls.return_value.get_agent_card = AsyncMock(return_value=MagicMock())
            mock_ctx.get.return_value.user_id = "test-user"
            yield mock_resolver_cls, mock_ctx, mock_httpx

    async def test_no_provider_unauthenticated(self, patched_resolver_env):
        """When no auth_provider is set, card is fetched with empty headers."""
        _, _, mock_httpx = patched_resolver_env
        client = _AuthenticatedA2ABaseClient(base_url=_AGENT_URL, auth_provider=None)
        await client._resolve_agent_card()
        _, httpx_kwargs = mock_httpx.AsyncClient.call_args
        assert httpx_kwargs.get("headers", {}) == {}

    async def test_plain_provider_calls_authenticate(
        self, patched_resolver_env, bearer_auth_provider
    ):
        """Provider without mixin → authenticate() called, headers extracted."""
        _, _, mock_httpx = patched_resolver_env
        client = _AuthenticatedA2ABaseClient(
            base_url=_AGENT_URL, auth_provider=bearer_auth_provider
        )
        await client._resolve_agent_card()
        bearer_auth_provider.authenticate.assert_awaited_once()
        _, httpx_kwargs = mock_httpx.AsyncClient.call_args
        assert httpx_kwargs["headers"]["Authorization"] == "Bearer test_token"

    async def test_mixin_provider_calls_authenticate_for_discovery(
        self, patched_resolver_env, mixin_auth_provider
    ):
        """Provider with mixin → authenticate_for_discovery() called; authenticate() NOT called."""
        _, _, mock_httpx = patched_resolver_env
        client = _AuthenticatedA2ABaseClient(base_url=_AGENT_URL, auth_provider=mixin_auth_provider)
        await client._resolve_agent_card()
        mixin_auth_provider.authenticate.assert_not_awaited()
        _, httpx_kwargs = mock_httpx.AsyncClient.call_args
        assert httpx_kwargs["headers"]["Authorization"] == "Bearer discovery_token"


# ---------------------------------------------------------------------------
# Tests: _AuthenticatedA2ABaseClient — __aenter__ (call phase)
# ---------------------------------------------------------------------------


class TestAuthenticatedA2ABaseClientCallPhase:
    async def test_call_auth_uses_auth_interceptor(
        self, bearer_auth_provider, patched_base_client_env
    ):
        """When auth_provider is set, AuthInterceptor is added for RPC calls."""
        mock_httpx, mock_factory = patched_base_client_env
        client = _AuthenticatedA2ABaseClient(
            base_url=_AGENT_URL, auth_provider=bearer_auth_provider
        )
        with _skip_agent_card_resolution(client):
            async with client:
                create_kw = mock_factory.return_value.create.call_args.kwargs
                assert len(create_kw["interceptors"]) == 1

    async def test_no_auth_empty_interceptors(self, patched_base_client_env):
        """When no auth_provider, no interceptors are added."""
        mock_httpx, mock_factory = patched_base_client_env
        client = _AuthenticatedA2ABaseClient(base_url=_AGENT_URL, auth_provider=None)
        with _skip_agent_card_resolution(client):
            async with client:
                create_kw = mock_factory.return_value.create.call_args.kwargs
                assert create_kw.get("interceptors") == []

    async def test_task_httpx_client_has_no_default_headers(
        self, bearer_auth_provider, patched_base_client_env
    ):
        """The long-lived task httpx client carries no default auth headers."""
        mock_httpx, _ = patched_base_client_env
        client = _AuthenticatedA2ABaseClient(
            base_url=_AGENT_URL, auth_provider=bearer_auth_provider
        )
        with _skip_agent_card_resolution(client):
            async with client:
                _, httpx_kwargs = mock_httpx.AsyncClient.call_args
                assert httpx_kwargs.get("headers", {}) == {}


# ---------------------------------------------------------------------------
# Tests: AuthenticatedA2AClientFunctionGroup
# ---------------------------------------------------------------------------


class TestAuthenticatedA2AClientFunctionGroup:
    async def test_uses_authenticated_base_client(self, a2a_config, mock_builder, patched_fg_env):
        fg = AuthenticatedA2AClientFunctionGroup(config=a2a_config, builder=mock_builder)
        result = await fg.__aenter__()

        assert result is fg
        patched_fg_env.assert_called_once()
        assert patched_fg_env.call_args.kwargs["auth_provider"] is None

    async def test_resolves_and_passes_auth_provider(self, mock_builder, patched_fg_env):
        mock_auth_provider = MagicMock()
        mock_builder.get_auth_provider.return_value = mock_auth_provider
        config = AuthenticatedA2AClientConfig(url=_AGENT_URL, auth_provider="my_auth")

        fg = AuthenticatedA2AClientFunctionGroup(config=config, builder=mock_builder)
        await fg.__aenter__()

        mock_builder.get_auth_provider.assert_awaited_once_with("my_auth")
        assert patched_fg_env.call_args.kwargs["auth_provider"] is mock_auth_provider

    async def test_raises_when_no_user_id(self, a2a_config, mock_builder):
        with patch(f"{_MODULE}.Context") as mock_ctx:
            mock_ctx.get.return_value.user_id = None
            fg = AuthenticatedA2AClientFunctionGroup(config=a2a_config, builder=mock_builder)
            with pytest.raises(RuntimeError, match="User ID not found in context"):
                await fg.__aenter__()


# ---------------------------------------------------------------------------
# Tests: AuthenticatedA2AClientFunctionGroup — registry path
# ---------------------------------------------------------------------------


class TestAuthenticatedA2AClientFunctionGroupRegistry:
    @pytest.fixture
    def registry_config(self):
        return AuthenticatedA2AClientConfig(
            registry=AgentCardRegistryLookup(deployment_id="dep-001"),
            auth_provider="my_auth",
        )

    @pytest.fixture
    def registry_config_external(self):
        return AuthenticatedA2AClientConfig(
            registry=AgentCardRegistryLookup(external_id="ext-001"),
        )

    async def test_registry_fetches_card_and_derives_base_url(
        self, registry_config, mock_builder, patched_fg_env
    ):
        mock_auth_provider = MagicMock()
        mock_builder.get_auth_provider.return_value = mock_auth_provider

        mock_card = MagicMock()
        mock_card.url = "https://agent.example.com/a2a/"

        mock_registry = AsyncMock()
        mock_registry.get = AsyncMock(return_value=mock_card)

        with patch(
            f"{_MODULE}.get_default_registry", new_callable=AsyncMock, return_value=mock_registry
        ):
            fg = AuthenticatedA2AClientFunctionGroup(config=registry_config, builder=mock_builder)
            await fg.__aenter__()

            mock_registry.get.assert_awaited_once_with(
                deployment_id="dep-001",
                external_id=None,
            )
            # Verify base_url derived from card.url
            assert patched_fg_env.call_args.kwargs["base_url"] == "https://agent.example.com/a2a/"

    async def test_registry_external_id_path(
        self, registry_config_external, mock_builder, patched_fg_env
    ):
        mock_card = MagicMock()
        mock_card.url = "https://agent.example.com/a2a/"

        mock_registry = AsyncMock()
        mock_registry.get = AsyncMock(return_value=mock_card)

        with patch(
            f"{_MODULE}.get_default_registry", new_callable=AsyncMock, return_value=mock_registry
        ):
            fg = AuthenticatedA2AClientFunctionGroup(
                config=registry_config_external, builder=mock_builder
            )
            await fg.__aenter__()

            mock_registry.get.assert_awaited_once_with(
                deployment_id=None,
                external_id="ext-001",
            )

    async def test_registry_pre_resolved_card_skips_discovery(self, registry_config, mock_builder):
        """When registry provides a card, _AuthenticatedA2ABaseClient skips _resolve_agent_card."""
        mock_auth_provider = MagicMock()
        mock_builder.get_auth_provider.return_value = mock_auth_provider

        mock_card = MagicMock()
        mock_card.url = "https://agent.example.com/a2a/"
        mock_card.security_schemes = None

        mock_registry = AsyncMock()
        mock_registry.get = AsyncMock(return_value=mock_card)

        with (
            patch(
                f"{_MODULE}.get_default_registry",
                new_callable=AsyncMock,
                return_value=mock_registry,
            ),
            patch(f"{_MODULE}.Context") as mock_ctx,
            patch(f"{_MODULE}.httpx") as mock_httpx,
            patch(f"{_MODULE}.ClientFactory") as mock_factory,
            patch.object(AuthenticatedA2AClientFunctionGroup, "_register_functions"),
        ):
            mock_ctx.get.return_value.user_id = "test-user"
            mock_httpx.AsyncClient.return_value = MagicMock(aclose=AsyncMock())
            mock_httpx.Timeout = MagicMock()
            mock_factory.return_value.create.return_value = MagicMock(aclose=AsyncMock())

            fg = AuthenticatedA2AClientFunctionGroup(config=registry_config, builder=mock_builder)
            await fg.__aenter__()

            # The client should have the pre-resolved card, not call _resolve_agent_card
            assert fg._client._agent_card is mock_card
