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

from datarobot_genai.dragent.plugins.auth_a2a_client import AuthenticatedA2AClientConfig
from datarobot_genai.dragent.plugins.auth_a2a_client import AuthenticatedA2AClientFunctionGroup
from datarobot_genai.dragent.plugins.auth_a2a_client import _AuthenticatedA2ABaseClient
from datarobot_genai.dragent.plugins.auth_a2a_client import _extract_auth_headers

_AGENT_URL = "http://agent.example.com"

_MODULE = "datarobot_genai.dragent.auth_a2a_client"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@contextmanager
def _skip_agent_card_resolution(client):
    """Patch _resolve_agent_card to set a mock agent card without network access."""

    async def _set_mock_card():
        client._agent_card = MagicMock()

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
        yield mock_httpx


@pytest.fixture
def patched_fg_env():
    """Patch Context, _AuthenticatedA2ABaseClient,
    and _register_functions for function group tests.
    """
    with (
        patch(f"{_MODULE}.Context") as mock_ctx,
        patch(f"{_MODULE}._AuthenticatedA2ABaseClient") as mock_cls,
        patch.object(AuthenticatedA2AClientFunctionGroup, "_register_functions"),
    ):
        mock_ctx.get.return_value.user_id = "test-user"
        yield mock_cls


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
# Tests: AuthenticatedA2AClientConfig
# ---------------------------------------------------------------------------


class TestAuthenticatedA2AClientConfig:
    def test_is_a2a_client_config(self, a2a_config):
        assert isinstance(a2a_config, A2AClientConfig)


# ---------------------------------------------------------------------------
# Tests: _AuthenticatedA2ABaseClient
# ---------------------------------------------------------------------------


class TestAuthenticatedA2ABaseClient:
    async def test_injects_bearer_headers(self, bearer_auth_provider, patched_base_client_env):
        client = _AuthenticatedA2ABaseClient(
            base_url=_AGENT_URL, auth_provider=bearer_auth_provider
        )
        with _skip_agent_card_resolution(client):
            async with client:
                _, httpx_kwargs = patched_base_client_env.AsyncClient.call_args
                assert httpx_kwargs["headers"]["Authorization"] == "Bearer test_token"

    async def test_no_auth_uses_empty_headers(self, patched_base_client_env):
        client = _AuthenticatedA2ABaseClient(base_url=_AGENT_URL, auth_provider=None)
        with _skip_agent_card_resolution(client):
            async with client:
                _, httpx_kwargs = patched_base_client_env.AsyncClient.call_args
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
