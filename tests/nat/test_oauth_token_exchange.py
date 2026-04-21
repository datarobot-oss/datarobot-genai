# Copyright 2025 DataRobot, Inc. and its affiliates.
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

from unittest.mock import AsyncMock, patch

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from datarobot_genai.nat.datarobot_oauth_token_exchange import (
    GRANT_TYPE_TOKEN_EXCHANGE,
    SUBJECT_TOKEN_TYPE_ACCESS_TOKEN,
    OAuth2TokenExchangeConfig,
    OAuth2TokenExchangeProvider,
)

_PATCH_TARGET = "datarobot_genai.nat.datarobot_oauth_token_exchange.httpx.AsyncClient"


def _response(json: dict) -> httpx.Response:
    return httpx.Response(200, json=json, request=httpx.Request("POST", "https://fake"))


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def base_config_kwargs():
    return {
        "client_id": "my-client",
        "client_secret": SecretStr("my-secret"),
        "token_url": "https://auth.example.com/oauth2/token",
    }


@pytest.fixture
def config(base_config_kwargs):
    return OAuth2TokenExchangeConfig(**base_config_kwargs)


@pytest.fixture
def provider(config):
    return OAuth2TokenExchangeProvider(config=config)


@pytest.fixture
def mock_http(token_response):
    """Yields a mock AsyncClient pre-configured with a successful exchange response."""
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.post.return_value = _response(token_response)
    with patch(_PATCH_TARGET, return_value=client):
        yield client


@pytest.fixture
def token_response():
    return {"access_token": "exchanged-token-xyz", "token_type": "Bearer", "expires_in": 3600}


# ── Config tests ──────────────────────────────────────────────────────────────

class TestOAuth2TokenExchangeConfig:
    def test_valid_minimal(self, config):
        assert config.token_url == "https://auth.example.com/oauth2/token"
        assert config.client_id == "my-client"

    def test_rejects_no_endpoint_source(self):
        with pytest.raises(ValidationError, match="Cannot resolve the token endpoint"):
            OAuth2TokenExchangeConfig(client_id="c", client_secret=SecretStr("s"))

    @pytest.mark.parametrize("field", ["token_url", "discovery_url", "issuer_url"])
    def test_rejects_http_non_localhost(self, field):
        with pytest.raises(ValidationError, match="must be HTTPS"):
            OAuth2TokenExchangeConfig(
                client_id="c", client_secret=SecretStr("s"),
                **{field: "http://remote.example.com/token"},
            )

    def test_allows_http_localhost(self):
        cfg = OAuth2TokenExchangeConfig(
            client_id="c", client_secret=SecretStr("s"),
            token_url="http://localhost:8080/token",
        )
        assert cfg.token_url == "http://localhost:8080/token"

    def test_forbids_extra_fields(self, base_config_kwargs):
        with pytest.raises(ValidationError):
            OAuth2TokenExchangeConfig(**base_config_kwargs, unknown_field="boom")


# ── Provider tests ────────────────────────────────────────────────────────────

class TestOAuth2TokenExchangeProvider:
    @pytest.mark.asyncio
    async def test_exchange_request_shape(self, provider, mock_http, token_response):
        """POST body must follow RFC 8693 §2.1."""
        result = await provider.authenticate(subject_token="original-token")

        body = mock_http.post.call_args.kwargs["data"]
        assert body["grant_type"] == GRANT_TYPE_TOKEN_EXCHANGE
        assert body["subject_token"] == "original-token"
        assert body["subject_token_type"] == SUBJECT_TOKEN_TYPE_ACCESS_TOKEN

        assert result.credentials[0].token.get_secret_value() == "exchanged-token-xyz"
        assert result.token_expires_at is not None
        assert result.raw == token_response

    @pytest.mark.asyncio
    async def test_audience_and_scopes_included(self, base_config_kwargs, mock_http):
        cfg = OAuth2TokenExchangeConfig(
            **base_config_kwargs,
            audience="https://api.downstream.com",
            scopes=["read", "write"],
        )
        await OAuth2TokenExchangeProvider(config=cfg).authenticate(subject_token="tok")

        body = mock_http.post.call_args.kwargs["data"]
        assert body["audience"] == "https://api.downstream.com"
        assert body["scope"] == "read write"

    @pytest.mark.asyncio
    async def test_missing_subject_token_raises(self, provider):
        with pytest.raises(ValueError, match="subject_token is required"):
            await provider.authenticate()

    @pytest.mark.asyncio
    async def test_no_expiry_when_expires_in_absent(self, provider):
        client = AsyncMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        client.post.return_value = _response({"access_token": "tok", "token_type": "Bearer"})

        with patch(_PATCH_TARGET, return_value=client):
            result = await provider.authenticate(subject_token="tok")

        assert result.token_expires_at is None

    @pytest.mark.asyncio
    async def test_resolves_token_url_via_oidc_discovery(self):
        """token_url is derived from issuer_url via OIDC discovery."""
        cfg = OAuth2TokenExchangeConfig(
            client_id="c", client_secret=SecretStr("s"),
            issuer_url="https://issuer.example.com",
        )
        client = AsyncMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        client.get.return_value = _response({"token_endpoint": "https://issuer.example.com/oauth2/token"})
        client.post.return_value = _response({"access_token": "new-tok", "token_type": "Bearer"})

        with patch(_PATCH_TARGET, return_value=client):
            result = await OAuth2TokenExchangeProvider(config=cfg).authenticate(subject_token="tok")

        client.get.assert_called_once_with(
            "https://issuer.example.com/.well-known/openid-configuration"
        )
        assert result.credentials[0].token.get_secret_value() == "new-tok"

    @pytest.mark.asyncio
    async def test_real_world_discovery_response(self):
        """Real-world OIDC discovery response works with our implementation."""
        discovery_response = {
            "issuer": "https://idp.example.com/oauth2/default",
            "authorization_endpoint": "https://idp.example.com/oauth2/default/v1/authorize",
            "token_endpoint": "https://idp.example.com/oauth2/default/v1/token",
            "jwks_uri": "https://idp.example.com/oauth2/default/v1/keys",
            "grant_types_supported": [
                "authorization_code",
                "implicit",
                "refresh_token",
                "urn:ietf:params:oauth:grant-type:jwt-bearer"
            ],
            "introspection_endpoint": "https://idp.example.com/oauth2/default/v1/introspect",
        }
        cfg = OAuth2TokenExchangeConfig(
            client_id="test-client",
            client_secret=SecretStr("test-secret"),
            issuer_url="https://idp.example.com/oauth2/default",
        )
        client = AsyncMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        client.get.return_value = _response(discovery_response)
        client.post.return_value = _response({"access_token": "exchanged-token", "token_type": "Bearer"})

        with patch(_PATCH_TARGET, return_value=client):
            result = await OAuth2TokenExchangeProvider(config=cfg).authenticate(subject_token="incoming-token")

        # Verify discovery was called with derived URL
        client.get.assert_called_once_with(
            "https://idp.example.com/oauth2/default/.well-known/openid-configuration"
        )
        # Verify token exchange was POSTed to the correct token endpoint
        assert client.post.call_args[0][0] == "https://idp.example.com/oauth2/default/v1/token"
        assert result.credentials[0].token.get_secret_value() == "exchanged-token"
