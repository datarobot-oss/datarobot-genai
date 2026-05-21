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

"""Tests for the XAA auth provider and its agent-card parsing helpers."""

import base64
import json
from unittest.mock import patch
from urllib.parse import parse_qs

import pytest
import respx
from a2a.types import AgentCapabilities
from a2a.types import AgentCard
from a2a.types import AgentExtension
from a2a.types import ClientCredentialsOAuthFlow
from a2a.types import OAuth2SecurityScheme
from a2a.types import OAuthFlows
from a2a.types import SecurityScheme
from httpx import Response
from nat.data_models.authentication import BearerTokenCred

from datarobot_genai.dragent.plugins.okta_a2a_auth import ApiTokenExchange
from datarobot_genai.dragent.plugins.okta_a2a_auth import (
    OAuth2CrossApplicationAccessAuthProviderConfig,
)
from datarobot_genai.dragent.plugins.okta_a2a_auth import (
    OAuth2CrossApplicationAccessOAuth2AuthProvider,
)
from datarobot_genai.dragent.plugins.okta_a2a_auth import OktaTokenExchange
from datarobot_genai.dragent.plugins.okta_a2a_auth import XAATokenExchangeImpl
from datarobot_genai.dragent.plugins.okta_a2a_auth import _get_token_exchange_impl
from datarobot_genai.dragent.plugins.okta_a2a_auth import _make_client_assertion
from datarobot_genai.dragent.plugins.okta_a2a_auth import _parse_cross_app_params
from datarobot_genai.dragent.plugins.okta_a2a_auth import get_token_exchange

_MODULE = "datarobot_genai.dragent.plugins.okta_a2a_auth"

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

_TRUSTED_ISSUER = "https://okta.example.com"
_AGENT_TOKEN_URL = "https://okta.example.com/oauth2/ausYYY/v1/token"
_TARGET_AUDIENCE = "https://api.agent.example.com"
_EXCHANGE_AUDIENCE = "https://okta.example.com/oauth2/ausYYY"
_AUTH_METHOD = "private_key_jwt"
_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:jwt-bearer"
_TARGET_ISSUER = "https://okta.example.com/oauth2/ausYYY"

_FAKE_JWK = {
    "kty": "RSA",
    "kid": "test-kid",
    "n": "abc",
    "e": "AQAB",
    "d": "def",
    "p": "ghi",
    "q": "jkl",
    "dp": "mno",
    "dq": "pqr",
    "qi": "stu",
}
_FAKE_JWK_B64 = base64.b64encode(json.dumps(_FAKE_JWK).encode()).decode()


def _make_agent_card() -> AgentCard:
    """Build a minimal AgentCard with the JWT Bearer Cross-Application Access extension."""
    cc_flow = ClientCredentialsOAuthFlow(
        token_url=_AGENT_TOKEN_URL,
        scopes={"read_data": "Read access"},
    )
    security_scheme = SecurityScheme(
        root=OAuth2SecurityScheme(
            type="oauth2",
            flows=OAuthFlows(client_credentials=cc_flow),
        )
    )
    extension = AgentExtension(
        uri="urn:ietf:params:oauth:grant-type:jwt-bearer",
        description=(
            "Two-Step Cross-Application Access execution parameters. "
            "Step 1: RFC 8693 Token Exchange prerequisite. "
            "Step 2: RFC 7523 JWT Bearer Grant."
        ),
        params={
            "ref": {"scheme": "oauth2", "flow": "clientCredentials"},
            "tokenEndpointAuthMethod": _AUTH_METHOD,
            "tokenExchange": {
                "grantType": "urn:ietf:params:oauth:grant-type:token-exchange",
                "requestedTokenType": "urn:ietf:params:oauth:token-type:id-jag",
                "trustedIssuer": _TRUSTED_ISSUER,
                "audience": _EXCHANGE_AUDIENCE,
            },
            "tokenRequest": {
                "grantType": _GRANT_TYPE,
                "audience": _TARGET_AUDIENCE,
            },
        },
    )
    return AgentCard(
        name="Test Agent",
        description="Test",
        url="https://agent.example.com/",
        version="1.0.0",
        skills=[],
        capabilities=AgentCapabilities(streaming=False, extensions=[extension]),
        default_input_modes=["text"],
        default_output_modes=["text"],
        security_schemes={"oauth2": security_scheme},
        security=[{"oauth2": ["read_data"]}],
    )


# ---------------------------------------------------------------------------
# Tests: discovery phase
# ---------------------------------------------------------------------------


class TestDiscovery:
    @pytest.fixture
    def provider(self):
        return OAuth2CrossApplicationAccessOAuth2AuthProvider(
            config=OAuth2CrossApplicationAccessAuthProviderConfig()
        )

    async def test_returns_bearer(self, provider):
        with patch(f"{_MODULE}.Context") as mock_ctx:
            mock_ctx.get.return_value.metadata.headers = {
                "x-datarobot-external-access-token": "my-okta-token"
            }
            headers = await provider.authenticate_for_discovery()
        assert headers == {"Authorization": "Bearer my-okta-token"}

    async def test_raises_when_header_missing(self, provider):
        with patch(f"{_MODULE}.Context") as mock_ctx:
            mock_ctx.get.return_value.metadata.headers = {}
            with pytest.raises(RuntimeError, match="x-datarobot-external-access-token"):
                await provider.authenticate_for_discovery()


# ---------------------------------------------------------------------------
# Tests: fallback token headers
# ---------------------------------------------------------------------------


class TestFallbackHeaders:
    @pytest.fixture
    def provider(self):
        return OAuth2CrossApplicationAccessOAuth2AuthProvider(
            config=OAuth2CrossApplicationAccessAuthProviderConfig()
        )

    async def test_authorization_fallback(self, provider):
        with patch(f"{_MODULE}.Context") as mock_ctx:
            mock_ctx.get.return_value.metadata.headers = {
                "authorization": "Bearer fallback-token-123"
            }
            headers = await provider.authenticate_for_discovery()
        assert headers == {"Authorization": "Bearer fallback-token-123"}

    async def test_primary_takes_precedence(self, provider):
        with patch(f"{_MODULE}.Context") as mock_ctx:
            mock_ctx.get.return_value.metadata.headers = {
                "x-datarobot-external-access-token": "primary-token",
                "authorization": "Bearer fallback-token",
            }
            headers = await provider.authenticate_for_discovery()
        assert headers == {"Authorization": "Bearer primary-token"}

    async def test_strips_bearer_prefix_case_insensitive(self, provider):
        with patch(f"{_MODULE}.Context") as mock_ctx:
            mock_ctx.get.return_value.metadata.headers = {"authorization": "BEARER my-token"}
            headers = await provider.authenticate_for_discovery()
        assert headers == {"Authorization": "Bearer my-token"}

    async def test_fallback_disabled_when_empty_list(self):
        provider = OAuth2CrossApplicationAccessOAuth2AuthProvider(
            config=OAuth2CrossApplicationAccessAuthProviderConfig(fallback_token_headers=[])
        )
        with patch(f"{_MODULE}.Context") as mock_ctx:
            mock_ctx.get.return_value.metadata.headers = {"authorization": "Bearer token"}
            with pytest.raises(RuntimeError, match="x-datarobot-external-access-token"):
                await provider.authenticate_for_discovery()


# ---------------------------------------------------------------------------
# Tests: set_agent_card / param parsing
# ---------------------------------------------------------------------------


class TestSetAgentCard:
    def test_parses_params(self):
        provider = OAuth2CrossApplicationAccessOAuth2AuthProvider(
            config=OAuth2CrossApplicationAccessAuthProviderConfig()
        )
        provider.set_agent_card(_make_agent_card())

        params = provider._flow_params
        assert params is not None
        assert params.token_url == _AGENT_TOKEN_URL
        assert params.trusted_issuer == _TRUSTED_ISSUER
        assert params.exchange_audience == _EXCHANGE_AUDIENCE
        assert params.target_audience == _TARGET_AUDIENCE
        assert params.token_endpoint_auth_method == _AUTH_METHOD
        assert params.id_jag_scopes == ["read_data"]


class TestParseCrossAppParams:
    def test_camel_case_params(self):
        """Extension params in camelCase (canonical — matches agent card generation)."""
        params = _parse_cross_app_params(_make_agent_card())

        assert params.token_url == _AGENT_TOKEN_URL
        assert params.trusted_issuer == _TRUSTED_ISSUER
        assert params.exchange_audience == _EXCHANGE_AUDIENCE
        assert params.target_audience == _TARGET_AUDIENCE
        assert params.token_endpoint_auth_method == _AUTH_METHOD
        assert params.id_jag_scopes == ["read_data"]

    def test_snake_case_params(self):
        """Extension params in snake_case (backward compatibility)."""
        cc_flow = ClientCredentialsOAuthFlow(
            token_url=_AGENT_TOKEN_URL,
            scopes={"read_data": "Read access"},
        )
        security_scheme = SecurityScheme(
            root=OAuth2SecurityScheme(type="oauth2", flows=OAuthFlows(client_credentials=cc_flow))
        )
        extension = AgentExtension(
            uri="urn:ietf:params:oauth:grant-type:jwt-bearer",
            description="",
            params={
                "token_endpoint_auth_method": _AUTH_METHOD,
                "token_exchange": {
                    "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
                    "requested_token_type": "urn:ietf:params:oauth:token-type:id-jag",
                    "trusted_issuer": _TRUSTED_ISSUER,
                    "audience": _EXCHANGE_AUDIENCE,
                },
                "token_request": {
                    "grant_type": _GRANT_TYPE,
                    "audience": _TARGET_AUDIENCE,
                },
            },
        )
        card = AgentCard(
            name="Test Agent",
            description="Test",
            url="https://agent.example.com/",
            version="1.0.0",
            skills=[],
            capabilities=AgentCapabilities(streaming=False, extensions=[extension]),
            default_input_modes=["text"],
            default_output_modes=["text"],
            security_schemes={"oauth2": security_scheme},
            security=[{"oauth2": ["read_data"]}],
        )
        params = _parse_cross_app_params(card)
        assert params.token_url == _AGENT_TOKEN_URL
        assert params.trusted_issuer == _TRUSTED_ISSUER
        assert params.exchange_audience == _EXCHANGE_AUDIENCE
        assert params.target_audience == _TARGET_AUDIENCE
        assert params.token_endpoint_auth_method == _AUTH_METHOD

    def test_scopes_come_from_card(self):
        cc_flow = ClientCredentialsOAuthFlow(
            token_url=_AGENT_TOKEN_URL,
            scopes={"dr.impersonation": "Impersonation scope", "openid": "OpenID scope"},
        )
        security_scheme = SecurityScheme(
            root=OAuth2SecurityScheme(type="oauth2", flows=OAuthFlows(client_credentials=cc_flow))
        )
        extension = AgentExtension(
            uri="urn:ietf:params:oauth:grant-type:jwt-bearer",
            description="",
            params={
                "token_endpoint_auth_method": _AUTH_METHOD,
                "token_exchange": {
                    "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
                    "requested_token_type": "urn:ietf:params:oauth:token-type:id-jag",
                    "trusted_issuer": _TRUSTED_ISSUER,
                    "audience": _EXCHANGE_AUDIENCE,
                },
                "token_request": {
                    "grant_type": _GRANT_TYPE,
                    "audience": _TARGET_AUDIENCE,
                },
            },
        )
        card = AgentCard(
            name="Test Agent",
            description="Test",
            url="https://agent.example.com/",
            version="1.0.0",
            skills=[],
            capabilities=AgentCapabilities(streaming=False, extensions=[extension]),
            default_input_modes=["text"],
            default_output_modes=["text"],
            security_schemes={"oauth2": security_scheme},
            security=[{"oauth2": ["dr.impersonation"]}],
        )
        params = _parse_cross_app_params(card)
        assert set(params.id_jag_scopes) == {"dr.impersonation", "openid"}


class TestConfigSerialization:
    def test_private_jwk_survives_json_roundtrip(self):
        cfg = OAuth2CrossApplicationAccessAuthProviderConfig(
            principal_id="principal_abc",
            private_jwk=_FAKE_JWK_B64,
        )
        payload = cfg.model_dump(mode="json")
        assert payload["private_jwk"] == _FAKE_JWK_B64
        restored = OAuth2CrossApplicationAccessAuthProviderConfig.model_validate(payload)
        assert restored.private_jwk is not None
        assert restored.private_jwk.get_secret_value() == _FAKE_JWK_B64


class TestTokenExchangeImplSelection:
    def test_default_is_okta_sdk(self, monkeypatch):
        monkeypatch.delenv("XAA_TOKEN_EXCHANGE_IMPL", raising=False)
        assert _get_token_exchange_impl() is XAATokenExchangeImpl.OKTA_SDK

    def test_http_from_env(self, monkeypatch):
        monkeypatch.setenv("XAA_TOKEN_EXCHANGE_IMPL", "http")
        assert _get_token_exchange_impl() is XAATokenExchangeImpl.HTTP

    def test_invalid_raises(self, monkeypatch):
        monkeypatch.setenv("XAA_TOKEN_EXCHANGE_IMPL", "banana")
        with pytest.raises(ValueError, match="XAA_TOKEN_EXCHANGE_IMPL"):
            _get_token_exchange_impl()

    def test_dispatches_http(self, monkeypatch):
        monkeypatch.setenv("XAA_TOKEN_EXCHANGE_IMPL", "http")
        exchange = get_token_exchange(OAuth2CrossApplicationAccessAuthProviderConfig())
        assert isinstance(exchange, ApiTokenExchange)

    def test_dispatches_okta_sdk(self, monkeypatch):
        monkeypatch.setenv("XAA_TOKEN_EXCHANGE_IMPL", "okta_sdk")
        exchange = get_token_exchange(OAuth2CrossApplicationAccessAuthProviderConfig())
        assert isinstance(exchange, OktaTokenExchange)


class TestMakeClientAssertion:
    """Tests for _make_client_assertion error sanitization."""

    def test_invalid_jwk_raises_without_key_material(self):
        bad_jwk = {"kty": "RSA", "n": "INVALID"}
        with pytest.raises(ValueError, match="Failed to load RSA private key"):
            _make_client_assertion(
                principal_id="principal_abc",
                token_url="https://token.example.com",
                private_jwk=bad_jwk,
            )

    def test_no_chained_cause(self):
        """'raise … from None' prevents JWK material in traceback."""
        bad_jwk = {"kty": "RSA", "n": "INVALID"}
        with pytest.raises(ValueError) as exc_info:
            _make_client_assertion(
                principal_id="principal_abc",
                token_url="https://token.example.com",
                private_jwk=bad_jwk,
            )
        assert exc_info.value.__cause__ is None
        assert exc_info.value.__suppress_context__ is True


# ---------------------------------------------------------------------------
# Tests: authenticate (call phase) — delegates to get_token_exchange
# ---------------------------------------------------------------------------


class TestAuthenticate:
    """Tests for the provider's authenticate() method.

    authenticate() now delegates to get_token_exchange().exchange_token(),
    so we mock the module-level _make_client_assertion and use respx for HTTP.
    The default impl is OKTA_SDK, so we force HTTP for these tests.
    """

    _ORG_AS_TOKEN_URL = _TRUSTED_ISSUER + "/oauth2/v1/token"
    _CUSTOM_AS_TOKEN_URL = _AGENT_TOKEN_URL

    @pytest.fixture
    def provider_with_card(self, monkeypatch):
        monkeypatch.setenv("XAA_TOKEN_EXCHANGE_IMPL", "http")
        config = OAuth2CrossApplicationAccessAuthProviderConfig(
            principal_id="principal_abc",
            private_jwk=_FAKE_JWK_B64,
        )
        provider = OAuth2CrossApplicationAccessOAuth2AuthProvider(config=config)
        provider.set_agent_card(_make_agent_card())
        return provider

    async def test_returns_bearer_cred(self, provider_with_card):
        with (
            patch(f"{_MODULE}.Context") as mock_ctx,
            patch(f"{_MODULE}._make_client_assertion", side_effect=["a1", "a2"]),
            respx.mock as mock_http,
        ):
            mock_ctx.get.return_value.metadata.headers = {
                "x-datarobot-external-access-token": "incoming-token"
            }
            mock_http.post(self._ORG_AS_TOKEN_URL).mock(
                return_value=Response(200, json={"access_token": "id-jag-token"})
            )
            mock_http.post(self._CUSTOM_AS_TOKEN_URL).mock(
                return_value=Response(200, json={"access_token": "final-scoped-token"})
            )
            result = await provider_with_card.authenticate(user_id="test-user")

        assert result is not None
        cred = result.credentials[0]
        assert isinstance(cred, BearerTokenCred)
        assert cred.token.get_secret_value() == "final-scoped-token"

    async def test_raises_on_step1_error(self, provider_with_card):
        with (
            patch(f"{_MODULE}.Context") as mock_ctx,
            patch(f"{_MODULE}._make_client_assertion", return_value="jwt"),
            respx.mock as mock_http,
        ):
            mock_ctx.get.return_value.metadata.headers = {
                "x-datarobot-external-access-token": "bad-token"
            }
            mock_http.post(self._ORG_AS_TOKEN_URL).mock(
                return_value=Response(
                    400,
                    json={
                        "error": "invalid_grant",
                        "error_description": "Token exchange failed: subject token expired",
                    },
                )
            )
            with pytest.raises(RuntimeError, match="Step 1.*subject token expired"):
                await provider_with_card.authenticate()

    async def test_raises_on_step2_error(self, provider_with_card):
        with (
            patch(f"{_MODULE}.Context") as mock_ctx,
            patch(f"{_MODULE}._make_client_assertion", return_value="jwt"),
            respx.mock as mock_http,
        ):
            mock_ctx.get.return_value.metadata.headers = {
                "x-datarobot-external-access-token": "ok-token"
            }
            mock_http.post(self._ORG_AS_TOKEN_URL).mock(
                return_value=Response(200, json={"access_token": "id-jag"})
            )
            mock_http.post(self._CUSTOM_AS_TOKEN_URL).mock(
                return_value=Response(
                    401,
                    json={"error": "unauthorized_client", "error_description": "Invalid assertion"},
                )
            )
            with pytest.raises(RuntimeError, match="Step 2.*Invalid assertion"):
                await provider_with_card.authenticate()

    async def test_raises_before_set_agent_card(self):
        provider = OAuth2CrossApplicationAccessOAuth2AuthProvider(
            config=OAuth2CrossApplicationAccessAuthProviderConfig()
        )
        with patch(f"{_MODULE}.Context") as mock_ctx:
            mock_ctx.get.return_value.metadata.headers = {
                "x-datarobot-external-access-token": "token"
            }
            with pytest.raises(RuntimeError, match="set_agent_card"):
                await provider.authenticate()

    @pytest.mark.parametrize(
        "principal_id,private_jwk_b64,match",
        [
            (None, _FAKE_JWK_B64, "principal_id"),
            ("principal_abc", None, "private_jwk"),
        ],
    )
    async def test_raises_without_credentials(
        self, principal_id, private_jwk_b64, match, monkeypatch
    ):
        monkeypatch.setenv("XAA_TOKEN_EXCHANGE_IMPL", "http")
        config = OAuth2CrossApplicationAccessAuthProviderConfig(
            principal_id=principal_id,
            private_jwk=private_jwk_b64,
        )
        provider = OAuth2CrossApplicationAccessOAuth2AuthProvider(config=config)
        provider.set_agent_card(_make_agent_card())

        with patch(f"{_MODULE}.Context") as mock_ctx:
            mock_ctx.get.return_value.metadata.headers = {
                "x-datarobot-external-access-token": "token"
            }
            with pytest.raises(ValueError, match=match):
                await provider.authenticate()


# ---------------------------------------------------------------------------
# End-to-end: card parse → authenticate → token exchange
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Full flow: card parsed → authenticate() delegates to ApiTokenExchange → BearerTokenCred.

    Mocks only HTTP (respx) and _make_client_assertion.
    """

    _ORG_AS_TOKEN_URL = _TRUSTED_ISSUER + "/oauth2/v1/token"
    _CUSTOM_AS_TOKEN_URL = _AGENT_TOKEN_URL

    @pytest.fixture
    def provider(self, monkeypatch):
        monkeypatch.setenv("XAA_TOKEN_EXCHANGE_IMPL", "http")
        config = OAuth2CrossApplicationAccessAuthProviderConfig(
            principal_id="0oa_test_principal",
            private_jwk=_FAKE_JWK_B64,
        )
        p = OAuth2CrossApplicationAccessOAuth2AuthProvider(config=config)
        p.set_agent_card(_make_agent_card())
        return p

    @pytest.fixture
    def incoming_token(self):
        return "user-okta-access-token-xyz"

    @pytest.fixture
    def mock_context(self, incoming_token):
        with patch(f"{_MODULE}.Context") as ctx:
            ctx.get.return_value.metadata.headers = {
                "x-datarobot-external-access-token": incoming_token,
            }
            yield ctx

    async def test_happy_path(self, provider, incoming_token, mock_context):
        assertion_calls: list[tuple[str, str, dict]] = []

        def _fake_assertion(principal_id: str, token_url: str, private_jwk: dict) -> str:
            assertion_calls.append((principal_id, token_url, private_jwk))
            return f"signed-jwt-for-{token_url}"

        with (
            patch(f"{_MODULE}._make_client_assertion", side_effect=_fake_assertion),
            respx.mock as mock_http,
        ):
            step1_route = mock_http.post(self._ORG_AS_TOKEN_URL).mock(
                return_value=Response(200, json={"access_token": "id-jag-token"})
            )
            step2_route = mock_http.post(self._CUSTOM_AS_TOKEN_URL).mock(
                return_value=Response(200, json={"access_token": "final-scoped-agent-token"})
            )

            result = await provider.authenticate(user_id="test-user")

        # _make_client_assertion called once per step with correct token URLs
        assert len(assertion_calls) == 2
        assert assertion_calls[0][1] == self._ORG_AS_TOKEN_URL
        assert assertion_calls[1][1] == self._CUSTOM_AS_TOKEN_URL

        # Step 1: correct form fields
        assert step1_route.called
        step1_body = parse_qs(step1_route.calls.last.request.content.decode())
        assert step1_body["grant_type"] == ["urn:ietf:params:oauth:grant-type:token-exchange"]
        assert step1_body["subject_token"] == [incoming_token]
        assert step1_body["audience"] == [_EXCHANGE_AUDIENCE]
        assert step1_body["resource"] == [_TARGET_AUDIENCE]
        assert step1_body["scope"] == ["read_data"]
        assert step1_body["client_assertion"] == [f"signed-jwt-for-{self._ORG_AS_TOKEN_URL}"]

        # Step 2: correct form fields
        assert step2_route.called
        step2_body = parse_qs(step2_route.calls.last.request.content.decode())
        assert step2_body["grant_type"] == ["urn:ietf:params:oauth:grant-type:jwt-bearer"]
        assert step2_body["assertion"] == ["id-jag-token"]
        assert step2_body["client_assertion"] == [f"signed-jwt-for-{self._CUSTOM_AS_TOKEN_URL}"]

        # Final result
        assert result is not None
        cred = result.credentials[0]
        assert isinstance(cred, BearerTokenCred)
        assert cred.token.get_secret_value() == "final-scoped-agent-token"
