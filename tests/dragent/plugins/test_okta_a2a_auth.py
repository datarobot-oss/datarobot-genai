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

"""Tests for OAuth2CrossApplicationAccessOAuth2AuthProvider and its agent-card parsing helpers."""

import base64
import json
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from a2a.types import AgentCapabilities
from a2a.types import AgentCard
from a2a.types import AgentExtension
from a2a.types import ClientCredentialsOAuthFlow
from a2a.types import OAuth2SecurityScheme
from a2a.types import OAuthFlows
from a2a.types import SecurityScheme
from nat.data_models.authentication import BearerTokenCred

from datarobot_genai.dragent.plugins.okta_a2a_auth import (
    OAuth2CrossApplicationAccessAuthProviderConfig,
)
from datarobot_genai.dragent.plugins.okta_a2a_auth import (
    OAuth2CrossApplicationAccessOAuth2AuthProvider,
)
from datarobot_genai.dragent.plugins.okta_a2a_auth import _parse_cross_app_params

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

# Derived: custom-AS base URL = agent token URL with /v1/token stripped
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
# Tests: OAuth2CrossApplicationAccessOAuth2AuthProvider — discovery phase
# ---------------------------------------------------------------------------


class TestOAuth2CrossApplicationAccessAuthProviderDiscovery:
    @pytest.fixture
    def provider(self):
        return OAuth2CrossApplicationAccessOAuth2AuthProvider(
            config=OAuth2CrossApplicationAccessAuthProviderConfig()
        )

    async def test_authenticate_for_discovery_returns_bearer(self, provider):
        """authenticate_for_discovery() returns the Okta token as Authorization: Bearer."""
        with patch(f"{_MODULE}.Context") as mock_ctx:
            mock_ctx.get.return_value.metadata.headers = {
                "x-datarobot-external-access-token": "my-okta-token"
            }
            headers = await provider.authenticate_for_discovery()

        assert headers == {"Authorization": "Bearer my-okta-token"}

    async def test_authenticate_for_discovery_raises_when_header_missing(self, provider):
        """authenticate_for_discovery() raises RuntimeError when header is absent."""
        with patch(f"{_MODULE}.Context") as mock_ctx:
            mock_ctx.get.return_value.metadata.headers = {}
            with pytest.raises(RuntimeError, match="x-datarobot-external-access-token"):
                await provider.authenticate_for_discovery()


# ---------------------------------------------------------------------------
# Tests: OAuth2CrossApplicationAccessOAuth2AuthProvider — fallback token headers
# ---------------------------------------------------------------------------


class TestOAuth2CrossApplicationAccessAuthProviderFallbackHeaders:
    """Tests for the fallback_token_headers local-dev feature."""

    @pytest.fixture
    def provider(self):
        return OAuth2CrossApplicationAccessOAuth2AuthProvider(
            config=OAuth2CrossApplicationAccessAuthProviderConfig()
        )

    async def test_extracts_token_from_authorization_fallback(self, provider):
        """Falls back to Authorization header when okta_token_header is absent."""
        with patch(f"{_MODULE}.Context") as mock_ctx:
            mock_ctx.get.return_value.metadata.headers = {
                "authorization": "Bearer fallback-token-123"
            }
            headers = await provider.authenticate_for_discovery()

        assert headers == {"Authorization": "Bearer fallback-token-123"}

    async def test_primary_header_takes_precedence_over_fallback(self, provider):
        """Primary header wins even if fallback is also present."""
        with patch(f"{_MODULE}.Context") as mock_ctx:
            mock_ctx.get.return_value.metadata.headers = {
                "x-datarobot-external-access-token": "primary-token",
                "authorization": "Bearer fallback-token",
            }
            headers = await provider.authenticate_for_discovery()

        assert headers == {"Authorization": "Bearer primary-token"}

    async def test_strips_bearer_prefix_case_insensitive(self, provider):
        """Bearer prefix is stripped regardless of casing."""
        with patch(f"{_MODULE}.Context") as mock_ctx:
            mock_ctx.get.return_value.metadata.headers = {"authorization": "BEARER my-token"}
            headers = await provider.authenticate_for_discovery()

        assert headers == {"Authorization": "Bearer my-token"}

    async def test_fallback_disabled_when_empty_list(self):
        """No fallback attempted when fallback_token_headers is empty."""
        provider = OAuth2CrossApplicationAccessOAuth2AuthProvider(
            config=OAuth2CrossApplicationAccessAuthProviderConfig(fallback_token_headers=[])
        )
        with patch(f"{_MODULE}.Context") as mock_ctx:
            mock_ctx.get.return_value.metadata.headers = {"authorization": "Bearer token"}
            with pytest.raises(RuntimeError, match="x-datarobot-external-access-token"):
                await provider.authenticate_for_discovery()


# ---------------------------------------------------------------------------
# Tests: OAuth2CrossApplicationAccessOAuth2AuthProvider — set_agent_card / param parsing
# ---------------------------------------------------------------------------


class TestOAuth2CrossApplicationAccessAuthProviderSetAgentCard:
    def test_set_agent_card_parses_params(self):
        """set_agent_card() populates _CrossAppFlowParams with all card fields."""
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
    def test_parse_cross_app_params_happy_path(self):
        """_parse_cross_app_params extracts full configuration context from the card."""
        params = _parse_cross_app_params(_make_agent_card())

        assert params.token_url == _AGENT_TOKEN_URL
        assert params.trusted_issuer == _TRUSTED_ISSUER
        assert params.exchange_audience == _EXCHANGE_AUDIENCE
        assert params.target_audience == _TARGET_AUDIENCE
        assert params.token_endpoint_auth_method == _AUTH_METHOD
        assert params.id_jag_scopes == ["read_data"]

    def test_parse_cross_app_params_custom_scopes(self):
        """_parse_cross_app_params uses caller-supplied scopes when provided."""
        params = _parse_cross_app_params(_make_agent_card(), id_jag_scopes=["openid", "profile"])

        assert params.id_jag_scopes == ["openid", "profile"]


class TestOAuth2CrossApplicationAccessAuthProviderConfigSerialization:
    def test_private_jwk_survives_json_roundtrip(self):
        """model_dump(mode='json') must emit the real JWK, not SecretStr's redacted placeholder."""
        cfg = OAuth2CrossApplicationAccessAuthProviderConfig(
            principal_id="principal_abc",
            private_jwk=_FAKE_JWK_B64,
        )
        payload = cfg.model_dump(mode="json")
        assert payload["private_jwk"] == _FAKE_JWK_B64
        restored = OAuth2CrossApplicationAccessAuthProviderConfig.model_validate(payload)
        assert restored.private_jwk is not None
        assert restored.private_jwk.get_secret_value() == _FAKE_JWK_B64


# ---------------------------------------------------------------------------
# Tests: OAuth2CrossApplicationAccessOAuth2AuthProvider — authenticate (call phase)
# ---------------------------------------------------------------------------


class TestOAuth2CrossApplicationAccessAuthProviderAuthenticate:
    @pytest.fixture
    def provider_with_card(self):
        config = OAuth2CrossApplicationAccessAuthProviderConfig(
            principal_id="principal_abc",
            private_jwk=_FAKE_JWK_B64,
        )
        provider = OAuth2CrossApplicationAccessOAuth2AuthProvider(config=config)
        provider.set_agent_card(_make_agent_card())
        return provider

    async def test_authenticate_returns_bearer_cred(self, provider_with_card):
        """authenticate() calls start()/resume() and returns the final scoped token."""
        mock_token_result = MagicMock(access_token="final-scoped-token")
        mock_flow = MagicMock()
        mock_flow.start = AsyncMock()
        mock_flow.resume = AsyncMock(return_value=mock_token_result)

        with (
            patch(f"{_MODULE}.Context") as mock_ctx,
            patch.object(provider_with_card, "_build_cross_app_flow", return_value=mock_flow),
        ):
            mock_ctx.get.return_value.metadata.headers = {
                "x-datarobot-external-access-token": "incoming-okta-token"
            }
            result = await provider_with_card.authenticate(user_id="test-user")

        assert result is not None
        assert len(result.credentials) == 1
        cred = result.credentials[0]
        assert isinstance(cred, BearerTokenCred)
        assert cred.token.get_secret_value() == "final-scoped-token"

        mock_flow.start.assert_awaited_once_with(
            token="incoming-okta-token",
            audience=_EXCHANGE_AUDIENCE,
        )
        mock_flow.resume.assert_awaited_once()

    async def test_authenticate_raises_on_sdk_error(self, provider_with_card):
        """authenticate() propagates exceptions from the SDK flow."""
        mock_flow = MagicMock()
        mock_flow.start = AsyncMock(side_effect=RuntimeError("token exchange failed"))

        with (
            patch(f"{_MODULE}.Context") as mock_ctx,
            patch.object(provider_with_card, "_build_cross_app_flow", return_value=mock_flow),
        ):
            mock_ctx.get.return_value.metadata.headers = {
                "x-datarobot-external-access-token": "bad-token"
            }
            with pytest.raises(RuntimeError, match="token exchange failed"):
                await provider_with_card.authenticate()

    async def test_authenticate_raises_before_set_agent_card(self):
        """authenticate() raises RuntimeError if called before set_agent_card()."""
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
    async def test_authenticate_raises_without_credentials(
        self, principal_id, private_jwk_b64, match
    ):
        """authenticate() raises ValueError when a required credential is absent."""
        config = OAuth2CrossApplicationAccessAuthProviderConfig(
            principal_id=principal_id,
            private_jwk=private_jwk_b64,
        )
        provider = OAuth2CrossApplicationAccessOAuth2AuthProvider(config=config)
        provider.set_agent_card(_make_agent_card())

        with pytest.raises(ValueError, match=match):
            await provider.authenticate()


# ---------------------------------------------------------------------------
# End-to-end: card parse → set_agent_card → authenticate → SDK wiring
# ---------------------------------------------------------------------------


class TestCrossAppAccessEndToEnd:
    """Mocked end-to-end happy path from raw AgentCard to final BearerTokenCred.

    Mocks only the Okta SDK boundary (CrossAppAccessFlow, OAuth2Client, etc.)
    and the NAT Context.  Everything else — card parsing, _CrossAppFlowParams
    construction, _build_cross_app_flow wiring — runs as production code.

    GIVEN an agent card with full Cross-Application Access extension,
          a provider configured with principal_id + private_jwk,
          and an incoming Okta access token in the request headers,
    WHEN  set_agent_card() then authenticate() are called,
    THEN  the Okta SDK is invoked with the correct parameters at every step
          and a BearerTokenCred with the final scoped token is returned.
    """

    # -- Fixtures: pure setup, no logic in the test body --

    @pytest.fixture
    def agent_card(self):
        """Full agent card matching the reference agentcard.json."""
        return _make_agent_card()

    @pytest.fixture
    def provider_config(self):
        return OAuth2CrossApplicationAccessAuthProviderConfig(
            principal_id="0oa_test_principal",
            private_jwk=_FAKE_JWK_B64,
        )

    @pytest.fixture
    def provider(self, provider_config, agent_card):
        """Return provider with the agent card already parsed."""
        p = OAuth2CrossApplicationAccessOAuth2AuthProvider(config=provider_config)
        p.set_agent_card(agent_card)
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

    @pytest.fixture
    def mock_sdk(self):
        """Patch all Okta SDK classes at the module level; yield a namespace."""
        mock_flow_instance = MagicMock()
        mock_flow_instance.start = AsyncMock()
        mock_flow_instance.resume = AsyncMock(
            return_value=MagicMock(access_token="final-scoped-agent-token")
        )

        with (
            patch(f"{_MODULE}.LocalKeyProvider") as mock_key_provider_cls,
            patch(f"{_MODULE}.OAuth2ClientConfiguration") as mock_config_cls,
            patch(f"{_MODULE}.OAuth2Client") as mock_client_cls,
            patch(f"{_MODULE}.ClientAssertionAuthorization") as mock_client_auth_cls,
            patch(f"{_MODULE}.JWTBearerClaims") as mock_claims_cls,
            patch(f"{_MODULE}.CrossAppAccessTarget") as mock_target_cls,
            patch(
                f"{_MODULE}.CrossAppAccessFlow", return_value=mock_flow_instance
            ) as mock_flow_cls,
        ):
            yield {
                "LocalKeyProvider": mock_key_provider_cls,
                "OAuth2ClientConfiguration": mock_config_cls,
                "OAuth2Client": mock_client_cls,
                "ClientAssertionAuthorization": mock_client_auth_cls,
                "JWTBearerClaims": mock_claims_cls,
                "CrossAppAccessTarget": mock_target_cls,
                "CrossAppAccessFlow": mock_flow_cls,
                "flow": mock_flow_instance,
            }

    # -- The test --

    async def test_happy_path_card_to_token(self, provider, incoming_token, mock_context, mock_sdk):
        result = await provider.authenticate(user_id="test-user")

        # -- Step 0: private_key_jwt → LocalKeyProvider initialized --
        mock_sdk["LocalKeyProvider"].from_jwk.assert_called_once_with(_FAKE_JWK, algorithm="RS256")

        # -- Step 0: JWT client assertion targets the originating AS token endpoint --
        mock_sdk["JWTBearerClaims"].assert_called_once_with(
            issuer="0oa_test_principal",
            subject="0oa_test_principal",
            audience=_TRUSTED_ISSUER.rstrip("/") + "/oauth2/v1/token",
            expires_in=60,
        )

        # -- Step 0: OAuth2ClientConfiguration uses the originating AS (trusted_issuer) --
        mock_sdk["OAuth2ClientConfiguration"].assert_called_once_with(
            issuer=_TRUSTED_ISSUER,
            scope=["read_data"],
            client_authorization=mock_sdk["ClientAssertionAuthorization"].return_value,
        )

        # -- Step 0: CrossAppAccessTarget uses the resource AS (exchange_audience) --
        mock_sdk["CrossAppAccessTarget"].assert_called_once_with(
            issuer=_EXCHANGE_AUDIENCE,
        )

        # -- Step 0: CrossAppAccessFlow wired with client + target --
        mock_sdk["CrossAppAccessFlow"].assert_called_once_with(
            client=mock_sdk["OAuth2Client"].return_value,
            target=mock_sdk["CrossAppAccessTarget"].return_value,
        )

        # -- Step 1: flow.start() exchanges the user token for an ID-JAG --
        mock_sdk["flow"].start.assert_awaited_once_with(
            token=incoming_token,
            audience=_EXCHANGE_AUDIENCE,
        )

        # -- Step 2: flow.resume() exchanges the ID-JAG for the final token --
        mock_sdk["flow"].resume.assert_awaited_once()

        # -- Final result: BearerTokenCred with the scoped agent token --
        assert result is not None
        assert len(result.credentials) == 1
        cred = result.credentials[0]
        assert isinstance(cred, BearerTokenCred)
        assert cred.token.get_secret_value() == "final-scoped-agent-token"
