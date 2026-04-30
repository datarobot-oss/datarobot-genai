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

"""Tests for OktaTokenExchangeAuthProvider and its agent-card parsing helpers."""

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

from datarobot_genai.dragent.plugins.okta_a2a_auth import OktaTokenExchangeAuthProvider
from datarobot_genai.dragent.plugins.okta_a2a_auth import OktaTokenExchangeAuthProviderConfig
from datarobot_genai.dragent.plugins.okta_a2a_auth import _parse_cross_app_params

_MODULE = "datarobot_genai.dragent.plugins.okta_a2a_auth"

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

_TRUSTED_ISSUER = "https://okta.example.com"
_AGENT_TOKEN_URL = "https://okta.example.com/oauth2/ausYYY/v1/token"
_AGENT_AUDIENCE = "https://api.agent.example.com"
_SUBJECT_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:access_token"
_REQUESTED_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:id-jag"
_AUTH_METHOD = "private_key_jwt"

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
    """Build a minimal AgentCard that contains the RFC 8693 extension."""
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
        uri="urn:ietf:params:oauth:grant-type:token-exchange",
        description="RFC 8693 two-step token exchange",
        params={
            "subject_token_constraints": {"trusted_issuer": _TRUSTED_ISSUER},
            "token_exchange_request": {
                "audience": _AGENT_AUDIENCE,
                "subject_token_type": _SUBJECT_TOKEN_TYPE,
                "requested_token_type": _REQUESTED_TOKEN_TYPE,
                "token_endpoint_auth_method": _AUTH_METHOD,
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
# Tests: OktaTokenExchangeAuthProvider — discovery phase
# ---------------------------------------------------------------------------


class TestOktaTokenExchangeAuthProviderDiscovery:
    @pytest.fixture
    def provider(self):
        return OktaTokenExchangeAuthProvider(config=OktaTokenExchangeAuthProviderConfig())

    async def test_authenticate_for_discovery_returns_bearer(self, provider):
        """authenticate_for_discovery() returns the Okta token as Authorization: Bearer."""
        with patch(f"{_MODULE}.Context") as mock_ctx:
            mock_ctx.get.return_value.metadata.headers = {
                "x-datarobot-okta-access-token": "my-okta-token"
            }
            headers = await provider.authenticate_for_discovery()

        assert headers == {"Authorization": "Bearer my-okta-token"}

    async def test_authenticate_for_discovery_raises_when_header_missing(self, provider):
        """authenticate_for_discovery() raises RuntimeError when header is absent."""
        with patch(f"{_MODULE}.Context") as mock_ctx:
            mock_ctx.get.return_value.metadata.headers = {}
            with pytest.raises(RuntimeError, match="x-datarobot-okta-access-token"):
                await provider.authenticate_for_discovery()


# ---------------------------------------------------------------------------
# Tests: OktaTokenExchangeAuthProvider — set_agent_card / param parsing
# ---------------------------------------------------------------------------


class TestOktaTokenExchangeAuthProviderSetAgentCard:
    def test_set_agent_card_parses_params(self):
        """set_agent_card() populates _CrossAppFlowParams from the agent card."""
        provider = OktaTokenExchangeAuthProvider(config=OktaTokenExchangeAuthProviderConfig())

        provider.set_agent_card(_make_agent_card())

        params = provider._flow_params
        assert params is not None
        assert params.issuer == _TRUSTED_ISSUER
        assert params.target_issuer == _TARGET_ISSUER
        assert params.id_jag_scopes == ["read_data"]


class TestParseCrossAppParams:
    def test_parse_cross_app_params_happy_path(self):
        """_parse_cross_app_params extracts issuer and target_issuer correctly."""
        params = _parse_cross_app_params(_make_agent_card())

        assert params.issuer == _TRUSTED_ISSUER
        assert params.target_issuer == _TARGET_ISSUER
        assert params.id_jag_scopes == ["read_data"]

    def test_parse_cross_app_params_custom_scopes(self):
        """_parse_cross_app_params uses caller-supplied scopes when provided."""
        params = _parse_cross_app_params(_make_agent_card(), id_jag_scopes=["openid", "profile"])

        assert params.id_jag_scopes == ["openid", "profile"]


# ---------------------------------------------------------------------------
# Tests: OktaTokenExchangeAuthProvider — authenticate (call phase)
# ---------------------------------------------------------------------------


class TestOktaTokenExchangeAuthProviderAuthenticate:
    @pytest.fixture
    def provider_with_card(self):
        config = OktaTokenExchangeAuthProviderConfig(
            principal_id="principal_abc",
            private_jwk=_FAKE_JWK_B64,
        )
        provider = OktaTokenExchangeAuthProvider(config=config)
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
                "x-datarobot-okta-access-token": "incoming-okta-token"
            }
            result = await provider_with_card.authenticate(user_id="test-user")

        assert result is not None
        assert len(result.credentials) == 1
        cred = result.credentials[0]
        assert isinstance(cred, BearerTokenCred)
        assert cred.token.get_secret_value() == "final-scoped-token"

        mock_flow.start.assert_awaited_once_with(
            token="incoming-okta-token",
            audience=_TARGET_ISSUER,
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
                "x-datarobot-okta-access-token": "bad-token"
            }
            with pytest.raises(RuntimeError, match="token exchange failed"):
                await provider_with_card.authenticate()

    async def test_authenticate_raises_before_set_agent_card(self):
        """authenticate() raises RuntimeError if called before set_agent_card()."""
        provider = OktaTokenExchangeAuthProvider(config=OktaTokenExchangeAuthProviderConfig())

        with patch(f"{_MODULE}.Context") as mock_ctx:
            mock_ctx.get.return_value.metadata.headers = {"x-datarobot-okta-access-token": "token"}
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
        config = OktaTokenExchangeAuthProviderConfig(
            principal_id=principal_id,
            private_jwk=private_jwk_b64,
        )
        provider = OktaTokenExchangeAuthProvider(config=config)
        provider.set_agent_card(_make_agent_card())

        with pytest.raises(ValueError, match=match):
            await provider.authenticate()
