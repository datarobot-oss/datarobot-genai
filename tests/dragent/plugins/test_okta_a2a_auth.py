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
        assert params.id_jag_scopes == ["read_data"]  # from agent card securitySchemes scopes


class TestParseCrossAppParams:
    def test_parse_cross_app_params_happy_path(self):
        """_parse_cross_app_params extracts full configuration context from the card."""
        params = _parse_cross_app_params(_make_agent_card())

        assert params.token_url == _AGENT_TOKEN_URL
        assert params.trusted_issuer == _TRUSTED_ISSUER
        assert params.exchange_audience == _EXCHANGE_AUDIENCE
        assert params.target_audience == _TARGET_AUDIENCE
        assert params.token_endpoint_auth_method == _AUTH_METHOD
        assert params.id_jag_scopes == ["read_data"]  # from agent card securitySchemes scopes

    def test_parse_cross_app_params_scopes_come_from_card(self):
        """id_jag_scopes are sourced from securitySchemes, not config."""
        # Build a card with a different scope to confirm it flows through
        from a2a.types import AgentCapabilities
        from a2a.types import AgentExtension
        from a2a.types import ClientCredentialsOAuthFlow
        from a2a.types import OAuth2SecurityScheme
        from a2a.types import OAuthFlows
        from a2a.types import SecurityScheme

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


class TestMakeClientAssertion:
    """Tests for _make_client_assertion error sanitization."""

    @pytest.fixture
    def provider(self):
        return OAuth2CrossApplicationAccessOAuth2AuthProvider(
            config=OAuth2CrossApplicationAccessAuthProviderConfig(
                principal_id="principal_abc",
                private_jwk=_FAKE_JWK_B64,
            )
        )

    def test_invalid_jwk_raises_value_error_without_key_material(self, provider):
        """_make_client_assertion raises ValueError — not the raw crypto exception — on bad JWK."""
        bad_jwk = {"kty": "RSA", "n": "INVALID"}
        with pytest.raises(ValueError, match="Failed to load RSA private key"):
            provider._make_client_assertion(
                token_url="https://token.example.com", private_jwk=bad_jwk
            )

    def test_value_error_has_no_chained_cause(self, provider):
        """The raised ValueError uses 'raise … from None' so the original exception
        (which may contain JWK material in its args) is not surfaced in any traceback
        display: __cause__ is None and __suppress_context__ is True.
        """
        bad_jwk = {"kty": "RSA", "n": "INVALID"}
        with pytest.raises(ValueError) as exc_info:
            provider._make_client_assertion(
                token_url="https://token.example.com", private_jwk=bad_jwk
            )
        assert exc_info.value.__cause__ is None
        assert exc_info.value.__suppress_context__ is True


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

    # Derived Step 1 URL: trusted_issuer + "/oauth2/v1/token"
    _ORG_AS_TOKEN_URL = _TRUSTED_ISSUER + "/oauth2/v1/token"
    # Step 2 URL is token_url from agent card securitySchemes
    _CUSTOM_AS_TOKEN_URL = _AGENT_TOKEN_URL

    async def test_authenticate_returns_bearer_cred(self, provider_with_card):
        """authenticate() makes two HTTP token requests and returns the final scoped token."""
        with (
            patch(f"{_MODULE}.Context") as mock_ctx,
            patch.object(
                provider_with_card,
                "_make_client_assertion",
                side_effect=["assertion-step1", "assertion-step2"],
            ),
            respx.mock as mock_http,
        ):
            mock_ctx.get.return_value.metadata.headers = {
                "x-datarobot-external-access-token": "incoming-okta-token"
            }
            # Step 1: org AS returns an ID-JAG
            mock_http.post(self._ORG_AS_TOKEN_URL).mock(
                return_value=Response(200, json={"access_token": "id-jag-token"})
            )
            # Step 2: custom AS returns the final scoped token
            mock_http.post(self._CUSTOM_AS_TOKEN_URL).mock(
                return_value=Response(200, json={"access_token": "final-scoped-token"})
            )

            result = await provider_with_card.authenticate(user_id="test-user")

        assert result is not None
        assert len(result.credentials) == 1
        cred = result.credentials[0]
        assert isinstance(cred, BearerTokenCred)
        assert cred.token.get_secret_value() == "final-scoped-token"

    async def test_authenticate_raises_on_http_error(self, provider_with_card):
        """authenticate() raises RuntimeError with Okta error details on HTTP failures."""
        with (
            patch(f"{_MODULE}.Context") as mock_ctx,
            patch.object(
                provider_with_card,
                "_make_client_assertion",
                return_value="assertion-jwt",
            ),
            respx.mock as mock_http,
        ):
            mock_ctx.get.return_value.metadata.headers = {
                "x-datarobot-external-access-token": "bad-token"
            }
            # Step 1 returns a structured Okta error response
            mock_http.post(self._ORG_AS_TOKEN_URL).mock(
                return_value=Response(
                    400,
                    json={
                        "error": "invalid_grant",
                        "error_description": "Token exchange failed: subject token expired",
                    },
                )
            )

            with pytest.raises(
                RuntimeError,
                match="Step 1.*Token exchange failed: subject token expired",
            ):
                await provider_with_card.authenticate()

    async def test_authenticate_raises_on_step2_http_error(self, provider_with_card):
        """authenticate() includes step number and Okta error in RuntimeError on Step 2 failure."""
        with (
            patch(f"{_MODULE}.Context") as mock_ctx,
            patch.object(
                provider_with_card,
                "_make_client_assertion",
                return_value="assertion-jwt",
            ),
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

    Mocks only the HTTP boundary (via ``respx``) and ``_make_client_assertion``
    to avoid needing real RSA key material.  Everything else — card parsing,
    _CrossAppFlowParams construction, authenticate() request logic — runs as
    production code.

    GIVEN an agent card with full Cross-Application Access extension,
          a provider configured with principal_id + private_jwk,
          and an incoming Okta access token in the request headers,
    WHEN  set_agent_card() then authenticate() are called,
    THEN  the correct token URLs receive the expected form fields and
          a BearerTokenCred with the final scoped token is returned.
    """

    # Derived from test constants
    _ORG_AS_TOKEN_URL = _TRUSTED_ISSUER + "/oauth2/v1/token"
    _CUSTOM_AS_TOKEN_URL = _AGENT_TOKEN_URL  # from agent card securitySchemes

    # -- Fixtures --

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

    # -- The test --

    async def test_happy_path_card_to_token(self, provider, incoming_token, mock_context):
        """Full flow: card parsed → two HTTP token requests → BearerTokenCred returned."""
        assertion_calls: list[tuple[str, dict]] = []

        def _fake_assertion(token_url: str, private_jwk: dict) -> str:
            assertion_calls.append((token_url, private_jwk))
            return f"signed-jwt-for-{token_url}"

        with (
            patch.object(provider, "_make_client_assertion", side_effect=_fake_assertion),
            respx.mock as mock_http,
        ):
            # Step 1: org AS returns an ID-JAG
            step1_route = mock_http.post(self._ORG_AS_TOKEN_URL).mock(
                return_value=Response(200, json={"access_token": "id-jag-token"})
            )
            # Step 2: custom AS returns the final scoped token
            step2_route = mock_http.post(self._CUSTOM_AS_TOKEN_URL).mock(
                return_value=Response(200, json={"access_token": "final-scoped-agent-token"})
            )

            result = await provider.authenticate(user_id="test-user")

        # -- _make_client_assertion called once per step with the correct token URL --
        assert len(assertion_calls) == 2
        assert assertion_calls[0][0] == self._ORG_AS_TOKEN_URL  # Step 1 aud
        assert assertion_calls[1][0] == self._CUSTOM_AS_TOKEN_URL  # Step 2 aud

        # -- Step 1 request: correct URL and key form fields (URL-decoded) --
        assert step1_route.called
        step1_body = parse_qs(step1_route.calls.last.request.content.decode())
        assert step1_body["grant_type"] == ["urn:ietf:params:oauth:grant-type:token-exchange"]
        assert step1_body["subject_token"] == [incoming_token]
        assert step1_body["audience"] == [_EXCHANGE_AUDIENCE]
        assert step1_body["resource"] == [_TARGET_AUDIENCE]
        assert step1_body["scope"] == ["read_data"]  # from agent card securitySchemes
        assert step1_body["client_assertion"] == [f"signed-jwt-for-{self._ORG_AS_TOKEN_URL}"]

        # -- Step 2 request: correct URL and key form fields (URL-decoded) --
        assert step2_route.called
        step2_body = parse_qs(step2_route.calls.last.request.content.decode())
        assert step2_body["grant_type"] == ["urn:ietf:params:oauth:grant-type:jwt-bearer"]
        assert step2_body["assertion"] == ["id-jag-token"]
        assert step2_body["client_assertion"] == [f"signed-jwt-for-{self._CUSTOM_AS_TOKEN_URL}"]

        # -- Final result: BearerTokenCred with the scoped agent token --
        assert result is not None
        assert len(result.credentials) == 1
        cred = result.credentials[0]
        assert isinstance(cred, BearerTokenCred)
        assert cred.token.get_secret_value() == "final-scoped-agent-token"
