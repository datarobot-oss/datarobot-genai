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

"""Tests for XAA token exchange implementations (HTTP and Okta SDK).

Verifies that both ``ApiTokenExchange`` (HTTP) and ``OktaTokenExchange``
(SDK) send the correct HTTP form fields to the Okta authorization servers.

The HTTP tests use ``respx`` to intercept ``httpx`` requests.
The SDK tests inject a fake ``NetworkInterface`` into the real
``okta-client-python`` stack to capture the raw HTTP requests the SDK sends.
"""

import base64
import json
from dataclasses import dataclass
from unittest.mock import Mock
from unittest.mock import patch
from urllib.parse import parse_qs

import jwt as pyjwt
import pytest
import respx
from cryptography.hazmat.primitives.asymmetric import rsa
from httpx import Response

from datarobot_genai.dragent.plugins.okta_a2a_auth import _CLIENT_ASSERTION_TYPE
from datarobot_genai.dragent.plugins.okta_a2a_auth import _REQUESTED_TOKEN_TYPE
from datarobot_genai.dragent.plugins.okta_a2a_auth import _SUBJECT_TOKEN_TYPE
from datarobot_genai.dragent.plugins.okta_a2a_auth import _TOKEN_EXCHANGE_GRANT_TYPE
from datarobot_genai.dragent.plugins.okta_a2a_auth import ApiTokenExchange
from datarobot_genai.dragent.plugins.okta_a2a_auth import (
    OAuth2CrossApplicationAccessAuthProviderConfig,
)
from datarobot_genai.dragent.plugins.okta_a2a_auth import OktaTokenExchange
from datarobot_genai.dragent.plugins.okta_a2a_auth import _CrossAppFlowParams

try:
    from okta_client.authfoundation.networking import HTTPRequest as OktaHTTPRequest
    from okta_client.authfoundation.networking import RawResponse as OktaRawResponse

    _HAS_OKTA_SDK = True
except ImportError:
    _HAS_OKTA_SDK = False

_MODULE = "datarobot_genai.dragent.plugins.okta_a2a_auth"

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

_TRUSTED_ISSUER = "https://okta.example.com"
_AGENT_TOKEN_URL = "https://okta.example.com/oauth2/ausYYY/v1/token"
_TARGET_AUDIENCE = "https://api.agent.example.com"
_EXCHANGE_AUDIENCE = "https://okta.example.com/oauth2/ausYYY"
_AUTH_METHOD = "private_key_jwt"

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


def _generate_test_rsa_jwk() -> dict:
    """Generate a real RSA JWK for the SDK's LocalKeyProvider."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk_dict = json.loads(pyjwt.algorithms.RSAAlgorithm.to_jwk(key, as_dict=False))
    jwk_dict["kid"] = "test-kid"
    jwk_dict["alg"] = "RS256"
    return jwk_dict


if _HAS_OKTA_SDK:
    # Pre-generate once per module so all tests use the same key material.
    _REAL_JWK = _generate_test_rsa_jwk()
    _REAL_JWK_B64 = base64.b64encode(json.dumps(_REAL_JWK).encode()).decode()


# ---------------------------------------------------------------------------
# Tests: HTTP impl — verify ALL form fields sent to Okta
# ---------------------------------------------------------------------------


class TestApiTokenExchangeFormFields:
    """Verify that ApiTokenExchange sends every required form field.

    This catches regressions where a field like ``subject_token_type``,
    ``requested_token_type``, or ``client_assertion_type`` is accidentally
    dropped or set to the wrong value.
    """

    _ORG_AS_TOKEN_URL = _TRUSTED_ISSUER + "/oauth2/v1/token"
    _CUSTOM_AS_TOKEN_URL = _AGENT_TOKEN_URL

    @pytest.fixture
    def flow_params(self) -> _CrossAppFlowParams:
        return _CrossAppFlowParams(
            token_url=_AGENT_TOKEN_URL,
            trusted_issuer=_TRUSTED_ISSUER,
            exchange_audience=_EXCHANGE_AUDIENCE,
            target_audience=_TARGET_AUDIENCE,
            token_endpoint_auth_method=_AUTH_METHOD,
            id_jag_scopes=["dr.impersonation"],
        )

    @pytest.fixture
    def exchange(self) -> ApiTokenExchange:
        config = OAuth2CrossApplicationAccessAuthProviderConfig(
            principal_id="0oa_test_principal",
            private_jwk=_FAKE_JWK_B64,
        )
        return ApiTokenExchange(config)

    async def test_step1_sends_all_required_fields(self, exchange, flow_params):
        """Step 1 (RFC 8693) must include all token-exchange parameters."""
        with (
            patch(f"{_MODULE}._make_client_assertion", side_effect=["assertion1", "assertion2"]),
            respx.mock as mock_http,
        ):
            step1 = mock_http.post(self._ORG_AS_TOKEN_URL).mock(
                return_value=Response(200, json={"access_token": "id-jag"})
            )
            mock_http.post(self._CUSTOM_AS_TOKEN_URL).mock(
                return_value=Response(200, json={"access_token": "final"})
            )
            await exchange.exchange_token(flow_params, "user-access-token")

        body = parse_qs(step1.calls.last.request.content.decode())
        assert body["grant_type"] == ["urn:ietf:params:oauth:grant-type:token-exchange"]
        assert body["subject_token"] == ["user-access-token"]
        assert body["subject_token_type"] == ["urn:ietf:params:oauth:token-type:access_token"]
        assert body["requested_token_type"] == ["urn:ietf:params:oauth:token-type:id-jag"]
        assert body["audience"] == [_EXCHANGE_AUDIENCE]
        assert body["resource"] == [_TARGET_AUDIENCE]
        assert body["scope"] == ["dr.impersonation"]
        assert body["client_assertion_type"] == [
            "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"
        ]
        assert body["client_assertion"] == ["assertion1"]

    async def test_step2_sends_all_required_fields(self, exchange, flow_params):
        """Step 2 (RFC 7523) must include the ID-JAG as assertion."""
        with (
            patch(f"{_MODULE}._make_client_assertion", side_effect=["assertion1", "assertion2"]),
            respx.mock as mock_http,
        ):
            mock_http.post(self._ORG_AS_TOKEN_URL).mock(
                return_value=Response(200, json={"access_token": "the-id-jag"})
            )
            step2 = mock_http.post(self._CUSTOM_AS_TOKEN_URL).mock(
                return_value=Response(200, json={"access_token": "final"})
            )
            await exchange.exchange_token(flow_params, "user-access-token")

        body = parse_qs(step2.calls.last.request.content.decode())
        assert body["grant_type"] == ["urn:ietf:params:oauth:grant-type:jwt-bearer"]
        assert body["assertion"] == ["the-id-jag"]
        assert body["client_assertion_type"] == [
            "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"
        ]
        assert body["client_assertion"] == ["assertion2"]

    async def test_step1_uses_correct_token_url(self, exchange, flow_params):
        """Step 1 must POST to {trusted_issuer}/oauth2/v1/token."""
        with (
            patch(f"{_MODULE}._make_client_assertion", return_value="jwt"),
            respx.mock as mock_http,
        ):
            step1 = mock_http.post(self._ORG_AS_TOKEN_URL).mock(
                return_value=Response(200, json={"access_token": "jag"})
            )
            mock_http.post(self._CUSTOM_AS_TOKEN_URL).mock(
                return_value=Response(200, json={"access_token": "final"})
            )
            await exchange.exchange_token(flow_params, "token")

        assert step1.called
        assert str(step1.calls.last.request.url) == self._ORG_AS_TOKEN_URL

    async def test_step2_uses_agent_card_token_url(self, exchange, flow_params):
        """Step 2 must POST to the tokenUrl from the agent card."""
        with (
            patch(f"{_MODULE}._make_client_assertion", return_value="jwt"),
            respx.mock as mock_http,
        ):
            mock_http.post(self._ORG_AS_TOKEN_URL).mock(
                return_value=Response(200, json={"access_token": "jag"})
            )
            step2 = mock_http.post(self._CUSTOM_AS_TOKEN_URL).mock(
                return_value=Response(200, json={"access_token": "final"})
            )
            await exchange.exchange_token(flow_params, "token")

        assert step2.called
        assert str(step2.calls.last.request.url) == self._CUSTOM_AS_TOKEN_URL

    async def test_client_assertion_audience_matches_target_url(self, exchange, flow_params):
        """_make_client_assertion must be called with the correct token URL for each step."""
        calls: list[tuple[str, str]] = []

        def _record(principal_id: str, token_url: str, private_jwk: dict) -> str:
            calls.append((principal_id, token_url))
            return "jwt"

        with (
            patch(f"{_MODULE}._make_client_assertion", side_effect=_record),
            respx.mock as mock_http,
        ):
            mock_http.post(self._ORG_AS_TOKEN_URL).mock(
                return_value=Response(200, json={"access_token": "jag"})
            )
            mock_http.post(self._CUSTOM_AS_TOKEN_URL).mock(
                return_value=Response(200, json={"access_token": "final"})
            )
            await exchange.exchange_token(flow_params, "token")

        assert calls[0] == ("0oa_test_principal", self._ORG_AS_TOKEN_URL)
        assert calls[1] == ("0oa_test_principal", self._CUSTOM_AS_TOKEN_URL)

    async def test_multi_scope_joined_with_space(self, flow_params):
        """Multiple scopes must be space-joined in the form body."""
        flow_params.id_jag_scopes = ["dr.impersonation", "openid"]
        config = OAuth2CrossApplicationAccessAuthProviderConfig(
            principal_id="p", private_jwk=_FAKE_JWK_B64
        )
        exchange = ApiTokenExchange(config)

        with (
            patch(f"{_MODULE}._make_client_assertion", return_value="jwt"),
            respx.mock as mock_http,
        ):
            step1 = mock_http.post(self._ORG_AS_TOKEN_URL).mock(
                return_value=Response(200, json={"access_token": "jag"})
            )
            mock_http.post(self._CUSTOM_AS_TOKEN_URL).mock(
                return_value=Response(200, json={"access_token": "final"})
            )
            await exchange.exchange_token(flow_params, "token")

        body = parse_qs(step1.calls.last.request.content.decode())
        assert body["scope"] == ["dr.impersonation openid"]

    def test_get_xaa_token_exchange_request_payload(self) -> None:
        cross_app_flow_params = Mock()
        cross_app_flow_params.id_jag_scopes = ["dsafafds"]
        subject_token = Mock()
        client_assertion = Mock()
        output = ApiTokenExchange.get_xaa_token_exchange_request_payload(
            cross_app_flow_params,
            subject_token,
            client_assertion,
        )

        expected_payload = {
            "grant_type": _TOKEN_EXCHANGE_GRANT_TYPE,
            "subject_token": subject_token,
            "subject_token_type": _SUBJECT_TOKEN_TYPE,
            "requested_token_type": _REQUESTED_TOKEN_TYPE,
            "audience": cross_app_flow_params.exchange_audience,
            "scope": " ".join(cross_app_flow_params.id_jag_scopes),
            "client_assertion_type": _CLIENT_ASSERTION_TYPE,
            "client_assertion": client_assertion,
            "resource": cross_app_flow_params.target_audience,
        }
        assert output == expected_payload

    def test_get_xaa_token_exchange_request_payload_without_resource_in_payload(self) -> None:
        cross_app_flow_params = Mock()
        cross_app_flow_params.target_audience = None
        cross_app_flow_params.id_jag_scopes = ["dsafafds"]
        output = ApiTokenExchange.get_xaa_token_exchange_request_payload(
            cross_app_flow_params,
            Mock(),
            Mock(),
        )

        assert "resource" not in output


# ---------------------------------------------------------------------------
# Tests: SDK impl — verify actual HTTP requests sent by okta-client-python
# ---------------------------------------------------------------------------
#
# Instead of mocking every SDK class, we inject a fake NetworkInterface
# that captures the raw HTTP requests the SDK sends.  This lets real SDK
# objects (LocalKeyProvider, OAuth2Client, CrossAppAccessFlow, …) be
# constructed and exercised — we only control the network boundary.
#
# The SDK uses urllib internally via its NetworkInterface protocol:
#
#   class NetworkInterface(Protocol):
#       def send(self, request: HTTPRequest) -> RawResponse: ...
#
# We implement a _FakeOktaNetwork that routes requests to canned JSON
# responses (discovery metadata, JWKS, token exchange, JWT bearer) and
# records every request for later assertion.


@dataclass
class _CapturedRequest:
    """A single HTTP request captured by _FakeOktaNetwork."""

    method: str
    url: str
    headers: dict[str, str]
    body: bytes | None

    @property
    def form_fields(self) -> dict[str, list[str]]:
        """Parse URL-encoded body into a dict of lists."""
        return parse_qs((self.body or b"").decode())


class _FakeOktaNetwork:
    """In-memory NetworkInterface that records requests and serves canned responses.

    Routes:
    - GET  {issuer}/.well-known/oauth-authorization-server → discovery metadata
    - GET  {issuer}/oauth2/v1/keys                         → empty JWKS
    - POST {issuer}/oauth2/v1/token                        → Step 1 token exchange
    - GET  {custom_as}/.well-known/oauth-authorization-server → custom AS discovery
    - GET  {custom_as}/v1/keys                             → empty JWKS
    - POST {custom_as}/v1/token                            → Step 2 JWT bearer
    """

    def __init__(
        self,
        trusted_issuer: str,
        custom_as_issuer: str,
        step1_access_token: str | None = None,
        step2_access_token: str = "final-scoped-from-sdk",
    ) -> None:
        self.trusted_issuer = trusted_issuer.rstrip("/")
        self.custom_as_issuer = custom_as_issuer.rstrip("/")
        # The SDK parses the ID-JAG as a JWT, so we need a valid JWT token.
        # Create a minimal unsigned JWT (alg=none) if no explicit value given.
        if step1_access_token is None:
            self.step1_access_token = pyjwt.encode(
                {"sub": "user", "iss": trusted_issuer, "typ": "oauth-id-jag+jwt"},
                key="",
                algorithm="none",
            )
        else:
            self.step1_access_token = step1_access_token
        self.step2_access_token = step2_access_token
        self.requests: list[_CapturedRequest] = []

    # -- NetworkInterface protocol ------------------------------------------

    def send(self, request: "OktaHTTPRequest") -> "OktaRawResponse":
        captured = _CapturedRequest(
            method=request.method.value
            if hasattr(request.method, "value")
            else str(request.method),
            url=request.url,
            headers=dict(request.headers),
            body=request.body,
        )
        self.requests.append(captured)

        url = request.url.split("?")[0]  # strip query params

        # --- Org AS discovery ---
        if url == f"{self.trusted_issuer}/.well-known/oauth-authorization-server":
            return self._json_response(
                {
                    "issuer": self.trusted_issuer,
                    "authorization_endpoint": f"{self.trusted_issuer}/oauth2/v1/authorize",
                    "token_endpoint": f"{self.trusted_issuer}/oauth2/v1/token",
                    "jwks_uri": f"{self.trusted_issuer}/oauth2/v1/keys",
                    "grant_types_supported": [
                        "urn:ietf:params:oauth:grant-type:token-exchange",
                    ],
                }
            )

        # --- Org AS JWKS ---
        if url == f"{self.trusted_issuer}/oauth2/v1/keys":
            return self._json_response({"keys": []})

        # --- Org AS token endpoint (Step 1) ---
        if url == f"{self.trusted_issuer}/oauth2/v1/token":
            return self._json_response(
                {
                    "access_token": self.step1_access_token,
                    "token_type": "Bearer",
                    "expires_in": 300,
                    "issued_token_type": "urn:ietf:params:oauth:token-type:id-jag",
                }
            )

        # --- Custom AS discovery ---
        if url == f"{self.custom_as_issuer}/.well-known/oauth-authorization-server":
            return self._json_response(
                {
                    "issuer": self.custom_as_issuer,
                    "authorization_endpoint": f"{self.custom_as_issuer}/v1/authorize",
                    "token_endpoint": f"{self.custom_as_issuer}/v1/token",
                    "jwks_uri": f"{self.custom_as_issuer}/v1/keys",
                    "grant_types_supported": [
                        "urn:ietf:params:oauth:grant-type:jwt-bearer",
                    ],
                }
            )

        # --- Custom AS JWKS ---
        if url == f"{self.custom_as_issuer}/v1/keys":
            return self._json_response({"keys": []})

        # --- Custom AS token endpoint (Step 2) ---
        if url == f"{self.custom_as_issuer}/v1/token":
            return self._json_response(
                {
                    "access_token": self.step2_access_token,
                    "token_type": "Bearer",
                    "expires_in": 3600,
                }
            )

        raise ValueError(f"_FakeOktaNetwork: unexpected request {request.method} {url}")

    @staticmethod
    def _json_response(body: dict, status: int = 200) -> "OktaRawResponse":
        return OktaRawResponse(
            status_code=status,
            headers={"content-type": "application/json"},
            body=json.dumps(body).encode(),
        )

    # -- Helpers for assertions ---------------------------------------------

    def post_requests(self) -> list[_CapturedRequest]:
        return [r for r in self.requests if r.method == "POST"]

    @property
    def step1_request(self) -> _CapturedRequest:
        """The POST to the org AS token endpoint (Step 1)."""
        posts = self.post_requests()
        assert len(posts) >= 1, "Expected at least one POST request (Step 1)"
        return posts[0]

    @property
    def step2_request(self) -> _CapturedRequest:
        """The POST to the custom AS token endpoint (Step 2)."""
        posts = self.post_requests()
        assert len(posts) >= 2, "Expected at least two POST requests (Step 1 + Step 2)"
        return posts[1]


@pytest.mark.skipif(not _HAS_OKTA_SDK, reason="okta-client-python not installed")
class TestOktaTokenExchangeSdkHttpRequests:
    """Verify OktaTokenExchange sends correct HTTP requests via the SDK.

    Uses a real ``okta-client-python`` SDK stack with a fake network layer
    injected via ``OAuth2Client(network=_FakeOktaNetwork(...))``.  This
    exercises the full code path — ``LocalKeyProvider``, ``JWTBearerClaims``,
    ``OAuth2ClientConfiguration``, ``CrossAppAccessFlow`` — and asserts on
    the **actual HTTP form fields** that reach the wire.

    Catches regressions like:
    - subject_token_type defaulting to id_token instead of access_token
    - missing ``resource`` parameter causing 'invalid_target'
    - wrong audience in client assertion JWT
    """

    @pytest.fixture
    def flow_params(self) -> _CrossAppFlowParams:
        return _CrossAppFlowParams(
            token_url=_AGENT_TOKEN_URL,
            trusted_issuer=_TRUSTED_ISSUER,
            exchange_audience=_EXCHANGE_AUDIENCE,
            target_audience=_TARGET_AUDIENCE,
            token_endpoint_auth_method=_AUTH_METHOD,
            id_jag_scopes=["dr.impersonation"],
        )

    @pytest.fixture
    def config(self) -> OAuth2CrossApplicationAccessAuthProviderConfig:
        return OAuth2CrossApplicationAccessAuthProviderConfig(
            principal_id="wlp_principal_123",
            private_jwk=_REAL_JWK_B64,
        )

    @pytest.fixture
    def exchange(self, config) -> OktaTokenExchange:
        return OktaTokenExchange(config)

    @pytest.fixture
    def fake_network(self) -> _FakeOktaNetwork:
        return _FakeOktaNetwork(
            trusted_issuer=_TRUSTED_ISSUER,
            custom_as_issuer=_EXCHANGE_AUDIENCE,
        )

    async def _run_exchange(self, exchange, flow_params, fake_network, subject_token="user-token"):
        """Run OktaTokenExchange with the fake network injected into OAuth2Client."""
        with patch(f"{_MODULE}.OAuth2Client") as mock_client_cls:
            # Intercept OAuth2Client construction to inject our fake network
            original_cls = pytest.importorskip("okta_client.authfoundation").OAuth2Client

            def _patched_client(*, configuration, **kwargs):
                return original_cls(configuration=configuration, network=fake_network, **kwargs)

            mock_client_cls.side_effect = _patched_client
            return await exchange.exchange_token(flow_params, subject_token)

    async def test_step1_subject_token_type_is_access_token(
        self, exchange, flow_params, fake_network
    ):
        """Step 1 must send subject_token_type=access_token (not id_token).

        The SDK defaults to id_token, which causes Okta to reject actual
        access tokens with 'subject_token is invalid'.
        """
        await self._run_exchange(exchange, flow_params, fake_network)

        body = fake_network.step1_request.form_fields
        assert body["subject_token_type"] == ["urn:ietf:params:oauth:token-type:access_token"]

    async def test_step1_sends_resource_parameter(self, exchange, flow_params, fake_network):
        """Step 1 must include resource=target_audience.

        Without this, Okta returns 'invalid_target: Token Exchange requests
        must include a valid audience of the authorization server'.
        """
        await self._run_exchange(exchange, flow_params, fake_network)

        body = fake_network.step1_request.form_fields
        assert body["resource"] == [_TARGET_AUDIENCE]

    async def test_step1_sends_subject_token(self, exchange, flow_params, fake_network):
        """Step 1 must forward the caller's access token as subject_token."""
        await self._run_exchange(
            exchange, flow_params, fake_network, subject_token="my-okta-access-token"
        )

        body = fake_network.step1_request.form_fields
        assert body["subject_token"] == ["my-okta-access-token"]

    async def test_step1_sends_audience(self, exchange, flow_params, fake_network):
        """Step 1 audience must be the custom AS issuer (exchange_audience)."""
        await self._run_exchange(exchange, flow_params, fake_network)

        body = fake_network.step1_request.form_fields
        assert body["audience"] == [_EXCHANGE_AUDIENCE]

    async def test_step1_sends_scope(self, exchange, flow_params, fake_network):
        """Step 1 must include the scopes from the agent card."""
        await self._run_exchange(exchange, flow_params, fake_network)

        body = fake_network.step1_request.form_fields
        assert body["scope"] == ["dr.impersonation"]

    async def test_step1_sends_requested_token_type_id_jag(
        self, exchange, flow_params, fake_network
    ):
        """Step 1 must request an ID-JAG token type."""
        await self._run_exchange(exchange, flow_params, fake_network)

        body = fake_network.step1_request.form_fields
        assert body["requested_token_type"] == ["urn:ietf:params:oauth:token-type:id-jag"]

    async def test_step1_grant_type_is_token_exchange(self, exchange, flow_params, fake_network):
        """Step 1 must use the RFC 8693 token exchange grant type."""
        await self._run_exchange(exchange, flow_params, fake_network)

        body = fake_network.step1_request.form_fields
        assert body["grant_type"] == ["urn:ietf:params:oauth:grant-type:token-exchange"]

    async def test_step1_client_assertion_jwt_claims(self, exchange, flow_params, fake_network):
        """Step 1 client assertion JWT must have correct iss, sub, aud claims."""
        await self._run_exchange(exchange, flow_params, fake_network)

        body = fake_network.step1_request.form_fields
        assertion_jwt = body["client_assertion"][0]
        # Decode without verification — we only check the claims structure
        claims = pyjwt.decode(assertion_jwt, options={"verify_signature": False})

        assert claims["iss"] == "wlp_principal_123"
        assert claims["sub"] == "wlp_principal_123"
        org_token_url = _TRUSTED_ISSUER + "/oauth2/v1/token"
        assert claims["aud"] == org_token_url

    async def test_step1_posts_to_org_as_token_url(self, exchange, flow_params, fake_network):
        """Step 1 must POST to {trusted_issuer}/oauth2/v1/token."""
        await self._run_exchange(exchange, flow_params, fake_network)

        expected_url = _TRUSTED_ISSUER + "/oauth2/v1/token"
        assert fake_network.step1_request.url == expected_url

    async def test_step2_sends_id_jag_as_assertion(self, exchange, flow_params, fake_network):
        """Step 2 must send the ID-JAG from Step 1 as the assertion."""
        await self._run_exchange(exchange, flow_params, fake_network)

        body = fake_network.step2_request.form_fields
        # The assertion must be the JWT returned by Step 1
        assertion = body["assertion"][0]
        assert assertion == fake_network.step1_access_token

    async def test_step2_grant_type_is_jwt_bearer(self, exchange, flow_params, fake_network):
        """Step 2 must use the RFC 7523 JWT bearer grant type."""
        await self._run_exchange(exchange, flow_params, fake_network)

        body = fake_network.step2_request.form_fields
        assert body["grant_type"] == ["urn:ietf:params:oauth:grant-type:jwt-bearer"]

    async def test_step2_client_assertion_audience_is_custom_as(
        self, exchange, flow_params, fake_network
    ):
        """Step 2 client assertion aud must be the custom AS token endpoint."""
        await self._run_exchange(exchange, flow_params, fake_network)

        body = fake_network.step2_request.form_fields
        assertion_jwt = body["client_assertion"][0]
        claims = pyjwt.decode(assertion_jwt, options={"verify_signature": False})

        custom_as_token_url = _EXCHANGE_AUDIENCE + "/v1/token"
        assert claims["aud"] == custom_as_token_url

    async def test_step2_posts_to_custom_as_token_url(self, exchange, flow_params, fake_network):
        """Step 2 must POST to the custom AS token endpoint."""
        await self._run_exchange(exchange, flow_params, fake_network)

        custom_as_token_url = _EXCHANGE_AUDIENCE + "/v1/token"
        assert fake_network.step2_request.url == custom_as_token_url

    async def test_returns_final_access_token(self, exchange, flow_params, fake_network):
        """exchange_token() must return the access token from Step 2."""
        result = await self._run_exchange(exchange, flow_params, fake_network)
        assert result == "final-scoped-from-sdk"

    async def test_multi_scope(self, exchange, flow_params, fake_network):
        """Multiple scopes must be sent correctly."""
        flow_params.id_jag_scopes = ["dr.impersonation", "openid"]
        await self._run_exchange(exchange, flow_params, fake_network)

        body = fake_network.step1_request.form_fields
        scope_value = body["scope"][0]
        assert set(scope_value.split()) == {"dr.impersonation", "openid"}

    async def test_raises_when_okta_sdk_not_installed(self, flow_params):
        """Must raise RuntimeError with install instructions when SDK is missing."""
        config = OAuth2CrossApplicationAccessAuthProviderConfig(
            principal_id="p", private_jwk=_REAL_JWK_B64
        )
        exchange = OktaTokenExchange(config)

        with patch(f"{_MODULE}._HAS_OKTA_SDK", False):
            with pytest.raises(RuntimeError, match="okta-client-python is not installed"):
                await exchange.exchange_token(flow_params, "token")

    def test_get_oauth2_client_additional_parameters(self) -> None:
        cross_app_flow_params = Mock()

        output = OktaTokenExchange.get_oauth2_client_additional_parameters(cross_app_flow_params)
        assert output == {"resource": cross_app_flow_params.target_audience}

    def test_get_oauth2_client_additional_parameters_without_resource_param(self) -> None:
        cross_app_flow_params = Mock()
        cross_app_flow_params.target_audience = None

        output = OktaTokenExchange.get_oauth2_client_additional_parameters(cross_app_flow_params)
        assert "resource" not in output
