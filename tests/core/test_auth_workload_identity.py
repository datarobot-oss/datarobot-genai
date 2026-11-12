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

from __future__ import annotations

import logging
import time
from urllib.parse import parse_qs

import httpx
import jwt
import pytest
import requests
import responses
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm

from datarobot_genai.core.auth import workload_identity as wt
from datarobot_genai.core.auth.workload_identity import auth as wt_auth
from datarobot_genai.core.auth.workload_identity import source as wt_source

PUBLIC_API = "https://app.example.com/api/v2"
DISCOVERY = "https://app.example.com/.well-known/openid-configuration"
RFC8414 = "https://app.example.com/.well-known/oauth-authorization-server"
TOKEN_URL = "https://app.example.com/oauth2/token"
COLLECTOR = "https://app.example.com/otel/v1/traces"

_ENV_VARS = (
    wt.CLIENT_ID_ENV,
    wt.WORKLOAD_TOKEN_FILE_ENV,
    wt.WORKLOAD_ID_ENV,
    wt.PRIVATE_KEY_ENV,
    wt.KEY_ID_ENV,
    wt.KEY_ALG_ENV,
    wt.PUBLIC_API_ENDPOINT_ENV,
    wt.DISCOVERY_URL_ENV,
    wt.TOKEN_ENDPOINT_ENV,
    wt.ASSERTION_AUDIENCE_ENV,
    "OTEL_EXPORTER_OTLP_HEADERS",
)


@pytest.fixture(scope="module")
def rsa_key() -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture(scope="module")
def pem(rsa_key: rsa.RSAPrivateKey) -> str:
    return rsa_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()


@pytest.fixture
def clean_env(monkeypatch):
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    wt_source._reset_workload_token_source()
    yield monkeypatch
    wt_source._reset_workload_token_source()


@pytest.fixture
def identity_env(clean_env, pem):
    clean_env.setenv(wt.CLIENT_ID_ENV, "wl-123")
    clean_env.setenv(wt.PRIVATE_KEY_ENV, pem)
    clean_env.setenv(wt.KEY_ID_ENV, "kid-1")
    clean_env.setenv(wt.PUBLIC_API_ENDPOINT_ENV, PUBLIC_API)
    return clean_env


def _no_discovery(env) -> None:
    env.setenv(wt.TOKEN_ENDPOINT_ENV, TOKEN_URL)
    env.setenv(wt.ASSERTION_AUDIENCE_ENV, TOKEN_URL)


class FakeClock:
    def __init__(self, now: float = 1_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _token_ok(token: str = "tok-1", expires_in: int = 7 * 24 * 3600) -> dict:
    return {"access_token": token, "token_type": "bearer", "expires_in": expires_in}


def _source(clock: FakeClock, *, audience=None, scopes=(), **overrides) -> wt.WorkloadTokenSource:
    config = wt.WorkloadIdentityConfig.from_env()
    assert config is not None
    for k, v in overrides.items():
        object.__setattr__(config, k, v)
    src = wt.WorkloadTokenSource(config, audience=audience, scopes=scopes, clock=clock)
    # No background thread in unit tests; refreshes are driven explicitly.
    src._thread = object()  # type: ignore[assignment]
    return src


class TestConfigFromEnv:
    def test_absent_env_means_disabled(self, clean_env):
        # GIVEN no workload identity env
        # WHEN reading config / source
        # THEN identity stays off
        assert wt.WorkloadIdentityConfig.from_env() is None
        assert wt.get_workload_token_source() is None

    def test_partial_env_means_disabled(self, identity_env):
        identity_env.delenv(wt.CLIENT_ID_ENV)
        assert wt.WorkloadIdentityConfig.from_env() is None

    def test_reads_all_fields(self, identity_env):
        identity_env.setenv(wt.TOKEN_ENDPOINT_ENV, TOKEN_URL)
        identity_env.setenv(wt.ASSERTION_AUDIENCE_ENV, "https://issuer/")
        config = wt.WorkloadIdentityConfig.from_env()
        assert config is not None
        assert (config.client_id, config.key_id, config.api_endpoint) == (
            "wl-123",
            "kid-1",
            PUBLIC_API,
        )
        assert config.token_endpoint == TOKEN_URL
        assert config.assertion_audience == "https://issuer/"

    def test_repr_hides_key(self, identity_env, pem):
        config = wt.WorkloadIdentityConfig.from_env()
        assert "PRIVATE KEY" not in repr(config)

    def test_missing_key_id_means_disabled(self, identity_env):
        identity_env.delenv(wt.KEY_ID_ENV)
        assert wt.WorkloadIdentityConfig.from_env() is None

    def test_client_id_falls_back_to_workload_id(self, identity_env):
        # GIVEN no DATAROBOT_CLIENT_ID; the workload id is the client id
        identity_env.delenv(wt.CLIENT_ID_ENV)
        identity_env.setenv(wt.WORKLOAD_ID_ENV, "wl-from-workload-id")
        config = wt.WorkloadIdentityConfig.from_env()
        assert config is not None and config.client_id == "wl-from-workload-id"

    def test_jwk_key_is_rejected(self, identity_env, rsa_key):
        # GIVEN the pre-#1533 JWK form; only PKCS#8 PEM is accepted
        identity_env.setenv(wt.PRIVATE_KEY_ENV, RSAAlgorithm.to_jwk(rsa_key))
        assert wt.get_workload_token_source() is None

    def test_non_rsa_key_is_rejected(self, identity_env):
        from cryptography.hazmat.primitives.asymmetric import ec

        ec_pem = (
            ec.generate_private_key(ec.SECP256R1())
            .private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
            .decode()
        )
        identity_env.setenv(wt.PRIVATE_KEY_ENV, ec_pem)
        assert wt.get_workload_token_source() is None

    @pytest.mark.parametrize(
        "endpoint, expected",
        [
            (PUBLIC_API, [RFC8414, DISCOVERY]),
            (PUBLIC_API + "/", [RFC8414, DISCOVERY]),
            (
                "https://h.example.com/prefix/api/v2",
                [
                    "https://h.example.com/.well-known/oauth-authorization-server/prefix",
                    "https://h.example.com/prefix/.well-known/openid-configuration",
                ],
            ),
        ],
    )
    def test_discovery_urls_strip_api_v2(self, identity_env, endpoint, expected):
        # RFC 8414 inserts the well-known segment before the path; OIDC appends it.
        identity_env.setenv(wt.PUBLIC_API_ENDPOINT_ENV, endpoint)
        config = wt.WorkloadIdentityConfig.from_env()
        assert config is not None and config.discovery_urls() == expected

    def test_discovery_url_override_is_the_only_url(self, identity_env):
        identity_env.setenv(wt.DISCOVERY_URL_ENV, "https://other/.well-known/openid-configuration")
        config = wt.WorkloadIdentityConfig.from_env()
        assert config is not None
        assert config.discovery_urls() == ["https://other/.well-known/openid-configuration"]

    def test_bad_key_disables_without_leaking(self, identity_env, caplog):
        identity_env.setenv(wt.PRIVATE_KEY_ENV, "-----BEGIN PRIVATE KEY-----\nnot-a-key\n")
        with caplog.at_level(logging.DEBUG):
            assert wt.get_workload_token_source() is None
        assert "not-a-key" not in caplog.text


class TestTokenRequest:
    @responses.activate
    def test_discovers_and_requests_per_spec(self, identity_env, rsa_key):
        # GIVEN no RFC 8414 metadata, OIDC discovery and a token endpoint
        responses.get(RFC8414, status=404)
        responses.get(DISCOVERY, json={"token_endpoint": TOKEN_URL})
        responses.post(TOKEN_URL, json=_token_ok())
        clock = FakeClock()
        # WHEN a token is requested
        token = _source(clock).token()
        # THEN the client_credentials request matches the spec
        assert token == "tok-1"
        form = {k: v[0] for k, v in parse_qs(responses.calls[2].request.body).items()}
        assert form["grant_type"] == "client_credentials"
        assert form["client_id"] == "wl-123"
        assert form["audience"] == PUBLIC_API
        assert form["client_assertion_type"] == (
            "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"
        )
        assert "scope" not in form
        assertion = form["client_assertion"]
        assert jwt.get_unverified_header(assertion) == {
            "alg": "RS256",
            "typ": "JWT",
            "kid": "kid-1",
        }
        claims = jwt.decode(
            assertion,
            rsa_key.public_key(),
            algorithms=["RS256"],
            audience=TOKEN_URL,
            options={"verify_exp": False, "verify_iat": False},
        )
        assert claims["iss"] == claims["sub"] == "wl-123"
        assert claims["exp"] - claims["iat"] <= 300
        # iat is wall-clock time, not the injected monotonic clock
        assert abs(claims["iat"] - time.time()) < 60
        assert claims["jti"]

    @responses.activate
    def test_token_endpoint_and_assertion_audience_overrides(self, identity_env, rsa_key):
        # GIVEN explicit token endpoint and assertion audience
        identity_env.setenv(wt.TOKEN_ENDPOINT_ENV, TOKEN_URL)
        identity_env.setenv(wt.ASSERTION_AUDIENCE_ENV, "https://issuer.example.com/")
        responses.post(TOKEN_URL, json=_token_ok())
        # WHEN a token is requested
        assert _source(FakeClock()).token() == "tok-1"
        # THEN discovery is skipped and the assertion aud is the override
        assert len(responses.calls) == 1
        form = parse_qs(responses.calls[0].request.body)
        claims = jwt.decode(
            form["client_assertion"][0],
            rsa_key.public_key(),
            algorithms=["RS256"],
            audience="https://issuer.example.com/",
            options={"verify_exp": False, "verify_iat": False},
        )
        assert claims["aud"] == "https://issuer.example.com/"

    @responses.activate
    def test_unique_jti(self, identity_env):
        _no_discovery(identity_env)
        responses.post(TOKEN_URL, json=_token_ok(expires_in=60))
        clock = FakeClock()
        src = _source(clock)
        src.token()
        clock.now += 60
        src.refresh_if_due()
        jtis = {
            jwt.decode(
                parse_qs(c.request.body)["client_assertion"][0], options={"verify_signature": False}
            )["jti"]
            for c in responses.calls
        }
        assert len(jtis) == 2

    @responses.activate
    def test_error_response_does_not_leak(self, identity_env, pem, caplog):
        _no_discovery(identity_env)
        responses.post(TOKEN_URL, status=401, json={"error": "invalid_client"})
        with caplog.at_level(logging.DEBUG):
            with pytest.raises(wt.WorkloadTokenError, match="invalid_client") as err:
                _source(FakeClock()).token()
        assert "invalid_client" in caplog.text
        assert "PRIVATE KEY" not in str(err.value)
        assert "PRIVATE KEY" not in caplog.text
        assert "client_assertion=" not in caplog.text


class TestRefresh:
    @responses.activate
    def test_refreshes_at_default_interval(self, identity_env):
        # GIVEN a token that outlives the interval
        _no_discovery(identity_env)
        responses.post(TOKEN_URL, json=_token_ok("tok-1"))
        responses.post(TOKEN_URL, json=_token_ok("tok-2"))
        clock = FakeClock()
        src = _source(clock)
        assert src.token() == "tok-1"
        # WHEN less than an hour passes, THEN no refresh
        clock.now += 3599
        src.refresh_if_due()
        assert len(responses.calls) == 1
        # WHEN an hour passes, THEN it refreshes
        clock.now += 1
        src.refresh_if_due()
        assert src.token() == "tok-2"

    @responses.activate
    def test_refresh_interval_from_env(self, identity_env):
        # GIVEN a configured interval shorter than the token lifetime
        _no_discovery(identity_env)
        identity_env.setenv(wt.REFRESH_INTERVAL_ENV, "900")
        responses.post(TOKEN_URL, json=_token_ok())
        src = _source(FakeClock())
        src.token()
        # THEN it refreshes at the interval
        assert src.seconds_until_refresh() == pytest.approx(900)

    @pytest.mark.parametrize("raw", ["0", "-5", "soon"])
    def test_bad_refresh_interval_uses_default(self, identity_env, raw):
        identity_env.setenv(wt.REFRESH_INTERVAL_ENV, raw)
        config = wt.WorkloadIdentityConfig.from_env()
        assert config is not None
        assert config.refresh_interval == wt.DEFAULT_REFRESH_INTERVAL_SECONDS

    @responses.activate
    def test_interval_longer_than_lifetime_refreshes_before_expiry(self, identity_env):
        # GIVEN an interval longer than the token lifetime
        _no_discovery(identity_env)
        identity_env.setenv(wt.REFRESH_INTERVAL_ENV, "86400")
        responses.post(TOKEN_URL, json=_token_ok(expires_in=3600))
        src = _source(FakeClock())
        src.token()
        # THEN the lifetime wins
        assert src.seconds_until_refresh() == pytest.approx(2880)

    @responses.activate
    def test_short_lived_token_refreshes_before_expiry(self, identity_env):
        _no_discovery(identity_env)
        responses.post(TOKEN_URL, json=_token_ok(expires_in=600))
        src = _source(FakeClock())
        src.token()
        assert src.seconds_until_refresh() == pytest.approx(480)

    @responses.activate
    def test_keeps_token_through_failures_until_expiry(self, identity_env):
        # GIVEN a held 2-hour token
        _no_discovery(identity_env)
        responses.post(TOKEN_URL, json=_token_ok("tok-1", expires_in=7200))
        clock = FakeClock()
        src = _source(clock)
        assert src.token() == "tok-1"
        # WHEN refreshes fail
        responses.replace(responses.POST, TOKEN_URL, status=503)
        clock.now += 3600
        src.refresh_if_due()
        # THEN the old token is still used, and retries back off
        assert src.token() == "tok-1"
        assert src.seconds_until_refresh() == pytest.approx(30)
        clock.now += 30
        src.refresh_if_due()
        assert src.seconds_until_refresh() == pytest.approx(60)
        # WHEN it expires, THEN token() raises with the last failure
        clock.now = 1_000_000.0 + 7200
        with pytest.raises(wt.WorkloadTokenError, match="No valid workload token"):
            src.token()

    @responses.activate
    def test_backoff_caps_and_resets(self, identity_env):
        _no_discovery(identity_env)
        responses.post(TOKEN_URL, status=500)
        clock = FakeClock()
        src = _source(clock)
        for _ in range(10):
            src.refresh_if_due()
            clock.now += src.seconds_until_refresh()
        assert src._retry_delay == 300
        responses.replace(responses.POST, TOKEN_URL, json=_token_ok())
        src.refresh_if_due()
        assert src._retry_delay == 30

    @responses.activate
    def test_first_failure_does_not_block_each_call(self, identity_env):
        # GIVEN the first fetch failed
        _no_discovery(identity_env)
        responses.post(TOKEN_URL, status=500)
        src = _source(FakeClock())
        with pytest.raises(wt.WorkloadTokenError):
            src.token()
        # WHEN called again inside the backoff window, THEN it raises without a new request
        with pytest.raises(wt.WorkloadTokenError):
            src.token()
        assert len(responses.calls) == 1

    @responses.activate
    def test_short_lived_token_is_valid_until_refresh(self, identity_env):
        # GIVEN a 60s token
        _no_discovery(identity_env)
        responses.post(TOKEN_URL, json=_token_ok(expires_in=60))
        clock = FakeClock()
        src = _source(clock)
        assert src.token() == "tok-1"
        # THEN it is still served when its refresh (at 48s) comes due
        assert src.seconds_until_refresh() == pytest.approx(48)
        clock.now += 48
        assert src.token() == "tok-1"
        # AND it counts as expired 6s (10%) before its lifetime ends
        clock.now += 6
        with pytest.raises(wt.WorkloadTokenError):
            src.token()

    @responses.activate
    @pytest.mark.parametrize(
        "rfc8414",
        [
            {"body": "<html>not found</html>", "content_type": "text/html"},
            {"json": {"issuer": "https://app.example.com"}},
            {"json": ["not", "an", "object"]},
        ],
    )
    def test_discovery_falls_back_on_unusable_200(self, identity_env, rfc8414):
        responses.get(RFC8414, status=200, **rfc8414)
        responses.get(DISCOVERY, json={"token_endpoint": TOKEN_URL})
        responses.post(TOKEN_URL, json=_token_ok())
        assert _source(FakeClock()).token() == "tok-1"

    @responses.activate
    def test_discovery_without_token_endpoint_anywhere_raises(self, identity_env):
        responses.get(RFC8414, status=200, json={})
        responses.get(DISCOVERY, status=200, json={})
        with pytest.raises(wt.WorkloadTokenError, match="has no token_endpoint"):
            _source(FakeClock()).token()

    @responses.activate
    def test_discovery_is_retried_after_failure(self, identity_env):
        responses.get(RFC8414, status=404)
        responses.get(DISCOVERY, status=503)
        clock = FakeClock()
        src = _source(clock)
        with pytest.raises(wt.WorkloadTokenError):
            src.token()
        responses.replace(responses.GET, DISCOVERY, json={"token_endpoint": TOKEN_URL})
        responses.post(TOKEN_URL, json=_token_ok())
        clock.now += 30
        assert src.token() == "tok-1"


@pytest.fixture
def federation_env(clean_env, tmp_path):
    token_file = tmp_path / "token"
    token_file.write_text("sa-token-1\n")
    clean_env.setenv(wt.CLIENT_ID_ENV, "wl-123")
    clean_env.setenv(wt.WORKLOAD_TOKEN_FILE_ENV, str(token_file))
    clean_env.setenv(wt.PUBLIC_API_ENDPOINT_ENV, PUBLIC_API)
    clean_env.token_file = token_file  # type: ignore[attr-defined]
    return clean_env


def _form(call) -> dict[str, str]:
    return {k: v[0] for k, v in parse_qs(call.request.body).items()}


class TestModeDetection:
    def test_private_key_means_pkjwt(self, identity_env):
        config = wt.WorkloadIdentityConfig.from_env()
        assert config is not None and config.mode is wt.AuthMode.PRIVATE_KEY_JWT

    def test_token_file_means_federation(self, federation_env):
        config = wt.WorkloadIdentityConfig.from_env()
        assert config is not None and config.mode is wt.AuthMode.FEDERATION

    def test_both_prefer_federation(self, federation_env, pem, caplog):
        # GIVEN both a token file and a private key
        federation_env.setenv(wt.PRIVATE_KEY_ENV, pem)
        federation_env.setenv(wt.KEY_ID_ENV, "kid-1")
        with caplog.at_level(logging.INFO):
            config = wt.WorkloadIdentityConfig.from_env()
        # THEN federation wins (no secret), the key is dropped, and the choice is logged
        assert config is not None and config.mode is wt.AuthMode.FEDERATION
        assert config.private_key == ""
        assert "using federation" in caplog.text

    def test_federation_needs_client_id(self, federation_env):
        federation_env.delenv(wt.CLIENT_ID_ENV)
        assert wt.WorkloadIdentityConfig.from_env() is None


class TestFederation:
    @responses.activate
    def test_exchanges_projected_token_per_rfc8693(self, federation_env):
        # GIVEN OIDC discovery only (no RFC 8414) and the bridge's token endpoint
        responses.get(RFC8414, status=404)
        responses.get(DISCOVERY, json={"token_endpoint": TOKEN_URL})
        responses.post(TOKEN_URL, json=_token_ok("fed-1"))
        # WHEN a token is requested
        assert _source(FakeClock()).token() == "fed-1"
        # THEN it is a token exchange with no client credential and no resource
        form = _form(responses.calls[-1])
        assert form == {
            "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
            "subject_token": "sa-token-1",
            "subject_token_type": "urn:ietf:params:oauth:token-type:jwt",
            "client_id": "wl-123",
            "audience": PUBLIC_API,
        }

    @responses.activate
    def test_rereads_token_file_on_each_fetch(self, federation_env):
        federation_env.setenv(wt.TOKEN_ENDPOINT_ENV, TOKEN_URL)
        responses.post(TOKEN_URL, json=_token_ok(expires_in=60))
        clock = FakeClock()
        src = _source(clock)
        src.token()
        # WHEN the kubelet rotates the projected token
        federation_env.token_file.write_text("sa-token-2")
        clock.now += 60
        src.refresh_if_due()
        # THEN the next exchange presents the new one
        assert _form(responses.calls[-1])["subject_token"] == "sa-token-2"

    @responses.activate
    def test_token_endpoint_override_skips_discovery(self, federation_env):
        federation_env.setenv(wt.TOKEN_ENDPOINT_ENV, "https://bridge.local/oauth2/token")
        responses.post("https://bridge.local/oauth2/token", json=_token_ok())
        assert _source(FakeClock()).token() == "tok-1"
        assert len(responses.calls) == 1

    def test_unreadable_token_file(self, federation_env, caplog):
        federation_env.setenv(wt.TOKEN_ENDPOINT_ENV, TOKEN_URL)
        federation_env.token_file.unlink()
        with caplog.at_level(logging.DEBUG):
            with pytest.raises(wt.WorkloadTokenError, match="cannot read"):
                _source(FakeClock()).token()
        assert "cannot read DATAROBOT_WORKLOAD_TOKEN_FILE" in caplog.text


class TestAudienceAndScopes:
    @responses.activate
    def test_audience_and_scopes_in_client_credentials(self, identity_env):
        # GIVEN a source with its own audience and scopes
        _no_discovery(identity_env)
        responses.post(TOKEN_URL, json=_token_ok())
        src = _source(
            FakeClock(),
            audience="https://api.other/",
            scopes=["agents.read", "agents.write", "agents.read"],
        )
        # WHEN a token is requested
        src.token()
        # THEN both are sent, scopes space-separated and de-duplicated
        form = _form(responses.calls[-1])
        assert form["audience"] == "https://api.other/"
        assert form["scope"] == "agents.read agents.write"

    @responses.activate
    def test_scopes_in_token_exchange(self, federation_env):
        federation_env.setenv(wt.TOKEN_ENDPOINT_ENV, TOKEN_URL)
        responses.post(TOKEN_URL, json=_token_ok())
        _source(FakeClock(), scopes=("agents.read",)).token()
        form = _form(responses.calls[-1])
        assert form["scope"] == "agents.read"
        assert form["audience"] == PUBLIC_API

    @responses.activate
    def test_audience_override_keeps_discovery_on_api_endpoint(self, identity_env):
        # GIVEN discovery off the public API endpoint and a different audience
        identity_env.delenv(wt.ASSERTION_AUDIENCE_ENV, raising=False)
        responses.get(RFC8414, status=404)
        responses.get(DISCOVERY, json={"token_endpoint": TOKEN_URL})
        responses.post(TOKEN_URL, json=_token_ok())
        _source(FakeClock(), audience="https://api.other/").token()
        # THEN discovery still uses the endpoint, and only the audience changes
        assert responses.calls[0].request.url == RFC8414
        form = _form(responses.calls[-1])
        assert form["audience"] == "https://api.other/"
        # and the assertion aud stays the token endpoint
        claims = jwt.decode(form["client_assertion"], options={"verify_signature": False})
        assert claims["aud"] == TOKEN_URL

    @pytest.mark.parametrize("scopes", ["read write", ["has space"], [""], ['quo"te'], ["tab\t"]])
    def test_invalid_scopes_rejected(self, identity_env, scopes):
        config = wt.WorkloadIdentityConfig.from_env()
        assert config is not None
        with pytest.raises((TypeError, ValueError)):
            wt.WorkloadTokenSource(config, scopes=scopes)


class TestDiscoveryAndOverrides:
    @responses.activate
    def test_rfc8414_metadata_wins_when_served(self, identity_env):
        responses.get(RFC8414, json={"token_endpoint": TOKEN_URL})
        responses.post(TOKEN_URL, json=_token_ok())
        assert _source(FakeClock()).token() == "tok-1"
        assert [c.request.url for c in responses.calls] == [RFC8414, TOKEN_URL]

    @responses.activate
    def test_rfc8414_404_falls_back_quietly(self, identity_env, caplog):
        responses.get(RFC8414, status=404)
        responses.get(DISCOVERY, json={"token_endpoint": TOKEN_URL})
        responses.post(TOKEN_URL, json=_token_ok())
        with caplog.at_level(logging.INFO):
            assert _source(FakeClock()).token() == "tok-1"
        assert "oauth-authorization-server" not in caplog.text

    @responses.activate
    def test_rfc8414_connection_error_falls_back(self, identity_env):
        # GIVEN nothing answers at the RFC 8414 URL (responses raises ConnectionError)
        responses.get(DISCOVERY, json={"token_endpoint": TOKEN_URL})
        responses.post(TOKEN_URL, json=_token_ok())
        assert _source(FakeClock()).token() == "tok-1"

    @responses.activate
    def test_discovery_is_cached(self, identity_env):
        responses.get(RFC8414, status=404)
        responses.get(DISCOVERY, json={"token_endpoint": TOKEN_URL})
        responses.post(TOKEN_URL, json=_token_ok(expires_in=60))
        clock = FakeClock()
        src = _source(clock)
        src.token()
        clock.now += 60
        src.refresh_if_due()
        assert sum(c.request.method == "GET" for c in responses.calls) == 2

    @responses.activate
    def test_discovery_url_override_used_verbatim(self, identity_env):
        identity_env.setenv(wt.DISCOVERY_URL_ENV, "https://other.example.com/meta")
        responses.get("https://other.example.com/meta", json={"token_endpoint": TOKEN_URL})
        responses.post(TOKEN_URL, json=_token_ok())
        assert _source(FakeClock()).token() == "tok-1"
        assert responses.calls[0].request.url == "https://other.example.com/meta"

    @responses.activate
    def test_token_endpoint_override_keeps_discovered_assertion_aud(self, identity_env):
        # GIVEN a token endpoint override and discovery
        identity_env.setenv(wt.TOKEN_ENDPOINT_ENV, "https://enclave.local/oauth2/token")
        responses.get(RFC8414, status=404)
        responses.get(DISCOVERY, json={"token_endpoint": TOKEN_URL})
        responses.post("https://enclave.local/oauth2/token", json=_token_ok())
        # WHEN a token is requested
        assert _source(FakeClock()).token() == "tok-1"
        # THEN it goes to the override, but the assertion aud is discovery's token_endpoint
        call = responses.calls[-1]
        assert call.request.url == "https://enclave.local/oauth2/token"
        claims = jwt.decode(_form(call)["client_assertion"], options={"verify_signature": False})
        assert claims["aud"] == TOKEN_URL

    @responses.activate
    def test_assertion_audience_override_wins(self, identity_env):
        identity_env.setenv(wt.ASSERTION_AUDIENCE_ENV, "https://issuer.example.com/")
        responses.get(RFC8414, status=404)
        responses.get(DISCOVERY, json={"token_endpoint": TOKEN_URL})
        responses.post(TOKEN_URL, json=_token_ok())
        assert _source(FakeClock()).token() == "tok-1"
        claims = jwt.decode(
            _form(responses.calls[-1])["client_assertion"], options={"verify_signature": False}
        )
        assert claims["aud"] == "https://issuer.example.com/"


def _ec_pem(curve) -> tuple[str, object]:
    from cryptography.hazmat.primitives.asymmetric import ec

    key = ec.generate_private_key(curve)
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    return pem, key


def _signed_assertion(env, alg_env: str | None) -> tuple[dict, str]:
    """Request one token and return the assertion header and token string."""
    _no_discovery(env)
    if alg_env is not None:
        env.setenv(wt.KEY_ALG_ENV, alg_env)
    with responses.RequestsMock() as rsps:
        rsps.post(TOKEN_URL, json=_token_ok())
        assert _source(FakeClock()).token() == "tok-1"
        assertion = _form(rsps.calls[0])["client_assertion"]
    return jwt.get_unverified_header(assertion), assertion


class TestKeyAlg:
    @pytest.mark.parametrize("value", [None, ""])
    def test_defaults_to_rs256(self, identity_env, rsa_key, value):
        # GIVEN DATAROBOT_CLIENT_KEY_ALG unset or empty
        header, assertion = _signed_assertion(identity_env, value)
        # THEN the assertion is RS256
        assert header["alg"] == "RS256" and header["kid"] == "kid-1"
        jwt.decode(
            assertion,
            rsa_key.public_key(),
            algorithms=["RS256"],
            audience=TOKEN_URL,
            options={"verify_exp": False, "verify_iat": False},
        )

    def test_ps256(self, identity_env, rsa_key):
        header, assertion = _signed_assertion(identity_env, "PS256")
        assert header["alg"] == "PS256"
        jwt.decode(
            assertion,
            rsa_key.public_key(),
            algorithms=["PS256"],
            audience=TOKEN_URL,
            options={"verify_exp": False, "verify_iat": False},
        )

    def test_es256_with_p256_key(self, identity_env):
        from cryptography.hazmat.primitives.asymmetric import ec

        pem, key = _ec_pem(ec.SECP256R1())
        identity_env.setenv(wt.PRIVATE_KEY_ENV, pem)
        header, assertion = _signed_assertion(identity_env, "ES256")
        assert header["alg"] == "ES256"
        jwt.decode(
            assertion,
            key.public_key(),  # type: ignore[attr-defined]
            algorithms=["ES256"],
            audience=TOKEN_URL,
            options={"verify_exp": False, "verify_iat": False},
        )

    @pytest.mark.parametrize(
        "alg, use_ec",
        [("ES256", False), ("RS256", True), ("PS256", True), ("ES384", True)],
    )
    def test_alg_key_mismatch_is_rejected(self, identity_env, alg, use_ec, caplog):
        # GIVEN a key that doesn't fit the alg (ES384 with a P-256 key: wrong curve)
        from cryptography.hazmat.primitives.asymmetric import ec

        if use_ec:
            identity_env.setenv(wt.PRIVATE_KEY_ENV, _ec_pem(ec.SECP256R1())[0])
        identity_env.setenv(wt.KEY_ALG_ENV, alg)
        config = wt.WorkloadIdentityConfig.from_env()
        assert config is not None
        # THEN loading fails with WorkloadTokenError and no key material
        with pytest.raises(wt.WorkloadTokenError) as exc:
            wt.WorkloadTokenSource(config)
        assert "PRIVATE KEY" not in str(exc.value)
        with caplog.at_level(logging.DEBUG):
            assert wt.get_workload_token_source() is None
        assert "PRIVATE KEY" not in caplog.text

    @pytest.mark.parametrize("alg", ["none", "HS256", "EdDSA", "RS1024"])
    def test_unknown_alg_is_rejected(self, identity_env, alg):
        identity_env.setenv(wt.KEY_ALG_ENV, alg)
        config = wt.WorkloadIdentityConfig.from_env()
        assert config is not None
        with pytest.raises(wt.WorkloadTokenError, match="not a supported signing alg"):
            wt.WorkloadTokenSource(config)

    def test_federation_ignores_alg(self, federation_env):
        federation_env.setenv(wt.KEY_ALG_ENV, "HS256")
        federation_env.setenv(wt.TOKEN_ENDPOINT_ENV, TOKEN_URL)
        config = wt.WorkloadIdentityConfig.from_env()
        assert config is not None
        wt.WorkloadTokenSource(config)  # no key loaded, no error


class FixedToken:
    def __init__(self, token: str | None) -> None:
        self._token = token

    def token(self) -> str:
        if self._token is None:
            raise wt.WorkloadTokenError("No valid workload token: down")
        return self._token


class TestAuthAdapters:
    def test_requests_auth_sets_bearer(self):
        req = requests.Request("GET", "https://x.test/", headers={"X-Other": "1"}).prepare()
        out = wt_auth.WorkloadTokenAuth(FixedToken("tok"))(req)  # type: ignore[arg-type]
        assert out.headers["Authorization"] == "Bearer tok"
        assert out.headers["X-Other"] == "1"

    def test_requests_auth_raises_without_token(self):
        req = requests.Request("GET", "https://x.test/").prepare()
        with pytest.raises(wt.WorkloadTokenError):
            wt_auth.WorkloadTokenAuth(FixedToken(None))(req)  # type: ignore[arg-type]

    def test_httpx_auth_sync(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(request.headers)
            return httpx.Response(200)

        auth = wt_auth.WorkloadTokenHttpxAuth(FixedToken("tok"))  # type: ignore[arg-type]
        with httpx.Client(transport=httpx.MockTransport(handler), auth=auth) as client:
            client.get("https://x.test/")
        assert seen["authorization"] == "Bearer tok"

    def test_httpx_auth_async(self):
        import asyncio

        seen = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            seen.update(request.headers)
            return httpx.Response(200)

        async def run() -> None:
            auth = wt_auth.WorkloadTokenHttpxAuth(FixedToken("tok"))  # type: ignore[arg-type]
            transport = httpx.MockTransport(handler)
            async with httpx.AsyncClient(transport=transport, auth=auth) as client:
                await client.get("https://x.test/")

        asyncio.run(run())
        assert seen["authorization"] == "Bearer tok"

    def test_httpx_auth_raises_without_token(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("request must not be sent")

        auth = wt_auth.WorkloadTokenHttpxAuth(FixedToken(None))  # type: ignore[arg-type]
        with httpx.Client(transport=httpx.MockTransport(handler), auth=auth) as client:
            with pytest.raises(wt.WorkloadTokenError):
                client.get("https://x.test/")
