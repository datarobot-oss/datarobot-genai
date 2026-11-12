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

import os

import pytest
import requests
import responses
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from datarobot_genai.core.auth import workload_token as wt
from datarobot_genai.core.telemetry import workload_otlp as wo

PUBLIC_API = "https://app.example.com/api/v2"
TOKEN_URL = "https://app.example.com/oauth2/token"
COLLECTOR = "https://app.example.com/otel/v1/traces"


def _token_ok(token: str = "tok-1", expires_in: int = 7 * 24 * 3600) -> dict:
    return {"access_token": token, "token_type": "bearer", "expires_in": expires_in}


@pytest.fixture
def clean_env(monkeypatch):
    for var in (
        wt.CLIENT_ID_ENV,
        wt.WORKLOAD_ID_ENV,
        wt.PRIVATE_KEY_ENV,
        wt.KEY_ID_ENV,
        wt.PUBLIC_API_ENDPOINT_ENV,
        wt.DISCOVERY_URL_ENV,
        wt.TOKEN_ENDPOINT_ENV,
        wt.ASSERTION_AUDIENCE_ENV,
        wo.OTLP_CREDENTIAL_PROVIDER_ENV,
        "OTEL_EXPORTER_OTLP_HEADERS",
    ):
        monkeypatch.delenv(var, raising=False)
    wt._reset_workload_token_source()
    yield monkeypatch
    wt._reset_workload_token_source()


@pytest.fixture
def identity_env(clean_env):
    pem = (
        rsa.generate_private_key(public_exponent=65537, key_size=2048)
        .private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        .decode()
    )
    clean_env.setenv(wt.CLIENT_ID_ENV, "wl-123")
    clean_env.setenv(wt.PRIVATE_KEY_ENV, pem)
    clean_env.setenv(wt.KEY_ID_ENV, "kid-1")
    clean_env.setenv(wt.PUBLIC_API_ENDPOINT_ENV, PUBLIC_API)
    return clean_env


class TestOtlpIntegration:
    @responses.activate
    def test_otlp_exporter_sends_bearer_and_static_headers(self, identity_env):
        # GIVEN workload identity and today's static OTLP headers
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        identity_env.setenv(wt.TOKEN_ENDPOINT_ENV, TOKEN_URL)
        identity_env.setenv(wt.ASSERTION_AUDIENCE_ENV, TOKEN_URL)
        responses.post(TOKEN_URL, json=_token_ok("tok-1"))
        responses.post(COLLECTOR, status=200)
        exporter = OTLPSpanExporter(
            endpoint=COLLECTOR,
            headers={"X-DataRobot-Entity-Id": "workload-wl-123"},
            session=wo.workload_otlp_session(),
        )
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        # WHEN a span is exported
        provider.get_tracer("t").start_span("s").end()
        provider.shutdown()
        # THEN the collector gets the bearer token alongside the static header
        sent = [c.request for c in responses.calls if c.request.url == COLLECTOR]
        assert sent, "no export request"
        assert sent[0].headers["Authorization"] == "Bearer tok-1"
        assert sent[0].headers["X-DataRobot-Entity-Id"] == "workload-wl-123"

    @responses.activate
    def test_export_proceeds_without_bearer_when_token_fails(self, identity_env):
        # GIVEN the token endpoint is down
        identity_env.setenv(wt.TOKEN_ENDPOINT_ENV, TOKEN_URL)
        identity_env.setenv(wt.ASSERTION_AUDIENCE_ENV, TOKEN_URL)
        responses.post(TOKEN_URL, status=503)
        responses.post(COLLECTOR, status=200)
        session = wo.workload_otlp_session()
        assert session is not None
        # WHEN exporting, THEN the request goes out with the static headers only
        session.post(COLLECTOR, headers={"X-DataRobot-Api-Key": "k"}, data=b"")
        sent = [c.request for c in responses.calls if c.request.url == COLLECTOR]
        assert sent and "Authorization" not in sent[0].headers

    def test_entry_point_factory_returns_session_without_identity(self, clean_env):
        session = wo.otlp_session()
        assert isinstance(session, requests.Session)
        assert session.auth is None


class TestCredentialProviderEnv:
    def test_noop_without_identity(self, clean_env):
        assert wo.enable_otlp_credential_provider() is False
        assert wo.OTLP_CREDENTIAL_PROVIDER_ENV not in os.environ

    def test_noop_when_entry_point_missing(self, identity_env):
        identity_env.setattr(wo, "entry_points", lambda **_: [])
        assert wo.enable_otlp_credential_provider() is False

    def test_respects_existing_setting(self, identity_env):
        identity_env.setenv(wo.OTLP_CREDENTIAL_PROVIDER_ENV, "other")
        assert wo.enable_otlp_credential_provider() is False

    def test_sets_env_when_available(self, identity_env):
        # Empty counts as unset; monkeypatch restores it afterwards.
        identity_env.setenv(wo.OTLP_CREDENTIAL_PROVIDER_ENV, "")
        identity_env.setattr(wo, "entry_points", lambda **_: ["ep"])
        assert wo.enable_otlp_credential_provider() is True
        assert os.environ[wo.OTLP_CREDENTIAL_PROVIDER_ENV] == wo.OTLP_CREDENTIAL_PROVIDER_NAME

    @responses.activate
    def test_exporter_without_session_uses_credential_provider(self, identity_env):
        # GIVEN an exporter built with no session, as NAT builds it
        from importlib.metadata import entry_points

        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        if not entry_points(
            group=wo.OTLP_CREDENTIAL_PROVIDER_GROUP, name=wo.OTLP_CREDENTIAL_PROVIDER_NAME
        ):
            pytest.skip("datarobot-genai entry points not installed")
        identity_env.setenv(wt.TOKEN_ENDPOINT_ENV, TOKEN_URL)
        identity_env.setenv(wt.ASSERTION_AUDIENCE_ENV, TOKEN_URL)
        identity_env.setenv(wo.OTLP_CREDENTIAL_PROVIDER_ENV, "")
        responses.post(TOKEN_URL, json=_token_ok("tok-1"))
        responses.post(COLLECTOR, status=200)
        assert wo.enable_otlp_credential_provider() is True
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint=COLLECTOR)))
        # WHEN a span is exported
        provider.get_tracer("t").start_span("s").end()
        provider.shutdown()
        # THEN the SDK-loaded session added the bearer token
        sent = [c.request for c in responses.calls if c.request.url == COLLECTOR]
        assert sent and sent[0].headers["Authorization"] == "Bearer tok-1"
