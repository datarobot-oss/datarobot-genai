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

"""Tests for datarobot_genai.core.datarobot_otel."""

from __future__ import annotations

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import ProxyTracerProvider
from opentelemetry.util._once import Once

from datarobot_genai.core import datarobot_otel

_ENV_VARS = (
    "DATAROBOT_API_TOKEN",
    "MLOPS_DEPLOYMENT_ID",
    "DATAROBOT_ENDPOINT",
    "DATAROBOT_PUBLIC_API_ENDPOINT",
    "OTEL_SERVICE_NAME",
)


@pytest.fixture
def clean_env(monkeypatch):
    """Strip env vars + reset OTel global TracerProvider + module bootstrap flag."""
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    # OTel guards set_tracer_provider behind Once(); resetting both lets each
    # test exercise a fresh global slot without leaking to siblings.
    monkeypatch.setattr("opentelemetry.trace._TRACER_PROVIDER", None)
    monkeypatch.setattr("opentelemetry.trace._TRACER_PROVIDER_SET_ONCE", Once())
    monkeypatch.setitem(datarobot_otel._BOOTSTRAP_STATE, "installed", False)
    return monkeypatch


class TestEnvResolvers:
    def test_api_key_from_env(self, clean_env):
        clean_env.setenv("DATAROBOT_API_TOKEN", "tok")
        assert datarobot_otel.resolve_api_key_from_env() == "tok"

    def test_api_key_empty_when_unset(self, clean_env):
        assert datarobot_otel.resolve_api_key_from_env() == ""

    def test_entity_id_auto_prefixed(self, clean_env):
        clean_env.setenv("MLOPS_DEPLOYMENT_ID", "abc123")
        assert datarobot_otel.resolve_entity_id_from_env() == "deployment-abc123"

    def test_entity_id_empty_when_unset(self, clean_env):
        assert datarobot_otel.resolve_entity_id_from_env() == ""

    def test_endpoint_strips_api_path(self, clean_env):
        clean_env.setenv("DATAROBOT_ENDPOINT", "https://example.test/api/v2")
        assert (
            datarobot_otel.resolve_otel_endpoint_from_env() == "https://example.test/otel/v1/traces"
        )

    def test_public_endpoint_priority(self, clean_env):
        clean_env.setenv("DATAROBOT_PUBLIC_API_ENDPOINT", "https://public.test/api/v2")
        clean_env.setenv("DATAROBOT_ENDPOINT", "https://internal.test/api/v2")
        assert (
            datarobot_otel.resolve_otel_endpoint_from_env() == "https://public.test/otel/v1/traces"
        )

    def test_endpoint_empty_when_unset(self, clean_env):
        assert datarobot_otel.resolve_otel_endpoint_from_env() == ""

    def test_endpoint_empty_for_malformed_url(self, clean_env):
        clean_env.setenv("DATAROBOT_ENDPOINT", "not-a-url")
        assert datarobot_otel.resolve_otel_endpoint_from_env() == ""


class TestBootstrapOtelProvider:
    @staticmethod
    def _set_full_env(monkeypatch):
        monkeypatch.setenv("MLOPS_DEPLOYMENT_ID", "abc123")
        monkeypatch.setenv("DATAROBOT_API_TOKEN", "tok")
        monkeypatch.setenv("DATAROBOT_ENDPOINT", "https://example.test/api/v2")

    def test_skips_when_env_missing(self, clean_env):
        # No env set — bootstrap returns False and global provider remains
        # the default proxy.
        assert datarobot_otel.bootstrap_otel_provider_for_datarobot() is False
        assert isinstance(trace.get_tracer_provider(), ProxyTracerProvider)

    def test_skips_when_api_key_only(self, clean_env):
        clean_env.setenv("DATAROBOT_API_TOKEN", "tok")
        assert datarobot_otel.bootstrap_otel_provider_for_datarobot() is False
        assert isinstance(trace.get_tracer_provider(), ProxyTracerProvider)

    def test_installs_provider_when_env_present(self, clean_env):
        self._set_full_env(clean_env)
        assert datarobot_otel.bootstrap_otel_provider_for_datarobot() is True

        provider = trace.get_tracer_provider()
        assert isinstance(provider, TracerProvider)

        # Resource attributes include service.name derived from MLOPS_DEPLOYMENT_ID.
        attrs = provider.resource.attributes
        assert attrs["service.name"] == "deployment-abc123"
        assert attrs["telemetry.sdk.language"] == "python"

    def test_explicit_otel_service_name_wins(self, clean_env):
        self._set_full_env(clean_env)
        clean_env.setenv("OTEL_SERVICE_NAME", "my-pinned-service")
        datarobot_otel.bootstrap_otel_provider_for_datarobot()

        provider = trace.get_tracer_provider()
        assert provider.resource.attributes["service.name"] == "my-pinned-service"

    def test_exporter_endpoint_and_headers(self, clean_env, monkeypatch):
        # Capture the OTLPSpanExporter constructor args so we can assert on
        # the endpoint + DR auth headers without making a real HTTP call.
        captured = {}

        from opentelemetry.exporter.otlp.proto.http import trace_exporter as exporter_module

        real_exporter_cls = exporter_module.OTLPSpanExporter

        def spy_exporter(**kwargs):
            captured.update(kwargs)
            # Build the real exporter — the BatchSpanProcessor will keep a
            # reference but we won't actually export anything in the test.
            return real_exporter_cls(**kwargs)

        monkeypatch.setattr(
            "datarobot_genai.core.datarobot_otel.OTLPSpanExporter",
            spy_exporter,
            raising=False,
        )
        # The import is local inside the bootstrap function, so the symbol
        # doesn't exist at module level. Patch it on the source module instead.
        monkeypatch.setattr(exporter_module, "OTLPSpanExporter", spy_exporter)

        self._set_full_env(clean_env)
        datarobot_otel.bootstrap_otel_provider_for_datarobot()

        assert captured["endpoint"] == "https://example.test/otel/v1/traces"
        assert captured["headers"]["X-DataRobot-Api-Key"] == "tok"
        assert captured["headers"]["X-DataRobot-Entity-Id"] == "deployment-abc123"

    def test_idempotent_second_call(self, clean_env):
        self._set_full_env(clean_env)
        assert datarobot_otel.bootstrap_otel_provider_for_datarobot() is True
        provider_first = trace.get_tracer_provider()

        # Second call should detect the module flag and no-op.
        assert datarobot_otel.bootstrap_otel_provider_for_datarobot() is False
        assert trace.get_tracer_provider() is provider_first

    def test_skips_when_provider_already_set(self, clean_env):
        # Simulate DRMCP / another component installing its own provider before
        # instrument() runs. Bootstrap should detect the non-Proxy provider
        # and step aside.
        pre_existing = TracerProvider()
        trace.set_tracer_provider(pre_existing)

        self._set_full_env(clean_env)
        assert datarobot_otel.bootstrap_otel_provider_for_datarobot() is False
        assert trace.get_tracer_provider() is pre_existing
