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
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import ProxyTracerProvider
from opentelemetry.util._once import Once

from datarobot_genai.core.telemetry import datarobot_otel
from datarobot_genai.core.telemetry.nat_tracer import _NAT_TRACER_WRAPPED_ATTR

_ENV_VARS = (
    "DATAROBOT_API_TOKEN",
    "DATAROBOT_DEFAULT_USE_CASE",
    "DATAROBOT_USE_CASE_ID",
    "MLOPS_DEPLOYMENT_ID",
    "WORKLOAD_ID",
    "DATAROBOT_ENDPOINT",
    "DATAROBOT_PUBLIC_API_ENDPOINT",
    "OTEL_SERVICE_NAME",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_HEADERS",
)


@pytest.fixture
def clean_env(monkeypatch):
    """Strip env vars + reset OTel global TracerProvider + module bootstrap flag."""
    for var in _ENV_VARS:
        # delenv alone records no undo for an absent var, so writes would outlive the test.
        monkeypatch.setenv(var, "")
        monkeypatch.delenv(var)
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

    def test_entity_id_workload(self, clean_env):
        clean_env.setenv("WORKLOAD_ID", "wkl42")
        assert datarobot_otel.resolve_entity_id_from_env() == "workload-wkl42"

    def test_entity_id_deployment_wins_over_workload(self, clean_env):
        clean_env.setenv("MLOPS_DEPLOYMENT_ID", "dep1")
        clean_env.setenv("WORKLOAD_ID", "wkl2")
        assert datarobot_otel.resolve_entity_id_from_env() == "deployment-dep1"

    def test_endpoint_strips_api_path(self, clean_env):
        clean_env.setenv("DATAROBOT_ENDPOINT", "https://example.test/api/v2")
        assert (
            datarobot_otel.resolve_otel_traces_endpoint_from_env()
            == "https://example.test/otel/v1/traces"
        )

    def test_public_endpoint_priority(self, clean_env):
        clean_env.setenv("DATAROBOT_PUBLIC_API_ENDPOINT", "https://public.test/api/v2")
        clean_env.setenv("DATAROBOT_ENDPOINT", "https://internal.test/api/v2")
        assert (
            datarobot_otel.resolve_otel_traces_endpoint_from_env()
            == "https://public.test/otel/v1/traces"
        )

    def test_endpoint_empty_when_unset(self, clean_env):
        assert datarobot_otel.resolve_otel_traces_endpoint_from_env() == ""

    def test_endpoint_empty_for_malformed_url(self, clean_env):
        clean_env.setenv("DATAROBOT_ENDPOINT", "not-a-url")
        assert datarobot_otel.resolve_otel_traces_endpoint_from_env() == ""

    def test_explicit_otlp_base_url_appends_traces_path(self, clean_env):
        # OTEL_EXPORTER_OTLP_ENDPOINT is the standard OTel base URL; we append
        # /v1/traces rather than returning it verbatim.
        clean_env.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "https://collector.test:4318")
        assert (
            datarobot_otel.resolve_otel_traces_endpoint_from_env()
            == "https://collector.test:4318/v1/traces"
        )

    def test_explicit_otlp_base_url_strips_trailing_slash(self, clean_env):
        clean_env.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "https://collector.test:4318/")
        assert (
            datarobot_otel.resolve_otel_traces_endpoint_from_env()
            == "https://collector.test:4318/v1/traces"
        )

    def test_explicit_otlp_endpoint_wins_over_datarobot_endpoint(self, clean_env):
        # The standard OTel override takes precedence over the DR-derived
        # endpoint so an operator can point spans at any collector.
        clean_env.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "https://collector.test")
        clean_env.setenv("DATAROBOT_ENDPOINT", "https://example.test/api/v2")
        assert (
            datarobot_otel.resolve_otel_traces_endpoint_from_env()
            == "https://collector.test/v1/traces"
        )


class TestHeaderResolvers:
    def test_headers_from_datarobot_env(self, clean_env):
        clean_env.setenv("DATAROBOT_API_TOKEN", "tok")
        clean_env.setenv("MLOPS_DEPLOYMENT_ID", "abc123")
        assert datarobot_otel.resolve_datarobot_headers_from_env() == {
            "X-DataRobot-Api-Key": "tok",
            "X-DataRobot-Entity-Id": "deployment-abc123",
        }

    def test_headers_from_workload_env(self, clean_env):
        clean_env.setenv("DATAROBOT_API_TOKEN", "tok")
        clean_env.setenv("WORKLOAD_ID", "wkl42")
        assert datarobot_otel.resolve_datarobot_headers_from_env() == {
            "X-DataRobot-Api-Key": "tok",
            "X-DataRobot-Entity-Id": "workload-wkl42",
        }

    def test_headers_none_when_env_unset(self, clean_env):
        # Both DATAROBOT_API_TOKEN and an entity ID must be set; otherwise
        # the resolver returns None so callers can skip auth header injection.
        assert datarobot_otel.resolve_datarobot_headers_from_env() is None

    def test_headers_none_when_api_key_only(self, clean_env):
        clean_env.setenv("DATAROBOT_API_TOKEN", "tok")
        assert datarobot_otel.resolve_datarobot_headers_from_env() is None

    def test_headers_none_when_entity_id_only(self, clean_env):
        clean_env.setenv("MLOPS_DEPLOYMENT_ID", "abc123")
        assert datarobot_otel.resolve_datarobot_headers_from_env() is None

    def test_headers_parsed_from_otlp_env(self, clean_env):
        clean_env.setenv(
            "OTEL_EXPORTER_OTLP_HEADERS",
            "X-DataRobot-Api-Key=env-key,X-DataRobot-Entity-Id=deployment-env",
        )
        assert datarobot_otel.resolve_datarobot_headers_from_env() == {
            "X-DataRobot-Api-Key": "env-key",
            "X-DataRobot-Entity-Id": "deployment-env",
        }

    def test_headers_value_with_equals_preserved(self, clean_env):
        # A header value can legitimately contain '=' (e.g. base64 padding or
        # a token with '='). Splitting on the first '=' only must keep the
        # full value intact rather than truncating at the first '='.
        clean_env.setenv(
            "OTEL_EXPORTER_OTLP_HEADERS",
            "Authorization=Basic dXNlcjpwYXNz==,X-Token=a=b=c",
        )
        assert datarobot_otel.resolve_datarobot_headers_from_env() == {
            "Authorization": "Basic dXNlcjpwYXNz==",
            "X-Token": "a=b=c",
        }


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

    def test_skips_when_entity_id_only(self, clean_env):
        clean_env.setenv("MLOPS_DEPLOYMENT_ID", "abc123")
        assert datarobot_otel.bootstrap_otel_provider_for_datarobot() is False
        assert isinstance(trace.get_tracer_provider(), ProxyTracerProvider)

    def test_installs_provider_when_env_present(self, clean_env):
        self._set_full_env(clean_env)
        assert datarobot_otel.bootstrap_otel_provider_for_datarobot() is True

        provider = trace.get_tracer_provider()
        assert isinstance(provider, TracerProvider)
        assert getattr(provider, _NAT_TRACER_WRAPPED_ATTR, False)

        # Resource attributes include service.name derived from MLOPS_DEPLOYMENT_ID.
        attrs = provider.resource.attributes
        assert attrs["service.name"] == "deployment-abc123"
        assert attrs["telemetry.sdk.language"] == "python"

    def test_defaults_to_batch_span_processor(self, clean_env):
        self._set_full_env(clean_env)
        assert datarobot_otel.bootstrap_otel_provider_for_datarobot() is True

        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = trace.get_tracer_provider()
        active = provider._active_span_processor
        registered = getattr(active, "_span_processors", (active,))
        assert any(isinstance(p, BatchSpanProcessor) for p in registered)

    def test_simple_span_processor_env_switch(self, clean_env):
        # e2e sets DATAROBOT_OTEL_SPAN_PROCESSOR=simple so spans export
        # synchronously and don't bleed across the shared mock collector.
        self._set_full_env(clean_env)
        clean_env.setenv("DATAROBOT_OTEL_SPAN_PROCESSOR", "simple")
        assert datarobot_otel.bootstrap_otel_provider_for_datarobot() is True

        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        provider = trace.get_tracer_provider()
        active = provider._active_span_processor
        registered = getattr(active, "_span_processors", (active,))
        assert any(isinstance(p, SimpleSpanProcessor) for p in registered)
        assert not any(isinstance(p, BatchSpanProcessor) for p in registered)

    def test_installs_provider_with_workload_env(self, clean_env):
        clean_env.setenv("WORKLOAD_ID", "wkl42")
        clean_env.setenv("DATAROBOT_API_TOKEN", "tok")
        clean_env.setenv("DATAROBOT_ENDPOINT", "https://example.test/api/v2")
        assert datarobot_otel.bootstrap_otel_provider_for_datarobot() is True

        provider = trace.get_tracer_provider()
        assert isinstance(provider, TracerProvider)
        attrs = provider.resource.attributes
        assert attrs["service.name"] == "workload-wkl42"

    def test_service_name_deployment_wins_over_workload(self, clean_env):
        self._set_full_env(clean_env)
        clean_env.setenv("WORKLOAD_ID", "wkl99")
        datarobot_otel.bootstrap_otel_provider_for_datarobot()

        provider = trace.get_tracer_provider()
        assert provider.resource.attributes["service.name"] == "deployment-abc123"

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
            "datarobot_genai.core.telemetry.datarobot_otel.OTLPSpanExporter",
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

    def test_attaches_processor_to_existing_sdk_provider(self, clean_env, monkeypatch):
        # Simulate the dragent_fastapi server installing its own SDK provider
        # before NAT plugin discovery runs. Bootstrap should keep that provider
        # in place (we don't fight the FastAPI layer) but attach a DR-pointed
        # BatchSpanProcessor so framework spans still reach DR.
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        sentinel_resource = Resource.create(
            {"service.name": "preexisting-service", "test.sentinel": "yes"}
        )
        pre_existing = TracerProvider(resource=sentinel_resource)
        trace.set_tracer_provider(pre_existing)

        # Spy the exporter so we can assert on endpoint + DR auth headers.
        from opentelemetry.exporter.otlp.proto.http import trace_exporter as exporter_module

        real_exporter_cls = exporter_module.OTLPSpanExporter
        captured: dict = {}

        def spy_exporter(**kwargs):
            captured.update(kwargs)
            return real_exporter_cls(**kwargs)

        monkeypatch.setattr(exporter_module, "OTLPSpanExporter", spy_exporter)

        self._set_full_env(clean_env)
        assert datarobot_otel.bootstrap_otel_provider_for_datarobot() is True

        # SDK provider was kept (not replaced); get_tracer is patched for NAT joins.
        assert trace.get_tracer_provider() is pre_existing
        assert getattr(pre_existing, _NAT_TRACER_WRAPPED_ATTR, False)
        assert pre_existing.resource.attributes["test.sentinel"] == "yes"
        assert pre_existing.resource.attributes["service.name"] == "preexisting-service"

        # The DR exporter was constructed with the right endpoint + headers.
        assert captured["endpoint"] == "https://example.test/otel/v1/traces"
        assert captured["headers"]["X-DataRobot-Api-Key"] == "tok"
        assert captured["headers"]["X-DataRobot-Entity-Id"] == "deployment-abc123"

        # And the new BatchSpanProcessor is registered on the existing provider.
        # The SDK keeps span processors in a multiprocessor at
        # ``_active_span_processor._span_processors`` (tuple of processors).
        active = pre_existing._active_span_processor
        registered = getattr(active, "_span_processors", (active,))
        assert any(isinstance(p, BatchSpanProcessor) for p in registered)

    def test_skips_when_non_sdk_provider_installed(self, clean_env):
        # Some third-party non-SDK TracerProvider implementation. We can't
        # attach a span processor to an unknown provider type, so the
        # bootstrap should bail without modifying it.
        class _ThirdPartyProvider:
            def get_tracer(self, *args, **kwargs):  # pragma: no cover - not called
                raise AssertionError("should not be invoked in this test")

        third_party = _ThirdPartyProvider()
        # bypass set_tracer_provider's Once() guard by writing the private
        # global directly (the clean_env fixture already reset Once()).
        trace._TRACER_PROVIDER = third_party

        self._set_full_env(clean_env)
        assert datarobot_otel.bootstrap_otel_provider_for_datarobot() is False
        assert trace.get_tracer_provider() is third_party


class TestSetupUseCaseTracing:
    @staticmethod
    def _set_local_env(monkeypatch):
        # Local shape: credentials and endpoint present, no platform-injected ids.
        monkeypatch.setenv("DATAROBOT_API_TOKEN", "tok")
        monkeypatch.setenv("DATAROBOT_ENDPOINT", "https://example.test/api/v2")

    def test_builds_use_case_headers_from_argument(self, clean_env):
        self._set_local_env(clean_env)
        assert datarobot_otel.setup_use_case_tracing("uc123") is True
        assert os.environ["OTEL_EXPORTER_OTLP_HEADERS"] == (
            "X-DataRobot-Api-Key=tok,X-DataRobot-Entity-Id=experiment_container-uc123"
        )
        assert isinstance(trace.get_tracer_provider(), TracerProvider)

    def test_resolves_from_use_case_id_env(self, clean_env):
        self._set_local_env(clean_env)
        clean_env.setenv("DATAROBOT_USE_CASE_ID", "uc789")
        assert datarobot_otel.setup_use_case_tracing() is True
        assert "experiment_container-uc789" in os.environ["OTEL_EXPORTER_OTLP_HEADERS"]

    def test_falls_back_to_default_use_case_env_when_use_case_id_unset(self, clean_env):
        self._set_local_env(clean_env)
        clean_env.setenv("DATAROBOT_DEFAULT_USE_CASE", "uc456")
        assert datarobot_otel.setup_use_case_tracing() is True
        assert "experiment_container-uc456" in os.environ["OTEL_EXPORTER_OTLP_HEADERS"]

    def test_use_case_id_env_wins_over_default_use_case_env(self, clean_env):
        self._set_local_env(clean_env)
        clean_env.setenv("DATAROBOT_USE_CASE_ID", "uc-id-wins")
        clean_env.setenv("DATAROBOT_DEFAULT_USE_CASE", "uc-default-loses")
        assert datarobot_otel.setup_use_case_tracing() is True
        assert "experiment_container-uc-id-wins" in os.environ["OTEL_EXPORTER_OTLP_HEADERS"]

    def test_argument_wins_over_env(self, clean_env):
        self._set_local_env(clean_env)
        clean_env.setenv("DATAROBOT_USE_CASE_ID", "from-use-case-id-env")
        clean_env.setenv("DATAROBOT_DEFAULT_USE_CASE", "from-default-env")
        assert datarobot_otel.setup_use_case_tracing("from-arg") is True
        assert "experiment_container-from-arg" in os.environ["OTEL_EXPORTER_OTLP_HEADERS"]

    def test_returns_false_without_use_case_id(self, clean_env):
        self._set_local_env(clean_env)
        assert datarobot_otel.setup_use_case_tracing() is False
        assert "OTEL_EXPORTER_OTLP_HEADERS" not in os.environ
        assert isinstance(trace.get_tracer_provider(), ProxyTracerProvider)

    def test_returns_false_without_api_token(self, clean_env):
        clean_env.setenv("DATAROBOT_ENDPOINT", "https://example.test/api/v2")
        assert datarobot_otel.setup_use_case_tracing("uc123") is False
        assert "OTEL_EXPORTER_OTLP_HEADERS" not in os.environ

    def test_returns_false_in_hosted_runtime(self, clean_env):
        # A deployment already has its entity id; never re-point those traces.
        self._set_local_env(clean_env)
        clean_env.setenv("MLOPS_DEPLOYMENT_ID", "abc123")
        assert datarobot_otel.setup_use_case_tracing("uc123") is False
        assert "OTEL_EXPORTER_OTLP_HEADERS" not in os.environ

    def test_returns_false_in_workload_runtime(self, clean_env):
        self._set_local_env(clean_env)
        clean_env.setenv("WORKLOAD_ID", "wl123")
        assert datarobot_otel.setup_use_case_tracing("uc123") is False
        assert "OTEL_EXPORTER_OTLP_HEADERS" not in os.environ

    def test_explicit_argument_overrides_conflicting_preset_headers(self, clean_env):
        # An explicit use_case_id beats a preset OTEL_EXPORTER_OTLP_HEADERS that
        # names a different entity: the caller asked for this use case by name.
        self._set_local_env(clean_env)
        preset = "X-DataRobot-Api-Key=other,X-DataRobot-Entity-Id=deployment-preset"
        clean_env.setenv("OTEL_EXPORTER_OTLP_HEADERS", preset)
        assert datarobot_otel.setup_use_case_tracing("uc123") is True
        assert os.environ["OTEL_EXPORTER_OTLP_HEADERS"] == (
            "X-DataRobot-Api-Key=tok,X-DataRobot-Entity-Id=experiment_container-uc123"
        )

    def test_env_resolved_id_does_not_override_conflicting_preset_headers(self, clean_env):
        # An id resolved from the environment (not passed explicitly) must not
        # silently repoint a preset header that names a different entity.
        self._set_local_env(clean_env)
        preset = "X-DataRobot-Api-Key=other,X-DataRobot-Entity-Id=deployment-preset"
        clean_env.setenv("OTEL_EXPORTER_OTLP_HEADERS", preset)
        clean_env.setenv("DATAROBOT_USE_CASE_ID", "uc123")
        assert datarobot_otel.setup_use_case_tracing() is False
        assert os.environ["OTEL_EXPORTER_OTLP_HEADERS"] == preset

    def test_retarget_after_install_is_refused(self, clean_env):
        # The exporter copies its headers when built, so a later call cannot retarget it;
        # reporting success would send the caller to a use case receiving no spans.
        self._set_local_env(clean_env)
        assert datarobot_otel.setup_use_case_tracing("uc-first") is True
        assert datarobot_otel.setup_use_case_tracing("uc-second") is False
        assert "experiment_container-uc-first" in os.environ["OTEL_EXPORTER_OTLP_HEADERS"]

    def test_lowercase_preset_header_naming_same_entity(self, clean_env):
        # dragent/cli/commands.py:69 writes lowercase header names. A preset
        # already naming this use case must not raise and must report success
        # (previously raised KeyError when bootstrap logged the entity id).
        self._set_local_env(clean_env)
        preset = "x-datarobot-api-key=tok,x-datarobot-entity-id=experiment_container-uc123"
        clean_env.setenv("OTEL_EXPORTER_OTLP_HEADERS", preset)
        assert datarobot_otel.setup_use_case_tracing("uc123") is True

    def test_second_call_in_same_process_returns_true(self, clean_env):
        # bootstrap short-circuits on its idempotency flag on the second call;
        # that must still read as "tracing is on", not as failure.
        self._set_local_env(clean_env)
        assert datarobot_otel.setup_use_case_tracing("uc123") is True
        assert datarobot_otel.setup_use_case_tracing("uc123") is True

    def test_returns_false_without_resolvable_endpoint(self, clean_env):
        # Endpoint is checked before writing OTEL_EXPORTER_OTLP_HEADERS, so a
        # missing endpoint cannot strand DATAROBOT_API_TOKEN in the environment.
        clean_env.setenv("DATAROBOT_API_TOKEN", "tok")
        assert datarobot_otel.setup_use_case_tracing("uc123") is False
        assert "OTEL_EXPORTER_OTLP_HEADERS" not in os.environ
