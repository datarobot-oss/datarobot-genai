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

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import ProxyTracerProvider
from opentelemetry.util._once import Once

from datarobot_genai.core.telemetry import datarobot_otel
from datarobot_genai.core.telemetry.nat_tracer import _NAT_TRACER_WRAPPED_ATTR

_ENV_VARS = (
    "DATAROBOT_API_TOKEN",
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
    monkeypatch.setitem(datarobot_otel._BOOTSTRAP_ENTITY, "id", "")
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

    def test_entity_id_ignores_a_use_case_in_the_environment(self, clean_env):
        # Only the platform names an entity through the environment. A use case reaches
        # the exporter by being passed in, so no variable can switch export on.
        clean_env.setenv("DATAROBOT_USE_CASE_ID", "uc123")
        assert datarobot_otel.resolve_entity_id_from_env() == ""

    def test_entity_id_hosted_runtime_wins_over_use_case(self, clean_env):
        # A use case id left in a deployment's environment must not redirect that
        # deployment's traces.
        clean_env.setenv("DATAROBOT_USE_CASE_ID", "uc123")
        clean_env.setenv("WORKLOAD_ID", "wkl42")
        assert datarobot_otel.resolve_entity_id_from_env() == "workload-wkl42"
        clean_env.setenv("MLOPS_DEPLOYMENT_ID", "dep1")
        assert datarobot_otel.resolve_entity_id_from_env() == "deployment-dep1"

    def test_entity_id_ignores_blank_use_case(self, clean_env):
        clean_env.setenv("DATAROBOT_USE_CASE_ID", "  ")
        assert datarobot_otel.resolve_entity_id_from_env() == ""

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

    def test_headers_for_a_caller_supplied_entity(self, clean_env):
        # How a local run identifies itself: as an argument, not an env var.
        clean_env.setenv("DATAROBOT_API_TOKEN", "tok")
        headers = datarobot_otel.resolve_datarobot_headers_from_env("experiment_container-uc123")
        assert headers == {
            "X-DataRobot-Api-Key": "tok",
            "X-DataRobot-Entity-Id": "experiment_container-uc123",
        }

    def test_the_environment_wins_over_a_caller_supplied_entity(self, clean_env):
        # A deployment id must outrank a use case a caller asks for, or a helper called
        # from code that also runs deployed would redirect that deployment's traces.
        clean_env.setenv("DATAROBOT_API_TOKEN", "tok")
        clean_env.setenv("MLOPS_DEPLOYMENT_ID", "abc123")
        headers = datarobot_otel.resolve_datarobot_headers_from_env("experiment_container-uc123")
        assert headers["X-DataRobot-Entity-Id"] == "deployment-abc123"

    def test_no_api_key_for_a_use_case_pointed_at_another_collector(self, clean_env):
        # A local run naming a use case, plus an endpoint of the caller's own: deriving
        # headers would post this account's API key to a host DataRobot never named.
        clean_env.setenv("DATAROBOT_API_TOKEN", "tok")
        clean_env.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
        assert (
            datarobot_otel.resolve_datarobot_headers_from_env("experiment_container-uc123") is None
        )

    def test_a_deployment_still_reaches_its_own_collector(self, clean_env):
        # The guard above must not touch a hosted runtime: a deployment pointed at an
        # operator-chosen collector kept its derived headers before this existed.
        clean_env.setenv("DATAROBOT_API_TOKEN", "tok")
        clean_env.setenv("MLOPS_DEPLOYMENT_ID", "abc123")
        clean_env.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector.test:4318")
        assert datarobot_otel.resolve_datarobot_headers_from_env() == {
            "X-DataRobot-Api-Key": "tok",
            "X-DataRobot-Entity-Id": "deployment-abc123",
        }

    def test_headers_parsed_from_otlp_env(self, clean_env):
        clean_env.setenv(
            "OTEL_EXPORTER_OTLP_HEADERS",
            "X-DataRobot-Api-Key=env-key,X-DataRobot-Entity-Id=deployment-env",
        )
        assert datarobot_otel.resolve_datarobot_headers_from_env() == {
            "X-DataRobot-Api-Key": "env-key",
            "X-DataRobot-Entity-Id": "deployment-env",
        }

    def test_headers_skips_malformed_entries(self, clean_env):
        # A trailing comma is the common way to malform this variable, and the
        # parse runs at instrument() time, outside the bootstrap's try/except —
        # so a bad entry must be dropped rather than raise into agent startup.
        clean_env.setenv(
            "OTEL_EXPORTER_OTLP_HEADERS",
            "X-DataRobot-Api-Key=tok,X-DataRobot-Entity-Id=experiment_container-uc123,",
        )
        assert datarobot_otel.resolve_datarobot_headers_from_env() == {
            "X-DataRobot-Api-Key": "tok",
            "X-DataRobot-Entity-Id": "experiment_container-uc123",
        }

    def test_headers_empty_when_every_entry_is_malformed(self, clean_env):
        # Falsy is what matters: the bootstrap skips rather than installing an
        # exporter with no auth headers.
        clean_env.setenv("OTEL_EXPORTER_OTLP_HEADERS", "nonsense")
        assert not datarobot_otel.resolve_datarobot_headers_from_env()

    def test_entity_id_read_back_from_lowercase_headers(self, clean_env):
        # The dragent CLI and custom applications write these names lowercase, and
        # callers read the entity back to report where spans are going.
        clean_env.setenv(
            "OTEL_EXPORTER_OTLP_HEADERS",
            "x-datarobot-api-key=k,x-datarobot-entity-id=experiment_container-uc1",
        )
        headers = datarobot_otel.resolve_datarobot_headers_from_env() or {}
        assert datarobot_otel.resolve_entity_id_from_headers(headers) == "experiment_container-uc1"

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

    def test_lowercase_preset_header_names(self, clean_env):
        # dragent/cli/commands.py writes lowercase header names, and a local caller
        # bootstraps with whatever OTEL_EXPORTER_OTLP_HEADERS holds. Logging the entity
        # id must tolerate that casing rather than raise KeyError after installing.
        clean_env.setenv("DATAROBOT_ENDPOINT", "https://example.test/api/v2")
        clean_env.setenv(
            "OTEL_EXPORTER_OTLP_HEADERS",
            "x-datarobot-api-key=tok,x-datarobot-entity-id=experiment_container-uc123",
        )
        assert datarobot_otel.bootstrap_otel_provider_for_datarobot() is True
        assert isinstance(trace.get_tracer_provider(), TracerProvider)

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

    @pytest.mark.parametrize(
        ("entity_env", "entity_arg", "expected_entity_id"),
        [
            ({"MLOPS_DEPLOYMENT_ID": "abc123"}, "", "deployment-abc123"),
            ({}, "experiment_container-uc123", "experiment_container-uc123"),
        ],
    )
    def test_exporter_endpoint_and_headers(
        self, clean_env, monkeypatch, entity_env, entity_arg, expected_entity_id
    ):
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

        clean_env.setenv("DATAROBOT_API_TOKEN", "tok")
        clean_env.setenv("DATAROBOT_ENDPOINT", "https://example.test/api/v2")
        for var, value in entity_env.items():
            clean_env.setenv(var, value)
        datarobot_otel.bootstrap_otel_provider_for_datarobot(entity_arg)

        # The entity id is what routes spans to an entity, so assert it where it is
        # actually handed over: a local run's use case is as load-bearing as a
        # deployment's id, and neither is checked anywhere else.
        assert captured["endpoint"] == "https://example.test/otel/v1/traces"
        assert captured["headers"]["X-DataRobot-Api-Key"] == "tok"
        assert captured["headers"]["X-DataRobot-Entity-Id"] == expected_entity_id

    def test_idempotent_second_call(self, clean_env):
        self._set_full_env(clean_env)
        assert datarobot_otel.bootstrap_otel_provider_for_datarobot() is True
        provider_first = trace.get_tracer_provider()

        # Second call should detect the module flag and no-op.
        assert datarobot_otel.bootstrap_otel_provider_for_datarobot() is False
        assert trace.get_tracer_provider() is provider_first

    def test_provider_installed_reports_process_state(self, clean_env):
        # Unlike the bootstrap's return value, which reports what one call did,
        # this stays True for the process, which is what a caller needs to know before
        # relying on its spans reaching DataRobot.
        assert datarobot_otel.datarobot_otel_provider_installed() is False

        self._set_full_env(clean_env)
        datarobot_otel.bootstrap_otel_provider_for_datarobot()
        assert datarobot_otel.datarobot_otel_provider_installed() is True

        assert datarobot_otel.bootstrap_otel_provider_for_datarobot() is False
        assert datarobot_otel.datarobot_otel_provider_installed() is True

    def test_attaches_processor_to_existing_sdk_provider(self, clean_env, monkeypatch):
        # Simulate a host that set up its own SDK provider before handing control
        # to agent code. Bootstrap should keep that provider in place (we don't
        # fight the host) but attach a DR-pointed BatchSpanProcessor so framework
        # spans still reach DR.
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
