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

from unittest.mock import patch

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

from datarobot_genai.core import telemetry_bootstrap


@pytest.fixture(autouse=True)
def _reset_state():
    telemetry_bootstrap._reset_for_tests()
    yield
    telemetry_bootstrap._reset_for_tests()


def test_disabled_via_otel_enabled_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_HEADERS", raising=False)
    monkeypatch.setenv("OTEL_ENABLED", "false")

    assert telemetry_bootstrap.initialize_tracer_provider(service_name="svc") is False


def test_no_entity_id_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_HEADERS", raising=False)
    monkeypatch.delenv("OTEL_ENTITY_ID", raising=False)
    monkeypatch.delenv("MLOPS_RUNTIME_PARAM_OTEL_ENTITY_ID", raising=False)
    monkeypatch.setenv("OTEL_ENABLED", "true")

    assert telemetry_bootstrap.initialize_tracer_provider(service_name="svc") is False


def test_installs_provider_when_entity_id_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_HEADERS", raising=False)
    monkeypatch.setenv("OTEL_ENABLED", "true")
    monkeypatch.setenv("OTEL_ENTITY_ID", "test-entity")
    monkeypatch.setenv("OTEL_COLLECTOR_BASE_URL", "http://collector.test/otel")
    monkeypatch.setenv("DATAROBOT_API_TOKEN", "test-token")

    # Avoid hitting the real OTLP exporter constructor in CI: the exporter is
    # only built lazily inside initialize_tracer_provider, but constructing it
    # parses env URLs and is otherwise fine in-process. We still patch it to
    # keep the test isolated from network/registry imports.
    with patch(
        "opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter"
    ) as exporter_cls:
        assert telemetry_bootstrap.initialize_tracer_provider(service_name="svc") is True

    exporter_cls.assert_called_once()
    provider = trace.get_tracer_provider()
    assert isinstance(provider, TracerProvider)


def test_idempotent_second_call_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_HEADERS", raising=False)
    monkeypatch.setenv("OTEL_ENABLED", "true")
    monkeypatch.setenv("OTEL_ENTITY_ID", "test-entity")
    monkeypatch.setenv("DATAROBOT_API_TOKEN", "test-token")

    with patch(
        "opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter"
    ) as exporter_cls:
        assert telemetry_bootstrap.initialize_tracer_provider(service_name="svc") is True
        # Second call must be a no-op (no new exporter constructed).
        assert telemetry_bootstrap.initialize_tracer_provider(service_name="svc") is True

    assert exporter_cls.call_count == 1


def test_respects_preset_otlp_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_ENABLED", "true")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://preset.test/otel")
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_HEADERS", raising=False)
    monkeypatch.delenv("OTEL_ENTITY_ID", raising=False)
    monkeypatch.delenv("MLOPS_RUNTIME_PARAM_OTEL_ENTITY_ID", raising=False)

    with patch("opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter"):
        # When OTEL_EXPORTER_OTLP_ENDPOINT is already set externally, the
        # bootstrap should proceed even without an entity id and not overwrite
        # the preset value.
        assert telemetry_bootstrap.initialize_tracer_provider(service_name="svc") is True

    import os

    assert os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://preset.test/otel"


def test_mlops_runtime_param_prefix_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_HEADERS", raising=False)
    monkeypatch.setenv("OTEL_ENABLED", "true")
    monkeypatch.setenv("MLOPS_RUNTIME_PARAM_OTEL_ENTITY_ID", "runtime-entity")
    monkeypatch.setenv("OTEL_ENTITY_ID", "plain-entity")
    monkeypatch.setenv("DATAROBOT_API_TOKEN", "test-token")

    with patch("opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter"):
        assert telemetry_bootstrap.initialize_tracer_provider(service_name="svc") is True

    import os

    headers = os.environ["OTEL_EXPORTER_OTLP_HEADERS"]
    assert "X-DataRobot-Entity-Id=runtime-entity" in headers


def test_unwraps_datarobot_runtime_param_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deployed agents see runtime params as JSON envelopes; the bootstrap must unwrap."""
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_HEADERS", raising=False)
    monkeypatch.setenv("OTEL_ENABLED", "true")
    monkeypatch.setenv(
        "MLOPS_RUNTIME_PARAM_OTEL_ENTITY_ID",
        '{"type":"string","payload":"deployed-entity"}',
    )
    monkeypatch.setenv(
        "MLOPS_RUNTIME_PARAM_DATAROBOT_API_TOKEN",
        '{"type":"string","payload":"deployed-token"}',
    )

    with patch("opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter"):
        assert telemetry_bootstrap.initialize_tracer_provider(service_name="svc") is True

    import os

    headers = os.environ["OTEL_EXPORTER_OTLP_HEADERS"]
    assert "X-DataRobot-Entity-Id=deployed-entity" in headers
    assert "X-DataRobot-Api-Key=deployed-token" in headers


def test_setup_dragent_tracing_calls_instrument(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_ENABLED", "false")  # short-circuit provider install

    with patch("datarobot_genai.core.telemetry_agent.instrument") as instrument_call:
        telemetry_bootstrap.setup_dragent_tracing(service_name="svc")

    instrument_call.assert_called_once_with(framework="nat")


def test_setup_dragent_tracing_swallows_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_ENABLED", "false")

    with patch(
        "datarobot_genai.core.telemetry_bootstrap.initialize_tracer_provider",
        side_effect=RuntimeError("boom"),
    ):
        # Must not raise — agent execution should never be blocked by tracing.
        telemetry_bootstrap.setup_dragent_tracing(service_name="svc")
