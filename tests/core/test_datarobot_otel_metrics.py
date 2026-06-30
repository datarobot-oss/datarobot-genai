# Copyright 2026 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the OTLP metrics provider bootstrap (guard + idempotency).

The OpenTelemetry SDK is mocked so these never mutate the process-global
MeterProvider (which can only be set once per process).
"""

from unittest.mock import MagicMock

import pytest

from datarobot_genai.core.telemetry import datarobot_otel_metrics as m


@pytest.fixture(autouse=True)
def _reset() -> None:
    m._reset_for_testing()
    yield
    m._reset_for_testing()


def test_no_op_without_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", raising=False)
    assert m.bootstrap_metrics_provider() is False


def test_installs_once_then_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    import opentelemetry.exporter.otlp.proto.http.metric_exporter as http_exp
    import opentelemetry.metrics as otel_metrics
    import opentelemetry.sdk.metrics as sdk_metrics
    import opentelemetry.sdk.metrics.export as sdk_export

    calls = {"set": 0}
    monkeypatch.setattr(
        otel_metrics, "set_meter_provider", lambda _p: calls.__setitem__("set", calls["set"] + 1)
    )
    monkeypatch.setattr(http_exp, "OTLPMetricExporter", lambda **_kw: MagicMock())
    monkeypatch.setattr(sdk_export, "PeriodicExportingMetricReader", lambda *_a, **_kw: MagicMock())
    monkeypatch.setattr(sdk_metrics, "MeterProvider", lambda **_kw: MagicMock())

    first = m.bootstrap_metrics_provider(endpoint="http://localhost:4318/v1/metrics")
    second = m.bootstrap_metrics_provider(endpoint="http://localhost:4318/v1/metrics")

    assert first is True
    assert second is False
    assert calls["set"] == 1


def test_endpoint_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    import opentelemetry.exporter.otlp.proto.http.metric_exporter as http_exp
    import opentelemetry.metrics as otel_metrics
    import opentelemetry.sdk.metrics as sdk_metrics
    import opentelemetry.sdk.metrics.export as sdk_export

    monkeypatch.setenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", "http://collector:4318/v1/metrics")
    seen = {}
    monkeypatch.setattr(otel_metrics, "set_meter_provider", lambda _p: None)
    monkeypatch.setattr(http_exp, "OTLPMetricExporter", lambda **kw: seen.update(kw) or MagicMock())
    monkeypatch.setattr(sdk_export, "PeriodicExportingMetricReader", lambda *_a, **_kw: MagicMock())
    monkeypatch.setattr(sdk_metrics, "MeterProvider", lambda **_kw: MagicMock())

    assert m.bootstrap_metrics_provider() is True
    assert seen["endpoint"] == "http://collector:4318/v1/metrics"
