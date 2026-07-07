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

The SDK constructors are patched on the module under test (it binds them via
``from ... import`` at import time) so these never mutate the process-global
MeterProvider (which can only be set once per process).
"""

from unittest.mock import MagicMock

import pytest

from datarobot_genai.drmcpbase import datarobot_otel_metrics as m


@pytest.fixture(autouse=True)
def _reset() -> None:
    m._reset_for_testing()
    yield
    m._reset_for_testing()


@pytest.fixture
def sdk_stubs(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Stub the SDK pieces bootstrap uses; capture what it passes to them."""
    seen: dict = {"set_provider_calls": 0, "exporter_kwargs": None, "resource_attrs": None}

    def _set_provider(_p: object) -> None:
        seen["set_provider_calls"] += 1

    def _exporter(**kwargs: object) -> MagicMock:
        seen["exporter_kwargs"] = kwargs
        return MagicMock()

    monkeypatch.setattr(m.metrics, "set_meter_provider", _set_provider)
    monkeypatch.setattr(m, "OTLPMetricExporter", _exporter)
    monkeypatch.setattr(m, "PeriodicExportingMetricReader", lambda *_a, **_kw: MagicMock())
    monkeypatch.setattr(m, "MeterProvider", lambda **_kw: MagicMock())
    monkeypatch.setattr(
        m.Resource,
        "create",
        classmethod(lambda _cls, attrs: seen.__setitem__("resource_attrs", dict(attrs))),
    )
    return seen


def test_no_op_without_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", raising=False)
    assert m.bootstrap_metrics_provider() is False


def test_installs_once_then_idempotent(sdk_stubs: dict) -> None:
    first = m.bootstrap_metrics_provider(endpoint="http://localhost:4318/v1/metrics")
    second = m.bootstrap_metrics_provider(endpoint="http://localhost:4318/v1/metrics")

    assert first is True
    assert second is False
    assert sdk_stubs["set_provider_calls"] == 1


def test_endpoint_from_env(monkeypatch: pytest.MonkeyPatch, sdk_stubs: dict) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", "http://collector:4318/v1/metrics")

    assert m.bootstrap_metrics_provider() is True
    assert sdk_stubs["exporter_kwargs"] == {"endpoint": "http://collector:4318/v1/metrics"}


def test_resource_attributes_merged_over_service_name(sdk_stubs: dict) -> None:
    assert (
        m.bootstrap_metrics_provider(
            endpoint="http://collector:4318/v1/metrics",
            resource_attributes={"datarobot.service.name": "test-mcp"},
        )
        is True
    )
    assert sdk_stubs["resource_attrs"]["datarobot.service.name"] == "test-mcp"
    assert "service.name" in sdk_stubs["resource_attrs"]
