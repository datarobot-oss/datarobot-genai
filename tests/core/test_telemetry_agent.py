# Copyright 2025 DataRobot, Inc. and its affiliates.
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
from opentelemetry.trace import ProxyTracerProvider
from opentelemetry.util._once import Once

from datarobot_genai.core.telemetry import datarobot_otel
from datarobot_genai.core.telemetry.agent import instrument

_EXPORT_ENV_VARS = (
    "MLOPS_DEPLOYMENT_ID",
    "WORKLOAD_ID",
    "DATAROBOT_API_TOKEN",
    "DATAROBOT_ENDPOINT",
    "DATAROBOT_PUBLIC_API_ENDPOINT",
    "DATAROBOT_USE_CASE_ID",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_HEADERS",
)


@pytest.fixture(autouse=True)
def no_ambient_otel_export(monkeypatch):
    """Keep a developer's own DataRobot env from turning these into real exports.

    ``instrument()`` installs the exporter whenever the environment resolves to
    an endpoint and headers, which a working shell often does.
    """
    for var in _EXPORT_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    # OTel guards set_tracer_provider behind Once(); resetting both (and the
    # bootstrap flag) lets a test observe a fresh global slot.
    monkeypatch.setattr("opentelemetry.trace._TRACER_PROVIDER", None)
    monkeypatch.setattr("opentelemetry.trace._TRACER_PROVIDER_SET_ONCE", Once())
    monkeypatch.setitem(datarobot_otel._BOOTSTRAP_STATE, "installed", False)
    monkeypatch.setitem(datarobot_otel._BOOTSTRAP_ENTITY, "id", "")


def test_instrument_idempotent() -> None:
    instrument()
    instrument()  # idempotent


def test_instrument_skips_bootstrap_without_deployment_id(monkeypatch) -> None:
    monkeypatch.delenv("MLOPS_DEPLOYMENT_ID", raising=False)
    monkeypatch.delenv("WORKLOAD_ID", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_HEADERS", raising=False)
    with patch(
        "datarobot_genai.core.telemetry.datarobot_otel.bootstrap_otel_provider_for_datarobot"
    ) as mock:
        instrument()
    mock.assert_not_called()


def test_instrument_bootstraps_when_deployment_id_set(monkeypatch) -> None:
    monkeypatch.setenv("MLOPS_DEPLOYMENT_ID", "abc123")
    with patch(
        "datarobot_genai.core.telemetry.datarobot_otel.bootstrap_otel_provider_for_datarobot"
    ) as mock:
        instrument()
    mock.assert_called_once()


def test_instrument_never_bootstraps_outside_a_hosted_runtime(monkeypatch) -> None:
    # GIVEN a full DataRobot environment and a use case id, but no deployment or workload,
    # THEN instrument() installs nothing: local tracing is opt-in through
    # trace_to_use_case(), so no environment variable switches export on here.
    monkeypatch.setenv("DATAROBOT_USE_CASE_ID", "uc123")
    monkeypatch.setenv("DATAROBOT_API_TOKEN", "tok")
    monkeypatch.setenv("DATAROBOT_ENDPOINT", "https://example.test/api/v2")

    instrument()

    assert isinstance(trace.get_tracer_provider(), ProxyTracerProvider)
    assert datarobot_otel.datarobot_otel_entity_id() == ""
