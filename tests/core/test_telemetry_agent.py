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
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.util._once import Once

from datarobot_genai.core.telemetry import datarobot_otel
from datarobot_genai.core.telemetry.agent import instrument

_EXPORT_ENV_VARS = (
    "MLOPS_DEPLOYMENT_ID",
    "WORKLOAD_ID",
    "DATAROBOT_API_TOKEN",
    "DATAROBOT_ENDPOINT",
    "DATAROBOT_PUBLIC_API_ENDPOINT",
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


def test_instrument_idempotent() -> None:
    instrument()
    instrument()  # idempotent


def test_instrument_bootstraps_without_hosted_runtime_env(monkeypatch) -> None:
    # GIVEN no MLOPS_DEPLOYMENT_ID / WORKLOAD_ID, WHEN instrument() runs, THEN it
    # still asks the bootstrap: a local run's framework and LLM spans need the
    # exporter just as much as a deployment's. Whether one is installed is the
    # bootstrap's own call, made from the endpoint and headers it resolves.
    with patch(
        "datarobot_genai.core.telemetry.datarobot_otel.bootstrap_otel_provider_for_datarobot"
    ) as mock:
        instrument()
    mock.assert_called_once()


def test_instrument_installs_exporter_from_local_otlp_env(monkeypatch) -> None:
    # GIVEN only the OTLP variables a local run sets, WHEN instrument() runs,
    # THEN a real SDK provider is installed. This is the behaviour the gate
    # removal exists for, so it is asserted end to end rather than mocked.
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "https://example.test/otel")
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_HEADERS",
        "X-DataRobot-Api-Key=tok,X-DataRobot-Entity-Id=experiment_container-uc123",
    )

    instrument()

    assert isinstance(trace.get_tracer_provider(), TracerProvider)
    assert datarobot_otel.datarobot_otel_provider_installed() is True


def test_instrument_installs_nothing_without_export_env() -> None:
    # GIVEN nothing configuring OTel export, THEN instrument() leaves the global
    # provider alone rather than installing an exporter nobody asked for.
    instrument()

    assert not isinstance(trace.get_tracer_provider(), TracerProvider)
    assert datarobot_otel.datarobot_otel_provider_installed() is False
