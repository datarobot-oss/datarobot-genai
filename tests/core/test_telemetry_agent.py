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

from datarobot_genai.core.telemetry.agent import instrument


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


def test_instrument_bootstraps_when_otel_headers_set(monkeypatch) -> None:
    # Local experimentation: not a hosted runtime, but _bridge_pulumi_otel_env
    # has set OTEL_EXPORTER_OTLP_HEADERS, so the SDK provider must still install.
    monkeypatch.delenv("MLOPS_DEPLOYMENT_ID", raising=False)
    monkeypatch.delenv("WORKLOAD_ID", raising=False)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "X-DataRobot-Entity-Id=experiment_container-x")
    with patch(
        "datarobot_genai.core.telemetry.datarobot_otel.bootstrap_otel_provider_for_datarobot"
    ) as mock:
        instrument()
    mock.assert_called_once()
