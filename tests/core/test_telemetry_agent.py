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


def test_instrument_idempotent() -> None:
    instrument()
    instrument()  # idempotent


@pytest.mark.parametrize(
    "export_env",
    [
        pytest.param(
            {
                "OTEL_EXPORTER_OTLP_ENDPOINT": "https://example.test/otel",
                "OTEL_EXPORTER_OTLP_HEADERS": (
                    "X-DataRobot-Api-Key=tok,X-DataRobot-Entity-Id=experiment_container-uc123"
                ),
            },
            id="local-otlp-env",
        ),
        pytest.param({"MLOPS_DEPLOYMENT_ID": "abc123"}, id="deployment"),
        pytest.param({"WORKLOAD_ID": "wkl42"}, id="workload"),
    ],
)
def test_instrument_bootstraps_in_any_runtime(monkeypatch, export_env) -> None:
    # The bootstrap is not gated on the hosted-runtime env: a local run's
    # framework and LLM spans need the exporter just as much as a deployment's.
    # Whether one is installed is the bootstrap's own call, made from the
    # endpoint and headers the environment resolves to.
    monkeypatch.setenv("DATAROBOT_API_TOKEN", "tok")
    monkeypatch.setenv("DATAROBOT_ENDPOINT", "https://example.test/api/v2")
    for name, value in export_env.items():
        monkeypatch.setenv(name, value)
    with patch(
        "datarobot_genai.core.telemetry.datarobot_otel.bootstrap_otel_provider_for_datarobot"
    ) as mock:
        instrument()
    mock.assert_called_once()
