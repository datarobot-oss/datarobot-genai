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

from unittest.mock import patch

import pytest
from opentelemetry import trace
from opentelemetry.util._once import Once

from datarobot_genai.core.telemetry import datarobot_otel
from datarobot_genai.core.telemetry.use_case import trace_to_use_case

_ENV_VARS = (
    "DATAROBOT_API_TOKEN",
    "DATAROBOT_ENDPOINT",
    "DATAROBOT_PUBLIC_API_ENDPOINT",
    "DATAROBOT_USE_CASE_ID",
    "DATAROBOT_DEFAULT_USE_CASE",
    "MLOPS_DEPLOYMENT_ID",
    "WORKLOAD_ID",
    "OTEL_SDK_DISABLED",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_HEADERS",
)


@pytest.fixture
def clean_env(monkeypatch):
    """Strip the env this reads, and reset the global provider + bootstrap flag."""
    for var in _ENV_VARS:
        # setenv first: delenv alone records no undo for an already-absent var, and
        # trace_to_use_case writes DATAROBOT_USE_CASE_ID, which would then outlive
        # the test and leak into the rest of the session.
        monkeypatch.setenv(var, "")
        monkeypatch.delenv(var)
    monkeypatch.setattr("opentelemetry.trace._TRACER_PROVIDER", None)
    monkeypatch.setattr("opentelemetry.trace._TRACER_PROVIDER_SET_ONCE", Once())
    monkeypatch.setitem(datarobot_otel._BOOTSTRAP_STATE, "installed", False)
    return monkeypatch


def _datarobot_env(monkeypatch) -> None:
    monkeypatch.setenv("DATAROBOT_API_TOKEN", "tok")
    monkeypatch.setenv("DATAROBOT_ENDPOINT", "https://example.test/api/v2")


class TestTraceToUseCase:
    def test_exports_to_the_given_use_case(self, clean_env):
        # GIVEN a use case id and DataRobot credentials, WHEN tracing starts, THEN
        # spans are attributed to that use case as an experiment container.
        _datarobot_env(clean_env)

        tracing = trace_to_use_case("Quickstart", use_case_id="uc123")

        assert tracing.exporting
        assert tracing.entity_id == "experiment_container-uc123"
        assert tracing.use_case_id == "uc123"
        assert "dr xp --entity-id uc123" in str(tracing)

    def test_looks_the_use_case_up_by_name_when_none_is_given(self, clean_env):
        # GIVEN no use case id from the caller, THEN one is looked up by exact name.
        _datarobot_env(clean_env)
        match = type("UseCase", (), {"id": "uc-found", "name": "Quickstart"})()
        other = type("UseCase", (), {"id": "uc-other", "name": "Quickstart extended"})()

        with patch("datarobot.UseCase") as use_case_api:
            use_case_api.list.return_value = [other, match]
            tracing = trace_to_use_case("Quickstart")

        use_case_api.create.assert_not_called()
        assert tracing.use_case_id == "uc-found"

    def test_creates_the_use_case_when_the_name_is_not_found(self, clean_env):
        _datarobot_env(clean_env)

        with patch("datarobot.UseCase") as use_case_api:
            use_case_api.list.return_value = []
            use_case_api.create.return_value = type("UseCase", (), {"id": "uc-new"})()
            tracing = trace_to_use_case("Quickstart")

        use_case_api.create.assert_called_once_with(name="Quickstart")
        assert tracing.use_case_id == "uc-new"

    def test_leaves_an_already_configured_environment_alone(self, clean_env):
        # GIVEN OTLP headers naming another entity (a codespace, a custom
        # application), THEN they are reported and never overwritten, and no use
        # case is looked up.
        _datarobot_env(clean_env)
        preset = "X-DataRobot-Api-Key=k,X-DataRobot-Entity-Id=custom_application-app9"
        clean_env.setenv("OTEL_EXPORTER_OTLP_HEADERS", preset)

        with patch("datarobot.UseCase") as use_case_api:
            tracing = trace_to_use_case("Quickstart")

        use_case_api.list.assert_not_called()
        assert tracing.exporting
        assert tracing.entity_id == "custom_application-app9"
        assert tracing.use_case_id == ""
        assert "already set" in str(tracing)

    def test_reports_the_deployment_entity_when_one_outranks_the_use_case(self, clean_env):
        # GIVEN a use case id and a deployment id, THEN spans go to the deployment
        # and the summary must not offer a `dr xp` command for the use case.
        _datarobot_env(clean_env)
        clean_env.setenv("MLOPS_DEPLOYMENT_ID", "dep7")

        tracing = trace_to_use_case("Quickstart", use_case_id="uc123")

        assert tracing.entity_id == "deployment-dep7"
        assert "dr xp" not in str(tracing)
        assert "deployment-dep7" in str(tracing)

    def test_does_nothing_when_the_sdk_is_disabled(self, clean_env):
        # GIVEN the reader opted out, THEN no use case is created and nothing exports.
        _datarobot_env(clean_env)
        clean_env.setenv("OTEL_SDK_DISABLED", "true")

        with patch("datarobot.UseCase") as use_case_api:
            tracing = trace_to_use_case("Quickstart")

        use_case_api.list.assert_not_called()
        assert not tracing.exporting
        assert "OTEL_SDK_DISABLED" in str(tracing)

    def test_reports_missing_credentials_instead_of_exporting(self, clean_env):
        # GIVEN a use case but no endpoint, THEN it reports rather than claiming success.
        clean_env.setenv("DATAROBOT_USE_CASE_ID", "uc123")

        tracing = trace_to_use_case("Quickstart")

        assert not tracing.exporting
        assert "did not resolve" in str(tracing)

    def test_reports_a_failed_use_case_lookup_instead_of_raising(self, clean_env):
        # The caller is a notebook cell; a DataRobot API failure must not end the run.
        _datarobot_env(clean_env)

        with patch("datarobot.UseCase") as use_case_api:
            use_case_api.list.side_effect = RuntimeError("403 Forbidden")
            tracing = trace_to_use_case("Quickstart")

        assert not tracing.exporting
        assert "403 Forbidden" in str(tracing)
        assert isinstance(trace.get_tracer_provider(), trace.ProxyTracerProvider)
