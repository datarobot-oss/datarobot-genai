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


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Strip the env this reads, and reset the global provider + bootstrap flag.

    Autouse: the function under test creates DataRobot resources, so a test that
    forgot to ask for this would reach whatever account the shell points at.
    """
    for var in _ENV_VARS:
        # setenv first: delenv alone records no undo for an already-absent var, and
        # trace_to_use_case writes DATAROBOT_USE_CASE_ID, which would then outlive
        # the test and leak into the rest of the session.
        monkeypatch.setenv(var, "")
        monkeypatch.delenv(var)
    monkeypatch.setattr("opentelemetry.trace._TRACER_PROVIDER", None)
    monkeypatch.setattr("opentelemetry.trace._TRACER_PROVIDER_SET_ONCE", Once())
    monkeypatch.setitem(datarobot_otel._BOOTSTRAP_STATE, "installed", False)
    monkeypatch.setitem(datarobot_otel._BOOTSTRAP_ENTITY, "id", "")
    return monkeypatch


def _datarobot_env(monkeypatch) -> None:
    monkeypatch.setenv("DATAROBOT_API_TOKEN", "tok")
    monkeypatch.setenv("DATAROBOT_ENDPOINT", "https://example.test/api/v2")


class TestTraceToUseCase:
    def test_exports_to_the_given_use_case(self, clean_env):
        # GIVEN a use case id and DataRobot credentials, THEN spans are attributed to
        # that use case and its id comes back for the caller to display.
        _datarobot_env(clean_env)

        assert trace_to_use_case("Quickstart", use_case_id="uc123") == "uc123"
        assert datarobot_otel.datarobot_otel_entity_id() == "experiment_container-uc123"

    def test_honours_a_use_case_id_already_in_the_environment(self, clean_env):
        # GIVEN the caller exported the use case they want, THEN it is used as-is: a
        # lookup would create a second use case and quietly export somewhere else.
        _datarobot_env(clean_env)
        clean_env.setenv("DATAROBOT_USE_CASE_ID", "uc-from-env")

        with patch("datarobot.UseCase") as use_case_api:
            assert trace_to_use_case("Quickstart") == "uc-from-env"

        use_case_api.list.assert_not_called()
        use_case_api.create.assert_not_called()

    def test_repeating_the_call_returns_where_spans_actually_go(self, clean_env):
        # The exporter keeps its entity for the life of the process, so a second call
        # asking for another use case must report the first, not the one it asked for.
        _datarobot_env(clean_env)

        assert trace_to_use_case("Quickstart", use_case_id="uc-first") == "uc-first"
        assert trace_to_use_case("Quickstart", use_case_id="uc-second") == "uc-first"

    def test_looks_the_use_case_up_by_name_when_none_is_given(self, clean_env):
        # GIVEN no use case id from the caller, THEN one is looked up by exact name.
        _datarobot_env(clean_env)
        match = type("UseCase", (), {"id": "uc-found", "name": "Quickstart"})()
        other = type("UseCase", (), {"id": "uc-other", "name": "Quickstart extended"})()

        with patch("datarobot.UseCase") as use_case_api:
            use_case_api.list.return_value = [other, match]
            assert trace_to_use_case("Quickstart") == "uc-found"

        use_case_api.create.assert_not_called()

    def test_creates_the_use_case_when_the_name_is_not_found(self, clean_env):
        _datarobot_env(clean_env)

        with patch("datarobot.UseCase") as use_case_api:
            use_case_api.list.return_value = []
            use_case_api.create.return_value = type("UseCase", (), {"id": "uc-new"})()
            assert trace_to_use_case("Quickstart") == "uc-new"

        use_case_api.create.assert_called_once_with(name="Quickstart")

    @pytest.mark.parametrize(
        ("var", "value"),
        [
            ("OTEL_EXPORTER_OTLP_HEADERS", "X-DataRobot-Entity-Id=custom_application-app9"),
            ("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"),
            ("OTEL_SDK_DISABLED", "true"),
        ],
    )
    def test_creates_nothing_when_the_environment_already_decided(self, clean_env, var, value):
        # GIVEN either OTLP variable or the kill switch, THEN nothing is configured or
        # created: the runtime already named the entity, exporting would post this
        # account's API key to a host DataRobot never named, or the reader opted out.
        _datarobot_env(clean_env)
        clean_env.setenv(var, value)

        with patch("datarobot.UseCase") as use_case_api:
            assert trace_to_use_case("Quickstart") == ""

        use_case_api.list.assert_not_called()
        use_case_api.create.assert_not_called()

    def test_reports_no_use_case_inside_a_deployment(self, clean_env):
        # GIVEN a hosted runtime, THEN the platform's entity wins, nothing is created to
        # litter the account, and no use case id is offered for spans it never receives.
        _datarobot_env(clean_env)
        clean_env.setenv("MLOPS_DEPLOYMENT_ID", "dep7")

        with patch("datarobot.UseCase") as use_case_api:
            assert trace_to_use_case("Quickstart", use_case_id="uc123") == ""

        use_case_api.list.assert_not_called()
        use_case_api.create.assert_not_called()
        assert datarobot_otel.datarobot_otel_entity_id() == "deployment-dep7"

    def test_reports_missing_credentials_instead_of_exporting(self, clean_env):
        # GIVEN no endpoint, THEN nothing is created: a use case it could never send
        # spans to is just account litter.
        with patch("datarobot.UseCase") as use_case_api:
            assert trace_to_use_case("Quickstart") == ""

        use_case_api.list.assert_not_called()
        use_case_api.create.assert_not_called()

    def test_a_failed_use_case_lookup_does_not_raise(self, clean_env):
        # The caller is a notebook cell; a DataRobot API failure must not end the run.
        _datarobot_env(clean_env)

        with patch("datarobot.UseCase") as use_case_api:
            use_case_api.list.side_effect = RuntimeError("403 Forbidden")
            assert trace_to_use_case("Quickstart") == ""

        assert isinstance(trace.get_tracer_provider(), trace.ProxyTracerProvider)
