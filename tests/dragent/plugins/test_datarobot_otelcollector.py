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

"""Tests for the datarobot_otelcollector NAT telemetry exporter."""

from __future__ import annotations

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from datarobot_genai.core.datarobot_otel import resolve_datarobot_headers_from_env
from datarobot_genai.dragent.plugins.datarobot_otelcollector import (
    DataRobotOtelCollectorTelemetryExporter,
)
from datarobot_genai.dragent.plugins.datarobot_otelcollector import (
    datarobot_otelcollector_telemetry_exporter,
)

ENDPOINT = "https://staging.datarobot.com/otel/v1/traces"
ENTITY_ID = "deployment-abc123"
API_KEY = "test-key-do-not-use"

_ENV_VARS = (
    "DATAROBOT_API_TOKEN",
    "MLOPS_DEPLOYMENT_ID",
    "DATAROBOT_ENDPOINT",
    "DATAROBOT_PUBLIC_API_ENDPOINT",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_HEADERS",
)


@pytest.fixture
def clean_env(monkeypatch):
    """Strip all DataRobot/MLOps env vars so default_factory paths see a
    deterministic empty environment. Individual tests opt-in to specific
    values with monkeypatch.setenv afterwards.
    """
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def _make_config(**overrides) -> DataRobotOtelCollectorTelemetryExporter:
    base = {
        "endpoint": ENDPOINT,
        "project": "test-agent",
    }
    base.update(overrides)
    return DataRobotOtelCollectorTelemetryExporter(**base)


def _set_datarobot_env(clean_env, *, api_key=API_KEY, entity_id=ENTITY_ID):
    clean_env.setenv("DATAROBOT_API_TOKEN", api_key)
    clean_env.setenv("MLOPS_DEPLOYMENT_ID", entity_id.removeprefix("deployment-"))


class TestConfigValidation:
    def test_explicit_values_valid(self, clean_env):
        cfg = _make_config()
        assert cfg.endpoint == ENDPOINT
        assert cfg.extra_headers == {}


class TestEnvDerivation:
    def test_endpoint_derived_from_env(self, clean_env):
        clean_env.setenv("DATAROBOT_ENDPOINT", "https://example.test/api/v2")
        cfg = DataRobotOtelCollectorTelemetryExporter(project="test-agent")
        assert cfg.endpoint == "https://example.test/otel/v1/traces"

    def test_public_endpoint_takes_priority_over_endpoint(self, clean_env):
        # resolve_otel_traces_endpoint_from_env puts DATAROBOT_PUBLIC_API_ENDPOINT
        # ahead of DATAROBOT_ENDPOINT — same precedence here.
        clean_env.setenv("DATAROBOT_PUBLIC_API_ENDPOINT", "https://public.test/api/v2")
        clean_env.setenv("DATAROBOT_ENDPOINT", "https://internal.test/api/v2")
        cfg = DataRobotOtelCollectorTelemetryExporter(project="test-agent")
        assert cfg.endpoint == "https://public.test/otel/v1/traces"

    def test_endpoint_strips_api_path_segment(self, clean_env):
        # DATAROBOT_ENDPOINT typically includes the /api/v2 suffix; the OTel
        # collector lives at /otel/v1/traces off the host, not under /api.
        clean_env.setenv("DATAROBOT_ENDPOINT", "https://example.test/api/v2")
        cfg = DataRobotOtelCollectorTelemetryExporter(project="test-agent")
        assert cfg.endpoint == "https://example.test/otel/v1/traces"

    def test_endpoint_empty_when_env_unset(self, clean_env):
        # No silent fallback to app.datarobot.com — empty is the explicit
        # "not configured" signal so an unconfigured env never targets
        # the public DR endpoint by accident.
        cfg = DataRobotOtelCollectorTelemetryExporter(project="test-agent")
        assert cfg.endpoint == ""

    def test_explicit_otlp_base_url_appends_traces_path(self, clean_env):
        # OTEL_EXPORTER_OTLP_ENDPOINT is the standard OTel base URL; we append
        # /v1/traces rather than returning it verbatim.
        clean_env.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "https://collector.test:4318")
        cfg = DataRobotOtelCollectorTelemetryExporter(project="test-agent")
        assert cfg.endpoint == "https://collector.test:4318/v1/traces"

    def test_explicit_otlp_base_url_strips_trailing_slash(self, clean_env):
        clean_env.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "https://collector.test:4318/")
        cfg = DataRobotOtelCollectorTelemetryExporter(project="test-agent")
        assert cfg.endpoint == "https://collector.test:4318/v1/traces"

    def test_explicit_otlp_endpoint_wins_over_datarobot_endpoint(self, clean_env):
        # The standard OTel override takes precedence over the DR-derived
        # endpoint so an operator can point spans at any collector.
        clean_env.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "https://collector.test")
        clean_env.setenv("DATAROBOT_ENDPOINT", "https://example.test/api/v2")
        cfg = DataRobotOtelCollectorTelemetryExporter(project="test-agent")
        assert cfg.endpoint == "https://collector.test/v1/traces"


class TestHeaderResolvers:
    def test_headers_none_when_datarobot_env_incomplete(self, clean_env):
        assert resolve_datarobot_headers_from_env() is None

    def test_headers_none_when_api_key_only(self, clean_env):
        clean_env.setenv("DATAROBOT_API_TOKEN", API_KEY)
        assert resolve_datarobot_headers_from_env() is None

    def test_headers_none_when_entity_id_only(self, clean_env):
        clean_env.setenv("MLOPS_DEPLOYMENT_ID", "abc123")
        assert resolve_datarobot_headers_from_env() is None


class TestExporterFactory:
    # The @register_telemetry_exporter decorator wraps the function in
    # asynccontextmanager, so each call below uses ``async with``.

    async def test_emits_both_datarobot_headers(self, clean_env):
        _set_datarobot_env(clean_env)
        cfg = _make_config()
        with patch(
            "datarobot_genai.dragent.plugins.datarobot_otelcollector.DataRobotOTLPSpanAdapterExporter"
        ) as mock_exporter:
            async with datarobot_otelcollector_telemetry_exporter(cfg, builder=MagicMock()):
                pass

        mock_exporter.assert_called_once()
        kwargs = mock_exporter.call_args.kwargs
        assert kwargs["endpoint"] == ENDPOINT
        assert kwargs["headers"]["X-DataRobot-Api-Key"] == API_KEY
        assert kwargs["headers"]["X-DataRobot-Entity-Id"] == ENTITY_ID

    async def test_extra_headers_override_defaults(self, clean_env):
        _set_datarobot_env(clean_env)
        cfg = _make_config(
            extra_headers={
                "X-DataRobot-Entity-Id": "deployment-override",
                "X-Custom-Trace": "yes",
            },
        )
        with patch(
            "datarobot_genai.dragent.plugins.datarobot_otelcollector.DataRobotOTLPSpanAdapterExporter"
        ) as mock_exporter:
            async with datarobot_otelcollector_telemetry_exporter(cfg, builder=MagicMock()):
                pass

        kwargs = mock_exporter.call_args.kwargs
        assert kwargs["headers"]["X-DataRobot-Entity-Id"] == "deployment-override"
        assert kwargs["headers"]["X-Custom-Trace"] == "yes"
        # API key still present because extra_headers didn't override it.
        assert kwargs["headers"]["X-DataRobot-Api-Key"] == API_KEY

    async def test_secret_value_extracted_for_header(self, clean_env):
        # The header dict should contain the resolved secret string, not a
        # SecretStr wrapper — OTLP exporter expects str values.
        _set_datarobot_env(clean_env)
        cfg = _make_config()
        with patch(
            "datarobot_genai.dragent.plugins.datarobot_otelcollector.DataRobotOTLPSpanAdapterExporter"
        ) as mock_exporter:
            async with datarobot_otelcollector_telemetry_exporter(cfg, builder=MagicMock()):
                pass

        api_key_header = mock_exporter.call_args.kwargs["headers"]["X-DataRobot-Api-Key"]
        assert isinstance(api_key_header, str)
        assert api_key_header == API_KEY

    async def test_project_becomes_service_name(self, clean_env):
        # Resource attributes are independent of auth headers, but the exporter
        # factory still needs resolvable headers — supply DR env here.
        _set_datarobot_env(clean_env)
        # The inherited ``project`` field should populate the OTel
        # ``service.name`` resource attribute, matching NAT's built-in
        # otelcollector exporter.
        cfg = _make_config(project="my-agent")
        with patch(
            "datarobot_genai.dragent.plugins.datarobot_otelcollector.DataRobotOTLPSpanAdapterExporter"
        ) as mock_exporter:
            async with datarobot_otelcollector_telemetry_exporter(cfg, builder=MagicMock()):
                pass

        attrs = mock_exporter.call_args.kwargs["resource_attributes"]
        assert attrs["service.name"] == "my-agent"
        assert attrs["telemetry.sdk.language"] == "python"
        assert attrs["telemetry.sdk.name"] == "opentelemetry"
        assert "telemetry.sdk.version" in attrs

    async def test_resource_attributes_override_defaults(self, clean_env):
        _set_datarobot_env(clean_env)
        # Explicit resource_attributes win over the defaults derived from
        # ``project`` / SDK metadata.
        cfg = _make_config(
            project="my-agent",
            resource_attributes={"service.name": "override-name", "env": "prod"},
        )
        with patch(
            "datarobot_genai.dragent.plugins.datarobot_otelcollector.DataRobotOTLPSpanAdapterExporter"
        ) as mock_exporter:
            async with datarobot_otelcollector_telemetry_exporter(cfg, builder=MagicMock()):
                pass

        attrs = mock_exporter.call_args.kwargs["resource_attributes"]
        assert attrs["service.name"] == "override-name"
        assert attrs["env"] == "prod"
        # SDK defaults still merged in.
        assert attrs["telemetry.sdk.language"] == "python"


class TestOtlpHeadersEnvOverride:
    """When OTEL_EXPORTER_OTLP_HEADERS is set, the standard OTel header env var
    wins over the DataRobot auth headers derived from env.
    """

    async def test_headers_parsed_from_env(self, clean_env):
        clean_env.setenv(
            "OTEL_EXPORTER_OTLP_HEADERS",
            "X-DataRobot-Api-Key=env-key,X-DataRobot-Entity-Id=deployment-env",
        )
        cfg = _make_config()
        with patch(
            "datarobot_genai.dragent.plugins.datarobot_otelcollector.DataRobotOTLPSpanAdapterExporter"
        ) as mock_exporter:
            async with datarobot_otelcollector_telemetry_exporter(cfg, builder=MagicMock()):
                pass

        headers = mock_exporter.call_args.kwargs["headers"]
        # Env-provided headers replace the DR auth headers derived from env.
        assert headers["X-DataRobot-Api-Key"] == "env-key"
        assert headers["X-DataRobot-Entity-Id"] == "deployment-env"

    async def test_env_headers_take_precedence_over_datarobot_env(self, clean_env):
        # DR env still supplies API key + entity id, but the OTel header var
        # must override them entirely.
        _set_datarobot_env(clean_env)
        clean_env.setenv("OTEL_EXPORTER_OTLP_HEADERS", "X-DataRobot-Api-Key=env-key")
        cfg = _make_config()
        with patch(
            "datarobot_genai.dragent.plugins.datarobot_otelcollector.DataRobotOTLPSpanAdapterExporter"
        ) as mock_exporter:
            async with datarobot_otelcollector_telemetry_exporter(cfg, builder=MagicMock()):
                pass

        headers = mock_exporter.call_args.kwargs["headers"]
        assert headers["X-DataRobot-Api-Key"] == "env-key"
        # The env-derived value did not leak through.
        assert headers["X-DataRobot-Api-Key"] != API_KEY

    async def test_extra_headers_still_merged_over_env_headers(self, clean_env):
        # extra_headers always win on collision, even over env-supplied headers.
        clean_env.setenv("OTEL_EXPORTER_OTLP_HEADERS", "X-DataRobot-Api-Key=env-key")
        cfg = _make_config(extra_headers={"X-DataRobot-Api-Key": "extra-key", "X-Custom": "v"})
        with patch(
            "datarobot_genai.dragent.plugins.datarobot_otelcollector.DataRobotOTLPSpanAdapterExporter"
        ) as mock_exporter:
            async with datarobot_otelcollector_telemetry_exporter(cfg, builder=MagicMock()):
                pass

        headers = mock_exporter.call_args.kwargs["headers"]
        assert headers["X-DataRobot-Api-Key"] == "extra-key"
        assert headers["X-Custom"] == "v"

    async def test_header_value_with_equals_preserved(self, clean_env):
        # A header value can legitimately contain '=' (e.g. base64 padding or
        # a token with '='). Splitting on the first '=' only must keep the
        # full value intact rather than truncating at the first '='.
        clean_env.setenv(
            "OTEL_EXPORTER_OTLP_HEADERS",
            "Authorization=Basic dXNlcjpwYXNz==,X-Token=a=b=c",
        )
        cfg = _make_config()
        with patch(
            "datarobot_genai.dragent.plugins.datarobot_otelcollector.DataRobotOTLPSpanAdapterExporter"
        ) as mock_exporter:
            async with datarobot_otelcollector_telemetry_exporter(cfg, builder=MagicMock()):
                pass

        headers = mock_exporter.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Basic dXNlcjpwYXNz=="
        assert headers["X-Token"] == "a=b=c"

    async def test_datarobot_env_headers_used_when_otlp_headers_unset(self, clean_env):
        # Sanity check the default path: with no OTEL_EXPORTER_OTLP_HEADERS,
        # the DR auth headers are derived from env as before.
        _set_datarobot_env(clean_env)
        cfg = _make_config()
        with patch(
            "datarobot_genai.dragent.plugins.datarobot_otelcollector.DataRobotOTLPSpanAdapterExporter"
        ) as mock_exporter:
            async with datarobot_otelcollector_telemetry_exporter(cfg, builder=MagicMock()):
                pass

        headers = mock_exporter.call_args.kwargs["headers"]
        assert headers["X-DataRobot-Api-Key"] == API_KEY
        assert headers["X-DataRobot-Entity-Id"] == ENTITY_ID
