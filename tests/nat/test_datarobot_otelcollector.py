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
from pydantic import ValidationError

from datarobot_genai.nat.datarobot_otelcollector import DataRobotOtelCollectorTelemetryExporter
from datarobot_genai.nat.datarobot_otelcollector import datarobot_otelcollector_telemetry_exporter
from datarobot_genai.nat.datarobot_otelcollector import prune_exporter_if_env_missing

ENDPOINT = "https://staging.datarobot.com/otel/v1/traces"
ENTITY_ID = "deployment-abc123"
API_KEY = "test-key-do-not-use"

_ENV_VARS = (
    "DATAROBOT_API_TOKEN",
    "MLOPS_DEPLOYMENT_ID",
    "DATAROBOT_ENDPOINT",
    "DATAROBOT_PUBLIC_API_ENDPOINT",
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
        "datarobot_api_key": API_KEY,
        "datarobot_entity_id": ENTITY_ID,
        "project": "test-agent",
    }
    base.update(overrides)
    return DataRobotOtelCollectorTelemetryExporter(**base)


class TestConfigValidation:
    def test_explicit_values_valid(self, clean_env):
        cfg = _make_config()
        assert cfg.endpoint == ENDPOINT
        assert cfg.datarobot_api_key.get_secret_value() == API_KEY
        assert cfg.datarobot_entity_id == ENTITY_ID
        assert cfg.extra_headers == {}

    def test_explicit_invalid_entity_id_rejected(self, clean_env):
        # Bare ID without the 'deployment-' prefix should still fail
        # validation when explicitly provided — relaxation only allows
        # empty so prune_exporter_if_env_missing can drop the entry.
        with pytest.raises(ValidationError) as excinfo:
            _make_config(datarobot_entity_id="abc123")
        assert "deployment-<id>" in str(excinfo.value)

    def test_entity_id_with_prefix_valid(self, clean_env):
        cfg = _make_config(datarobot_entity_id="deployment-xyz")
        assert cfg.datarobot_entity_id == "deployment-xyz"

    def test_api_key_not_logged_in_repr(self, clean_env):
        cfg = _make_config(datarobot_api_key="super-secret-token")
        rendered = repr(cfg)
        assert "super-secret-token" not in rendered
        # Pydantic's SecretStr renders as '**********' or similar.
        assert "**" in rendered or "Secret" in rendered


class TestEnvDerivation:
    def test_all_three_fields_derived_from_env(self, clean_env):
        clean_env.setenv("MLOPS_DEPLOYMENT_ID", "abc123")
        clean_env.setenv("DATAROBOT_API_TOKEN", "tok-derived")
        clean_env.setenv("DATAROBOT_ENDPOINT", "https://example.test/api/v2")
        cfg = DataRobotOtelCollectorTelemetryExporter(project="test-agent")
        assert cfg.datarobot_entity_id == "deployment-abc123"
        assert cfg.datarobot_api_key.get_secret_value() == "tok-derived"
        assert cfg.endpoint == "https://example.test/otel/v1/traces"

    def test_entity_id_auto_prefixed(self, clean_env):
        clean_env.setenv("MLOPS_DEPLOYMENT_ID", "6a1716687ee09515373a0ee5")
        cfg = DataRobotOtelCollectorTelemetryExporter(
            datarobot_api_key=API_KEY,
            endpoint=ENDPOINT,
            project="test-agent",
        )
        assert cfg.datarobot_entity_id == "deployment-6a1716687ee09515373a0ee5"

    def test_explicit_entity_id_wins_over_env(self, clean_env):
        clean_env.setenv("MLOPS_DEPLOYMENT_ID", "from-env")
        cfg = _make_config(datarobot_entity_id="deployment-explicit")
        assert cfg.datarobot_entity_id == "deployment-explicit"

    def test_public_endpoint_takes_priority_over_endpoint(self, clean_env):
        # resolve_datarobot_endpoint puts DATAROBOT_PUBLIC_API_ENDPOINT
        # ahead of DATAROBOT_ENDPOINT — same precedence here.
        clean_env.setenv("DATAROBOT_PUBLIC_API_ENDPOINT", "https://public.test/api/v2")
        clean_env.setenv("DATAROBOT_ENDPOINT", "https://internal.test/api/v2")
        cfg = DataRobotOtelCollectorTelemetryExporter(
            datarobot_api_key=API_KEY,
            datarobot_entity_id=ENTITY_ID,
            project="test-agent",
        )
        assert cfg.endpoint == "https://public.test/otel/v1/traces"

    def test_endpoint_strips_api_path_segment(self, clean_env):
        # DATAROBOT_ENDPOINT typically includes the /api/v2 suffix; the OTel
        # collector lives at /otel/v1/traces off the host, not under /api.
        clean_env.setenv("DATAROBOT_ENDPOINT", "https://example.test/api/v2")
        cfg = DataRobotOtelCollectorTelemetryExporter(
            datarobot_api_key=API_KEY,
            datarobot_entity_id=ENTITY_ID,
            project="test-agent",
        )
        assert cfg.endpoint == "https://example.test/otel/v1/traces"

    def test_endpoint_empty_when_env_unset(self, clean_env):
        # No silent fallback to app.datarobot.com — empty is the explicit
        # "not configured" signal that prune_exporter_if_env_missing relies on.
        cfg = DataRobotOtelCollectorTelemetryExporter(
            datarobot_api_key=API_KEY,
            datarobot_entity_id=ENTITY_ID,
            project="test-agent",
        )
        assert cfg.endpoint == ""

    def test_entity_id_empty_when_env_unset(self, clean_env):
        cfg = DataRobotOtelCollectorTelemetryExporter(
            datarobot_api_key=API_KEY,
            endpoint=ENDPOINT,
            project="test-agent",
        )
        assert cfg.datarobot_entity_id == ""


class TestPruneExporterIfEnvMissing:
    def _yaml(self, **fields):
        entry = {"_type": "datarobot_otelcollector"}
        entry.update(fields)
        return {"telemetry": {"tracing": {"otelcollector": entry}}}

    def test_prunes_when_env_missing_and_no_overrides(self, clean_env):
        cfg = self._yaml(project="agent")
        prune_exporter_if_env_missing(cfg)
        assert cfg["telemetry"]["tracing"] == {}

    def test_keeps_when_user_pinned_both_values(self, clean_env):
        cfg = self._yaml(
            project="agent",
            datarobot_api_key="pinned-key",
            datarobot_entity_id="deployment-pinned",
        )
        prune_exporter_if_env_missing(cfg)
        assert "otelcollector" in cfg["telemetry"]["tracing"]

    def test_keeps_when_only_api_key_pinned_but_deployment_in_env(self, clean_env):
        clean_env.setenv("MLOPS_DEPLOYMENT_ID", "abc123")
        cfg = self._yaml(project="agent", datarobot_api_key="pinned-key")
        prune_exporter_if_env_missing(cfg)
        assert "otelcollector" in cfg["telemetry"]["tracing"]

    def test_keeps_when_env_present(self, clean_env):
        clean_env.setenv("MLOPS_DEPLOYMENT_ID", "abc123")
        clean_env.setenv("DATAROBOT_API_TOKEN", "tok")
        cfg = self._yaml(project="agent")
        prune_exporter_if_env_missing(cfg)
        assert "otelcollector" in cfg["telemetry"]["tracing"]

    def test_unrelated_exporters_untouched(self, clean_env):
        cfg = {
            "telemetry": {
                "tracing": {
                    "otelcollector": {"_type": "datarobot_otelcollector", "project": "agent"},
                    "builtin": {"_type": "otelcollector", "endpoint": "http://x"},
                }
            }
        }
        prune_exporter_if_env_missing(cfg)
        assert "otelcollector" not in cfg["telemetry"]["tracing"]
        assert "builtin" in cfg["telemetry"]["tracing"]

    def test_no_telemetry_block_is_noop(self, clean_env):
        cfg = {"workflow": {"_type": "react_agent"}}
        prune_exporter_if_env_missing(cfg)
        assert cfg == {"workflow": {"_type": "react_agent"}}

    def test_empty_tracing_block_is_noop(self, clean_env):
        cfg = {"telemetry": {"tracing": {}}}
        prune_exporter_if_env_missing(cfg)
        assert cfg == {"telemetry": {"tracing": {}}}


class TestExporterFactory:
    # The @register_telemetry_exporter decorator wraps the function in
    # asynccontextmanager, so each call below uses ``async with``.

    async def test_emits_both_datarobot_headers(self):
        cfg = _make_config()
        with patch(
            "datarobot_genai.nat.datarobot_otelcollector.OTLPSpanAdapterExporter"
        ) as mock_exporter:
            async with datarobot_otelcollector_telemetry_exporter(cfg, builder=MagicMock()):
                pass

        mock_exporter.assert_called_once()
        kwargs = mock_exporter.call_args.kwargs
        assert kwargs["endpoint"] == ENDPOINT
        assert kwargs["headers"]["X-DataRobot-Api-Key"] == API_KEY
        assert kwargs["headers"]["X-DataRobot-Entity-Id"] == ENTITY_ID

    async def test_extra_headers_override_defaults(self):
        cfg = _make_config(
            extra_headers={
                "X-DataRobot-Entity-Id": "deployment-override",
                "X-Custom-Trace": "yes",
            },
        )
        with patch(
            "datarobot_genai.nat.datarobot_otelcollector.OTLPSpanAdapterExporter"
        ) as mock_exporter:
            async with datarobot_otelcollector_telemetry_exporter(cfg, builder=MagicMock()):
                pass

        kwargs = mock_exporter.call_args.kwargs
        assert kwargs["headers"]["X-DataRobot-Entity-Id"] == "deployment-override"
        assert kwargs["headers"]["X-Custom-Trace"] == "yes"
        # API key still present because extra_headers didn't override it.
        assert kwargs["headers"]["X-DataRobot-Api-Key"] == API_KEY

    async def test_secret_value_extracted_for_header(self):
        # The header dict should contain the resolved secret string, not a
        # SecretStr wrapper — OTLP exporter expects str values.
        cfg = _make_config()
        with patch(
            "datarobot_genai.nat.datarobot_otelcollector.OTLPSpanAdapterExporter"
        ) as mock_exporter:
            async with datarobot_otelcollector_telemetry_exporter(cfg, builder=MagicMock()):
                pass

        api_key_header = mock_exporter.call_args.kwargs["headers"]["X-DataRobot-Api-Key"]
        assert isinstance(api_key_header, str)
        assert api_key_header == API_KEY

    async def test_project_becomes_service_name(self):
        # The inherited ``project`` field should populate the OTel
        # ``service.name`` resource attribute, matching NAT's built-in
        # otelcollector exporter.
        cfg = _make_config(project="my-agent")
        with patch(
            "datarobot_genai.nat.datarobot_otelcollector.OTLPSpanAdapterExporter"
        ) as mock_exporter:
            async with datarobot_otelcollector_telemetry_exporter(cfg, builder=MagicMock()):
                pass

        attrs = mock_exporter.call_args.kwargs["resource_attributes"]
        assert attrs["service.name"] == "my-agent"
        assert attrs["telemetry.sdk.language"] == "python"
        assert attrs["telemetry.sdk.name"] == "opentelemetry"
        assert "telemetry.sdk.version" in attrs

    async def test_resource_attributes_override_defaults(self):
        # Explicit resource_attributes win over the defaults derived from
        # ``project`` / SDK metadata.
        cfg = _make_config(
            project="my-agent",
            resource_attributes={"service.name": "override-name", "env": "prod"},
        )
        with patch(
            "datarobot_genai.nat.datarobot_otelcollector.OTLPSpanAdapterExporter"
        ) as mock_exporter:
            async with datarobot_otelcollector_telemetry_exporter(cfg, builder=MagicMock()):
                pass

        attrs = mock_exporter.call_args.kwargs["resource_attributes"]
        assert attrs["service.name"] == "override-name"
        assert attrs["env"] == "prod"
        # SDK defaults still merged in.
        assert attrs["telemetry.sdk.language"] == "python"
