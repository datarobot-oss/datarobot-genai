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

ENDPOINT = "https://staging.datarobot.com/otel/v1/traces"
ENTITY_ID = "deployment-abc123"
API_KEY = "test-key-do-not-use"


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
    def test_minimal_config_valid(self):
        cfg = _make_config()
        assert cfg.endpoint == ENDPOINT
        assert cfg.datarobot_api_key.get_secret_value() == API_KEY
        assert cfg.datarobot_entity_id == ENTITY_ID
        assert cfg.extra_headers == {}

    def test_api_key_required(self):
        with pytest.raises(ValidationError):
            DataRobotOtelCollectorTelemetryExporter(
                endpoint=ENDPOINT,
                datarobot_entity_id=ENTITY_ID,
                project="test-agent",
            )

    def test_entity_id_required(self):
        with pytest.raises(ValidationError):
            DataRobotOtelCollectorTelemetryExporter(
                endpoint=ENDPOINT,
                datarobot_api_key=API_KEY,
                project="test-agent",
            )

    def test_entity_id_must_have_deployment_prefix(self):
        # Bare ID without the 'deployment-' prefix should fail validation,
        # matching the DataRobot external-agent-monitoring skill convention.
        with pytest.raises(ValidationError) as excinfo:
            _make_config(datarobot_entity_id="abc123")
        assert "deployment-<id>" in str(excinfo.value)

    def test_entity_id_with_prefix_valid(self):
        cfg = _make_config(datarobot_entity_id="deployment-xyz")
        assert cfg.datarobot_entity_id == "deployment-xyz"

    def test_api_key_not_logged_in_repr(self):
        cfg = _make_config(datarobot_api_key="super-secret-token")
        rendered = repr(cfg)
        assert "super-secret-token" not in rendered
        # Pydantic's SecretStr renders as '**********' or similar.
        assert "**" in rendered or "Secret" in rendered


class TestExporterFactory:
    # The @register_telemetry_exporter decorator wraps the function in
    # asynccontextmanager, so each call below uses ``async with``.

    async def test_emits_both_datarobot_headers(self):
        cfg = _make_config()
        with patch(
            "datarobot_genai.nat.datarobot_otelcollector.OTLPSpanAdapterExporter"
        ) as mock_exporter:
            async with datarobot_otelcollector_telemetry_exporter(
                cfg, builder=MagicMock()
            ):
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
            async with datarobot_otelcollector_telemetry_exporter(
                cfg, builder=MagicMock()
            ):
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
            async with datarobot_otelcollector_telemetry_exporter(
                cfg, builder=MagicMock()
            ):
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
            async with datarobot_otelcollector_telemetry_exporter(
                cfg, builder=MagicMock()
            ):
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
            async with datarobot_otelcollector_telemetry_exporter(
                cfg, builder=MagicMock()
            ):
                pass

        attrs = mock_exporter.call_args.kwargs["resource_attributes"]
        assert attrs["service.name"] == "override-name"
        assert attrs["env"] == "prod"
        # SDK defaults still merged in.
        assert attrs["telemetry.sdk.language"] == "python"
