# Copyright 2025 DataRobot, Inc.
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

import os
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from opentelemetry.trace import Span
from opentelemetry.trace import SpanContext

from datarobot_genai.drmcp.core import telemetry


def test_initialize_telemetry_disabled() -> None:
    mcp_mock = MagicMock()
    mock_config = MagicMock()
    mock_config.otel_enabled = False
    with patch("datarobot_genai.drmcp.core.telemetry.get_config", return_value=mock_config):
        result = telemetry.initialize_telemetry(mcp_mock)
        assert result is None


def test_initialize_telemetry_enabled() -> None:
    mock_config = MagicMock()
    mock_config.mcp_server_name = "test-service"
    mock_config.otel_enabled = True
    mock_config.otel_collector_base_url = "http://test-collector:4318"
    mock_config.otel_entity_id = "test-entity"
    mock_config.otel_attributes = {"custom.attr": "test-value"}

    mock_credentials = MagicMock()
    mock_credentials.datarobot.application_api_token = "test-app-token"

    with (
        patch("datarobot_genai.drmcp.core.telemetry.get_config", return_value=mock_config),
        patch(
            "datarobot_genai.drmcp.core.telemetry.get_credentials", return_value=mock_credentials
        ),
        patch("datarobot_genai.drmcp.core.telemetry._setup_otel_exporter") as mock_exporter,
        patch(
            "datarobot_genai.drmcp.core.telemetry._setup_http_instrumentors"
        ) as mock_instrumentors,
        patch("opentelemetry.sdk.resources.Resource.create") as mock_resource_create,
        patch.dict("os.environ", {}, clear=True),
    ):
        mcp_mock = MagicMock()
        telemetry.initialize_telemetry(mcp_mock)

        mock_exporter.assert_called_once()
        mock_instrumentors.assert_called_once()
        expected = {
            "custom.attr": "test-value",
            "datarobot.service.name": "test-service",
        }
        mock_resource_create.assert_called_once_with(expected)
        assert os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://test-collector:4318"
        assert (
            os.environ["OTEL_EXPORTER_OTLP_HEADERS"]
            == "X-DataRobot-Api-Key=test-app-token,X-DataRobot-Entity-Id=test-entity"
        )


def test_setup_otel_env_variables() -> None:
    mock_config = MagicMock()
    mock_config.otel_collector_base_url = "http://test-collector:4318"
    mock_config.otel_entity_id = "test-entity"

    mock_credentials = MagicMock()
    mock_credentials.datarobot.application_api_token = "test-app-token"

    with (
        patch("datarobot_genai.drmcp.core.telemetry.get_config", return_value=mock_config),
        patch(
            "datarobot_genai.drmcp.core.telemetry.get_credentials", return_value=mock_credentials
        ),
        patch.dict("os.environ", {}, clear=True),
    ):
        telemetry._setup_otel_env_variables()

        assert os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://test-collector:4318"
        assert (
            os.environ["OTEL_EXPORTER_OTLP_HEADERS"]
            == "X-DataRobot-Api-Key=test-app-token,X-DataRobot-Entity-Id=test-entity"
        )


@pytest.mark.asyncio
async def test_trace_execution_async() -> None:
    mock_config = MagicMock()
    mock_config.otel_attributes = {"custom.attr": "test-value"}

    mock_span = MagicMock(spec=Span)
    mock_span_context = MagicMock(spec=SpanContext)
    mock_span_context.is_valid = True
    mock_span_context.trace_id = 123456
    mock_span.get_span_context.return_value = mock_span_context

    with (
        patch("opentelemetry.trace.get_tracer") as mock_get_tracer,
        patch("datarobot_genai.drmcp.core.telemetry.get_config", return_value=mock_config),
    ):
        mock_tracer = MagicMock()
        mock_tracer.start_span.return_value = mock_span
        mock_get_tracer.return_value = mock_tracer

        @telemetry.trace_execution("test_tool")
        async def test_async_function(param1: str, param2: int) -> str:
            return f"{param1}-{param2}"

        result = await test_async_function("test", 123)

        assert result == "test-123"
        mock_span.set_attribute.assert_any_call("mcp.type", "tool")
        mock_span.set_attribute.assert_any_call("tool.name", "test_tool")
        mock_span.set_attribute.assert_any_call("tool.param.param1", "test")
        mock_span.set_attributes.assert_any_call({"custom.attr": "test-value"})
        mock_span.set_attribute.assert_any_call("tool.param.param2", 123)
        mock_span.set_attribute.assert_any_call("tool.success", True)


def test_trace_execution_sync() -> None:
    mock_config = MagicMock()
    mock_config.otel_attributes = {"custom.attr": "test-value"}

    mock_span = MagicMock(spec=Span)
    with (
        patch("opentelemetry.trace.get_tracer") as mock_get_tracer,
        patch("datarobot_genai.drmcp.core.telemetry.get_config", return_value=mock_config),
    ):
        mock_tracer = MagicMock()
        mock_tracer.start_span.return_value = mock_span
        mock_get_tracer.return_value = mock_tracer

        @telemetry.trace_execution()
        def test_sync_function(param1: str) -> str:
            return f"result-{param1}"

        result = test_sync_function("test")

        assert result == "result-test"
        mock_span.set_attribute.assert_any_call("mcp.type", "tool")
        mock_span.set_attribute.assert_any_call("tool.name", "test_sync_function")
        mock_span.set_attribute.assert_any_call("tool.param.param1", "test")
        mock_span.set_attributes.assert_any_call({"custom.attr": "test-value"})
        mock_span.set_attribute.assert_any_call("tool.success", True)


def test_get_trace_id() -> None:
    mock_span = MagicMock(spec=Span)
    mock_span_context = MagicMock(spec=SpanContext)
    mock_span_context.is_valid = True
    mock_span_context.trace_id = 123456
    mock_span.get_span_context.return_value = mock_span_context

    with patch("opentelemetry.trace.get_current_span", return_value=mock_span):
        trace_id = telemetry.get_trace_id()
        assert trace_id is not None

    # Test invalid context
    mock_span_context.is_valid = False
    with patch("opentelemetry.trace.get_current_span", return_value=mock_span):
        trace_id = telemetry.get_trace_id()
        assert trace_id is None

    # Test no current span
    with patch("opentelemetry.trace.get_current_span", return_value=None):
        trace_id = telemetry.get_trace_id()
        assert trace_id is None
