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
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from opentelemetry.trace import Span
from opentelemetry.trace import SpanContext

from datarobot_genai.drmcp.core import telemetry
from datarobot_genai.drmcp.core.telemetry import OpenTelemetryMiddleware
from datarobot_genai.drmcp.core.telemetry import with_otel_context


def test_initialize_telemetry_disabled() -> None:
    mcp_mock = MagicMock()
    mock_config = MagicMock()
    mock_config.otel_enabled = False
    with patch("datarobot_genai.drmcp.core.telemetry.get_config", return_value=mock_config):
        result = telemetry.initialize_telemetry(mcp_mock)
        assert result is None


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
        patch("datarobot_genai.drmcp.core.telemetry._setup_otel_logging", return_value=None),
        patch("opentelemetry.exporter.otlp.proto.http._log_exporter.OTLPLogExporter"),
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


class TestWithOtelContext:
    """Test cases for with_otel_context function."""

    @patch("datarobot_genai.drmcp.core.telemetry.extract")
    @patch("datarobot_genai.drmcp.core.telemetry.attach")
    @patch("datarobot_genai.drmcp.core.telemetry.detach")
    def test_with_otel_context_success(self, mock_detach, mock_attach, mock_extract):
        """Test successful context management."""
        # Mock the context operations
        mock_token = Mock()
        mock_attach.return_value = mock_token
        mock_extract.return_value = {}

        # Test context manager
        carrier = {}
        with with_otel_context(carrier):
            mock_extract.assert_called_once_with(carrier)
            mock_attach.assert_called_once()

        mock_detach.assert_called_once_with(mock_token)

    @patch("datarobot_genai.drmcp.core.telemetry.extract")
    @patch("datarobot_genai.drmcp.core.telemetry.attach")
    @patch("datarobot_genai.drmcp.core.telemetry.detach")
    def test_with_otel_context_exception(self, mock_detach, mock_attach, mock_extract):
        """Test context management with exception."""
        # Mock the context operations
        mock_token = Mock()
        mock_attach.return_value = mock_token
        mock_extract.return_value = {}

        # Test context manager with exception
        carrier = {}
        with pytest.raises(ValueError):
            with with_otel_context(carrier):
                raise ValueError("Test exception")

        mock_detach.assert_called_once_with(mock_token)


class TestOpenTelemetryMiddleware:
    """Test cases for OpenTelemetryMiddleware class."""

    def test_middleware_initialization(self):
        """Test OpenTelemetryMiddleware initialization."""
        middleware = OpenTelemetryMiddleware()

        assert middleware.tracer is not None

    def test_middleware_initialization_with_custom_tracer(self):
        """Test OpenTelemetryMiddleware initialization with custom tracer name."""
        middleware = OpenTelemetryMiddleware(tracer_name="custom_tracer")

        assert middleware.tracer is not None

    @pytest.mark.asyncio
    @patch("datarobot_genai.drmcp.core.telemetry.tracer")
    async def test_on_request_success(self, mock_tracer):
        """Test on_request method with successful request."""
        middleware = OpenTelemetryMiddleware()

        # Mock context and response
        mock_context = Mock()
        mock_context.method = "GET"
        mock_context.source = "test"
        mock_context.type = "request"
        mock_response = Mock()

        # Mock the span and context manager
        mock_span = Mock()
        mock_tracer.start_as_current_span.return_value.__enter__ = Mock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = Mock(return_value=None)

        # Mock the call_next function
        mock_call_next = AsyncMock(return_value=mock_response)

        result = await middleware.on_request(mock_context, mock_call_next)

        assert result == mock_response
        mock_tracer.start_as_current_span.assert_called_once()
        mock_span.set_attribute.assert_called()
        mock_span.set_status.assert_called_once()

    @pytest.mark.asyncio
    @patch("datarobot_genai.drmcp.core.telemetry.tracer")
    async def test_on_request_exception(self, mock_tracer):
        """Test on_request method with exception."""
        middleware = OpenTelemetryMiddleware()

        # Mock context
        mock_context = Mock()
        mock_context.method = "GET"
        mock_context.source = "test"
        mock_context.type = "request"

        # Mock the span and context manager
        mock_span = Mock()
        mock_tracer.start_as_current_span.return_value.__enter__ = Mock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = Mock(return_value=None)

        # Mock the call_next function to raise an exception
        mock_call_next = AsyncMock(side_effect=Exception("Test error"))

        with pytest.raises(Exception, match="Test error"):
            await middleware.on_request(mock_context, mock_call_next)

        mock_tracer.start_as_current_span.assert_called_once()
        mock_span.set_attribute.assert_called()
        mock_span.record_exception.assert_called_once()
        mock_span.set_status.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_call_tool_success(self):
        """Test on_call_tool method with successful tool call."""
        middleware = OpenTelemetryMiddleware()

        # Mock context and response
        mock_context = Mock()
        mock_context.message = Mock()
        mock_context.message.name = "test_tool"
        mock_context.message.arguments = {"param1": "value1"}
        mock_response = Mock()
        mock_response.content = [{"type": "text", "text": "Success"}]

        # Mock the span and context manager
        mock_span = Mock()
        middleware.tracer.start_as_current_span.return_value.__enter__ = Mock(
            return_value=mock_span
        )
        middleware.tracer.start_as_current_span.return_value.__exit__ = Mock(return_value=None)

        # Mock the call_next function
        mock_call_next = AsyncMock(return_value=mock_response)

        result = await middleware.on_call_tool(mock_context, mock_call_next)

        assert result == mock_response
        middleware.tracer.start_as_current_span.assert_called_once()
        mock_span.set_attributes.assert_called_once()
        mock_span.set_attribute.assert_called()
        mock_span.set_status.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_call_tool_success_no_content(self):
        """Test on_call_tool method with successful tool call but no content."""
        middleware = OpenTelemetryMiddleware()

        # Mock context and response
        mock_context = Mock()
        mock_context.message = Mock()
        mock_context.message.name = "test_tool"
        mock_context.message.arguments = {"param1": "value1"}
        mock_response = Mock()
        mock_response.content = []

        # Mock the span and context manager
        mock_span = Mock()
        middleware.tracer.start_as_current_span.return_value.__enter__ = Mock(
            return_value=mock_span
        )
        middleware.tracer.start_as_current_span.return_value.__exit__ = Mock(return_value=None)

        # Mock the call_next function
        mock_call_next = AsyncMock(return_value=mock_response)

        result = await middleware.on_call_tool(mock_context, mock_call_next)

        assert result == mock_response
        middleware.tracer.start_as_current_span.assert_called_once()
        mock_span.set_attributes.assert_called_once()
        mock_span.set_status.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_call_tool_exception(self):
        """Test on_call_tool method with exception."""
        middleware = OpenTelemetryMiddleware()

        # Mock context
        mock_context = Mock()
        mock_context.message = Mock()
        mock_context.message.name = "test_tool"
        mock_context.message.arguments = {"param1": "value1"}

        # Mock the span and context manager
        mock_span = Mock()
        middleware.tracer.start_as_current_span.return_value.__enter__ = Mock(
            return_value=mock_span
        )
        middleware.tracer.start_as_current_span.return_value.__exit__ = Mock(return_value=None)

        # Mock the call_next function to raise an exception
        mock_call_next = AsyncMock(side_effect=Exception("Test error"))

        with pytest.raises(Exception, match="Test error"):
            await middleware.on_call_tool(mock_context, mock_call_next)

        middleware.tracer.start_as_current_span.assert_called_once()
        mock_span.set_attributes.assert_called_once()
        mock_span.set_attribute.assert_called()
        mock_span.record_exception.assert_called_once()
        mock_span.set_status.assert_called_once()
