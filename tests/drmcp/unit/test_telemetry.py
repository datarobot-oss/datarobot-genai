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
from datarobot_genai.drmcp.core.telemetry import initialize_telemetry as _real_initialize_telemetry
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
    mock_config.otel_exporter_otlp_endpoint = ""
    mock_config.otel_exporter_otlp_headers = ""

    mock_credentials = MagicMock()
    mock_credentials.datarobot.datarobot_api_token = "test-app-token"

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


def test_setup_otel_env_variables_bridges_config_fields() -> None:
    """When Config has standard OTel fields (e.g. from pulumi_config.json),
    they are bridged to os.environ via setdefault.
    """
    mock_config = MagicMock()
    mock_config.otel_exporter_otlp_endpoint = "https://config.example.com/otel"
    mock_config.otel_exporter_otlp_headers = "x-key=config123"
    mock_config.otel_collector_base_url = "http://fallback:4318"
    mock_config.otel_entity_id = "fallback-entity"

    with (
        patch("datarobot_genai.drmcp.core.telemetry.get_config", return_value=mock_config),
        patch.dict("os.environ", {}, clear=True),
    ):
        telemetry._setup_otel_env_variables()

        assert os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] == "https://config.example.com/otel"
        assert os.environ["OTEL_EXPORTER_OTLP_HEADERS"] == "x-key=config123"


def test_setup_otel_env_variables_setdefault_no_override() -> None:
    """Explicit env vars are NOT overridden by Config values (setdefault semantics)."""
    mock_config = MagicMock()
    mock_config.otel_exporter_otlp_endpoint = "https://config.example.com/otel"
    mock_config.otel_exporter_otlp_headers = "x-key=config123"

    with (
        patch("datarobot_genai.drmcp.core.telemetry.get_config", return_value=mock_config),
        patch.dict(
            "os.environ",
            {
                "OTEL_EXPORTER_OTLP_ENDPOINT": "https://explicit.example.com/otel",
                "OTEL_EXPORTER_OTLP_HEADERS": "x-key=explicit",
            },
            clear=True,
        ),
    ):
        telemetry._setup_otel_env_variables()

        assert os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] == "https://explicit.example.com/otel"
        assert os.environ["OTEL_EXPORTER_OTLP_HEADERS"] == "x-key=explicit"


def test_setup_otel_env_variables_entity_id_headers_do_not_orphan_endpoint() -> None:
    """Regression: when the Config validator assembles headers from
    ``otel_entity_id``, the endpoint must still be bridged (previously the
    "already set" early-return fired on the headers it had just written,
    leaving exporters on the OTLP localhost default).
    """
    mock_config = MagicMock()
    mock_config.otel_exporter_otlp_endpoint = ""
    mock_config.otel_exporter_otlp_headers = (
        "x-datarobot-entity-id=deployment-abc,x-datarobot-api-key=tok"
    )
    mock_config.otel_collector_base_url = "https://app.example.com/otel"
    mock_config.otel_entity_id = "deployment-abc"

    with (
        patch("datarobot_genai.drmcp.core.telemetry.get_config", return_value=mock_config),
        patch.dict("os.environ", {}, clear=True),
    ):
        telemetry._setup_otel_env_variables()

        assert os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] == "https://app.example.com/otel"
        assert (
            os.environ["OTEL_EXPORTER_OTLP_HEADERS"]
            == "x-datarobot-entity-id=deployment-abc,x-datarobot-api-key=tok"
        )


def test_setup_otel_env_variables_deployment_shape_runtime_params() -> None:
    """Deployment shape, no mocks on the config: only ``MLOPS_RUNTIME_PARAM_*``
    envelopes + DataRobot credentials in the environment (what a deployment
    container actually gets when OTel is configured via runtime parameters).
    The real ``MCPServerConfig`` must resolve them, the bridge must populate
    both OTLP env vars, and a no-arg metric exporter must land on the
    collector's ``/v1/metrics`` — proving the whole chain without a network.
    """
    import json

    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter

    from datarobot_genai.drmcp.core.config import MCPServerConfig

    deployment_env = {
        "DATAROBOT_ENDPOINT": "https://app.example.com/api/v2",
        "DATAROBOT_API_TOKEN": "test-token-123",
        "MLOPS_DEPLOYMENT_ID": "deadbeef12345678",
        "MLOPS_RUNTIME_PARAM_OTEL_ENTITY_ID": json.dumps(
            {"type": "string", "payload": "deployment-deadbeef12345678"}
        ),
    }
    with patch.dict("os.environ", deployment_env, clear=True):
        config = MCPServerConfig()
        assert config.otel_entity_id == "deployment-deadbeef12345678"
        # The collector default derives from DATAROBOT_ENDPOINT at config-module
        # import time (fine in a deployment where env precedes the process, but
        # not pinnable to a literal here) — anchor on the resolved value.
        collector_base = config.otel_collector_base_url
        assert collector_base.endswith("/otel")

        with patch("datarobot_genai.drmcp.core.telemetry.get_config", return_value=config):
            telemetry._setup_otel_env_variables()

        assert os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] == collector_base
        assert "x-datarobot-entity-id=deployment-deadbeef12345678" in (
            os.environ["OTEL_EXPORTER_OTLP_HEADERS"]
        )

        exporter = OTLPMetricExporter()
        assert exporter._endpoint == collector_base + "/v1/metrics"


def test_initialize_telemetry_proceeds_with_config_headers() -> None:
    """initialize_telemetry should NOT skip when otel_exporter_otlp_headers is set
    in Config, even if otel_entity_id is empty.
    """
    mcp_mock = MagicMock()
    mock_config = MagicMock()
    mock_config.otel_enabled = True
    mock_config.otel_entity_id = ""
    mock_config.otel_exporter_otlp_headers = "x-key=from-config"
    mock_config.otel_exporter_otlp_endpoint = "https://config.example.com/otel"
    mock_config.mcp_server_name = "test-mcp"
    mock_config.otel_attributes = {}
    mock_config.otel_enabled_http_instrumentors = False

    with (
        patch("datarobot_genai.drmcp.core.telemetry.get_config", return_value=mock_config),
        patch("datarobot_genai.drmcp.core.telemetry._setup_otel_env_variables"),
        patch("datarobot_genai.drmcp.core.telemetry._setup_otel_exporter"),
        patch("datarobot_genai.drmcp.core.telemetry._setup_otel_logging"),
        patch("datarobot_genai.drmcp.core.telemetry.trace") as mock_trace,
        patch.dict("os.environ", {}, clear=True),
    ):
        _real_initialize_telemetry(mcp_mock)
        mock_trace.set_tracer_provider.assert_called_once()


def test_initialize_telemetry_installs_metrics_leg() -> None:
    """The config-gated startup path wires metrics next to traces and logs."""
    mcp_mock = MagicMock()
    mock_config = MagicMock()
    mock_config.otel_enabled = True
    mock_config.otel_entity_id = "entity"
    mock_config.otel_exporter_otlp_headers = ""
    mock_config.mcp_server_name = "test-mcp"
    mock_config.otel_attributes = {}
    mock_config.otel_enabled_http_instrumentors = False

    with (
        patch("datarobot_genai.drmcp.core.telemetry.get_config", return_value=mock_config),
        patch("datarobot_genai.drmcp.core.telemetry._setup_otel_env_variables"),
        patch("datarobot_genai.drmcp.core.telemetry._setup_otel_exporter"),
        patch("datarobot_genai.drmcp.core.telemetry.bootstrap_metrics_provider") as mock_bootstrap,
        patch("datarobot_genai.drmcp.core.telemetry._setup_otel_logging"),
        patch("datarobot_genai.drmcp.core.telemetry.trace"),
        patch.dict("os.environ", {}, clear=True),
    ):
        _real_initialize_telemetry(mcp_mock)

    mock_bootstrap.assert_called_once_with(
        resource_attributes={"datarobot.service.name": "test-mcp"}
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
        mock_span.set_attribute.assert_any_call("gen_ai.tool.name", "test_tool")
        mock_span.set_attribute.assert_any_call("gen_ai.operation.name", "execute_tool")
        mock_span.set_attribute.assert_any_call("tool.param.param1", "test")
        mock_span.set_attributes.assert_any_call({"custom.attr": "test-value"})
        mock_span.set_attribute.assert_any_call("tool.param.param2", 123)
        mock_span.set_attribute.assert_any_call(
            "gen_ai.tool.call.arguments", '{"param1": "test", "param2": 123}'
        )
        mock_span.set_status.assert_called()


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
        mock_span.set_attribute.assert_any_call("gen_ai.tool.name", "test_sync_function")
        mock_span.set_attribute.assert_any_call("gen_ai.operation.name", "execute_tool")
        mock_span.set_attribute.assert_any_call("tool.param.param1", "test")
        mock_span.set_attributes.assert_any_call({"custom.attr": "test-value"})
        mock_span.set_attribute.assert_any_call("gen_ai.tool.call.arguments", '{"param1": "test"}')
        mock_span.set_status.assert_called()


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
        mock_span.set_attribute.assert_any_call("error.type", "Exception")
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
        mock_span.set_attribute.assert_any_call("error.type", "Exception")
        mock_span.record_exception.assert_called_once()
        mock_span.set_status.assert_called_once()
