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

import functools
import inspect
import logging
import os
from collections.abc import Callable
from collections.abc import Iterator
from collections.abc import Mapping
from contextlib import contextmanager
from typing import Any
from typing import TypeVar

from fastmcp import FastMCP
from fastmcp.server.middleware import CallNext
from fastmcp.server.middleware import Middleware
from fastmcp.server.middleware import MiddlewareContext
from fastmcp.tools.tool import ToolResult
from opentelemetry import trace
from opentelemetry._logs import set_logger_provider
from opentelemetry.context import attach
from opentelemetry.context import detach
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.propagate import extract
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs import LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.trace import Span
from opentelemetry.trace import SpanContext
from opentelemetry.trace import Status
from opentelemetry.trace import StatusCode
from opentelemetry.trace import format_trace_id
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.base import RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from .config import get_config
from .credentials import get_credentials

root_logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

# Track instrumentation state
_INSTRUMENTED = False


@contextmanager
def with_otel_context(carrier: Mapping[str, str]) -> Iterator[Any]:
    ctx = extract(carrier)
    token = attach(ctx)
    try:
        yield
    finally:
        detach(token)


class OpenTelemetryMiddleware(Middleware):
    def __init__(self, tracer_name: str = "fastmcp") -> None:
        self.tracer = trace.get_tracer(tracer_name)

    async def on_request(self, context: MiddlewareContext, call_next: CallNext[Any, Any]) -> Any:
        with tracer.start_as_current_span(f"mcp.request.{context.method}") as span:
            span.set_attribute("mcp.source", context.source)
            span.set_attribute("mcp.type", context.type)
            span.set_attribute("mcp.method", context.method or "")
            try:
                result = await call_next(context)
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
            return result

    async def on_call_tool(
        self, context: MiddlewareContext, call_next: CallNext[Any, ToolResult]
    ) -> ToolResult:
        tool_name = context.message.name

        with self.tracer.start_as_current_span(f"tool.{tool_name}") as span:
            span.set_attributes(
                {
                    "mcp.tool.name": tool_name,
                    "mcp.tool.arguments": str(context.message.arguments),
                }
            )
            try:
                result = await call_next(context)
                span.set_attribute("mcp.tool.success", True)
                if hasattr(result, "content"):
                    span.set_attribute("mcp.tool.content_length", len(str(result.content)))

                span.set_status(Status(StatusCode.OK))
                return result

            except Exception as e:
                span.set_attribute("mcp.tool.success", False)
                span.set_attribute("mcp.tool.error", str(e))
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise


class OtelASGIMiddleware(BaseHTTPMiddleware):
    """ASGI middleware extracts trace_id and parent span id from raw http request
    and set them in the context, so downsream trace can link to them
    if avaialble.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        with with_otel_context(request.headers):
            response = await call_next(request)
            return response


def _setup_otel_env_variables() -> None:
    """Set up OpenTelemetry environment variables for DataRobot integration."""
    # do not override if already set
    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") or os.environ.get(
        "OTEL_EXPORTER_OTLP_HEADERS"
    ):
        root_logger.info(
            "OTEL_EXPORTER_OTLP_ENDPOINT or OTEL_EXPORTER_OTLP_HEADERS already set, skipping"
        )
        return

    credentials = get_credentials()

    config = get_config()
    otlp_endpoint = config.otel_collector_base_url
    entity_id = config.otel_entity_id

    otlp_headers = (
        f"X-DataRobot-Api-Key={credentials.datarobot.application_api_token},"
        f"X-DataRobot-Entity-Id={entity_id}"
    )
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = otlp_endpoint
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = otlp_headers
    root_logger.info(
        f"Using OTEL_EXPORTER_OTLP_ENDPOINT: {otlp_endpoint} with X-DataRobot-Entity-Id {entity_id}"
    )


def _setup_otel_exporter() -> None:
    """Set up OpenTelemetry exporter with SimpleSpanProcessor."""
    otlp_exporter = OTLPSpanExporter()
    span_processor = SimpleSpanProcessor(otlp_exporter)
    provider = trace.get_tracer_provider()
    # mypy: TracerProvider has add_span_processor at runtime; typing may lag
    if hasattr(provider, "add_span_processor"):
        provider.add_span_processor(span_processor)


class _ExcludeOtelLogsFilter(logging.Filter):
    """A logging filter to exclude logs from the opentelemetry library."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not record.name.startswith("opentelemetry")


def _setup_otel_logging(resource: Resource) -> LoggerProvider:
    logger_provider = LoggerProvider(resource=resource)
    set_logger_provider(logger_provider)
    exporter = OTLPLogExporter()
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(exporter))
    handler = LoggingHandler(level=logging.INFO, logger_provider=logger_provider)
    handler.addFilter(_ExcludeOtelLogsFilter())
    logging.getLogger().addHandler(handler)
    return logger_provider


def _setup_http_instrumentors() -> None:
    """Set up HTTP client instrumentors.

    This function is idempotent - it will only instrument clients once.
    """
    # Use a local variable to avoid global statement warning
    instrumented = _INSTRUMENTED
    if instrumented:
        root_logger.debug("HTTP clients already instrumented")
        return

    root_logger.info("Setting up HTTP client instrumentation")
    try:
        # Instrument requests library
        RequestsInstrumentor().instrument()
        root_logger.debug("Instrumented requests library")
    except Exception as e:
        root_logger.warning(f"Failed to instrument requests: {e}")

    try:
        # Instrument aiohttp client
        AioHttpClientInstrumentor().instrument()
        root_logger.debug("Instrumented aiohttp client")
    except Exception as e:
        root_logger.warning(f"Failed to instrument aiohttp: {e}")

    try:
        # Instrument httpx
        HTTPXClientInstrumentor().instrument()
        root_logger.debug("Instrumented httpx")
    except Exception as e:
        root_logger.warning(f"Failed to instrument httpx: {e}")

    globals()["_INSTRUMENTED"] = True
    root_logger.info("HTTP client instrumentation complete")


def _prepare_shared_attributes(
    attributes: dict[str, Any],
) -> dict[str, str]:
    """Set custom attributes on a span."""
    # Flatten nested attributes
    flattened_attrs = {}
    for key, value in attributes.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flattened_attrs[f"{key}.{sub_key}"] = sub_value

        elif isinstance(value, (str, int, float, bool)):
            flattened_attrs[key] = value
    return flattened_attrs


def _set_otel_attributes(span: Span, attributes: dict[str, Any]) -> None:
    """Set custom attributes on a span."""
    attrs = _prepare_shared_attributes(attributes=attributes)
    span.set_attributes(attrs)


def initialize_telemetry(mcp: FastMCP) -> None:
    """Initialize OpenTelemetry for the FastMCP application."""
    config = get_config()

    # If OpenTelemetry is disabled, return None
    if not config.otel_enabled:
        root_logger.info("OpenTelemetry is disabled")
        return None

    # If OTEL_ENTITY_ID is not set, skip telemetry
    if not config.otel_entity_id and not os.environ.get("OTEL_EXPORTER_OTLP_HEADERS"):
        root_logger.info(
            "Neither OTEL_ENTITY_ID nor OTEL_EXPORTER_OTLP_HEADERS is set, skipping telemetry"
        )
        return None

    resource_attrs = {"datarobot.service.name": config.mcp_server_name}
    if config.otel_attributes:
        resource_attrs.update(_prepare_shared_attributes(config.otel_attributes))
    resource = Resource.create(resource_attrs)

    # Set up tracer provider with service name from config
    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)

    # Setup environment
    _setup_otel_env_variables()

    # Setup OTEL exporter
    _setup_otel_exporter()
    _setup_otel_logging(resource)

    # Setup HTTP client instrumentation
    if config.otel_enabled_http_instrumentors:
        _setup_http_instrumentors()

    mcp.add_middleware(OpenTelemetryMiddleware(__name__))


"""Helper functions for OpenTelemetry instrumentation of tools."""


def _add_parameters_to_span(
    span: Span, func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> None:
    """Add function parameters as span attributes.

    Only adds simple types (str, int, float, bool) to avoid complex object serialization.
    Skips 'self' parameter for methods.
    """
    # Get parameter names
    sig = inspect.signature(func)
    param_names = list(sig.parameters.keys())

    # Skip 'self' parameter for methods (only if first param is named 'self')
    start_idx = 1 if args and param_names and param_names[0] == "self" else 0
    param_names = param_names[start_idx:]
    args = args[start_idx:]

    # Add positional arguments
    for name, value in zip(param_names, args):
        if isinstance(value, (str, int, float, bool)):
            span.set_attribute(f"tool.param.{name}", value)

    # Add keyword arguments
    for name, value in kwargs.items():
        if isinstance(value, (str, int, float, bool)):
            span.set_attribute(f"tool.param.{name}", value)


def get_trace_id() -> str | None:
    """Get the current trace ID if available."""
    current_span = trace.get_current_span()
    if not current_span:
        return None

    context: SpanContext = current_span.get_span_context()
    if not context.is_valid:
        return None

    return str(format_trace_id(context.trace_id))


T = TypeVar("T", bound=Callable[..., Any])


def trace_execution(trace_name: str | None = None, trace_type: str = "tool") -> Callable[[T], T]:
    """Trace tool execution.

    Args:
        trace_name: Optional name for the span. If not provided, uses the function name.
        trace_type: Optional type for the span. If not provided, uses "tool".

    Example:
        @trace_execution()
        async def my_tool(self, param1: str) -> str:
            return "result"

        @trace_execution("custom_name")
        async def another_tool(self, param1: str) -> str:
            return "result"
    """

    def decorator(func: T) -> T:
        def _create_span_for_tool(
            trace_name: str | None,
            trace_type: str,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> Span:
            # Get span name from decorator arg, function name, or class method name
            span_name = trace_name
            if not span_name:
                if (
                    args
                    and hasattr(args[0], "__class__")
                    and not isinstance(args[0], (str, int, float, bool))
                ):
                    # If it's a method, include the class name
                    span_name = f"{args[0].__class__.__name__}.{func.__name__}"
                else:
                    # Just use the function name without any prefix
                    span_name = func.__name__

            # Start a new span
            tracer = trace.get_tracer(__name__)
            span = tracer.start_span(f"{trace_type}.{span_name}")

            # Add standard attributes
            span.set_attribute("mcp.type", trace_type)
            span.set_attribute(f"{trace_type}.name", span_name)

            # Add tool parameters as span attributes
            _add_parameters_to_span(span, func, args, kwargs)

            # Add configured attributes from config
            config = get_config()
            if config.otel_attributes:
                _set_otel_attributes(span, config.otel_attributes)

            return span

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            span = _create_span_for_tool(trace_name, trace_type, args, kwargs)
            try:
                result = await func(*args, **kwargs)
                span.set_attribute(f"{trace_type}.success", True)
                return result
            except Exception as e:
                span.set_attribute(f"{trace_type}.success", False)
                span.record_exception(e)
                raise e
            finally:
                span.end()

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            span = _create_span_for_tool(trace_name, trace_type, args, kwargs)
            try:
                result = func(*args, **kwargs)
                span.set_attribute(f"{trace_type}.success", True)
                return result
            except Exception as e:
                span.set_attribute(f"{trace_type}.success", False)
                span.record_exception(e)
                raise e
            finally:
                span.end()

        # Use appropriate wrapper based on whether the function is async
        return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper  # type: ignore[return-value]

    return decorator
