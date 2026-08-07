# Copyright 2026 DataRobot, Inc.
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

"""Tests for MCP tool-call metrics (drmcpbase/tool_metrics.py)."""

from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from fastmcp.exceptions import ToolError as FastMCPToolError
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from datarobot_genai.drmcp.core.telemetry import OpenTelemetryMiddleware
from datarobot_genai.drmcpbase import tool_metrics
from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.exceptions import ToolErrorKind
from datarobot_genai.drmcputils.exceptions import tool_error_kind_from_message


def _reader_and_instruments() -> tuple[Any, Any]:
    """Build a fresh InMemoryMetricReader + instruments on their own provider.

    Dependency-injected so tests never touch the global MeterProvider (which
    OpenTelemetry only allows setting once per process).
    """
    reader = InMemoryMetricReader()
    meter = MeterProvider(metric_readers=[reader]).get_meter("test")
    return reader, tool_metrics.build_instruments(meter)


def _points(reader: Any, name: str) -> list[Any]:
    data = reader.get_metrics_data()
    out: list[Any] = []
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == name:
                    out.extend(metric.data.data_points)
    return out


# --- kind parsing ---------------------------------------------------------------


class TestToolErrorKindFromMessage:
    def test_every_kind_prefix_parses(self) -> None:
        # GIVEN each client-facing "[kind] message" shape log_execution emits
        # WHEN the message is parsed
        # THEN the matching enum member comes back
        for kind in ToolErrorKind:
            assert tool_error_kind_from_message(f"[{kind.value}] boom") is kind

    def test_leading_whitespace_is_tolerated(self) -> None:
        assert tool_error_kind_from_message("  [upstream] api broke") is ToolErrorKind.UPSTREAM

    @pytest.mark.parametrize(
        "message",
        ["plain failure", "[weird] unrecognized", "prefix [validation] buried", ""],
    )
    def test_non_prefixed_messages_parse_to_none(self, message: str) -> None:
        # GIVEN a message without a leading recognized kind prefix
        # WHEN parsed THEN no kind is inferred
        assert tool_error_kind_from_message(message) is None


class TestResolveErrorKind:
    def test_drmcputils_tool_error_uses_its_kind_attribute(self) -> None:
        # GIVEN a drmcputils ToolError carrying a structured kind
        error = ToolError("bad input", kind=ToolErrorKind.VALIDATION)
        # WHEN resolved THEN the attribute wins (no message parsing needed)
        assert tool_metrics.resolve_error_kind(error) == "validation"

    def test_fastmcp_error_falls_back_to_message_prefix(self) -> None:
        # GIVEN the FastMCP-facing error log_execution raises (kind only in the message)
        error = FastMCPToolError("[not_found] issue X does not exist")
        # WHEN resolved THEN the prefix is recovered
        assert tool_metrics.resolve_error_kind(error) == "not_found"

    def test_unrecognizable_error_is_unknown(self) -> None:
        # GIVEN an arbitrary exception with no kind information
        # WHEN resolved THEN it lands in the unknown bucket
        assert tool_metrics.resolve_error_kind(RuntimeError("boom")) == "unknown"


# --- metric emission ---------------------------------------------------------------


class TestRecordToolCall:
    def test_success_emits_total_and_duration_no_failure(self) -> None:
        # GIVEN instruments bound to an in-memory reader
        reader, instruments = _reader_and_instruments()

        # WHEN one successful call is recorded
        tool_metrics.record_tool_call("vdb_query", 0.25, None, instruments=instruments)

        # THEN the counter and histogram carry semconv + outcome labels
        total = _points(reader, tool_metrics.TOOL_CALLS_TOTAL)
        assert any(
            p.attributes.get("gen_ai.tool.name") == "vdb_query"
            and p.attributes.get("mcp.method.name") == "tools/call"
            and p.attributes.get("gen_ai.operation.name") == "execute_tool"
            and p.attributes.get("outcome") == "success"
            and p.value == 1
            for p in total
        )
        duration = _points(reader, tool_metrics.MCP_SERVER_OPERATION_DURATION)
        assert any(
            p.attributes.get("gen_ai.tool.name") == "vdb_query"
            and p.attributes.get("mcp.method.name") == "tools/call"
            and "error.type" not in p.attributes
            and p.count == 1
            for p in duration
        )
        # AND no failure counter datapoint exists
        assert _points(reader, tool_metrics.TOOL_CALL_FAILURES) == []

    def test_failure_emits_failure_with_kind_and_type(self) -> None:
        # GIVEN instruments bound to an in-memory reader
        reader, instruments = _reader_and_instruments()
        error = ToolError("api broke", kind=ToolErrorKind.UPSTREAM)

        # WHEN one failing call is recorded
        tool_metrics.record_tool_call("predict", 1.5, error, instruments=instruments)

        # THEN the failure counter carries tool + kind + exception type
        failures = _points(reader, tool_metrics.TOOL_CALL_FAILURES)
        assert any(
            p.attributes.get("gen_ai.tool.name") == "predict"
            and p.attributes.get("error.kind") == "upstream"
            and p.attributes.get("error.type") == "ToolError"
            and p.value == 1
            for p in failures
        )
        # AND duration includes error.type per MCP semconv
        duration = _points(reader, tool_metrics.MCP_SERVER_OPERATION_DURATION)
        assert any(
            p.attributes.get("gen_ai.tool.name") == "predict"
            and p.attributes.get("error.type") == "ToolError"
            for p in duration
        )
        # AND the call still counts in the denominator with outcome=failure
        total = _points(reader, tool_metrics.TOOL_CALLS_TOTAL)
        assert any(p.attributes.get("outcome") == "failure" for p in total)

    def test_get_instruments_never_needs_guarding(self) -> None:
        # GIVEN no bootstrapped SDK provider (OTel serves its no-op provider)
        # WHEN recording through the lazily-built global instruments
        # THEN nothing raises (records land in the no-op provider)
        tool_metrics.record_tool_call("noop_tool", 0.1, None)
        tool_metrics.record_tool_call("noop_tool", 0.1, RuntimeError("x"))

    def test_an_unknown_tool_name_never_becomes_a_label(self) -> None:
        """A caller-invented tool name must not mint a time series.

        ``on_call_tool`` fires before resolution, so looping ``tools/call`` over random
        names would otherwise add a series per name to all three instruments.
        """
        # GIVEN a call naming a tool the server does not have
        reader, instruments = _reader_and_instruments()

        class NotFoundError(Exception):
            """Stands in for FastMCP's NotFoundError, matched by class name."""

        # WHEN the resulting NotFoundError is recorded
        tool_metrics.record_tool_call(
            "does_not_exist_aaa", 0.1, NotFoundError("Unknown tool"), instruments=instruments
        )

        # THEN the invented name is nowhere in the labels, but the call is still counted
        names = {
            p.attributes.get("gen_ai.tool.name")
            for name in (
                tool_metrics.TOOL_CALLS_TOTAL,
                tool_metrics.TOOL_CALL_FAILURES,
                tool_metrics.MCP_SERVER_OPERATION_DURATION,
            )
            for p in _points(reader, name)
        }
        assert names == {tool_metrics.UNKNOWN_TOOL_NAME}

    def test_a_real_tool_failure_keeps_its_name(self) -> None:
        """Only unresolvable names are collapsed — ordinary failures stay attributable."""
        reader, instruments = _reader_and_instruments()

        tool_metrics.record_tool_call("predict", 0.1, RuntimeError("boom"), instruments=instruments)

        failures = _points(reader, tool_metrics.TOOL_CALL_FAILURES)
        assert any(p.attributes.get("gen_ai.tool.name") == "predict" for p in failures)

    def test_recording_never_breaks_the_tool_call(self) -> None:
        """A broken meter must not turn a successful tool call into a client error.

        ``record_tool_call`` runs on the success path *before* the result is returned.
        """
        # GIVEN instruments whose counter raises (a misbehaving exporter/meter)
        broken = tool_metrics.ToolCallInstruments(
            calls_total=Mock(add=Mock(side_effect=RuntimeError("meter exploded"))),
            server_operation_duration=Mock(),
            call_failures=Mock(),
        )

        # WHEN a successful call is recorded through it
        # THEN nothing propagates to the caller
        tool_metrics.record_tool_call("predict", 0.1, None, instruments=broken)


# --- middleware wiring ---------------------------------------------------------------


class TestMiddlewareRecordsMetrics:
    @pytest.mark.asyncio
    async def test_on_call_tool_success_records_one_call(self) -> None:
        # GIVEN the OTel middleware and a tool call that succeeds
        middleware = OpenTelemetryMiddleware()
        context = Mock()
        context.message.name = "vdb_query"
        context.message.arguments = {"q": "hello"}
        call_next = AsyncMock(return_value=Mock())

        # WHEN the call flows through on_call_tool
        with patch("datarobot_genai.drmcp.core.telemetry.record_tool_call") as record:
            await middleware.on_call_tool(context, call_next)

        # THEN exactly one success is recorded with a measured duration
        record.assert_called_once()
        name, duration, error = record.call_args.args
        assert name == "vdb_query"
        assert duration >= 0
        assert error is None

    @pytest.mark.asyncio
    async def test_on_call_tool_failure_records_the_exception_and_reraises(self) -> None:
        # GIVEN the OTel middleware and a tool call that fails
        middleware = OpenTelemetryMiddleware()
        context = Mock()
        context.message.name = "predict"
        context.message.arguments = {}
        boom = FastMCPToolError("[upstream] api broke")
        call_next = AsyncMock(side_effect=boom)

        # WHEN the call flows through on_call_tool
        with patch("datarobot_genai.drmcp.core.telemetry.record_tool_call") as record:
            with pytest.raises(FastMCPToolError):
                await middleware.on_call_tool(context, call_next)

        # THEN the failure is recorded with the original exception
        record.assert_called_once()
        name, duration, error = record.call_args.args
        assert name == "predict"
        assert duration >= 0
        assert error is boom
