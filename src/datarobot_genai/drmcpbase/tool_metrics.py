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

"""OTel metrics for MCP tool calls (shared by User MCP and Global MCP servers).

Until now tool calls were observable through spans only — per-tool call
counts, latency, and failure taxonomy had to be re-derived by scanning
traces. This module adds SLI instruments (mirroring the sandbox convention in
``drtools/core/sandbox/observability.py``) plus the OTel-recommended server
operation duration histogram:

- ``mcp.server.operation.duration``     histogram (OTel GenAI MCP semconv)
- ``datarobot.mcp.tool.calls_total``    counter, by tool + outcome
- ``datarobot.mcp.tool.call_failure_total`` counter, by tool + error.kind + error.type

Latency follows `OpenTelemetry GenAI MCP semantic conventions
<https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/mcp.md>`_
(``mcp.server.operation.duration`` with ``mcp.method.name=tools/call``,
``gen_ai.tool.name``, ``gen_ai.operation.name=execute_tool``, and ``error.type``
on failure). MCP attribute registry:
https://opentelemetry.io/docs/specs/semconv/registry/attributes/mcp/

``error.kind`` on the failure counter carries the
:class:`~datarobot_genai.drmcputils.exceptions.ToolErrorKind` value. The
middleware sees failures AFTER ``log_execution`` converted them to FastMCP
``ToolError`` strings, so the kind is recovered from the exception's ``kind``
attribute when present, else parsed from the ``"[kind] …"`` message prefix, else
recorded as ``"unknown"``.

Metrics ship through the OTLP MeterProvider ``bootstrap_metrics_provider`` installs
during ``initialize_telemetry``; without OTLP env it resolves to a no-op provider. That
makes recording cheap, not infallible, so :func:`record_tool_call` swallows its own
errors — it runs on the success path before the tool's result is returned.
"""

import logging
from dataclasses import dataclass
from typing import Any

from opentelemetry import metrics

from datarobot_genai.drmcputils.exceptions import ToolErrorKind
from datarobot_genai.drmcputils.exceptions import tool_error_kind_from_message

logger = logging.getLogger(__name__)

# Dotted, package-namespaced meter name.
METER_NAME = "datarobot_genai.mcp"

# Stand-in for a tools/call naming a tool this server does not have — see
# _label_safe_tool_name. FastMCP raises NotFoundError before resolution, and the class is
# matched by name to keep this module independent of the FastMCP exception hierarchy.
UNKNOWN_TOOL_NAME = "<unknown>"
_NOT_FOUND_ERROR = "NotFoundError"

# OTel GenAI MCP semconv (server-side operation latency).
MCP_SERVER_OPERATION_DURATION = "mcp.server.operation.duration"

# DataRobot SLI counters (not part of upstream MCP semconv; mirror sandbox.* shape).
TOOL_CALLS_TOTAL = "datarobot.mcp.tool.calls_total"
TOOL_CALL_FAILURES = "datarobot.mcp.tool.call_failure_total"

MCP_METHOD_TOOLS_CALL = "tools/call"
GEN_AI_OPERATION_EXECUTE_TOOL = "execute_tool"

OUTCOME_SUCCESS = "success"
OUTCOME_FAILURE = "failure"

# Failure bucket for exceptions carrying no recognizable ToolErrorKind.
ERROR_KIND_UNKNOWN = "unknown"

# OTel advisory boundaries for mcp.server.operation.duration (semconv mcp.md).
_MCP_SERVER_OPERATION_DURATION_BUCKETS = [
    0.01,
    0.02,
    0.05,
    0.1,
    0.2,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    30.0,
    60.0,
    120.0,
    300.0,
]


@dataclass
class ToolCallInstruments:
    """OTel instruments backing per-tool MCP SLIs and semconv latency."""

    calls_total: Any  # Counter: denominator + success-rate numerator
    server_operation_duration: Any  # Histogram (seconds): mcp.server.operation.duration
    call_failures: Any  # Counter: failure-taxonomy SLI (by kind + type)


def _tool_call_base_attributes(tool_name: str) -> dict[str, str]:
    return {
        "mcp.method.name": MCP_METHOD_TOOLS_CALL,
        "gen_ai.tool.name": tool_name,
        "gen_ai.operation.name": GEN_AI_OPERATION_EXECUTE_TOOL,
    }


def build_instruments(meter: Any) -> ToolCallInstruments:
    """Create the tool-call instruments on ``meter`` (an OTel ``Meter``)."""
    return ToolCallInstruments(
        calls_total=meter.create_counter(
            TOOL_CALLS_TOTAL,
            unit="1",
            description=(
                "MCP tool calls, labeled by gen_ai.tool.name and outcome (success|failure)."
            ),
        ),
        server_operation_duration=meter.create_histogram(
            MCP_SERVER_OPERATION_DURATION,
            unit="s",
            description=(
                "MCP server operation duration for tools/call "
                "(mcp.method.name, gen_ai.tool.name, gen_ai.operation.name, error.type)."
            ),
            explicit_bucket_boundaries_advisory=_MCP_SERVER_OPERATION_DURATION_BUCKETS,
        ),
        call_failures=meter.create_counter(
            TOOL_CALL_FAILURES,
            unit="1",
            description=(
                "MCP tool call failures, labeled by gen_ai.tool.name, error.kind "
                "(ToolErrorKind value or 'unknown'), and error.type (exception class)."
            ),
        ),
    )


# Lazily-initialized singletons keyed on the active global MeterProvider (the
# same pattern as the sandbox instruments): if the real SDK provider replaces
# the no-op/proxy provider after first use, instruments are rebuilt against it.
_STATE: dict[str, Any] = {"instruments": None, "provider": None}


def get_instruments() -> ToolCallInstruments:
    """Build (and cache) instruments from the global meter."""
    provider = metrics.get_meter_provider()
    if _STATE["instruments"] is None or _STATE["provider"] is not provider:
        _STATE["instruments"] = build_instruments(metrics.get_meter(METER_NAME))
        _STATE["provider"] = provider
    return _STATE["instruments"]


def _label_safe_tool_name(tool_name: str, error: BaseException | None) -> str:
    """Substitute a sentinel for tool names that never existed.

    ``on_call_tool`` fires before the tool is resolved, so an unknown name still reaches
    the recorder and would become a label value. Every label value is a new time series,
    and the name here is caller-supplied: a client looping ``tools/call`` over random
    names could mint unbounded series on all three instruments. A ``NotFoundError`` means
    the name matched nothing this server registers, so there is nothing to attribute the
    call to — the sentinel keeps the failure counted without the cardinality.
    """
    if error is not None and type(error).__name__ == _NOT_FOUND_ERROR:
        return UNKNOWN_TOOL_NAME
    return tool_name


def resolve_error_kind(error: BaseException) -> str:
    """Resolve the ``error.kind`` label value for a tool failure.

    Prefers the exception's own ``kind`` (drmcputils ``ToolError``); falls
    back to the ``"[kind] …"`` message prefix ``log_execution`` puts on the
    FastMCP-facing error; else ``"unknown"``.
    """
    kind = getattr(error, "kind", None)
    if isinstance(kind, ToolErrorKind):
        return kind.value
    parsed = tool_error_kind_from_message(str(error))
    if parsed is not None:
        return parsed.value
    return ERROR_KIND_UNKNOWN


def record_tool_call(
    tool_name: str,
    duration_s: float,
    error: BaseException | None,
    *,
    instruments: ToolCallInstruments | None = None,
) -> None:
    """Emit the SLI and semconv metrics for one MCP tools/call.

    Never raises. The success path calls this *before* returning the tool's result, so an
    exception escaping here would turn a tool call that worked into a client-visible error
    — telemetry is not worth that. "No-op without OTLP env" is not the same guarantee: it
    only holds in the configuration where nothing is recorded at all.
    """
    try:
        _record_tool_call(tool_name, duration_s, error, instruments=instruments)
    except Exception:
        logger.debug("Failed to record MCP tool-call metrics for %r", tool_name, exc_info=True)


def _record_tool_call(
    tool_name: str,
    duration_s: float,
    error: BaseException | None,
    *,
    instruments: ToolCallInstruments | None = None,
) -> None:
    instruments = instruments or get_instruments()
    outcome = OUTCOME_SUCCESS if error is None else OUTCOME_FAILURE
    base_attrs = _tool_call_base_attributes(_label_safe_tool_name(tool_name, error))

    instruments.calls_total.add(
        1,
        {**base_attrs, "outcome": outcome},
    )

    duration_attrs = dict(base_attrs)
    if error is not None:
        duration_attrs["error.type"] = type(error).__name__
    instruments.server_operation_duration.record(duration_s, duration_attrs)

    if error is not None:
        instruments.call_failures.add(
            1,
            {
                **base_attrs,
                "error.kind": resolve_error_kind(error),
                "error.type": type(error).__name__,
            },
        )
