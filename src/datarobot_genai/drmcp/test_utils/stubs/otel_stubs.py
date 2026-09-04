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

"""REST stubs for DataRobot OTel MCP tools (integration tests).

Chained into :func:`datarobot_genai.drmcp.test_utils.stubs.dr_client_stubs.stub_get`
via the same fall-through-on-``None`` pattern as
:func:`datarobot_genai.drmcp.test_utils.stubs.workload_stubs.workload_stub_get`.

Tests under ``tests/drmcp/unit/otel_tools/`` build their own oversized/small trace
fixtures (``tests/drmcp/unit/otel_tools/trace_factory.py``) because tier-2 stubs
under ``src/`` cannot import from ``tests/``. This module's canned trace is
deliberately small — tier 2 proves registration, the generated JSON schema, and
serialization over the real MCP protocol, not truncation math (that is tier 1's
job, already covered against realistic fixtures). It still carries one
byte-identical duplicate group (``completion`` / ``gen_ai.task.output`` /
``traceloop.entity.output``) so ``otel_trace_get(view="payloads")`` and
``otel_span_payload_get``'s ``dropped_as_duplicate`` reporting exercise a real,
non-trivial path end to end.
"""

from __future__ import annotations

from typing import Any

from datarobot_genai.drmcp.test_utils.stubs.stub_rest_response import StubRestResponse

# Entity used by every otel_* stub route below. entity_type/entity_id are not
# matched against the URL path (mirrors workload_stub_get's own
# id-agnostic routing) except where a test specifically needs "no data for
# this entity" — see STUB_OTEL_EMPTY_ENTITY_ID and _otel_stats_response.
STUB_OTEL_ENTITY_TYPE = "deployment"
STUB_OTEL_ENTITY_ID = "a" * 24  # 24 hex chars, per require_object_id.
STUB_OTEL_EMPTY_ENTITY_ID = "e" * 24  # Valid id, but no OTel data for it (stats only).
STUB_OTEL_TRACE_ID = "b" * 32  # 32 hex chars, per require_trace_id.
STUB_OTEL_SPAN_ID_OK = "span-ok"
STUB_OTEL_SPAN_ID_ERROR = "span-error"
STUB_OTEL_MISSING_SPAN_ID = "span-does-not-exist"

_DUPLICATED_TEXT = "A" * 250  # >= truncation.PAYLOAD_MIN_CHARS (200), so it counts as payload.

_TRACE_SPANS: list[dict[str, Any]] = [
    {
        "span_id": STUB_OTEL_SPAN_ID_OK,
        "parent_span_id": None,
        "name": "root.span",
        "status_code": "OK",
        "status_message": "",
        "kind": "SERVER",
        "service_name": "stub-deployment",
        "duration": 1.2,
        "start_time": 1756000000000,
        "attributes": {"gen_ai.system": "openai"},
    },
    {
        "span_id": STUB_OTEL_SPAN_ID_ERROR,
        "parent_span_id": STUB_OTEL_SPAN_ID_OK,
        "name": "llm.chat",
        "status_code": "ERROR",
        "status_message": "RateLimitError: rate limit exceeded",
        "kind": "CLIENT",
        "service_name": "stub-deployment",
        "duration": 3.4,
        "start_time": 1756000000500,
        "prompt": "stub prompt text",
        "completion": _DUPLICATED_TEXT,
        "attributes": {
            # Byte-identical to 'completion' above: dropped as derived (§3), and
            # reported under dropped_as_duplicate since 'completion' survives.
            "gen_ai.task.output": _DUPLICATED_TEXT,
            "traceloop.entity.output": _DUPLICATED_TEXT,
            "gen_ai.usage.input_tokens": 42,
        },
    },
]

_TRACE_LIST_ROW: dict[str, Any] = {
    "trace_id": STUB_OTEL_TRACE_ID,
    "spans_count": len(_TRACE_SPANS),
    "error_spans_count": 1,
    "duration": 4.6,
    "cost": 0.002,
    "root_span_name": _TRACE_SPANS[0]["name"],
    "root_service_name": _TRACE_SPANS[0]["service_name"],
    "root_user_id": "stub_user_1",
    "timestamp": "2026-08-24T00:00:00Z",
    "prompt": "stub prompt text",
    "completion": _DUPLICATED_TEXT[:80],
    "gen_ai_usage_input_tokens": 120,
    "gen_ai_usage_output_tokens": 45,
    "tools": [{"name": "search", "call_count": 1}],
}

_LOG_LINE: dict[str, Any] = {
    "timestamp": "2026-08-24T00:00:00Z",
    "level": "error",
    "message": "Stub error message",
    "stacktrace": "Traceback (most recent call last):\n  ...\nRateLimitError",
    "span_id": STUB_OTEL_SPAN_ID_ERROR,
    "trace_id": STUB_OTEL_TRACE_ID,
}

_METRIC_CATALOG_ROW: dict[str, Any] = {
    "otel_name": "gen_ai.tokens.total",
    "description": "Total tokens consumed",
    "metric_type": "counter",
    "units": "1",
}

_AUTOCOLLECTED_METRIC_ROW: dict[str, Any] = {
    "otel_name": "cpu.usage",
    "display_name": "CPU Usage",
    "aggregation": "average",
    "level": "pod",
    "unit": "percent",
    "aggregated_value": 12.5,
    "current_value": 14.0,
    "maximumMetricOtelName": "cpu.usage.max",
}

_CONFIGURED_METRIC_ROW: dict[str, Any] = {
    "otel_name": "custom.latency",
    "display_name": "Custom Latency",
    "aggregated_value": 250.0,
    "current_value": 240.0,
    "aggregation": "average",
    "unit": "ms",
    "buckets": None,
}

_ENTITY_STATS_ROWS: list[dict[str, Any]] = [
    {"user_id": "stub_user_1", "span_count": 10, "metric_count": 2, "log_count": 100},
    {"user_id": "stub_user_2", "span_count": 5, "metric_count": 1, "log_count": 20},
]


def _params_dict(params: dict[str, Any] | list[tuple[str, Any]] | None) -> dict[str, Any]:
    if params is None:
        return {}
    if isinstance(params, dict):
        return params
    out: dict[str, Any] = {}
    for key, value in params:
        if key in out:
            existing = out[key]
            out[key] = [existing, value] if not isinstance(existing, list) else [*existing, value]
        else:
            out[key] = value
    return out


def _paginate(items: list[dict[str, Any]], params: dict[str, Any]) -> dict[str, Any]:
    offset = int(params.get("offset", 0))
    limit = int(params.get("limit", 100))
    page = items[offset : offset + limit]
    return {
        "data": page,
        "count": len(page),
        "totalCount": len(items),
        "next": None,
        "previous": None,
    }


def _trace_envelope(spans: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "trace_id": STUB_OTEL_TRACE_ID,
        "span_count": len(spans),
        "duration": 4.6,
        "root_span_name": spans[0]["name"] if spans else None,
        "root_service_name": spans[0]["service_name"] if spans else None,
        "spans": spans,
        "metrics": {"prompt_guards": {}, "response_guards": {}},
        "count": len(spans),
        "offset": 0,
        "limit": 100,
        "total_count": len(spans),
        "next": None,
        "previous": None,
    }


def _otel_traces_response(segments: list[str], p: dict[str, Any]) -> StubRestResponse | None:
    # GET otel/{entityType}/{entityId}/traces/  -> ["otel", et, eid, "traces"]
    if len(segments) == 4 and segments[3] == "traces":
        return StubRestResponse(_paginate([_TRACE_LIST_ROW], p))
    # GET otel/{entityType}/{entityId}/traces/{traceId}/ -> [..., "traces", traceId]
    if len(segments) == 5 and segments[3] == "traces":
        return StubRestResponse(_trace_envelope(_TRACE_SPANS))
    return None


def _otel_logs_response(segments: list[str], p: dict[str, Any]) -> StubRestResponse | None:
    # GET otel/{entityType}/{entityId}/logs/ -> ["otel", et, eid, "logs"]
    if len(segments) == 4 and segments[3] == "logs":
        return StubRestResponse(_paginate([_LOG_LINE], p))
    return None


def _otel_metrics_response(segments: list[str], p: dict[str, Any]) -> StubRestResponse | None:
    # GET otel/{entityType}/{entityId}/metrics/{summary|values|autocollectedValues}/
    if len(segments) != 5 or segments[3] != "metrics":
        return None
    kind = segments[4]
    if kind == "summary":
        return StubRestResponse({"data": [_METRIC_CATALOG_ROW]})
    if kind == "values":
        return StubRestResponse(
            {
                "start_time": "2026-08-24T00:00:00Z",
                "end_time": "2026-08-24T01:00:00Z",
                "metric_aggregations": [_CONFIGURED_METRIC_ROW],
            }
        )
    if kind == "autocollectedValues":
        return StubRestResponse({"data": [_AUTOCOLLECTED_METRIC_ROW]})
    return None


def _otel_stats_response(segments: list[str], p: dict[str, Any]) -> StubRestResponse | None:
    # GET otel/stats/ -> ["otel", "stats"]
    if segments != ["otel", "stats"]:
        return None
    service_name = p.get("serviceName")
    if service_name == f"{STUB_OTEL_ENTITY_TYPE}-{STUB_OTEL_ENTITY_ID}":
        return StubRestResponse({"data": _ENTITY_STATS_ROWS})
    return StubRestResponse({"data": []})


def otel_stub_get(
    url: str, params: dict[str, Any] | list[tuple[str, Any]] | None = None, **kwargs: Any
) -> StubRestResponse | None:
    """Stub for ``rest_client.get()`` REST calls under ``otel/``.

    Returns ``None`` for any URL it does not recognize so
    :func:`datarobot_genai.drmcp.test_utils.stubs.dr_client_stubs.stub_get` can fall
    through to the next handler, matching :func:`workload_stub_get`'s contract.
    """
    del kwargs
    segments = url.rstrip("/").split("/")
    if not segments or segments[0] != "otel":
        return None
    p = _params_dict(params)
    for handler in (
        _otel_traces_response,
        _otel_logs_response,
        _otel_metrics_response,
        _otel_stats_response,
    ):
        response = handler(segments, p)
        if response is not None:
            return response
    return None
