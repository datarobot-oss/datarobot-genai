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

"""Fetch and pretty-print DataRobot OTEL traces for a deployed MCP server.

DataRobot exposes the OTEL telemetry a deployed MCP server emits through the
public API:

- ``GET /api/v2/otel/deployment/{id}/traces/`` — trace summaries (root span,
  duration, span/error counts, tools called).
- ``GET /api/v2/otel/deployment/{id}/traces/{traceId}/`` — the full span tree.

These helpers power the ``traces`` / ``trace <id>`` commands in
:mod:`datarobot_genai.drmcp.test_utils.test_interactive`, and are importable on
their own for scripts that want the same view (e.g. e2e demos showing how long
each tool call / sandbox execution took).

Units: trace-summary ``duration`` is milliseconds; span ``duration`` is
nanoseconds; ``timestamp``/``startTime`` are epoch milliseconds.
"""

from __future__ import annotations

import datetime
import os
import re
from typing import Any

import httpx

# directAccess URL of a deployed MCP server, e.g.
# https://staging.datarobot.com/api/v2/deployments/<24-hex>/directAccess/mcp
_DEPLOYMENT_URL_RE = re.compile(r"/deployments/([a-f0-9]{24})/directAccess", re.IGNORECASE)

_TRACES_PAGE_LIMIT = 100


def deployment_id_from_url(url: str | None) -> str | None:
    """Extract the deployment id from a directAccess MCP URL, else ``None``."""
    if not url:
        return None
    match = _DEPLOYMENT_URL_RE.search(url)
    return match.group(1) if match else None


def _api_base(datarobot_endpoint: str | None = None) -> str:
    endpoint = datarobot_endpoint or os.environ.get("DATAROBOT_ENDPOINT", "")
    return endpoint.rstrip("/")


def _headers(datarobot_api_token: str | None = None) -> dict[str, str]:
    token = datarobot_api_token or os.environ.get("DATAROBOT_API_TOKEN", "")
    return {"Authorization": f"Bearer {token}"}


def fetch_traces(
    deployment_id: str,
    *,
    limit: int = 10,
    datarobot_endpoint: str | None = None,
    datarobot_api_token: str | None = None,
) -> list[dict[str, Any]]:
    """Return the most recent ``limit`` trace summaries for a deployment."""
    limit = min(max(limit, 1), _TRACES_PAGE_LIMIT)
    url = f"{_api_base(datarobot_endpoint)}/otel/deployment/{deployment_id}/traces/"
    response = httpx.get(
        url,
        params={"limit": limit},
        headers=_headers(datarobot_api_token),
        timeout=30,
    )
    response.raise_for_status()
    data: list[dict[str, Any]] = response.json().get("data") or []
    return data


def fetch_trace(
    deployment_id: str,
    trace_id: str,
    *,
    datarobot_endpoint: str | None = None,
    datarobot_api_token: str | None = None,
) -> dict[str, Any]:
    """Return the full span tree for one trace."""
    url = f"{_api_base(datarobot_endpoint)}/otel/deployment/{deployment_id}/traces/{trace_id}/"
    response = httpx.get(url, headers=_headers(datarobot_api_token), timeout=30)
    response.raise_for_status()
    body: dict[str, Any] = response.json()
    return body


def _fmt_ms(duration_ms: float) -> str:
    if duration_ms >= 1000:
        return f"{duration_ms / 1000:.1f}s"
    return f"{duration_ms:.0f}ms"


def _fmt_epoch_ms(epoch_ms: float) -> str:
    dt = datetime.datetime.fromtimestamp(epoch_ms / 1000, tz=datetime.UTC)
    return dt.strftime("%H:%M:%S")


def format_traces_table(traces: list[dict[str, Any]]) -> str:
    """Render trace summaries as an aligned, human-readable table."""
    if not traces:
        return "(no traces recorded for this deployment yet)"
    rows = [("TIME(UTC)", "TRACE", "ROOT SPAN", "DURATION", "SPANS", "ERR", "TOOLS")]
    for t in traces:
        tools = ", ".join(
            f"{tool.get('name')}×{tool.get('callCount')}"
            if (tool.get("callCount") or 1) > 1
            else str(tool.get("name"))
            for tool in t.get("tools") or []
        )
        rows.append(
            (
                _fmt_epoch_ms(t.get("timestamp") or 0),
                str(t.get("traceId", ""))[:12],
                str(t.get("rootSpanName", "")),
                _fmt_ms(t.get("duration") or 0.0),
                str(t.get("spansCount", "")),
                str(t.get("errorSpansCount", "")),
                tools,
            )
        )
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return "\n".join(
        "  ".join(cell.ljust(w) for cell, w in zip(row, widths)).rstrip() for row in rows
    )


def format_trace_tree(trace: dict[str, Any]) -> str:
    """Render one trace's spans as an indented tree with durations and status."""
    spans = trace.get("spans") or []
    if not spans:
        return "(trace has no spans)"

    children: dict[str | None, list[dict[str, Any]]] = {}
    for span in spans:
        children.setdefault(span.get("parentSpanId"), []).append(span)
    for siblings in children.values():
        siblings.sort(key=lambda s: s.get("startTime") or 0)

    span_ids = {s.get("spanId") for s in spans}
    # Roots: no parent, or parent outside this trace (e.g. client-side span).
    roots = [
        s for pid, group in children.items() if pid is None or pid not in span_ids for s in group
    ]

    lines = [f"trace {trace.get('traceId', '')} ({len(spans)} spans)"]

    def _walk(span: dict[str, Any], depth: int) -> None:
        duration_ns = span.get("duration") or 0.0
        status = str(span.get("statusCode") or "")
        marker = "✗" if status.lower() == "error" else "✓"
        line = f"{'  ' * depth}{marker} {span.get('name')}  [{_fmt_ms(duration_ns / 1e6)}]"
        attrs = span.get("attributes") or {}
        tool_name = attrs.get("mcp.tool.name") or attrs.get("gen_ai.tool.name")
        if tool_name:
            line += f"  tool={tool_name}"
        lines.append(line)
        message = span.get("statusMessage")
        if message:
            lines.append(f"{'  ' * (depth + 1)}└ {str(message)[:160]}")
        for child in children.get(span.get("spanId"), []):
            _walk(child, depth + 1)

    for root in roots:
        _walk(root, 0)
    return "\n".join(lines)
