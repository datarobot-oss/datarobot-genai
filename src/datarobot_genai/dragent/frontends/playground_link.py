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

"""Make a deployed agent's trace visible in the Agentic Playground.

The playground walks down from buzok's ``chat_completion_deployment`` bridge span (under
``experiment_container-<use_case_id>``), but the agent's server span parents under an
intermediate hop (buzok's aiohttp client / the prediction router) that is never exported there,
so the agent subtree is unreachable. The agent can't learn the bridge span id, but it knows the
incoming ``traceparent`` parent (that intermediate span) and, via ``DATAROBOT_USE_CASE_ID``, the
use case - so it materializes that missing parent as an ``experiment_container`` span the
playground can walk into. Best-effort: a no-op without a use-case id or ``traceparent``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

_EXPERIMENT_CONTAINER_PREFIX = "experiment_container-"


def _resolve_use_case_id() -> str:
    """Use-case id from ``DATAROBOT_USE_CASE_ID`` (plain env or ``MLOPS_RUNTIME_PARAM_`` form)."""
    if direct := os.getenv("DATAROBOT_USE_CASE_ID", "").strip():
        return direct
    raw = os.getenv("MLOPS_RUNTIME_PARAM_DATAROBOT_USE_CASE_ID")
    if raw:
        try:
            payload = json.loads(raw).get("payload")
        except (ValueError, AttributeError):
            payload = None
        if isinstance(payload, str):
            return payload.strip()
    return ""


def _trace_ids_from_traceparent(traceparent: str | None) -> tuple[int, int] | None:
    """``(trace_id, parent_span_id)`` from a W3C ``traceparent``, or ``None``."""
    parts = (traceparent or "").split("-")
    if len(parts) < 4 or len(parts[1]) != 32 or len(parts[2]) != 16:
        return None
    try:
        trace_id, span_id = int(parts[1], 16), int(parts[2], 16)
    except ValueError:
        return None
    return (trace_id, span_id) if trace_id and span_id else None


def export_experiment_container_link_span(
    *,
    trace_id: int,
    span_id: int,
    use_case_id: str,
    api_key: str,
    start_time_ns: int,
    end_time_ns: int,
) -> bool:
    """Export a root span with the given ids under ``experiment_container-<uc>``.

    This is the agent subtree's missing parent, so the playground can walk into it. Built as a
    ``ReadableSpan`` and handed straight to the exporter (no SDK provider) to control the id.
    """
    from datarobot_genai.core.telemetry.datarobot_otel import resolve_otel_traces_endpoint_from_env

    endpoint = resolve_otel_traces_endpoint_from_env()
    if not (endpoint and use_case_id and api_key):
        return False

    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import ReadableSpan
    from opentelemetry.sdk.util.instrumentation import InstrumentationScope
    from opentelemetry.trace import SpanContext
    from opentelemetry.trace import SpanKind
    from opentelemetry.trace import TraceFlags

    entity_id = f"{_EXPERIMENT_CONTAINER_PREFIX}{use_case_id}"
    span = ReadableSpan(
        name="agent",
        context=SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            is_remote=False,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
        ),
        parent=None,  # root -> an experiment_container entry the playground walks from
        resource=Resource.create({"service.name": entity_id}),
        kind=SpanKind.INTERNAL,
        start_time=start_time_ns,
        end_time=end_time_ns,
        instrumentation_scope=InstrumentationScope("datarobot_genai.playground_link"),
    )
    exporter = OTLPSpanExporter(
        endpoint=endpoint,
        headers={"X-DataRobot-Api-Key": api_key, "X-DataRobot-Entity-Id": entity_id},
    )
    try:
        exporter.export([span])
    finally:
        exporter.shutdown()
    return True


class PlaygroundTraceLinkMiddleware:
    """Link the agent trace into the playground; no-op without a use case + ``traceparent``."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        use_case_id = _resolve_use_case_id() if scope.get("type") == "http" else ""
        headers = (
            {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
            if use_case_id
            else {}
        )
        ids = _trace_ids_from_traceparent(headers.get("traceparent")) if use_case_id else None
        if ids is None:
            await self.app(scope, receive, send)
            return

        trace_id, span_id = ids
        # The caller's forwarded bearer token has use-case access (like buzok's bridge export).
        api_key = (
            headers.get("authorization", "").removeprefix("Bearer ").removeprefix("bearer ").strip()
        )
        api_key = api_key or os.getenv("DATAROBOT_API_TOKEN", "").strip()
        start_time_ns = time.time_ns()
        try:
            await self.app(scope, receive, send)
        finally:
            # Export off the loop (blocking HTTP); response already sent, so no added latency.
            try:
                await asyncio.to_thread(
                    export_experiment_container_link_span,
                    trace_id=trace_id,
                    span_id=span_id,
                    use_case_id=use_case_id,
                    api_key=api_key,
                    start_time_ns=start_time_ns,
                    end_time_ns=time.time_ns(),
                )
            except Exception:
                logger.debug("playground trace-link export failed", exc_info=True)
