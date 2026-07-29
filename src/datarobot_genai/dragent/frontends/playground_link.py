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

The playground renders a chat's trace by walking *down* from buzok's
``chat_completion_deployment`` bridge span, which lives under
``experiment_container-<use_case_id>``. Buzok injects a W3C ``traceparent`` when it calls the
deployment, but the outbound hop (buzok's aiohttp client / the prediction router) creates its
own span and re-injects *its* span id as the parent. That intermediate span is never exported
to ``experiment_container``, so the agent's server span parents under a phantom and the whole
agent subtree is unreachable from the bridge span - the playground shows only the bridge span.

The agent never receives the use-case id (buzok strips ``tracing_context`` from the body) or
the bridge span id (the ``traceparent`` parent was overwritten by the phantom). So the only
thing the agent can do on its own is materialize that missing parent: given the use-case id
(passed by the app as ``DATAROBOT_USE_CASE_ID``) it exports a single span whose id equals the
incoming ``traceparent`` parent, tagged ``experiment_container-<use_case_id>``. That span
becomes an ``experiment_container`` entry the playground walks from, so the agent subtree
(which already parents under that id) renders as the chat's trace.

This is additive and best-effort: no use-case id, no ``traceparent``, or no export target ->
no-op, and any failure is swallowed so it never affects the response.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

_EXPERIMENT_CONTAINER_ENTITY_ID_PREFIX = "experiment_container-"


def _resolve_use_case_id() -> str:
    """Use-case id the deployed agent runs under, or ``""``.

    Prefers a plain ``DATAROBOT_USE_CASE_ID`` env var; falls back to the DataRobot custom-model
    runtime parameter form (``MLOPS_RUNTIME_PARAM_DATAROBOT_USE_CASE_ID`` holds
    ``{"type": ..., "payload": <value>}``), so the app only has to declare the runtime param.
    """
    direct = os.getenv("DATAROBOT_USE_CASE_ID", "").strip()
    if direct:
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
    """Return ``(trace_id, parent_span_id)`` from a W3C ``traceparent``, or ``None``."""
    if not traceparent:
        return None
    parts = traceparent.split("-")
    if len(parts) < 4 or len(parts[1]) != 32 or len(parts[2]) != 16:
        return None
    try:
        trace_id = int(parts[1], 16)
        span_id = int(parts[2], 16)
    except ValueError:
        return None
    if not trace_id or not span_id:
        return None
    return trace_id, span_id


def _api_key_from_authorization(authorization: str | None) -> str:
    # Use the caller's forwarded bearer token (it has use-case access, like buzok's bridge
    # span export); fall back to the deployment's token.
    if authorization:
        token = authorization.removeprefix("Bearer ").removeprefix("bearer ").strip()
        if token:
            return token
    return os.getenv("DATAROBOT_API_TOKEN", "").strip()


def export_experiment_container_link_span(
    *,
    trace_id: int,
    span_id: int,
    use_case_id: str,
    api_key: str,
    start_time_ns: int,
    end_time_ns: int,
) -> bool:
    """Export one span (``trace_id``/``span_id`` forced) under ``experiment_container-<uc>``.

    Materializes the missing parent of the agent's server span so the playground's
    subtree walk reaches the agent trace. Returns ``True`` when a span was exported.
    """
    from datarobot_genai.core.telemetry.datarobot_otel import resolve_otel_traces_endpoint_from_env

    endpoint = resolve_otel_traces_endpoint_from_env()
    if not (endpoint and use_case_id and api_key):
        return False

    from opentelemetry.context import Context as OTelContext
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.id_generator import IdGenerator

    entity_id = f"{_EXPERIMENT_CONTAINER_ENTITY_ID_PREFIX}{use_case_id}"

    class _FixedIdGenerator(IdGenerator):
        def generate_span_id(self) -> int:
            return span_id

        def generate_trace_id(self) -> int:
            return trace_id

    resource = Resource.create({"service.name": entity_id})
    provider = TracerProvider(resource=resource, id_generator=_FixedIdGenerator())
    exporter = OTLPSpanExporter(
        endpoint=endpoint,
        headers={"X-DataRobot-Api-Key": api_key, "X-DataRobot-Entity-Id": entity_id},
    )
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    try:
        # Empty context -> the span is a trace root (no parent), so the playground treats it as
        # an experiment_container entry and walks its children (the agent subtree).
        span = provider.get_tracer("datarobot_genai.playground_link").start_span(
            "agent",
            context=OTelContext(),
            start_time=start_time_ns,
        )
        span.end(end_time=end_time_ns)
    finally:
        provider.shutdown()
    return True


class PlaygroundTraceLinkMiddleware:
    """ASGI middleware that links the agent trace into the Agentic Playground per request.

    No-op unless ``DATAROBOT_USE_CASE_ID`` is set and the request carries a ``traceparent``.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        use_case_id = _resolve_use_case_id()
        headers = {
            k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])
        }
        ids = _trace_ids_from_traceparent(headers.get("traceparent")) if use_case_id else None

        if ids is None:
            await self.app(scope, receive, send)
            return

        trace_id, span_id = ids
        api_key = _api_key_from_authorization(headers.get("authorization"))
        start_time_ns = time.time_ns()
        try:
            await self.app(scope, receive, send)
        finally:
            # Export off the event loop: it opens an HTTP session and blocks. The response is
            # already sent by this point, so this only extends the request task, not latency.
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
                logger.debug("Failed to export playground trace-link span", exc_info=True)
