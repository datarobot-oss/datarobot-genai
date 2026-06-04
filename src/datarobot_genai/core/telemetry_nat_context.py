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

"""Bridge NAT workflow trace context into the OpenTelemetry SDK context."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import NonRecordingSpan
from opentelemetry.trace import SpanContext
from opentelemetry.trace import TraceFlags

logger = logging.getLogger(__name__)

# Attach tokens for NonRecordingSpan parents pushed by the NAT exporter bridge.
_otel_context_tokens: ContextVar[list[Any]] = ContextVar("nat_otel_context_tokens", default=[])


def push_nat_span_context(*, trace_id: int, span_id: int) -> None:
    """Attach a NonRecordingSpan so SDK spans share NAT's trace and parent."""
    span_context = SpanContext(
        trace_id=trace_id,
        span_id=span_id,
        is_remote=False,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    )
    ctx = trace.set_span_in_context(NonRecordingSpan(span_context))
    token = otel_context.attach(ctx)
    stack = list(_otel_context_tokens.get())
    stack.append(token)
    _otel_context_tokens.set(stack)


def pop_nat_span_context() -> None:
    """Pop one NAT-bridged OTel context level."""
    stack = list(_otel_context_tokens.get())
    if not stack:
        return
    token = stack.pop()
    _otel_context_tokens.set(stack)
    otel_context.detach(token)


def reset_nat_span_context() -> None:
    """Detach any remaining bridged OTel context levels."""
    while _otel_context_tokens.get():
        pop_nat_span_context()


def _workflow_trace_id_from_nat() -> int | None:
    try:
        from nat.builder.context import Context

        return Context.get().workflow_trace_id
    except Exception:
        logger.debug("Unable to read NAT workflow trace id", exc_info=True)
        return None


@contextmanager
def use_nat_workflow_trace_context() -> Iterator[None]:
    """Ensure SDK spans join the active NAT workflow trace when possible.

    When ``datarobot_otelcollector`` is active it keeps the OTel SDK context
    aligned with NAT intermediate steps. This fallback covers the gap when
    only ``instrument()`` is configured or the exporter bridge is unavailable.
    """
    current = trace.get_current_span()
    if current.get_span_context().is_valid:
        yield
        return

    trace_id = _workflow_trace_id_from_nat()
    if trace_id is None:
        yield
        return

    span_context = SpanContext(
        trace_id=trace_id,
        span_id=uuid.uuid4().int >> 64,
        is_remote=True,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    )
    token = otel_context.attach(trace.set_span_in_context(NonRecordingSpan(span_context)))
    try:
        yield
    finally:
        otel_context.detach(token)
