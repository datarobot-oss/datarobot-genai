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

NatSpanRef = tuple[int, int]

# Track NAT span hierarchy without mutating OTel context in the exporter callback.
# SDK spans attach this parent locally inside use_nat_workflow_trace_context().
_nat_parent_stack: ContextVar[list[NatSpanRef]] = ContextVar("nat_parent_stack", default=[])


def push_nat_span_context(*, trace_id: int, span_id: int) -> None:
    """Record the active NAT span as the SDK parent for this execution context."""
    stack = list(_nat_parent_stack.get())
    stack.append((trace_id, span_id))
    _nat_parent_stack.set(stack)


def pop_nat_span_context() -> None:
    """Remove one NAT span level from the current execution context."""
    stack = list(_nat_parent_stack.get())
    if not stack:
        return
    stack.pop()
    _nat_parent_stack.set(stack)


def reset_nat_span_context() -> None:
    """Clear any remaining NAT span levels for the current execution context."""
    _nat_parent_stack.set([])


def _current_nat_parent_span_context() -> SpanContext | None:
    stack = _nat_parent_stack.get()
    if not stack:
        return None
    trace_id, span_id = stack[-1]
    return SpanContext(
        trace_id=trace_id,
        span_id=span_id,
        is_remote=False,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    )


def _safe_detach(token: object) -> None:
    try:
        otel_context.detach(token)
    except ValueError:
        logger.debug(
            "Skipping OTel context detach for token from a different execution context",
            exc_info=True,
        )


def _workflow_trace_id_from_nat() -> int | None:
    try:
        from nat.builder.context import Context

        return Context.get().workflow_trace_id
    except Exception:
        logger.debug("Unable to read NAT workflow trace id", exc_info=True)
        return None


def _attach_span_context(span_context: SpanContext) -> Any:
    return otel_context.attach(trace.set_span_in_context(NonRecordingSpan(span_context)))


@contextmanager
def use_nat_workflow_trace_context() -> Iterator[None]:
    """Ensure SDK spans join the active NAT workflow trace when possible.

    When ``datarobot_otelcollector`` is active it keeps a NAT span stack aligned
    with intermediate steps. This helper attaches that parent (or falls back to
    ``workflow_trace_id``) only for the duration of the caller's scope so OTel
    context tokens are not leaked across async boundaries.
    """
    current = trace.get_current_span()
    if current.get_span_context().is_valid:
        yield
        return

    parent_context = _current_nat_parent_span_context()
    if parent_context is not None:
        token = _attach_span_context(parent_context)
        try:
            yield
        finally:
            _safe_detach(token)
        return

    trace_id = _workflow_trace_id_from_nat()
    if trace_id is None:
        yield
        return

    fallback_context = SpanContext(
        trace_id=trace_id,
        span_id=uuid.uuid4().int >> 64,
        is_remote=True,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    )
    token = _attach_span_context(fallback_context)
    try:
        yield
    finally:
        _safe_detach(token)
