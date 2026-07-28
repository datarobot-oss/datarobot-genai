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
import threading
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

_lock = threading.Lock()
# NAT exporter callbacks and memory ops can run in different asyncio tasks.
# Key stacks by workflow_run_id so both sides see the same active NAT span.
_run_parent_stacks: dict[str, list[NatSpanRef]] = {}

# Fallback for unit tests that push/pop without a NAT workflow run id.
_local_parent_stack: ContextVar[list[NatSpanRef]] = ContextVar("nat_local_parent_stack", default=[])

_RUNNER_PATCH_STATE: dict[str, bool] = {"patched": False}


def _seed_nat_root_span_id(context_state: Any, request_span_context: SpanContext) -> None:
    """Link the NAT lifecycle root span under the incoming request span.

    NAT's span exporter creates the workflow root span with no parent unless ``_root_span_id``
    is pre-set (its eager-trace-linking hook). Seeding it with the request span's id (the
    FastAPI server span, itself a child of the caller's bridge span) makes the NAT span tree —
    and the SDK spans that nest under it — join the request trace instead of floating as a
    disconnected root, so the caller (playground/deployment) sees the full agent trace.

    ``request_span_context`` is captured before ``Runner.__aenter__`` restores the workflow
    build-phase context (which would otherwise replace the ambient span with the first
    request's). Only applied when it belongs to this request's trace, and never overrides an id
    already set (e.g. the eval loop's eager linking).
    """
    try:
        if context_state._root_span_id.get() is not None:
            return
        workflow_trace_id = context_state.workflow_trace_id.get()
        if (
            request_span_context.is_valid
            and workflow_trace_id
            and request_span_context.trace_id == workflow_trace_id
        ):
            context_state._root_span_id.set(request_span_context.span_id)
    except Exception:
        logger.debug("Could not seed NAT root span id from the request span", exc_info=True)


def patch_nat_runner_context_isolation() -> None:
    """Make NAT's ``Runner`` respect the incoming request's trace context.

    ``Workflow.__init__`` snapshots ``contextvars.copy_context()`` at (lazy, first-request)
    build time and ``Runner.__aenter__`` re-applies that whole snapshot on every run. That
    causes two tracing defects on a warm server:

    * it overwrites each request's own ``workflow_trace_id`` / ``workflow_run_id`` with the
      first request's, so every later request joins the first request's trace;
    * the NAT lifecycle root span is left with no parent, so the agent span subtree floats
      instead of nesting under the caller's (bridge) span.

    We wrap ``__aenter__`` to (1) restore the per-request trace vars after the snapshot is
    applied and (2) seed ``_root_span_id`` from the request span so the NAT tree links under
    the caller. Idempotent, and a no-op when NAT is unavailable.
    """
    if _RUNNER_PATCH_STATE["patched"]:
        return
    try:
        from nat.runtime.runner import Runner
    except Exception:
        logger.debug(
            "nat.runtime.runner unavailable; skipping context-isolation patch", exc_info=True
        )
        return

    original_aenter = Runner.__aenter__

    async def _aenter_preserving_request_context(self: Any) -> Any:
        context_state = self._context_state
        per_request_vars = (context_state.workflow_trace_id, context_state.workflow_run_id)
        preserved = [(var, var.get()) for var in per_request_vars]
        # Capture the request span before the build-phase context is restored over it.
        request_span_context = trace.get_current_span().get_span_context()
        result = await original_aenter(self)
        for var, value in preserved:
            var.set(value)
        _seed_nat_root_span_id(context_state, request_span_context)
        return result

    Runner.__aenter__ = _aenter_preserving_request_context  # type: ignore[method-assign]
    _RUNNER_PATCH_STATE["patched"] = True
    logger.debug("Patched nat Runner.__aenter__ for per-request trace-context isolation")


def _workflow_run_id_from_nat() -> str | None:
    try:
        from nat.builder.context import Context

        return Context.get().workflow_run_id
    except Exception:
        logger.debug("Unable to read NAT workflow run id", exc_info=True)
        return None


def _stack_for_run(run_id: str | None) -> list[NatSpanRef] | None:
    if not run_id:
        return None
    with _lock:
        return _run_parent_stacks.setdefault(run_id, [])


def push_nat_span_context(
    *,
    trace_id: int,
    span_id: int,
    run_id: str | None = None,
) -> None:
    """Record the active NAT span as the SDK parent for a workflow run."""
    key = run_id or _workflow_run_id_from_nat()
    if key:
        stack = _stack_for_run(key)
        assert stack is not None
        stack.append((trace_id, span_id))
        return

    stack = list(_local_parent_stack.get())
    stack.append((trace_id, span_id))
    _local_parent_stack.set(stack)


def pop_nat_span_context(*, run_id: str | None = None) -> None:
    """Remove one NAT span level for a workflow run."""
    key = run_id or _workflow_run_id_from_nat()
    if key:
        with _lock:
            stack = _run_parent_stacks.get(key)
            if not stack:
                return
            stack.pop()
            if not stack:
                _run_parent_stacks.pop(key, None)
        return

    stack = list(_local_parent_stack.get())
    if not stack:
        return
    stack.pop()
    _local_parent_stack.set(stack)


def reset_nat_span_context(*, run_id: str | None = None) -> None:
    """Clear NAT span levels for one workflow run (or the local fallback stack)."""
    key = run_id or _workflow_run_id_from_nat()
    if key:
        with _lock:
            _run_parent_stacks.pop(key, None)
        return
    _local_parent_stack.set([])


def _stack_for_current_run() -> list[NatSpanRef]:
    key = _workflow_run_id_from_nat()
    if key:
        with _lock:
            return list(_run_parent_stacks.get(key, ()))

    with _lock:
        if len(_run_parent_stacks) == 1:
            return list(next(iter(_run_parent_stacks.values())))

    return list(_local_parent_stack.get())


def _current_nat_parent_span_context() -> SpanContext | None:
    stack = _stack_for_current_run()

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


def _resolve_workflow_parent_context() -> SpanContext | None:
    parent_context = _current_nat_parent_span_context()
    if parent_context is not None:
        return parent_context

    trace_id = _workflow_trace_id_from_nat()
    if trace_id is None:
        return None

    return SpanContext(
        trace_id=trace_id,
        span_id=uuid.uuid4().int >> 64,
        is_remote=True,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    )


@contextmanager
def use_nat_workflow_trace_context() -> Iterator[None]:
    """Ensure SDK spans join the active NAT workflow trace when possible.

    When ``datarobot_otelcollector`` is active it keeps a per-run NAT span stack
    aligned with intermediate steps. This helper attaches that parent (or falls
    back to ``workflow_trace_id``) only for the duration of the caller's scope.
    """
    parent_context = _resolve_workflow_parent_context()
    if parent_context is None:
        yield
        return

    current = trace.get_current_span().get_span_context()
    if current.is_valid and current.trace_id == parent_context.trace_id:
        yield
        return

    token = _attach_span_context(parent_context)
    try:
        yield
    finally:
        _safe_detach(token)
