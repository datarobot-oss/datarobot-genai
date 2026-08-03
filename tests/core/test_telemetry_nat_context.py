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

from __future__ import annotations

import uuid
from contextvars import ContextVar

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import INVALID_SPAN_CONTEXT
from opentelemetry.util._once import Once

from datarobot_genai.core.telemetry import memory
from datarobot_genai.core.telemetry import nat_context


@pytest.fixture
def memory_span_exporter(monkeypatch: pytest.MonkeyPatch) -> InMemorySpanExporter:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr("opentelemetry.trace._TRACER_PROVIDER", None)
    monkeypatch.setattr("opentelemetry.trace._TRACER_PROVIDER_SET_ONCE", Once())
    trace.set_tracer_provider(provider)
    monkeypatch.setattr(
        memory,
        "_tracer",
        trace.get_tracer("test.telemetry_nat_context"),
    )
    yield exporter
    exporter.clear()


def test_push_and_pop_nat_span_context_nests_sdk_spans(
    memory_span_exporter: InMemorySpanExporter,
) -> None:
    trace_id = uuid.uuid4().int
    parent_span_id = uuid.uuid4().int >> 64
    child_span_id = uuid.uuid4().int >> 64

    nat_context.push_nat_span_context(trace_id=trace_id, span_id=parent_span_id)
    nat_context.push_nat_span_context(trace_id=trace_id, span_id=child_span_id)
    with nat_context.use_nat_workflow_trace_context():
        with trace.get_tracer("test").start_as_current_span("search_memory"):
            pass
    nat_context.pop_nat_span_context()
    nat_context.pop_nat_span_context()

    span = memory_span_exporter.get_finished_spans()[0]
    assert span.context.trace_id == trace_id
    assert span.parent.span_id == child_span_id


def test_pop_without_push_is_safe() -> None:
    nat_context.pop_nat_span_context()
    nat_context.reset_nat_span_context()


def test_reset_nat_span_context_clears_stack(
    memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_id = uuid.uuid4().int
    parent_span_id = uuid.uuid4().int >> 64
    monkeypatch.setattr(
        nat_context,
        "_workflow_trace_id_from_nat",
        lambda: None,
    )
    nat_context.push_nat_span_context(trace_id=trace_id, span_id=parent_span_id)
    nat_context.reset_nat_span_context()

    with memory.trace_memory_operation("search_memory", store_name="mem0"):
        pass

    span = memory_span_exporter.get_finished_spans()[0]
    assert span.context.trace_id != trace_id


def test_use_nat_workflow_trace_context_joins_workflow_trace(
    memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow_trace_id = uuid.uuid4().int
    monkeypatch.setattr(
        nat_context,
        "_workflow_trace_id_from_nat",
        lambda: workflow_trace_id,
    )

    with nat_context.use_nat_workflow_trace_context():
        with trace.get_tracer("test").start_as_current_span("update_memory"):
            pass

    span = memory_span_exporter.get_finished_spans()[0]
    assert span.context.trace_id == workflow_trace_id


def test_trace_memory_operation_uses_existing_nat_context(
    memory_span_exporter: InMemorySpanExporter,
) -> None:
    trace_id = uuid.uuid4().int
    parent_span_id = uuid.uuid4().int >> 64

    nat_context.push_nat_span_context(trace_id=trace_id, span_id=parent_span_id)
    try:
        with memory.trace_memory_operation("delete_memory", store_name="mem0"):
            pass
    finally:
        nat_context.pop_nat_span_context()

    span = memory_span_exporter.get_finished_spans()[0]
    assert span.context.trace_id == trace_id
    assert span.parent.span_id == parent_span_id


def test_trace_memory_operation_fallback_uses_workflow_trace_id(
    memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow_trace_id = uuid.uuid4().int
    monkeypatch.setattr(
        nat_context,
        "_workflow_trace_id_from_nat",
        lambda: workflow_trace_id,
    )

    with memory.trace_memory_operation("search_memory", store_name="mem0"):
        pass

    span = memory_span_exporter.get_finished_spans()[0]
    assert span.context.trace_id == workflow_trace_id


def test_single_active_run_fallback_without_nat_context(
    memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "run-only"
    trace_id = uuid.uuid4().int
    parent_span_id = uuid.uuid4().int >> 64
    monkeypatch.setattr(
        nat_context,
        "_workflow_run_id_from_nat",
        lambda: None,
    )
    monkeypatch.setattr(
        nat_context,
        "_workflow_trace_id_from_nat",
        lambda: None,
    )

    nat_context.push_nat_span_context(
        trace_id=trace_id,
        span_id=parent_span_id,
        run_id=run_id,
    )
    try:
        with memory.trace_memory_operation("search_memory", store_name="mem0"):
            pass
    finally:
        nat_context.pop_nat_span_context(run_id=run_id)

    span = memory_span_exporter.get_finished_spans()[0]
    assert span.context.trace_id == trace_id
    assert span.parent.span_id == parent_span_id


def test_run_id_stack_is_visible_from_memory_context(
    memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "run-123"
    trace_id = uuid.uuid4().int
    parent_span_id = uuid.uuid4().int >> 64
    monkeypatch.setattr(
        nat_context,
        "_workflow_run_id_from_nat",
        lambda: run_id,
    )
    monkeypatch.setattr(
        nat_context,
        "_workflow_trace_id_from_nat",
        lambda: None,
    )

    nat_context.push_nat_span_context(
        trace_id=trace_id,
        span_id=parent_span_id,
        run_id=run_id,
    )
    try:
        with memory.trace_memory_operation("delete_memory", store_name="mem0"):
            pass
    finally:
        nat_context.pop_nat_span_context(run_id=run_id)

    span = memory_span_exporter.get_finished_spans()[0]
    assert span.context.trace_id == trace_id
    assert span.parent.span_id == parent_span_id


def test_use_nat_workflow_trace_context_overrides_unrelated_active_span(
    memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow_trace_id = uuid.uuid4().int
    monkeypatch.setattr(
        nat_context,
        "_workflow_trace_id_from_nat",
        lambda: workflow_trace_id,
    )

    with trace.get_tracer("other").start_as_current_span("framework-span"):
        with nat_context.use_nat_workflow_trace_context():
            with trace.get_tracer("test").start_as_current_span("search_memory"):
                pass

    spans = {span.name: span for span in memory_span_exporter.get_finished_spans()}
    span = spans["search_memory"]
    assert span.context.trace_id == workflow_trace_id


class _FakeContextState:
    """Minimal stand-in for NAT's ContextState: only the vars the patch re-pins."""

    def __init__(self) -> None:
        self.workflow_trace_id: ContextVar[int | None] = ContextVar("wtid", default=None)
        self.workflow_run_id: ContextVar[str | None] = ContextVar("wrid", default=None)
        self._root_span_id: ContextVar[int | None] = ContextVar("rsid", default=None)


def test_reassert_repins_request_trace_and_seeds_root(
    memory_span_exporter: InMemorySpanExporter,
) -> None:
    leaked_trace_id = uuid.uuid4().int
    context_state = _FakeContextState()
    context_state.workflow_trace_id.set(leaked_trace_id)  # NAT restored the build snapshot
    context_state.workflow_run_id.set("build-run")

    with trace.get_tracer("test").start_as_current_span("POST /chat/completions") as span:
        span_context = span.get_span_context()

    nat_context._reassert_request_trace_context(context_state, span_context, "request-run")

    # Re-pinned to this request's span, not the leaked build snapshot.
    assert context_state.workflow_trace_id.get() == span_context.trace_id
    assert context_state.workflow_trace_id.get() != leaked_trace_id
    # NAT root nests under the request span.
    assert context_state._root_span_id.get() == span_context.span_id
    # This request's run id survives the snapshot restore.
    assert context_state.workflow_run_id.get() == "request-run"


def test_reassert_keeps_preset_root_span_id(
    memory_span_exporter: InMemorySpanExporter,
) -> None:
    preset_root_span_id = uuid.uuid4().int >> 64
    context_state = _FakeContextState()
    context_state._root_span_id.set(preset_root_span_id)

    with trace.get_tracer("test").start_as_current_span("POST /chat/completions") as span:
        span_context = span.get_span_context()

    nat_context._reassert_request_trace_context(context_state, span_context, "request-run")

    # An already-set root span id (e.g. the eval loop's eager link) is not overwritten.
    assert context_state._root_span_id.get() == preset_root_span_id


def test_reassert_ignores_invalid_span() -> None:
    leaked_trace_id = uuid.uuid4().int
    context_state = _FakeContextState()
    context_state.workflow_trace_id.set(leaked_trace_id)

    nat_context._reassert_request_trace_context(context_state, INVALID_SPAN_CONTEXT, "request-run")

    # No valid request span: leave NAT's trace/root alone, but still restore run_id.
    assert context_state.workflow_trace_id.get() == leaked_trace_id
    assert context_state._root_span_id.get() is None
    assert context_state.workflow_run_id.get() == "request-run"
