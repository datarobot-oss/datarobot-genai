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

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util._once import Once

from datarobot_genai.core import telemetry_nat_context
from datarobot_genai.core.telemetry_nat_tracer import wrap_sdk_tracer_provider


@pytest.fixture
def memory_span_exporter(monkeypatch: pytest.MonkeyPatch) -> InMemorySpanExporter:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    wrap_sdk_tracer_provider(provider)
    monkeypatch.setattr("opentelemetry.trace._TRACER_PROVIDER", None)
    monkeypatch.setattr("opentelemetry.trace._TRACER_PROVIDER_SET_ONCE", Once())
    trace.set_tracer_provider(provider)
    yield exporter
    exporter.clear()


def test_wrapped_tracer_joins_pushed_nat_parent(memory_span_exporter: InMemorySpanExporter) -> None:
    trace_id = uuid.uuid4().int
    parent_span_id = uuid.uuid4().int >> 64

    telemetry_nat_context.push_nat_span_context(trace_id=trace_id, span_id=parent_span_id)
    try:
        with trace.get_tracer("test.framework").start_as_current_span("invoke_agent LangGraph"):
            pass
    finally:
        telemetry_nat_context.pop_nat_span_context()

    span = memory_span_exporter.get_finished_spans()[0]
    assert span.name == "invoke_agent LangGraph"
    assert span.context.trace_id == trace_id
    assert span.parent.span_id == parent_span_id


def test_wrapped_tracer_start_span_joins_nat_parent(
    memory_span_exporter: InMemorySpanExporter,
) -> None:
    trace_id = uuid.uuid4().int
    parent_span_id = uuid.uuid4().int >> 64

    telemetry_nat_context.push_nat_span_context(trace_id=trace_id, span_id=parent_span_id)
    try:
        span = trace.get_tracer("test.http").start_span("POST")
        span.end()
    finally:
        telemetry_nat_context.pop_nat_span_context()

    finished = memory_span_exporter.get_finished_spans()[0]
    assert finished.name == "POST"
    assert finished.context.trace_id == trace_id
    assert finished.parent.span_id == parent_span_id


def test_single_active_run_fallback_without_nat_context(
    memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "run-only"
    trace_id = uuid.uuid4().int
    parent_span_id = uuid.uuid4().int >> 64
    monkeypatch.setattr(telemetry_nat_context, "_workflow_run_id_from_nat", lambda: None)
    monkeypatch.setattr(telemetry_nat_context, "_workflow_trace_id_from_nat", lambda: None)

    telemetry_nat_context.push_nat_span_context(
        trace_id=trace_id,
        span_id=parent_span_id,
        run_id=run_id,
    )
    try:
        with trace.get_tracer("test.langchain").start_as_current_span(
            "ChatPromptTemplate.workflow"
        ):
            pass
    finally:
        telemetry_nat_context.pop_nat_span_context(run_id=run_id)

    span = memory_span_exporter.get_finished_spans()[0]
    assert span.context.trace_id == trace_id
    assert span.parent.span_id == parent_span_id
