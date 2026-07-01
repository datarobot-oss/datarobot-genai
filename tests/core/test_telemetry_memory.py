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

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind
from opentelemetry.util._once import Once

from datarobot_genai.core.telemetry import memory


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
        trace.get_tracer("test.telemetry_memory"),
    )
    yield exporter
    exporter.clear()


def test_trace_memory_operation_emits_gen_ai_attributes(
    memory_span_exporter: InMemorySpanExporter,
) -> None:
    with memory.trace_memory_operation(
        "search_memory",
        store_name="mem0",
        store_id="project-123",
        attributes={
            "gen_ai.memory.query.text": "hello",
            "memory.top_k": 3,
        },
    ):
        pass

    spans = memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "search_memory"
    assert span.kind == SpanKind.CLIENT
    assert span.attributes["gen_ai.operation.name"] == "search_memory"
    assert span.attributes["gen_ai.memory.store.name"] == "mem0"
    assert span.attributes["gen_ai.memory.store.id"] == "project-123"
    assert span.attributes["gen_ai.memory.query.text"] == "hello"
    assert span.attributes["memory.top_k"] == 3


def test_trace_memory_operation_records_exception(
    memory_span_exporter: InMemorySpanExporter,
) -> None:
    with pytest.raises(RuntimeError, match="boom"):
        with memory.trace_memory_operation(
            "update_memory",
            store_name="datarobot-memory",
        ):
            raise RuntimeError("boom")

    span = memory_span_exporter.get_finished_spans()[0]
    assert span.attributes["error.type"] == "RuntimeError"
    assert span.status.status_code.name == "ERROR"


def test_truncate_memory_text() -> None:
    assert memory.truncate_memory_text("short") == "short"
    long_text = "x" * 600
    truncated = memory.truncate_memory_text(long_text)
    assert len(truncated) == 512
    assert truncated.endswith("...")
