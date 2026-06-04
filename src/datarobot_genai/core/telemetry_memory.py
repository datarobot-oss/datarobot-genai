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

"""OpenTelemetry helpers for GenAI memory operations."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from typing import Literal

from opentelemetry import trace
from opentelemetry.trace import Span
from opentelemetry.trace import SpanKind
from opentelemetry.trace import Status
from opentelemetry.trace import StatusCode

_tracer = trace.get_tracer(__name__)

MemoryOperationName = Literal["update_memory", "search_memory", "delete_memory"]

_MAX_QUERY_LEN = 512


def truncate_memory_text(value: str, limit: int = _MAX_QUERY_LEN) -> str:
    """Truncate long memory query text for span attributes."""
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def _set_span_attributes(span: Span, attributes: dict[str, Any]) -> None:
    for key, value in attributes.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            span.set_attribute(key, value)


@contextmanager
def trace_memory_operation(
    operation_name: MemoryOperationName,
    *,
    store_name: str,
    store_id: str | None = None,
    attributes: dict[str, Any] | None = None,
) -> Iterator[Span]:
    """Create a GenAI memory span following OTel semantic conventions."""
    with _tracer.start_as_current_span(
        operation_name,
        kind=SpanKind.CLIENT,
    ) as span:
        attrs = {
            "gen_ai.operation.name": operation_name,
            "gen_ai.memory.store.name": store_name,
            **(attributes or {}),
        }
        if store_id:
            attrs["gen_ai.memory.store.id"] = store_id
        _set_span_attributes(span, attrs)
        try:
            yield span
            span.set_status(Status(StatusCode.OK))
        except Exception as exc:
            span.set_attribute("error.type", type(exc).__name__)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            span.record_exception(exc)
            raise
