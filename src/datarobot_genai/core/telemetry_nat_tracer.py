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

"""TracerProvider wrapper that parents SDK spans under the active NAT workflow trace."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider
from opentelemetry.trace import Link
from opentelemetry.trace import Span
from opentelemetry.trace import SpanKind
from opentelemetry.trace import Tracer
from opentelemetry.util.types import Attributes

from datarobot_genai.core.telemetry_nat_context import use_nat_workflow_trace_context


class NatWorkflowTracer(Tracer):
    """Tracer that joins SDK spans to the active NAT workflow trace when possible."""

    def __init__(self, delegate: Tracer) -> None:
        self._delegate = delegate

    def start_span(
        self,
        name: str,
        context: otel_context.Context | None = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Attributes = None,
        links: Link | None = None,
        start_time: int | None = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
    ) -> Span:
        with use_nat_workflow_trace_context():
            return self._delegate.start_span(
                name,
                context=context,
                kind=kind,
                attributes=attributes,
                links=links,
                start_time=start_time,
                record_exception=record_exception,
                set_status_on_exception=set_status_on_exception,
            )

    def start_as_current_span(
        self,
        name: str,
        context: otel_context.Context | None = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Attributes = None,
        links: Link | None = None,
        start_time: int | None = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
        end_on_exit: bool = True,
    ) -> AbstractContextManager[Span]:
        return _NatWorkflowSpanContextManager(
            self._delegate.start_as_current_span(
                name,
                context=context,
                kind=kind,
                attributes=attributes,
                links=links,
                start_time=start_time,
                record_exception=record_exception,
                set_status_on_exception=set_status_on_exception,
                end_on_exit=end_on_exit,
            )
        )


class _NatWorkflowSpanContextManager(AbstractContextManager[Span]):
    def __init__(self, delegate: AbstractContextManager[Span]) -> None:
        self._delegate = delegate
        self._nat_context: AbstractContextManager[None] | None = None

    def __enter__(self) -> Span:
        self._nat_context = use_nat_workflow_trace_context()
        self._nat_context.__enter__()
        return self._delegate.__enter__()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool | None:
        try:
            return self._delegate.__exit__(exc_type, exc_val, exc_tb)
        finally:
            if self._nat_context is not None:
                self._nat_context.__exit__(exc_type, exc_val, exc_tb)


_NAT_TRACER_WRAPPED_ATTR = "_nat_workflow_tracer_wrapped"


def wrap_sdk_tracer_provider(provider: SDKTracerProvider) -> SDKTracerProvider:
    """Patch an SDK provider so every tracer joins the NAT workflow trace."""
    if getattr(provider, _NAT_TRACER_WRAPPED_ATTR, False):
        return provider

    original_get_tracer = provider.get_tracer

    def get_tracer(
        instrumenting_module_name: str,
        instrumenting_library_version: str | None = None,
        schema_url: str | None = None,
        attributes: Attributes = None,
    ) -> Tracer:
        return NatWorkflowTracer(
            original_get_tracer(
                instrumenting_module_name,
                instrumenting_library_version,
                schema_url,
                attributes,
            )
        )

    provider.get_tracer = get_tracer  # type: ignore[method-assign]
    setattr(provider, _NAT_TRACER_WRAPPED_ATTR, True)
    return provider
