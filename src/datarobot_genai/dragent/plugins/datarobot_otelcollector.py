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

"""NAT OpenTelemetry collector exporter for the DataRobot OTel ingest endpoint.

NAT's built-in ``otelcollector`` exporter (see
``nat.plugins.opentelemetry.register``) exposes ``endpoint`` but not
``headers``, so it cannot authenticate against the DataRobot OTel ingest
endpoint that requires ``X-DataRobot-Api-Key`` and ``X-DataRobot-Entity-Id``.

This exporter mirrors the conventions documented in DataRobot's
``datarobot-external-agent-monitoring`` skill (see
``datarobot-oss/datarobot-agent-skills``), adapted to NAT's plugin model:

* Headers are passed directly to the exporter, not relying on ``OTEL_EXPORTER_OTLP_*`` env vars,
  which some frameworks misinterpret.
* ``endpoint`` defaults to ``<DR base host>/otel/v1/traces`` derived from
  ``DATAROBOT_PUBLIC_API_ENDPOINT`` / ``DATAROBOT_ENDPOINT`` (the OTel
  collector ingress lives at the same host as the DR API, off
  ``/otel/v1/traces``, not under ``/api/v2``).

Inside a DataRobot deployment, the minimal ``workflow.yaml`` is::

    telemetry:
      tracing:
        otelcollector:
          _type: datarobot_otelcollector
          project: "agent"

Explicit overrides for any of the three auto-derived fields are honored —
pin them in ``workflow.yaml`` to point at a non-default collector.

Span size. NAT's ``SpanExporter`` writes each function's input and output twice
(``input.value`` plus ``input.value_obj``, ``output.value`` plus
``output.value_obj``) and records a streaming function's output as the list of
every chunk it yielded, so a streamed answer lands in the trace as hundreds of
per-token envelopes, on the workflow span and again on the inner agent span. By
default this exporter drops the ``*.value_obj`` duplicates and coalesces a
streamed ``DRAgentEventResponse`` chunk list into one response (adjacent
per-token deltas merged, every other event kept) before NAT serializes it.
``drop_value_obj_attributes: false`` and ``coalesce_streamed_output: false``
restore NAT's raw output.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

from nat.builder.builder import Builder
from nat.cli.register_workflow import register_telemetry_exporter
from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.span import Span
from nat.data_models.telemetry_exporter import TelemetryExporterBaseConfig
from nat.observability.mixin.batch_config_mixin import BatchConfigMixin
from nat.observability.mixin.collector_config_mixin import CollectorConfigMixin
from nat.observability.processor.processor import Processor
from nat.plugins.opentelemetry import OTLPSpanAdapterExporter
from nat.plugins.opentelemetry.otel_span_exporter import get_opentelemetry_sdk_version
from pydantic import Field

from datarobot_genai.core.telemetry.datarobot_otel import resolve_datarobot_headers_from_env
from datarobot_genai.core.telemetry.datarobot_otel import resolve_otel_traces_endpoint_from_env
from datarobot_genai.core.telemetry.nat_context import pop_nat_span_context
from datarobot_genai.core.telemetry.nat_context import push_nat_span_context
from datarobot_genai.core.telemetry.nat_context import reset_nat_span_context
from datarobot_genai.dragent.frontends.converters import coalesce_dragent_event_responses
from datarobot_genai.dragent.frontends.response import DRAgentEventResponse

logger = logging.getLogger(__name__)

_VALUE_OBJ_ATTRIBUTE_KEYS = ("input.value_obj", "output.value_obj")


class DropValueObjAttributesProcessor(Processor[Span, Span]):
    """Remove ``input.value_obj`` / ``output.value_obj`` from a span before export.

    NAT's ``SpanExporter`` writes them as a second JSON serialization of the
    payload already carried by ``input.value`` / ``output.value``. Nothing in
    DataRobot's trace storage or viewer reads them, so they only double the
    span size.
    """

    # NAT infers a processor's types from the raw ``process`` signature; with this
    # module's postponed annotations those are the string "Span", which fails the
    # pipeline compatibility check, so state the types directly.
    @property
    def input_type(self) -> type[Span]:
        return Span

    @property
    def output_type(self) -> type[Span]:
        return Span

    async def process(self, item: Span) -> Span:
        for key in _VALUE_OBJ_ATTRIBUTE_KEYS:
            item.attributes.pop(key, None)
        return item


def _coalesce_streamed_output(event: IntermediateStep) -> IntermediateStep:
    """Return a copy of ``event`` whose streamed chunk-list output is coalesced.

    NAT records a streaming function's output as the list of every chunk it
    yielded. When that list holds only ``DRAgentEventResponse`` chunks, replace
    it with one coalesced response (see ``coalesce_dragent_event_responses``).
    Any other output shape, including an empty or mixed list, returns ``event``
    unchanged.

    The incoming step is shared with every other subscriber of NAT's event
    stream (the step adaptor streaming to clients, other exporters), so it is
    never mutated: a shallow ``model_copy`` chain replaces only
    ``payload.data.output``. ``event.UUID`` reads ``payload.UUID``, which the
    copy keeps, so NAT still pairs the copy with the span opened at START.
    """
    data = event.payload.data
    output = data.output if data is not None else None
    if (
        not isinstance(output, list)
        or not output
        or not all(isinstance(item, DRAgentEventResponse) for item in output)
    ):
        return event
    coalesced = coalesce_dragent_event_responses(output)
    return event.model_copy(
        update={
            "payload": event.payload.model_copy(
                update={"data": data.model_copy(update={"output": coalesced})}
            )
        }
    )


def _unescape_json_unicode(serialized: str) -> str:
    r"""Re-encode a JSON string so non-ASCII text is stored as-is instead of ``\uXXXX``.

    NAT serializes span payloads with ``json.dumps``'s default ``ensure_ascii=True``,
    which turns Japanese (and any other non-ASCII) text into ``\uXXXX`` escapes that
    are unreadable in the trace viewer. Round-tripping keeps the JSON semantically
    identical while emitting the characters literally.

    The escaped input is returned unchanged when it isn't valid JSON or when the
    decoded text contains lone surrogates, which can't be encoded as UTF-8 for export.
    """
    if "\\u" not in serialized:
        return serialized
    try:
        unescaped = json.dumps(json.loads(serialized), ensure_ascii=False)
        unescaped.encode("utf-8")
    except (ValueError, UnicodeEncodeError):
        return serialized
    return unescaped


class DataRobotOTLPSpanAdapterExporter(OTLPSpanAdapterExporter):
    r"""OTLP exporter that mirrors NAT span hierarchy into the OTel SDK context.

    NAT lifecycle spans and SDK spans (memory, framework auto-instrumentors)
    otherwise export as separate trace trees. This bridge keeps the SDK context
    aligned with NAT intermediate steps so memory spans nest under the active
    workflow span.

    It also keeps non-ASCII text in span payload attributes (``input.value``,
    ``output.value``, metadata) readable instead of ``\uXXXX``-escaped, and
    shrinks NAT's payload attributes: ``drop_value_obj_attributes`` removes the
    duplicate ``input.value_obj`` / ``output.value_obj`` keys through a span
    processor at the front of the export pipeline, and
    ``coalesce_streamed_output`` collapses a streamed ``DRAgentEventResponse``
    chunk list into one response before NAT writes ``output.value``. Both
    default on.
    """

    def __init__(
        self,
        *,
        drop_value_obj_attributes: bool = True,
        coalesce_streamed_output: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._coalesce_streamed_output = coalesce_streamed_output
        if drop_value_obj_attributes:
            # Position 0 runs before NAT's SpanToOtelProcessor, the same slot
            # NAT's own redaction exporter uses for its span processors.
            self.add_processor(
                DropValueObjAttributesProcessor(),
                name="drop_value_obj_attributes",
                position=0,
            )

    def _to_json_string(self, data: Any) -> str:
        return _unescape_json_unicode(super()._to_json_string(data))

    def _serialize_payload(self, input_value: Any) -> tuple[str, bool]:
        serialized, is_json = super()._serialize_payload(input_value)
        if is_json:
            serialized = _unescape_json_unicode(serialized)
        return serialized, is_json

    def _workflow_run_id(self) -> str | None:
        try:
            return self._context_state.workflow_run_id.get()
        except Exception:
            logger.debug("Unable to read workflow run id for NAT span bridge", exc_info=True)
            return None

    @staticmethod
    def _span_has_bridge_context(span: object | None) -> bool:
        context = getattr(span, "context", None)
        return bool(context and context.trace_id and context.span_id)

    def _process_start_event(self, event: IntermediateStep) -> None:
        super()._process_start_event(event)
        run_id = self._workflow_run_id()
        span = self._span_stack.get(event.UUID)
        if run_id and self._span_has_bridge_context(span):
            push_nat_span_context(
                trace_id=span.context.trace_id,
                span_id=span.context.span_id,
                run_id=run_id,
            )

    def _process_end_event(self, event: IntermediateStep) -> None:
        run_id = self._workflow_run_id()
        span = self._span_stack.get(event.UUID)
        should_pop = bool(run_id and self._span_has_bridge_context(span))
        if self._coalesce_streamed_output:
            try:
                event = _coalesce_streamed_output(event)
            except Exception:
                # One odd payload must not cost the span; export NAT's raw output instead.
                logger.warning(
                    "Failed to coalesce streamed output for step %s", event.UUID, exc_info=True
                )
        super()._process_end_event(event)
        if should_pop:
            pop_nat_span_context(run_id=run_id)

    def on_complete(self) -> None:
        run_id = self._workflow_run_id()
        if run_id:
            reset_nat_span_context(run_id=run_id)
        super().on_complete()


class DataRobotOtelCollectorTelemetryExporter(  # type: ignore[call-arg]
    BatchConfigMixin,
    CollectorConfigMixin,
    TelemetryExporterBaseConfig,
    name="datarobot_otelcollector",
):
    """Telemetry exporter for the DataRobot OTel collector.

    Inherits ``project`` and ``endpoint`` (from ``CollectorConfigMixin``,
    though ``endpoint`` is overridden below to auto-derive from env) and batch
    tuning fields (from ``BatchConfigMixin``). Adds the DataRobot-specific auth
    headers plus ``resource_attributes``.
    """

    endpoint: str = Field(
        default_factory=resolve_otel_traces_endpoint_from_env,
        description=(
            "OTLP/HTTP endpoint for the DataRobot OTel collector. If "
            "omitted, derived from DATAROBOT_PUBLIC_API_ENDPOINT / "
            "DATAROBOT_ENDPOINT by stripping any path (e.g. /api/v2) and "
            "appending /otel/v1/traces."
        ),
    )
    extra_headers: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Additional headers to forward to the OTel collector. Keys "
            "here override the DataRobot defaults on name collision."
        ),
    )
    resource_attributes: dict[str, str] = Field(
        default_factory=dict,
        description="The resource attributes to add to the span.",
    )
    drop_value_obj_attributes: bool = Field(
        default=True,
        description=(
            "Drop the input.value_obj / output.value_obj span attributes, a second "
            "serialization of input.value / output.value that nothing reads."
        ),
    )
    coalesce_streamed_output: bool = Field(
        default=True,
        description=(
            "Merge a streaming function's per-token DRAgentEventResponse chunks into "
            "one response before it is written to output.value. Non-delta events "
            "(tool calls, run boundaries) are kept."
        ),
    )


@register_telemetry_exporter(config_type=DataRobotOtelCollectorTelemetryExporter)
async def datarobot_otelcollector_telemetry_exporter(
    config: DataRobotOtelCollectorTelemetryExporter,
    builder: Builder,
) -> AsyncGenerator[DataRobotOTLPSpanAdapterExporter]:
    """Yield an OTLP span exporter pointed at the DataRobot OTel collector."""
    headers = dict(resolve_datarobot_headers_from_env() or {})
    # Caller-supplied headers win on collision; lets you e.g. add request-
    # specific X-DataRobot-* metadata without forking the exporter.
    headers.update(config.extra_headers)

    # Mirror nat.plugins.opentelemetry.register.otel_telemetry_exporter:
    # the inherited ``project`` field is the OTel ``service.name`` for spans.
    # User-supplied resource_attributes win on collision.
    default_resource_attributes = {
        "telemetry.sdk.language": "python",
        "telemetry.sdk.name": "opentelemetry",
        "telemetry.sdk.version": get_opentelemetry_sdk_version(),
        "service.name": config.project,
    }
    merged_resource_attributes = {
        **default_resource_attributes,
        **config.resource_attributes,
    }

    # Never log the secret. Sorted keys only.
    lower_headers = {k.lower(): v for k, v in headers.items()}
    logger.info(
        "Configuring datarobot_otelcollector exporter endpoint=%s entity_id=%s header_keys=%s",
        config.endpoint,
        lower_headers.get("x-datarobot-entity-id", ""),
        sorted(headers.keys()),
    )

    yield DataRobotOTLPSpanAdapterExporter(
        endpoint=config.endpoint,
        headers=headers,
        resource_attributes=merged_resource_attributes,
        batch_size=config.batch_size,
        flush_interval=config.flush_interval,
        max_queue_size=config.max_queue_size,
        drop_on_overflow=config.drop_on_overflow,
        shutdown_timeout=config.shutdown_timeout,
        drop_value_obj_attributes=config.drop_value_obj_attributes,
        coalesce_streamed_output=config.coalesce_streamed_output,
    )
