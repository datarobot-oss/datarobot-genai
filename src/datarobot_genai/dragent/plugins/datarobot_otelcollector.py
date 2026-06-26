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
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator

from nat.builder.builder import Builder
from nat.cli.register_workflow import register_telemetry_exporter
from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.telemetry_exporter import TelemetryExporterBaseConfig
from nat.observability.mixin.batch_config_mixin import BatchConfigMixin
from nat.observability.mixin.collector_config_mixin import CollectorConfigMixin
from nat.plugins.opentelemetry import OTLPSpanAdapterExporter
from nat.plugins.opentelemetry.otel_span_exporter import get_opentelemetry_sdk_version
from pydantic import Field

from datarobot_genai.core.datarobot_otel import resolve_datarobot_headers_from_env
from datarobot_genai.core.datarobot_otel import resolve_otel_traces_endpoint_from_env
from datarobot_genai.core.telemetry_nat_context import pop_nat_span_context
from datarobot_genai.core.telemetry_nat_context import push_nat_span_context
from datarobot_genai.core.telemetry_nat_context import reset_nat_span_context

logger = logging.getLogger(__name__)


class DataRobotOTLPSpanAdapterExporter(OTLPSpanAdapterExporter):
    """OTLP exporter that mirrors NAT span hierarchy into the OTel SDK context.

    NAT lifecycle spans and SDK spans (memory, framework auto-instrumentors)
    otherwise export as separate trace trees. This bridge keeps the SDK context
    aligned with NAT intermediate steps so memory spans nest under the active
    workflow span.
    """

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
    )
