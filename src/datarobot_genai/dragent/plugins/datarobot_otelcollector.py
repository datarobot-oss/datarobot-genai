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

* Headers are passed directly to the exporter — never via
  ``OTEL_EXPORTER_OTLP_*`` env vars, which some frameworks misinterpret.
* ``datarobot_entity_id`` uses the ``deployment-<id>`` shape. When omitted it
  is auto-derived from ``MLOPS_DEPLOYMENT_ID`` (the bare deployment ID
  exposed inside a DataRobot deployment) with the ``deployment-`` prefix
  auto-prepended.
* ``datarobot_api_key`` defaults to ``DATAROBOT_API_TOKEN``.
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

Local dev / incomplete config: when any of ``endpoint``,
``datarobot_api_key``, or ``datarobot_entity_id`` resolves empty -
the exporter silently drops spans instead of POSTing to a real
endpoint with bad auth (which the DataRobot ingest rejects with repeated
``401 Unauthorized``). This mirrors the no-op contract that
``bootstrap_otel_provider_for_datarobot`` already honors for framework
auto-instrumentor spans.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator

from nat.builder.builder import Builder
from nat.cli.register_workflow import register_telemetry_exporter
from nat.data_models.common import SerializableSecretStr
from nat.data_models.common import get_secret_value
from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.telemetry_exporter import TelemetryExporterBaseConfig
from nat.observability.exporter.base_exporter import BaseExporter
from nat.observability.mixin.batch_config_mixin import BatchConfigMixin
from nat.observability.mixin.collector_config_mixin import CollectorConfigMixin
from nat.plugins.opentelemetry import OTLPSpanAdapterExporter
from nat.plugins.opentelemetry.otel_span_exporter import get_opentelemetry_sdk_version
from pydantic import Field
from pydantic import field_validator

from datarobot_genai.core.datarobot_otel import ENTITY_ID_PREFIX
from datarobot_genai.core.datarobot_otel import resolve_api_key_from_env
from datarobot_genai.core.datarobot_otel import resolve_entity_id_from_env
from datarobot_genai.core.datarobot_otel import resolve_otel_endpoint_from_env

logger = logging.getLogger(__name__)


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
        default_factory=resolve_otel_endpoint_from_env,
        description=(
            "OTLP/HTTP endpoint for the DataRobot OTel collector. If "
            "omitted, derived from DATAROBOT_PUBLIC_API_ENDPOINT / "
            "DATAROBOT_ENDPOINT by stripping any path (e.g. /api/v2) and "
            "appending /otel/v1/traces."
        ),
    )
    datarobot_api_key: SerializableSecretStr = Field(
        default_factory=lambda: SerializableSecretStr(resolve_api_key_from_env()),
        description=(
            "DataRobot API key sent as the X-DataRobot-Api-Key header. If "
            "omitted, derived from the DATAROBOT_API_TOKEN env var."
        ),
    )
    datarobot_entity_id: str = Field(
        default_factory=resolve_entity_id_from_env,
        description=(
            "DataRobot entity identifier sent as the X-DataRobot-Entity-Id "
            "header. Must use the prefixed form 'deployment-<id>' (e.g. "
            "'deployment-abc123') for shell deployments created by the "
            "datarobot-external-agent-monitoring skill. If omitted, derived "
            "from MLOPS_DEPLOYMENT_ID with the 'deployment-' prefix "
            "auto-prepended."
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

    @field_validator("datarobot_entity_id")
    @classmethod
    def _validate_entity_id_prefix(cls, value: str) -> str:
        # Empty is permitted because the auto-derive path may legitimately
        # return an empty string when MLOPS_DEPLOYMENT_ID is not set (e.g.
        # local dev). Any non-empty value must still match the
        # 'deployment-<id>' shape documented in the DR
        # external-agent-monitoring skill.
        if value and not value.startswith(ENTITY_ID_PREFIX):
            raise ValueError(
                "datarobot_entity_id must be of the form 'deployment-<id>' "
                f"(got {value!r}). Inside a DataRobot deployment, omit the "
                "field — it is auto-derived from MLOPS_DEPLOYMENT_ID. For "
                "local dev, run create_shell_deployment.py from the "
                "datarobot-external-agent-monitoring skill."
            )
        return value


class _DroppingSpanExporter(BaseExporter):
    """Subscribes to the event stream but drops every span."""

    def export(self, event: IntermediateStep) -> None:
        return None


@register_telemetry_exporter(config_type=DataRobotOtelCollectorTelemetryExporter)
async def datarobot_otelcollector_telemetry_exporter(
    config: DataRobotOtelCollectorTelemetryExporter,
    builder: Builder,
) -> AsyncGenerator[BaseExporter, None]:
    """Yield an OTLP span exporter pointed at the DataRobot OTel collector.

    When the resolved config is incomplete (missing endpoint, api key, or
    entity id — i.e. local-dev case) yield a no-op exporter that drops spans
    instead of authenticating against a real endpoint and failing with 401.
    """
    api_key = get_secret_value(config.datarobot_api_key)
    if not api_key or not config.datarobot_entity_id or not config.endpoint:
        logger.info(
            "datarobot_otelcollector: DataRobot OTel ingest config incomplete - "
            "skipping span export."
        )
        yield _DroppingSpanExporter()
        return

    headers: dict[str, str] = {
        "X-DataRobot-Api-Key": api_key,
        "X-DataRobot-Entity-Id": config.datarobot_entity_id,
    }
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
    logger.debug(
        "Configuring datarobot_otelcollector exporter endpoint=%s entity_id=%s header_keys=%s",
        config.endpoint,
        config.datarobot_entity_id,
        sorted(headers.keys()),
    )

    yield OTLPSpanAdapterExporter(
        endpoint=config.endpoint,
        headers=headers,
        resource_attributes=merged_resource_attributes,
        batch_size=config.batch_size,
        flush_interval=config.flush_interval,
        max_queue_size=config.max_queue_size,
        drop_on_overflow=config.drop_on_overflow,
        shutdown_timeout=config.shutdown_timeout,
    )
