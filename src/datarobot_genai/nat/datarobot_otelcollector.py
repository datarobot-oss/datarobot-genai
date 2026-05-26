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
* ``datarobot_entity_id`` is required (``deployment-<id>`` shape).
* The endpoint convention is the base OTel URL (e.g.
  ``https://<host>/otel``) with ``/v1/traces`` appended for span export.

Example ``workflow.yaml``::

    telemetry:
      tracing:
        otelcollector:
          _type: datarobot_otelcollector
          endpoint: "${DATAROBOT_OTEL_ENDPOINT}/v1/traces"
          datarobot_api_key: "${DATAROBOT_API_TOKEN}"
          datarobot_entity_id: "${DATAROBOT_ENTITY_ID}"
          project: "agent"
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator

from nat.builder.builder import Builder
from nat.cli.register_workflow import register_telemetry_exporter
from nat.data_models.common import SerializableSecretStr
from nat.data_models.common import get_secret_value
from nat.data_models.telemetry_exporter import TelemetryExporterBaseConfig
from nat.observability.mixin.batch_config_mixin import BatchConfigMixin
from nat.observability.mixin.collector_config_mixin import CollectorConfigMixin
from nat.plugins.opentelemetry import OTLPSpanAdapterExporter
from nat.plugins.opentelemetry.otel_span_exporter import get_opentelemetry_sdk_version
from pydantic import Field
from pydantic import field_validator

logger = logging.getLogger(__name__)


class DataRobotOtelCollectorTelemetryExporter(  # type: ignore[call-arg]
    BatchConfigMixin,
    CollectorConfigMixin,
    TelemetryExporterBaseConfig,
    name="datarobot_otelcollector",
):
    """Telemetry exporter for the DataRobot OTel collector.

    Inherits ``endpoint``, ``project``, ``resource_attributes`` (from
    ``CollectorConfigMixin``), and batch tuning fields (from
    ``BatchConfigMixin``). Adds the DataRobot-specific auth headers.
    """

    datarobot_api_key: SerializableSecretStr = Field(
        ...,
        description=(
            "DataRobot API key sent as the X-DataRobot-Api-Key header. "
            "Provide via env var, e.g. ${DATAROBOT_API_TOKEN}."
        ),
    )
    datarobot_entity_id: str = Field(
        ...,
        description=(
            "DataRobot entity identifier sent as the X-DataRobot-Entity-Id "
            "header. Must use the prefixed form 'deployment-<id>' (e.g. "
            "'deployment-abc123') for shell deployments created by the "
            "datarobot-external-agent-monitoring skill."
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
        # Match the validation rule documented in the DR external agent
        # monitoring skill: must be the 'deployment-<id>' shape, not bare.
        if not value or not value.startswith("deployment-"):
            raise ValueError(
                "datarobot_entity_id must be of the form 'deployment-<id>' "
                f"(got {value!r}). Run create_shell_deployment.py from the "
                "datarobot-external-agent-monitoring skill if you do not "
                "have one yet."
            )
        return value


@register_telemetry_exporter(config_type=DataRobotOtelCollectorTelemetryExporter)
async def datarobot_otelcollector_telemetry_exporter(
    config: DataRobotOtelCollectorTelemetryExporter,
    builder: Builder,
) -> AsyncGenerator[OTLPSpanAdapterExporter, None]:
    """Yield an OTLP span exporter pointed at the DataRobot OTel collector."""
    headers: dict[str, str] = {
        "X-DataRobot-Api-Key": get_secret_value(config.datarobot_api_key),
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
        "Configuring datarobot_otelcollector exporter "
        "endpoint=%s entity_id=%s header_keys=%s",
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
