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

Outside a deployment (local dev, CI without DR env), the exporter is
silently pruned from the loaded config by
:func:`prune_exporter_if_env_missing` so listing it unconditionally is
safe. Explicit overrides for any of the three auto-derived fields are
still honored — pin them in ``workflow.yaml`` to point at a non-default
collector.
"""

from __future__ import annotations

import logging
import os
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

# Resolver helpers live in core/ so telemetry_agent can reuse them without
# importing from nat/. Keep underscore aliases so this module's existing
# Field(default_factory=…) bindings stay stable.
from datarobot_genai.core.datarobot_otel import (
    resolve_api_key_from_env as _resolve_api_key_from_env,
)
from datarobot_genai.core.datarobot_otel import (
    resolve_entity_id_from_env as _resolve_entity_id_from_env,
)
from datarobot_genai.core.datarobot_otel import (
    resolve_otel_endpoint_from_env as _resolve_otel_endpoint_from_env,
)

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

    endpoint: str = Field(
        default_factory=_resolve_otel_endpoint_from_env,
        description=(
            "OTLP/HTTP endpoint for the DataRobot OTel collector. If "
            "omitted, derived from DATAROBOT_PUBLIC_API_ENDPOINT / "
            "DATAROBOT_ENDPOINT by stripping any path (e.g. /api/v2) and "
            "appending /otel/v1/traces."
        ),
    )
    datarobot_api_key: SerializableSecretStr = Field(
        default_factory=lambda: SerializableSecretStr(_resolve_api_key_from_env()),
        description=(
            "DataRobot API key sent as the X-DataRobot-Api-Key header. If "
            "omitted, derived from the DATAROBOT_API_TOKEN env var."
        ),
    )
    datarobot_entity_id: str = Field(
        default_factory=_resolve_entity_id_from_env,
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
        # Empty is permitted so prune_exporter_if_env_missing can drop the
        # exporter at config-load time without a pydantic ValidationError.
        # Any non-empty value must still match the 'deployment-<id>' shape
        # documented in the DR external-agent-monitoring skill.
        if value and not value.startswith("deployment-"):
            raise ValueError(
                "datarobot_entity_id must be of the form 'deployment-<id>' "
                f"(got {value!r}). Inside a DataRobot deployment, omit the "
                "field — it is auto-derived from MLOPS_DEPLOYMENT_ID. For "
                "local dev, run create_shell_deployment.py from the "
                "datarobot-external-agent-monitoring skill."
            )
        return value


def prune_exporter_if_env_missing(config_yaml: dict) -> None:
    """Drop ``datarobot_otelcollector`` tracing entries from a YAML config
    dict when essential DataRobot env is not present (e.g. local dev).

    Mutates *config_yaml* in place. Called from
    :func:`datarobot_genai.nat.helpers.load_config` before NAT's pydantic
    validation runs, since NAT has no built-in way to skip a misconfigured
    exporter without raising.

    An entry is pruned only when the auto-derive path can't produce values:
    if the user pinned ``endpoint``, ``datarobot_api_key``, and
    ``datarobot_entity_id`` explicitly in the YAML, the entry is preserved
    regardless of env.
    """
    tracing = (config_yaml.get("telemetry") or {}).get("tracing") or {}
    if not tracing:
        return
    has_deployment = bool(os.getenv("MLOPS_DEPLOYMENT_ID"))
    has_api_key = bool(os.getenv("DATAROBOT_API_TOKEN"))
    has_endpoint = bool(
        os.getenv("DATAROBOT_PUBLIC_API_ENDPOINT") or os.getenv("DATAROBOT_ENDPOINT")
    )
    if has_deployment and has_api_key and has_endpoint:
        return
    for name in list(tracing.keys()):
        entry = tracing[name]
        if not isinstance(entry, dict) or entry.get("_type") != "datarobot_otelcollector":
            continue
        entity_id_pinned = bool(entry.get("datarobot_entity_id"))
        api_key_pinned = bool(entry.get("datarobot_api_key"))
        endpoint_pinned = bool(entry.get("endpoint"))
        if (
            (has_deployment or entity_id_pinned)
            and (has_api_key or api_key_pinned)
            and (has_endpoint or endpoint_pinned)
        ):
            continue
        logger.info(
            "Skipping datarobot_otelcollector tracing exporter %r: "
            "DataRobot deployment env (MLOPS_DEPLOYMENT_ID / "
            "DATAROBOT_API_TOKEN / DATAROBOT_(PUBLIC_)ENDPOINT) not fully "
            "set and the field is not pinned in workflow.yaml.",
            name,
        )
        tracing.pop(name)


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
