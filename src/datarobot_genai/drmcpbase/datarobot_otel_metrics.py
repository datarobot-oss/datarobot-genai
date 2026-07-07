# Copyright 2026 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bootstrap an OTel SDK ``MeterProvider`` so metrics actually export.

Sibling to ``datarobot_otel.py`` (which does the same for traces). genai wires
OTel traces + logs but no *metrics* provider, so SLI counters/histograms emitted
via ``opentelemetry.metrics.get_meter(...)`` go to the default no-op provider
and never leave the process. Call :func:`bootstrap_metrics_provider` once at
startup to point the global ``MeterProvider`` at an OTLP/HTTP collector.

Follows the same safety contract as the trace bootstrap: lazy SDK imports, a
no-op when no endpoint is configured, idempotent, and never raises.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Idempotency state. Mutable dict so the inner write needs no ``global``
# statement (mirrors ``_BOOTSTRAP_STATE`` in ``datarobot_otel.py``).
_STATE: dict[str, bool] = {"installed": False}

# Default export cadence. Short enough that a live demo surfaces datapoints
# within a few seconds; override via ``export_interval_ms``.
_DEFAULT_EXPORT_INTERVAL_MS = 10_000


def resolve_metrics_endpoint_from_env() -> str:
    """Metrics-specific OTLP endpoint from env (``""`` if unset).

    Deliberately honors only ``OTEL_EXPORTER_OTLP_METRICS_ENDPOINT`` — *not* the
    shared ``OTEL_EXPORTER_OTLP_ENDPOINT``. In this repo the shared var can hold a
    traces-specific URL (e.g. ``…/otel/v1/traces``, see ``core/telemetry/datarobot_otel.py``),
    and passing that to ``OTLPMetricExporter`` would send metrics to the wrong
    signal path. Callers that want the shared base endpoint should pass
    ``endpoint=`` explicitly (and include the ``/v1/metrics`` path themselves).
    """
    return os.getenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", "")


def _resolve_service_name() -> str:
    return os.getenv("OTEL_SERVICE_NAME") or "datarobot-sandbox"


def bootstrap_metrics_provider(
    endpoint: str | None = None,
    *,
    headers: dict[str, str] | None = None,
    export_interval_ms: int | None = None,
    resource_attributes: dict[str, Any] | None = None,
) -> bool:
    """Install a global OTLP/HTTP ``MeterProvider``; return whether it installed.

    Returns ``False`` (silently) when no endpoint is configured (the local-dev /
    CI shape), when already installed in this process, or when setup raises.

    ``resource_attributes`` are merged over the default ``service.name`` so a
    host process (e.g. the MCP server) can stamp metrics with the same resource
    identity it uses for traces and logs.
    """
    if _STATE["installed"]:
        return False

    endpoint = endpoint or resolve_metrics_endpoint_from_env()
    if not endpoint:
        logger.info(
            "Skipping OTel MeterProvider bootstrap: no OTEL_EXPORTER_OTLP_"
            "(METRICS_)ENDPOINT set and no endpoint passed."
        )
        return False

    try:
        from opentelemetry import metrics
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource

        exporter = (
            OTLPMetricExporter(endpoint=endpoint, headers=headers)
            if headers
            else OTLPMetricExporter(endpoint=endpoint)
        )
        reader = PeriodicExportingMetricReader(
            exporter,
            export_interval_millis=export_interval_ms or _DEFAULT_EXPORT_INTERVAL_MS,
        )
        resource = Resource.create(
            {"service.name": _resolve_service_name(), **(resource_attributes or {})}
        )
        metrics.set_meter_provider(MeterProvider(metric_readers=[reader], resource=resource))
    except Exception:
        # Never let telemetry setup take down the caller.
        logger.exception("Failed to bootstrap OTel MeterProvider")
        return False

    _STATE["installed"] = True
    logger.info("OTel MeterProvider installed → %s", endpoint)
    return True


def _reset_for_testing() -> None:
    """Clear the idempotency flag so tests can re-run the bootstrap."""
    _STATE["installed"] = False
