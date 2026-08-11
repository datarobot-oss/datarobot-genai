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

"""One call to attribute a local run's spans to a DataRobot use case.

Deployments and workloads are told which entity they are by the platform. A
notebook or a script is not, so it has to name one, and a use case is the
natural home for local experiments. This wraps that decision -- which use case,
and whether to touch the environment at all -- so callers do not reimplement it.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from datarobot_genai.core.telemetry.agent import instrument
from datarobot_genai.core.telemetry.datarobot_otel import EXPERIMENT_CONTAINER_ENTITY_ID_PREFIX
from datarobot_genai.core.telemetry.datarobot_otel import datarobot_otel_provider_installed
from datarobot_genai.core.telemetry.datarobot_otel import resolve_datarobot_headers_from_env
from datarobot_genai.core.telemetry.datarobot_otel import resolve_entity_id_from_headers

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UseCaseTracing:
    """The outcome of :func:`trace_to_use_case`, printable as a one-line summary."""

    entity_id: str = ""
    use_case_id: str = ""
    reason: str = ""

    @property
    def exporting(self) -> bool:
        return bool(self.entity_id)

    def __str__(self) -> str:
        if not self.exporting:
            return f"Traces are not being exported: {self.reason}."
        # Only offer the command when the use case really is where spans land: a
        # deployment or workload id in the environment outranks it.
        if self.entity_id == f"{EXPERIMENT_CONTAINER_ENTITY_ID_PREFIX}{self.use_case_id}":
            return f"View traces with:   dr xp --entity-id {self.use_case_id}"
        return f"Traces are going to {self.entity_id}, which this environment already set."


def trace_to_use_case(default_name: str, use_case_id: str = "") -> UseCaseTracing:
    """Export this process's spans to a DataRobot use case.

    Uses ``use_case_id`` when given, else the use case named ``default_name``,
    creating it on first use. Whatever has already configured
    OpenTelemetry wins -- a deployment, a workload, a codespace, or an
    ``OTEL_EXPORTER_OTLP_*`` value of your own is left exactly as it is -- so this is
    safe to call from code that also runs deployed. Calls :func:`instrument` for you;
    call the matching framework ``instrument()`` as well to collect framework spans.

    Never raises: a missing endpoint or API token comes back as a
    :class:`UseCaseTracing` that is not ``exporting`` and says why.
    """
    if os.getenv("OTEL_SDK_DISABLED", "").strip().lower() == "true":
        return UseCaseTracing(reason="OTEL_SDK_DISABLED is set")

    selected_id = ""
    if not os.getenv("OTEL_EXPORTER_OTLP_HEADERS"):
        try:
            selected_id = _resolve_use_case_id(default_name, use_case_id)
        except Exception as exc:  # noqa: BLE001 - reported, never raised at the caller
            logger.info("Could not resolve a use case for tracing: %s", exc)
            return UseCaseTracing(reason=f"no use case available ({exc})")
        os.environ["DATAROBOT_USE_CASE_ID"] = selected_id

    instrument()
    if not datarobot_otel_provider_installed():
        if not selected_id:
            return UseCaseTracing(reason="OpenTelemetry here points somewhere this run cannot use")
        return UseCaseTracing(reason="DataRobot endpoint and credentials did not resolve")

    entity_id = resolve_entity_id_from_headers(resolve_datarobot_headers_from_env() or {})
    if not entity_id:
        return UseCaseTracing(reason="OpenTelemetry is configured for another destination")
    return UseCaseTracing(entity_id=entity_id, use_case_id=selected_id)


def _resolve_use_case_id(default_name: str, use_case_id: str) -> str:
    if configured := use_case_id.strip():
        return configured

    # Imported here so the DataRobot client is only needed when a lookup happens.
    import datarobot as dr

    # `search` matches on part of a name, so compare the full name before reusing.
    existing = [
        use_case
        for use_case in dr.UseCase.list(search_params={"search": default_name})
        if use_case.name == default_name
    ]
    selected = existing[0] if existing else dr.UseCase.create(name=default_name)
    return str(selected.id)
