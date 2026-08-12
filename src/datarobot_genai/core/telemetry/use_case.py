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

from datarobot_genai.core.runtime import is_hosted_runtime
from datarobot_genai.core.telemetry.agent import instrument
from datarobot_genai.core.telemetry.datarobot_otel import EXPERIMENT_CONTAINER_ENTITY_ID_PREFIX
from datarobot_genai.core.telemetry.datarobot_otel import datarobot_otel_provider_installed
from datarobot_genai.core.telemetry.datarobot_otel import resolve_api_key_from_env
from datarobot_genai.core.telemetry.datarobot_otel import resolve_datarobot_headers_from_env
from datarobot_genai.core.telemetry.datarobot_otel import resolve_entity_id_from_headers
from datarobot_genai.core.telemetry.datarobot_otel import resolve_otel_traces_endpoint_from_env

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UseCaseTracing:
    """The outcome of :func:`trace_to_use_case`, printable as a one-line summary."""

    entity_id: str = ""
    reason: str = ""

    @property
    def exporting(self) -> bool:
        return bool(self.entity_id)

    def __str__(self) -> str:
        if not self.exporting:
            return f"Traces are not being exported: {self.reason}."
        if self.reason:
            return f"Traces are going to {self.entity_id}: {self.reason}."
        # Read the id back out of the entity so a second call prints the same line as
        # the first. Only a use case has a `dr xp` view; a deployment or workload id in
        # the environment outranks it and is reported as-is.
        if self.entity_id.startswith(EXPERIMENT_CONTAINER_ENTITY_ID_PREFIX):
            use_case_id = self.entity_id.removeprefix(EXPERIMENT_CONTAINER_ENTITY_ID_PREFIX)
            return f"View traces with:   dr xp --entity-id {use_case_id}"
        return f"Traces are going to {self.entity_id}."


def trace_to_use_case(default_name: str, use_case_id: str = "") -> UseCaseTracing:
    """Export this process's spans to a DataRobot use case.

    Uses ``use_case_id`` when given, else ``DATAROBOT_USE_CASE_ID`` or
    ``DATAROBOT_DEFAULT_USE_CASE`` from the environment, else the use case named
    ``default_name``, creating it on first use. Sets ``DATAROBOT_USE_CASE_ID`` to
    whichever it picked, so the rest of the process agrees. Calls :func:`instrument`
    for you; call the matching framework ``instrument()`` as well to collect
    framework spans.

    Chooses nothing and creates nothing when the process is already pointed
    somewhere: inside a deployment or workload the platform names the entity, and
    an ``OTEL_EXPORTER_OTLP_*`` variable of your own means your collector, which
    must not be handed DataRobot credentials. Both cases are reported rather than
    overridden, so this is safe in code that also runs deployed.

    A missing endpoint or API token comes back as a :class:`UseCaseTracing` that is
    not ``exporting`` and says why, rather than as an exception.
    """
    if reason := _skip_reason():
        return UseCaseTracing(reason=reason)

    wanted = use_case_id.strip() or _use_case_id_from_env()
    already_tracing = datarobot_otel_provider_installed()
    if not already_tracing and not is_hosted_runtime():
        try:
            os.environ["DATAROBOT_USE_CASE_ID"] = wanted or _use_case_id_by_name(default_name)
        except Exception as exc:  # noqa: BLE001 - reported, never raised at the caller
            logger.info("Could not resolve a use case for tracing: %s", exc)
            return UseCaseTracing(reason=f"no use case available ({exc})")

    instrument()
    if not datarobot_otel_provider_installed():
        return UseCaseTracing(reason="the OpenTelemetry provider could not be installed")

    entity_id = resolve_entity_id_from_headers(resolve_datarobot_headers_from_env() or {})
    if already_tracing and wanted and entity_id != _entity(wanted):
        # The exporter was built with the earlier entity and keeps it for the life of
        # the process, so a new id cannot take effect until the process restarts.
        return UseCaseTracing(
            entity_id=entity_id,
            reason=f"already tracing into {entity_id}; restart to use {wanted}",
        )
    return UseCaseTracing(entity_id=entity_id)


def _skip_reason() -> str:
    """Why this process must not be pointed at a use case, or "" to go ahead."""
    if os.getenv("OTEL_SDK_DISABLED", "").strip().lower() == "true":
        return "OTEL_SDK_DISABLED is set"
    if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") or os.getenv("OTEL_EXPORTER_OTLP_HEADERS"):
        # Exporting here would send this account's API key to whatever that variable
        # names. Leave the caller's own OpenTelemetry setup alone.
        return (
            "OTEL_EXPORTER_OTLP_* is set, so this run leaves your OpenTelemetry "
            "configuration alone; unset it to trace into a use case instead"
        )
    if not (resolve_api_key_from_env() and resolve_otel_traces_endpoint_from_env()):
        # Checked before any lookup, so a run that cannot export never creates a use
        # case it would have no way to send spans to.
        return "DataRobot endpoint and credentials did not resolve"
    return ""


def _entity(use_case_id: str) -> str:
    return f"{EXPERIMENT_CONTAINER_ENTITY_ID_PREFIX}{use_case_id}"


def _use_case_id_from_env() -> str:
    # DATAROBOT_DEFAULT_USE_CASE is what the app framework calls the same thing.
    return (
        os.getenv("DATAROBOT_USE_CASE_ID", "").strip()
        or os.getenv("DATAROBOT_DEFAULT_USE_CASE", "").strip()
    )


def _use_case_id_by_name(default_name: str) -> str:
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
