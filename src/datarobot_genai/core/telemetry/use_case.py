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

The platform tells a deployment or workload which entity it is. A notebook or a
script has to name one itself, and a use case is the natural home for local runs.
"""

from __future__ import annotations

import logging
import os

from datarobot_genai.core.runtime import is_hosted_runtime
from datarobot_genai.core.telemetry.agent import instrument
from datarobot_genai.core.telemetry.datarobot_otel import bootstrap_otel_provider_for_datarobot
from datarobot_genai.core.telemetry.datarobot_otel import datarobot_otel_entity_id
from datarobot_genai.core.telemetry.datarobot_otel import resolve_api_key_from_env
from datarobot_genai.core.telemetry.datarobot_otel import resolve_otel_traces_endpoint_from_env

logger = logging.getLogger(__name__)

# A use case is an "experiment container" to the OTel ingest, its pre-rename name.
ENTITY_PREFIX = "experiment_container-"


def trace_to_use_case(default_name: str, use_case_id: str = "") -> str:
    """Export this process's spans to a DataRobot use case, and return its id.

    Takes ``use_case_id``, else ``DATAROBOT_USE_CASE_ID`` or
    ``DATAROBOT_DEFAULT_USE_CASE``, else the use case named ``default_name``, created on
    first use. Also call your framework's ``instrument()`` for framework spans.

    Returns "" when spans are not going to a use case, having logged why: a deployment
    or workload is named by the platform, and an ``OTEL_EXPORTER_OTLP_*`` of your own is
    left alone. Never raises.
    """
    if reason := _skip_reason():
        logger.info("Not tracing to a use case: %s", reason)
        return ""

    # DATAROBOT_DEFAULT_USE_CASE is what the app framework calls the same thing.
    wanted = (
        use_case_id.strip()
        or os.getenv("DATAROBOT_USE_CASE_ID", "").strip()
        or os.getenv("DATAROBOT_DEFAULT_USE_CASE", "").strip()
    )
    if not datarobot_otel_entity_id() and not is_hosted_runtime():
        try:
            selected = wanted or _use_case_id_by_name(default_name)
        except Exception as exc:  # noqa: BLE001 - reported, never raised at the caller
            logger.warning("Not tracing: no use case available (%s)", exc)
            return ""
        # So the rest of the process, and anything it spawns, agrees on the use case.
        os.environ["DATAROBOT_USE_CASE_ID"] = selected
        # instrument() only bootstraps a hosted runtime, so a local run asks here.
        bootstrap_otel_provider_for_datarobot(f"{ENTITY_PREFIX}{selected}")

    instrument()
    # Read back what was installed, so a repeat call reports where spans really go
    # rather than what this call asked for.
    entity_id = datarobot_otel_entity_id()
    if not entity_id.startswith(ENTITY_PREFIX):
        logger.info("Not tracing to a use case; entity is %s", entity_id or "unset")
        return ""
    return entity_id.removeprefix(ENTITY_PREFIX)


def _skip_reason() -> str:
    """Why this process must not be pointed at a use case, or "" to go ahead."""
    if os.getenv("OTEL_SDK_DISABLED", "").strip().lower() == "true":
        return "OTEL_SDK_DISABLED is set"
    if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") or os.getenv("OTEL_EXPORTER_OTLP_HEADERS"):
        # Exporting would send this account's API key wherever that variable points.
        return (
            "OTEL_EXPORTER_OTLP_* is set, so this run leaves your OpenTelemetry "
            "configuration alone; unset it to trace into a use case instead"
        )
    if not (resolve_api_key_from_env() and resolve_otel_traces_endpoint_from_env()):
        # Before any lookup, so a run that cannot export never creates a use case.
        return "DataRobot endpoint and credentials did not resolve"
    return ""


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
