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

"""Shared TracerProvider + OTLP exporter bootstrap for DataRobot agent runtimes.

DRUM bootstraps the global ``TracerProvider`` outside this repo, so generated
``custom.py`` only needs to call :func:`instrument` from :mod:`telemetry_agent`.
The DRAgent execution path has no equivalent bootstrap; without one, every
span emitted by an instrumented library lands in the no-op default provider
and is dropped. This module fills that gap.
"""

from __future__ import annotations

import atexit
import logging
import os
from urllib.parse import quote
from urllib.parse import urlparse
from urllib.parse import urlunparse

from pydantic_core import PydanticUseDefault

from datarobot_genai.drtools.core.config_utils import extract_datarobot_runtime_param_payload
from datarobot_genai.drtools.core.constants import DEFAULT_DATAROBOT_ENDPOINT
from datarobot_genai.drtools.core.constants import RUNTIME_PARAM_ENV_VAR_NAME_PREFIX

logger = logging.getLogger(__name__)

# Idempotency guard so that multiple call sites (FastAPI build_app, inline
# entry) cooperate safely. Wrapped in a single-key dict to mutate without a
# ``global`` statement (matches the pattern in :mod:`telemetry_agent`).
_BOOTSTRAP_STATE: dict[str, bool] = {"installed": False}


def _read_runtime_param(name: str, default: str = "") -> str:
    """Return the value of a DataRobot runtime param env var, unwrapped.

    DataRobot deployments expose runtime parameters as
    ``MLOPS_RUNTIME_PARAM_<NAME>`` env vars whose value is a JSON envelope
    like ``{"type":"string","payload":"value"}``. Outside deployments the
    plain ``<NAME>`` env var is honoured. The payload is unwrapped via the
    same helper Pydantic validators use elsewhere so a deployed agent and a
    local dev shell see the same values.
    """
    raw = os.environ.get(RUNTIME_PARAM_ENV_VAR_NAME_PREFIX + name) or os.environ.get(name)
    if raw is None:
        return default
    try:
        unwrapped = extract_datarobot_runtime_param_payload(raw)
    except PydanticUseDefault:
        return default
    if unwrapped is None:
        return default
    return str(unwrapped)


def _default_otlp_endpoint() -> str:
    """Derive ``{DR_ENDPOINT_HOST}/otel`` from ``DATAROBOT_ENDPOINT``."""
    parsed = urlparse(_read_runtime_param("DATAROBOT_ENDPOINT", DEFAULT_DATAROBOT_ENDPOINT))
    return urlunparse((parsed.scheme, parsed.netloc, "otel", "", "", ""))


def _otel_enabled() -> bool:
    raw = _read_runtime_param("OTEL_ENABLED", "true").strip().lower()
    return raw not in {"false", "0", "no", "off"}


def _setup_otel_env_variables() -> bool:
    """Populate ``OTEL_EXPORTER_OTLP_ENDPOINT`` / ``OTEL_EXPORTER_OTLP_HEADERS``.

    Returns True when an exporter destination is known (either pre-set in env
    or derived from a DataRobot entity id), False when there is nowhere to
    send spans (in which case the caller should skip provider setup).
    """
    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") or os.environ.get(
        "OTEL_EXPORTER_OTLP_HEADERS"
    ):
        logger.debug(
            "OTEL_EXPORTER_OTLP_ENDPOINT or OTEL_EXPORTER_OTLP_HEADERS already set, skipping"
        )
        return True

    entity_id = _read_runtime_param("OTEL_ENTITY_ID")
    if not entity_id:
        logger.info("OTEL_ENTITY_ID is not set; skipping DataRobot OTLP exporter configuration")
        return False

    api_token = _read_runtime_param("DATAROBOT_API_TOKEN")
    endpoint = _read_runtime_param("OTEL_COLLECTOR_BASE_URL", _default_otlp_endpoint())

    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = endpoint
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = (
        f"X-DataRobot-Api-Key={quote(api_token, safe='')}"
        f",X-DataRobot-Entity-Id={quote(entity_id, safe='')}"
    )
    logger.info("Configured OTLP exporter: endpoint=%s, entity_id=%s", endpoint, entity_id)
    return True


def initialize_tracer_provider(*, service_name: str) -> bool:
    """Set the global ``TracerProvider`` with an OTLP exporter, once per process.

    Idempotent: subsequent calls are no-ops once a provider has been installed.
    Returns True when a provider was installed (now or earlier in the process),
    False when telemetry is disabled or no exporter destination is configured.
    """
    if _BOOTSTRAP_STATE["installed"]:
        return True

    if not _otel_enabled():
        logger.info("OpenTelemetry is disabled via OTEL_ENABLED")
        return False

    if not _setup_otel_env_variables():
        return False

    # Local imports defer the heavyweight SDK/exporter imports until we have
    # decided to actually wire up tracing.
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create({"datarobot.service.name": service_name}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    atexit.register(provider.shutdown)

    _BOOTSTRAP_STATE["installed"] = True
    logger.info("Installed global TracerProvider for service %s", service_name)
    return True


def setup_dragent_tracing(*, service_name: str) -> None:
    """Bootstrap the tracer provider and attach NAT instrumentors.

    Single entry point used by both the DRAgent FastAPI worker and the inline
    execution path. Swallows and logs any failure so a broken telemetry stack
    cannot prevent agent execution.
    """
    try:
        from datarobot_genai.core.telemetry_agent import instrument

        if not initialize_tracer_provider(service_name=service_name):
            return
        instrument(framework="nat")
    except Exception:
        logger.exception("Failed to initialize OpenTelemetry tracing for dragent")


def _reset_for_tests() -> None:
    """Test-only hook to clear the idempotency guard."""
    _BOOTSTRAP_STATE["installed"] = False
