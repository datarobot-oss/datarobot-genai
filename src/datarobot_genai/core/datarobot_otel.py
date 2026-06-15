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

"""DataRobot OTel ingest endpoint helpers.

Two responsibilities, intentionally co-located so a single import covers
both the NAT telemetry exporter
(``datarobot_genai.dragent.plugins.datarobot_otelcollector``) and
the framework-instrumentor bootstrap (``datarobot_genai.core.telemetry_agent``):

* ``resolve_*_from_env`` — read ``MLOPS_DEPLOYMENT_ID`` / ``DATAROBOT_API_TOKEN`` /
  ``DATAROBOT_(PUBLIC_)ENDPOINT`` and shape them into the values the OTel ingest
  expects (``deployment-<id>`` entity id; ``<host>/otel/v1/traces`` endpoint).
* ``bootstrap_otel_provider_for_datarobot`` — install a global OTel SDK
  ``TracerProvider`` pointed at the DataRobot ingest so framework
  auto-instrumentors (``LangchainInstrumentor``, ``CrewAIInstrumentor``,
  ``LlamaIndexInstrumentor``) actually export spans. NAT's own exporter
  pipeline (``OTLPSpanAdapterExporter``) does not touch the OTel SDK's global
  ``TracerProvider``, so without this bootstrap, framework spans go to the
  default no-op tracer and never reach DataRobot.
"""

from __future__ import annotations

import logging
import os
import urllib.parse

logger = logging.getLogger(__name__)

# Idempotency state for ``bootstrap_otel_provider_for_datarobot``. Mutable
# dict so the inner ``["installed"]`` write doesn't need a ``global``
# statement (mirrors ``_INSTRUMENTATION_STATE`` in ``telemetry_agent.py``).
# Repeated calls — which happen via plugin discovery + custom.py in the same
# process — short-circuit on this flag and don't trip OTel's "overriding"
# warning.
_BOOTSTRAP_STATE: dict[str, bool] = {"installed": False}

# The DataRobot OTel ingest expects entity ids (and deployment-derived service
# names) in the ``deployment-<id>`` shape. Single source of truth so the
# resolver that produces it and the validator that checks it can't drift.
ENTITY_ID_PREFIX = "deployment-"


def resolve_api_key_from_env() -> str:
    return os.getenv("DATAROBOT_API_TOKEN", "")


def resolve_entity_id_from_env() -> str:
    # MLOPS_DEPLOYMENT_ID holds the bare deployment ID inside a DR deployment;
    # auto-prepend the 'deployment-' prefix required by the OTel ingest path.
    # Mirrors the MLOPS_DEPLOYMENT_ID-driven pattern used by the A2A frontend.
    deployment_id = os.getenv("MLOPS_DEPLOYMENT_ID", "")
    if deployment_id:
        return f"{ENTITY_ID_PREFIX}{deployment_id}"
    # Fall back to OTEL_ENTITY_ID (set by Pulumi infra / af-component-agent)
    # so local dev and non-DRUM deployments can still export spans.
    return os.getenv("OTEL_ENTITY_ID", "")


def resolve_otel_endpoint_from_env() -> str:
    # Derive from the DR API base URL: e.g. https://app.datarobot.com/api/v2
    # → https://app.datarobot.com/otel/v1/traces. The OTel collector ingress
    # lives at the same host, off /otel/v1/traces, not under /api/v2. We
    # only honour explicitly set env vars (no built-in default) so an
    # unconfigured env never silently targets app.datarobot.com.
    base = os.getenv("DATAROBOT_PUBLIC_API_ENDPOINT") or os.getenv("DATAROBOT_ENDPOINT")
    if not base:
        return ""
    parsed = urllib.parse.urlsplit(base)
    if not parsed.scheme or not parsed.netloc:
        return ""
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "/otel/v1/traces", "", ""))


def _resolve_service_name() -> str:
    # Prefer the standard OTel env override, then a deployment-derived name,
    # then a generic fallback so spans always have *some* service identity.
    if name := os.getenv("OTEL_SERVICE_NAME"):
        return name
    if deployment_id := os.getenv("MLOPS_DEPLOYMENT_ID"):
        return f"{ENTITY_ID_PREFIX}{deployment_id}"
    return "datarobot-agent"


def bootstrap_otel_provider_for_datarobot() -> bool:
    """Ensure framework auto-instrumentor spans reach the DataRobot OTel ingest.

    Sibling to (not replacing) the NAT-side ``datarobot_otelcollector`` exporter:
    NAT pipes its own ``IntermediateStep``-derived spans through its own
    channel; this path serves framework auto-instrumentors that emit spans
    through the OTel SDK's standard ``trace.get_tracer(...)`` API.

    Two modes, picked at call time based on what's already in the global
    ``TracerProvider`` slot:

    * **install** — slot still holds the default ``ProxyTracerProvider``: we
      create a new SDK ``TracerProvider`` with the DataRobot exporter and set
      it globally.
    * **attach** — slot already holds an SDK ``TracerProvider`` (e.g. the
      ``dragent_fastapi`` server's startup telemetry layer installs one before
      NAT plugin discovery happens): we add our own ``BatchSpanProcessor`` to
      it instead of replacing it. The pre-existing provider's resource
      (including its ``service.name``) is kept — DataRobot's OTel ingest
      routes off the ``X-DataRobot-*`` headers we set on the exporter, not off
      resource attributes, so the merge is safe.

    Returns ``True`` when a processor was installed or attached by this call,
    ``False`` (silently) when:

    * the DataRobot deployment env is incomplete (``MLOPS_DEPLOYMENT_ID``,
      ``DATAROBOT_API_TOKEN``, or ``DATAROBOT_(PUBLIC_)ENDPOINT`` missing) —
      the local-dev / CI shape;
    * something other than an SDK ``TracerProvider`` or the default proxy is
      already installed (we can't attach to an unknown provider type);
    * this function has already run successfully in this process.
    """
    if _BOOTSTRAP_STATE["installed"]:
        return False

    api_key = resolve_api_key_from_env()
    entity_id = resolve_entity_id_from_env()
    endpoint = resolve_otel_endpoint_from_env()
    if not api_key or not entity_id or not endpoint:
        logger.info(
            "Skipping OTel TracerProvider bootstrap: DataRobot deployment env "
            "(MLOPS_DEPLOYMENT_ID / DATAROBOT_API_TOKEN / "
            "DATAROBOT_(PUBLIC_)ENDPOINT) not fully set."
        )
        return False

    # Imported lazily so test environments without the SDK don't pay the cost
    # at module import. All three packages are pinned by the dragent extra.
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.trace import ProxyTracerProvider

    try:
        # Inspect the global slot before building anything: a BatchSpanProcessor
        # spawns a worker thread (and the exporter an HTTP session) at
        # construction, so we must not build them on the skip path below.
        current = trace.get_tracer_provider()
        if not isinstance(current, (ProxyTracerProvider, TracerProvider)):
            logger.debug(
                "Skipping OTel TracerProvider bootstrap: non-SDK provider "
                "already installed (%s); cannot attach a span processor.",
                type(current).__name__,
            )
            return False

        exporter = OTLPSpanExporter(
            endpoint=endpoint,
            headers={
                "X-DataRobot-Api-Key": api_key,
                "X-DataRobot-Entity-Id": entity_id,
            },
        )
        processor = BatchSpanProcessor(exporter)

        from datarobot_genai.core.telemetry_nat_tracer import wrap_sdk_tracer_provider

        if isinstance(current, ProxyTracerProvider):
            sdk_version = _get_opentelemetry_sdk_version()
            resource = Resource.create(
                {
                    "telemetry.sdk.language": "python",
                    "telemetry.sdk.name": "opentelemetry",
                    "telemetry.sdk.version": sdk_version,
                    "service.name": _resolve_service_name(),
                }
            )
            provider = wrap_sdk_tracer_provider(TracerProvider(resource=resource))
            provider.add_span_processor(processor)
            trace.set_tracer_provider(provider)
            action = "installed"
        else:  # SDK TracerProvider already installed — attach to it.
            provider = wrap_sdk_tracer_provider(current)
            provider.add_span_processor(processor)
            action = "attached"
    except Exception:
        # Never let telemetry setup take down the agent. Log with traceback so
        # operators can diagnose without crashing user code.
        logger.exception("Failed to bootstrap DataRobot OTel TracerProvider")
        return False

    _BOOTSTRAP_STATE["installed"] = True
    logger.info(
        "DataRobot OTel span processor %s → %s (entity_id=%s)",
        action,
        endpoint,
        entity_id,
    )
    return True


def _get_opentelemetry_sdk_version() -> str:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version

    try:
        return version("opentelemetry-sdk")
    except PackageNotFoundError:
        return "unknown"
