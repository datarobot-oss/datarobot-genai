# Copyright 2025 DataRobot, Inc. and its affiliates.
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

"""Lightweight, idempotent client/framework instrumentation for agents."""

from __future__ import annotations

import importlib
import logging
import os

from datarobot_genai.core.runtime import is_hosted_runtime

# Suppress the "Attempting to instrument while already instrumented" warning
logging.getLogger("opentelemetry.instrumentation.instrumentor").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

# Internal instrumentation state to avoid 'global' mutation warnings
_INSTRUMENTATION_STATE = {"http": False, "openai": False, "threading": False}


def _instrument_threading() -> None:
    if _INSTRUMENTATION_STATE["threading"]:
        return
    try:
        threading_module = importlib.import_module("opentelemetry.instrumentation.threading")
        threading_instrumentor = getattr(threading_module, "ThreadingInstrumentor")
        threading_instrumentor().instrument()
        _INSTRUMENTATION_STATE["threading"] = True
    except Exception as e:
        logger.debug(f"threading instrumentation skipped: {e}")


def _instrument_http_clients() -> None:
    if _INSTRUMENTATION_STATE["http"]:
        return
    try:
        requests_module = importlib.import_module("opentelemetry.instrumentation.requests")
        requests_instrumentor = getattr(requests_module, "RequestsInstrumentor")
        requests_instrumentor().instrument()
    except Exception as e:
        logger.debug(f"requests instrumentation skipped: {e}")
    try:
        aiohttp_module = importlib.import_module("opentelemetry.instrumentation.aiohttp_client")
        aiohttp_instrumentor = getattr(aiohttp_module, "AioHttpClientInstrumentor")
        aiohttp_instrumentor().instrument()
    except Exception as e:
        logger.debug(f"aiohttp instrumentation skipped: {e}")
    try:
        httpx_module = importlib.import_module("opentelemetry.instrumentation.httpx")
        httpx_instrumentor = getattr(httpx_module, "HTTPXClientInstrumentor")
        httpx_instrumentor().instrument()
    except Exception as e:
        logger.debug(f"httpx instrumentation skipped: {e}")
    _INSTRUMENTATION_STATE["http"] = True


def _instrument_openai() -> None:
    if _INSTRUMENTATION_STATE["openai"]:
        return
    try:
        openai_module = importlib.import_module("opentelemetry.instrumentation.openai")
        openai_instrumentor = getattr(openai_module, "OpenAIInstrumentor")
        openai_instrumentor().instrument()
        _INSTRUMENTATION_STATE["openai"] = True
    except Exception as e:
        logger.debug(f"openai instrumentation skipped: {e}")


def instrument() -> None:
    """Idempotently instrument supported HTTP clients and OpenAI SDK.

    Also disables telemetry for some third-party libraries to avoid duplicate/undesired tracking.
    """
    # Some libraries collect telemetry data by default. Disable that.
    os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

    # Install a global OTel TracerProvider pointed at the DataRobot OTel
    # ingest before any instrumentor patches a framework. NAT's
    # datarobot_otelcollector exporter pipes its own IntermediateStep-derived
    # spans through a separate channel that does not touch the OTel SDK
    # global provider — so without this bootstrap, the framework
    # instrumentors patched below would emit spans through a no-op tracer
    # and nothing reaches DataRobot.
    #
    # TODO (BUZZOK-31396): Call bootstrap from the deployment/notebook entrypoint instead of
    # here so notebook hosts that already install their own TracerProvider
    # (via setup_otel_env_variables) are not double-bootstrapped. See
    # https://github.com/datarobot/datarobot-user-models/blob/master/public_dropin_environments/python311_genai_agents/run_agent.py#L188
    if is_hosted_runtime():
        from .datarobot_otel import bootstrap_otel_provider_for_datarobot

        bootstrap_otel_provider_for_datarobot()

    _instrument_threading()
    _instrument_http_clients()
    _instrument_openai()
