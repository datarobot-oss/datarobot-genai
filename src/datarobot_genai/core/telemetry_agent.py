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
from typing import Any
from typing import Literal
from typing import cast

# Suppress the "Attempting to instrument while already instrumented" warning
logging.getLogger("opentelemetry.instrumentation.instrumentor").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

# Internal instrumentation state to avoid 'global' mutation warnings
_INSTRUMENTATION_STATE = {"http": False, "openai": False, "threading": False}
_INSTRUMENTED_FRAMEWORKS: set[str] = set()


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


def _instrument_framework(framework: str) -> None:
    if framework in _INSTRUMENTED_FRAMEWORKS:
        return
    try:
        if framework == "crewai":
            crewai_module = importlib.import_module("opentelemetry.instrumentation.crewai")
            crewai_instrumentor = getattr(crewai_module, "CrewAIInstrumentor")
            crewai_instrumentor().instrument()
            os.environ.setdefault("CREWAI_TESTING", "true")
        elif framework == "langgraph":
            # Provided by opentelemetry-instrumentation-langchain
            langchain_module = importlib.import_module("opentelemetry.instrumentation.langchain")
            langchain_instrumentor = getattr(langchain_module, "LangchainInstrumentor")
            langchain_instrumentor().instrument()
        elif framework == "llamaindex":
            llamaindex_module = importlib.import_module("opentelemetry.instrumentation.llamaindex")
            llamaindex_instrumentor = getattr(llamaindex_module, "LlamaIndexInstrumentor")
            # LlamaIndex instrumentor lacks precise typing; cast to Any to avoid mypy complaints
            cast(Any, llamaindex_instrumentor()).instrument()
        elif framework == "nat":
            _instrument_framework("crewai")
            _instrument_framework("langgraph")
            _instrument_framework("llamaindex")
        _INSTRUMENTED_FRAMEWORKS.add(framework)
    except Exception as e:
        logger.debug(f"{framework} instrumentation skipped: {e}")


def instrument(
    framework: Literal["crewai", "langgraph", "llamaindex", "nat"] | None = None,
) -> None:
    """Idempotently instrument supported HTTP clients, OpenAI SDK, and optionally a framework.

    Also disables telemetry for some third-party libraries to avoid duplicate/undesired tracking.
    """
    # Some libraries collect telemetry data by default. Disable that.
    os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
    os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

    _instrument_threading()
    _instrument_http_clients()
    _instrument_openai()
    if framework:
        _instrument_framework(framework)
