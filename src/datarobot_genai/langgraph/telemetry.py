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

"""Idempotent LangChain/LangGraph auto-instrumentation for agent telemetry."""

from __future__ import annotations

import logging

from opentelemetry.instrumentation.langchain import LangchainInstrumentor

logger = logging.getLogger(__name__)

_INSTRUMENTED = {"langchain": False}


def instrument() -> None:
    """Idempotently enable the LangChain OpenTelemetry instrumentor."""
    if _INSTRUMENTED["langchain"]:
        logger.info("Langchain instrumentation already enabled")
        return
    try:
        LangchainInstrumentor().instrument()
        _INSTRUMENTED["langchain"] = True
    except Exception as e:
        logger.info(f"Langchain instrumentation failed: {e}")
