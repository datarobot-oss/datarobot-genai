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

"""Idempotent CrewAI auto-instrumentation for agent telemetry."""

from __future__ import annotations

import logging
import os

from opentelemetry.instrumentation.crewai import CrewAIInstrumentor

logger = logging.getLogger(__name__)

_INSTRUMENTED = {"crewai": False}


def instrument() -> None:
    """Idempotently enable the CrewAI OpenTelemetry instrumentor."""
    if _INSTRUMENTED["crewai"]:
        logger.info("CrewAI instrumentation already enabled")
        return
    try:
        CrewAIInstrumentor().instrument()
        os.environ.setdefault("CREWAI_TESTING", "true")
        _INSTRUMENTED["crewai"] = True
    except Exception as e:
        logger.info(f"CrewAI instrumentation failed: {e}")
