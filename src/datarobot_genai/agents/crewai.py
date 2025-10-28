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

from __future__ import annotations

import importlib
from typing import Any

from ragas import MultiTurnSample

from ..utils.urls import get_api_base

# Lazily-resolved handle to CrewAI's LLM class so tests can monkeypatch
# the symbol without requiring the extra to be installed at import time.
LLM: Any | None = None


def build_llm(
    *,
    api_base: str,
    api_key: str | None,
    model: str,
    deployment_id: str | None,
    timeout: int,
) -> Any:
    """Create a CrewAI LLM configured for DataRobot LLM Gateway or deployment."""
    base = get_api_base(api_base, deployment_id)
    # Resolve CrewAI's LLM lazily so importing this module doesn't require the extra
    if LLM is not None:
        llm_cls = LLM
    else:
        llm_cls = importlib.import_module("crewai").LLM
    return llm_cls(model=model, api_base=base, api_key=api_key, timeout=timeout)


def create_pipeline_interactions_from_messages(
    messages: list[Any] | None,
) -> MultiTurnSample | None:
    if not messages:
        return None
    return MultiTurnSample(user_input=messages)
