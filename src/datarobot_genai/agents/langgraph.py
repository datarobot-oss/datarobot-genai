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

from typing import Any

from langchain_core.messages import ToolMessage
from ragas import MultiTurnSample
from ragas.integrations.langgraph import convert_to_ragas_messages


def create_pipeline_interactions_from_events(
    events: list[dict[str, Any]] | None,
) -> MultiTurnSample | None:
    """Convert a list of LangGraph events into Ragas MultiTurnSample."""
    if not events:
        return None
    messages = []
    for e in events:
        for _, v in e.items():
            messages.extend(v.get("messages", []))
    messages = [m for m in messages if not isinstance(m, ToolMessage)]
    ragas_trace = convert_to_ragas_messages(messages)
    return MultiTurnSample(user_input=ragas_trace)
