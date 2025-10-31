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

from crewai import LLM
from ragas import MultiTurnSample
from ragas.messages import AIMessage
from ragas.messages import HumanMessage
from ragas.messages import ToolMessage

from datarobot_genai.core.utils.urls import get_api_base


def build_llm(
    *,
    api_base: str,
    api_key: str | None,
    model: str,
    deployment_id: str | None,
    timeout: int,
) -> LLM:
    """Create a CrewAI LLM configured for DataRobot LLM Gateway or deployment."""
    base = get_api_base(api_base, deployment_id)
    return LLM(model=model, api_base=base, api_key=api_key, timeout=timeout)


def create_pipeline_interactions_from_messages(
    messages: list[HumanMessage | AIMessage | ToolMessage] | None,
) -> MultiTurnSample | None:
    if not messages:
        return None
    return MultiTurnSample(user_input=messages)
