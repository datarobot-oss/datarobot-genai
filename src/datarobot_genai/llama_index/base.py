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

"""
Base class for LlamaIndex-based agents.

Provides a standard ``invoke`` that runs an AgentWorkflow, collects events,
and converts them into pipeline interactions. Subclasses provide the workflow
and response extraction logic.
"""

from __future__ import annotations

import abc
import inspect
from collections.abc import AsyncGenerator
from typing import Any

from openai.types.chat import CompletionCreateParams
from ragas import MultiTurnSample

from datarobot_genai.core.agents.base import BaseAgent
from datarobot_genai.core.agents.base import InvokeReturn
from datarobot_genai.core.agents.base import UsageMetrics
from datarobot_genai.core.agents.base import default_usage_metrics
from datarobot_genai.core.agents.base import extract_user_prompt_content
from datarobot_genai.core.agents.base import is_streaming

from .agent import create_pipeline_interactions_from_events


class LlamaIndexAgent(BaseAgent, abc.ABC):
    """Abstract base agent for LlamaIndex workflows."""

    @abc.abstractmethod
    def build_workflow(self) -> Any:
        """Return an AgentWorkflow instance ready to run."""
        raise NotImplementedError

    @abc.abstractmethod
    def extract_response_text(self, result_state: Any, events: list[Any]) -> str:
        """Extract final response text from workflow state and/or events."""
        raise NotImplementedError

    def make_input_message(self, completion_create_params: CompletionCreateParams) -> str:
        """Create an input string for the workflow from the user prompt."""
        user_prompt_content = extract_user_prompt_content(completion_create_params)
        return str(user_prompt_content)

    async def invoke(self, completion_create_params: CompletionCreateParams) -> InvokeReturn:
        """Run the LlamaIndex workflow with the provided completion parameters."""
        input_message = self.make_input_message(completion_create_params)

        workflow = self.build_workflow()
        handler = workflow.run(user_msg=input_message)

        events: list[Any] = []
        async for event in handler.stream_events():
            events.append(event)

        # Extract state from workflow context (supports sync/async get or attribute)
        state = None
        ctx = getattr(handler, "ctx", None)
        try:
            if ctx is not None:
                get = getattr(ctx, "get", None)
                if callable(get):
                    result = get("state")
                    state = await result if inspect.isawaitable(result) else result
                elif hasattr(ctx, "state"):
                    state = getattr(ctx, "state")
        except (AttributeError, TypeError):
            state = None
        response_text = self.extract_response_text(state, events)

        pipeline_interactions = create_pipeline_interactions_from_events(events)

        usage_metrics: UsageMetrics = default_usage_metrics()
        if is_streaming(completion_create_params):

            async def _gen() -> AsyncGenerator[tuple[str, MultiTurnSample | None, UsageMetrics]]:
                yield response_text, pipeline_interactions, usage_metrics

            return _gen()

        return response_text, pipeline_interactions, usage_metrics
