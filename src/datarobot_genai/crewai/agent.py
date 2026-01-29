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
Base class for CrewAI-based agents.

Manages MCP tool lifecycle and standardizes kickoff flow.

Note: This base does not capture pipeline interactions; it returns None by
default. Subclasses may implement message capture if they need interactions.
"""

from __future__ import annotations

import abc
import asyncio
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING
from typing import Any

from crewai import Crew
from crewai.events.event_bus import CrewAIEventsBus
from crewai.tools import BaseTool
from openai.types.chat import CompletionCreateParams

from datarobot_genai.core.agents.base import BaseAgent
from datarobot_genai.core.agents.base import InvokeReturn
from datarobot_genai.core.agents.base import UsageMetrics
from datarobot_genai.core.agents.base import default_usage_metrics
from datarobot_genai.core.agents.base import extract_user_prompt_content
from datarobot_genai.core.agents.base import is_streaming

from .mcp import mcp_tools_context

if TYPE_CHECKING:
    from ragas import MultiTurnSample
    from ragas.messages import AIMessage
    from ragas.messages import HumanMessage
    from ragas.messages import ToolMessage


def create_pipeline_interactions_from_messages(
    messages: list[HumanMessage | AIMessage | ToolMessage] | None,
) -> MultiTurnSample | None:
    if not messages:
        return None
    # Lazy import to reduce memory overhead when ragas is not used
    from ragas import MultiTurnSample

    return MultiTurnSample(user_input=messages)


class CrewAIAgent(BaseAgent[BaseTool], abc.ABC):
    """Abstract base agent for CrewAI workflows.

    Subclasses should define the ``agents`` and ``tasks`` properties
    and may override ``build_crewai_workflow`` to customize the workflow
    construction.
    """

    @property
    @abc.abstractmethod
    def agents(self) -> list[Any]:  # CrewAI Agent list
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def tasks(self) -> list[Any]:  # CrewAI Task list
        raise NotImplementedError

    def build_crewai_workflow(self) -> Any:
        """Create a CrewAI workflow instance.

        Default implementation constructs a Crew with provided agents and tasks.
        Subclasses can override to customize Crew options.
        """
        return Crew(agents=self.agents, tasks=self.tasks, verbose=self.verbose)

    @abc.abstractmethod
    def make_kickoff_inputs(self, user_prompt_content: str) -> dict[str, Any]:
        """Build the inputs dict for ``Crew.kickoff``.

        Subclasses must implement this to provide the exact inputs required
        by their CrewAI tasks.
        """
        raise NotImplementedError

    def _extract_pipeline_interactions(self) -> MultiTurnSample | None:
        """Extract pipeline interactions from event listener if available."""
        if not hasattr(self, "event_listener"):
            return None
        try:
            listener = getattr(self, "event_listener", None)
            messages = getattr(listener, "messages", None) if listener is not None else None
            return create_pipeline_interactions_from_messages(messages)
        except Exception:
            return None

    def _extract_usage_metrics(self, crew_output: Any) -> UsageMetrics:
        """Extract usage metrics from crew output."""
        token_usage = getattr(crew_output, "token_usage", None)
        if token_usage is not None:
            return {
                "completion_tokens": int(getattr(token_usage, "completion_tokens", 0)),
                "prompt_tokens": int(getattr(token_usage, "prompt_tokens", 0)),
                "total_tokens": int(getattr(token_usage, "total_tokens", 0)),
            }
        return default_usage_metrics()

    def _process_crew_output(
        self, crew_output: Any
    ) -> tuple[str, MultiTurnSample | None, UsageMetrics]:
        """Process crew output into response tuple."""
        response_text = str(crew_output.raw)
        pipeline_interactions = self._extract_pipeline_interactions()
        usage_metrics = self._extract_usage_metrics(crew_output)
        return response_text, pipeline_interactions, usage_metrics

    async def invoke(self, completion_create_params: CompletionCreateParams) -> InvokeReturn:
        """Run the CrewAI workflow with the provided completion parameters."""
        user_prompt_content = extract_user_prompt_content(completion_create_params)
        # Preserve prior template startup print for CLI parity
        try:
            print("Running agent with user prompt:", user_prompt_content, flush=True)
        except Exception:
            # Printing is best-effort; proceed regardless
            pass

        # Use MCP context manager to handle connection lifecycle
        with mcp_tools_context(
            authorization_context=self._authorization_context,
            forwarded_headers=self.forwarded_headers,
        ) as mcp_tools:
            # Set MCP tools for all agents if MCP is not configured this is effectively a no-op
            self.set_mcp_tools(mcp_tools)

            # If an event listener is provided by the subclass/template, register it
            if hasattr(self, "event_listener") and CrewAIEventsBus is not None:
                try:
                    listener = getattr(self, "event_listener")
                    setup_fn = getattr(listener, "setup_listeners", None)
                    if callable(setup_fn):
                        setup_fn(CrewAIEventsBus)
                except Exception:
                    # Listener is optional best-effort; proceed without failing invoke
                    pass

            crew = self.build_crewai_workflow()

            if is_streaming(completion_create_params):

                async def _gen() -> AsyncGenerator[
                    tuple[str, MultiTurnSample | None, UsageMetrics]
                ]:
                    crew_output = await asyncio.to_thread(
                        crew.kickoff,
                        inputs=self.make_kickoff_inputs(user_prompt_content),
                    )
                    yield self._process_crew_output(crew_output)

                return _gen()

            crew_output = crew.kickoff(inputs=self.make_kickoff_inputs(user_prompt_content))
            return self._process_crew_output(crew_output)
