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
import json
import threading
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING
from typing import Any

from crewai import Crew
from crewai.events import crewai_event_bus
from crewai.events.event_bus import CrewAIEventsBus
from crewai.events.event_listener import event_listener
from crewai.events.types.task_events import TaskCompletedEvent
from crewai.events.types.task_events import TaskStartedEvent
from crewai.tools import BaseTool
from openai.types.chat import CompletionCreateParams

from datarobot_genai.core.agents.base import BaseAgent
from datarobot_genai.core.agents.base import InvokeReturn
from datarobot_genai.core.agents.base import UsageMetrics
from datarobot_genai.core.agents.base import default_usage_metrics
from datarobot_genai.core.agents.base import extract_user_prompt_content
from datarobot_genai.core.agents.base import is_streaming

from .agent import create_pipeline_interactions_from_messages
from .mcp import mcp_tools_context

if TYPE_CHECKING:
    from ragas import MultiTurnSample


class CrewAIAgent(BaseAgent[BaseTool], abc.ABC):
    """Abstract base agent for CrewAI workflows.

    Subclasses should define the ``agents`` and ``tasks`` properties
    and may override ``build_crewai_workflow`` to customize the workflow
    construction.

    Args:
        emit_task_progress: When True, yields task progress events during
            execution.
        **kwargs: Passed to BaseAgent.
    """

    def __init__(self, *, emit_task_progress: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.emit_task_progress = emit_task_progress

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

            if self.emit_task_progress:
                return self._invoke_with_task_progress(crew, user_prompt_content)

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

    async def _invoke_with_task_progress(
        self, crew: Crew, user_prompt_content: str
    ) -> AsyncGenerator[tuple[str, MultiTurnSample | None, UsageMetrics], None]:
        """Run crew yielding task progress events, then final result."""
        loop = asyncio.get_running_loop()
        event_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

        def on_task_started(source: Any, event: Any) -> None:
            task_name = (event.task.name if event.task and event.task.name else None) or "Task"
            agent_name = (
                event.task.agent.role if event.task and event.task.agent else None
            ) or "Agent"
            loop.call_soon_threadsafe(
                event_queue.put_nowait,
                {"type": "task_started", "task_name": task_name, "agent_name": agent_name},
            )

        def on_task_completed(source: Any, event: Any) -> None:
            task_name = (event.task.name if event.task and event.task.name else None) or "Task"
            agent_name = (
                event.task.agent.role if event.task and event.task.agent else None
            ) or "Agent"
            loop.call_soon_threadsafe(
                event_queue.put_nowait,
                {"type": "task_completed", "task_name": task_name, "agent_name": agent_name},
            )

        crew_result: Any = None
        crew_error: Exception | None = None

        def run_crew() -> None:
            nonlocal crew_result, crew_error
            try:
                with crewai_event_bus.scoped_handlers():
                    # scoped_handlers() clears ALL handlers including CrewAI's default ones,
                    # re-register to preserve nice console logging of the Crew progress
                    event_listener.setup_listeners(crewai_event_bus)
                    crewai_event_bus.on(TaskStartedEvent)(on_task_started)
                    crewai_event_bus.on(TaskCompletedEvent)(on_task_completed)
                    crew_result = crew.kickoff(inputs=self.make_kickoff_inputs(user_prompt_content))
            except Exception as e:
                crew_error = e
            finally:
                loop.call_soon_threadsafe(event_queue.put_nowait, None)

        thread = threading.Thread(target=run_crew, daemon=True)
        thread.start()

        empty_usage = default_usage_metrics()
        while (event := await event_queue.get()) is not None:
            yield (json.dumps({"task_progress": event}), None, empty_usage)

        await asyncio.to_thread(thread.join)

        if crew_error:
            raise crew_error

        yield self._process_crew_output(crew_result)
