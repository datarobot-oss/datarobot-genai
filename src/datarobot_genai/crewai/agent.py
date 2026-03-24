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
import uuid
from typing import TYPE_CHECKING
from typing import Any

from ag_ui.core import EventType
from ag_ui.core import ReasoningEndEvent
from ag_ui.core import ReasoningMessageContentEvent
from ag_ui.core import ReasoningStartEvent
from ag_ui.core import RunAgentInput
from ag_ui.core import RunFinishedEvent
from ag_ui.core import RunStartedEvent
from ag_ui.core import StepFinishedEvent
from ag_ui.core import StepStartedEvent
from ag_ui.core import TextMessageChunkEvent
from ag_ui.core import TextMessageContentEvent
from ag_ui.core import TextMessageEndEvent
from ag_ui.core import TextMessageStartEvent
from crewai import Crew
from crewai.events import crewai_event_bus
from crewai.tools import BaseTool
from crewai.types.streaming import CrewStreamingOutput
from crewai.types.streaming import StreamChunkType

from datarobot_genai.core.agents.base import BaseAgent
from datarobot_genai.core.agents.base import InvokeReturn
from datarobot_genai.core.agents.base import UsageMetrics
from datarobot_genai.core.agents.base import default_usage_metrics
from datarobot_genai.core.agents.base import extract_user_prompt_content
from datarobot_genai.crewai.ragas_events import CrewAIRagasEventListener
from datarobot_genai.crewai.streaming_events import CrewAIStreamingEventListener

from .mcp import mcp_tools_context

if TYPE_CHECKING:
    from ragas import MultiTurnSample
    from ragas.messages import AIMessage
    from ragas.messages import HumanMessage
    from ragas.messages import ToolMessage


class CrewAIAgent(BaseAgent[BaseTool], abc.ABC):
    """Abstract base agent for CrewAI workflows.

    Subclasses should define the ``agents`` and ``tasks`` properties
    and may override ``crew`` to customize the workflow construction.
    """

    @property
    @abc.abstractmethod
    def agents(self) -> list[Any]:  # CrewAI Agent list
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def tasks(self) -> list[Any]:  # CrewAI Task list
        raise NotImplementedError

    def crew(self) -> Crew:
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

        Expected inputs:

        - ``topic`` (or similar): The user's prompt content. This is required
          and should be passed through to CrewAI tasks that use placeholders
          like ``{topic}`` in their descriptions.

        - ``chat_history`` (optional): Include this key with an empty string
          value (``""``) to opt into automatic chat history injection. When
          present, the base class will populate it with a plain-text summary
          of prior conversation turns. Use ``{chat_history}`` in agent
          goals/backstories or task descriptions to reference it.

        Returns
        -------
        dict[str, Any]
            A dictionary of inputs that will be passed to ``Crew.kickoff()``.
            The dictionary keys should match the placeholder names used in
            your CrewAI task descriptions and agent configurations.

        Examples
        --------
        Basic implementation (no history):

        .. code-block:: python

            def make_kickoff_inputs(self, user_prompt_content: str) -> dict[str, Any]:
                return {"topic": user_prompt_content}

        With chat history opt-in:

        .. code-block:: python

            def make_kickoff_inputs(self, user_prompt_content: str) -> dict[str, Any]:
                return {
                    "topic": user_prompt_content,
                    "chat_history": "",  # Will be auto-populated with prior turns
                }
        """
        raise NotImplementedError

    @classmethod
    def create_pipeline_interactions_from_messages(
        cls,
        messages: list[HumanMessage | AIMessage | ToolMessage] | None,
    ) -> MultiTurnSample | None:
        if not messages:
            return None
        # Lazy import to reduce memory overhead when ragas is not used
        from ragas import MultiTurnSample

        return MultiTurnSample(user_input=messages)

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

    async def invoke(self, run_agent_input: RunAgentInput) -> InvokeReturn:
        """Run the CrewAI workflow with the provided completion parameters."""
        user_prompt_content = extract_user_prompt_content(run_agent_input)
        # Preserve prior template startup print for CLI parity
        try:
            print("Running agent with user prompt:", user_prompt_content, flush=True)
        except Exception:
            # Printing is best-effort; proceed regardless
            pass

        thread_id = run_agent_input.thread_id
        run_id = run_agent_input.run_id

        zero_metrics: UsageMetrics = {
            "completion_tokens": 0,
            "prompt_tokens": 0,
            "total_tokens": 0,
        }
        # Partial AG-UI: workflow lifecycle + text message events
        yield (
            RunStartedEvent(type=EventType.RUN_STARTED, thread_id=thread_id, run_id=run_id),
            None,
            default_usage_metrics(),
        )

        pipeline_interactions: MultiTurnSample | None = None
        usage_metrics = default_usage_metrics()

        # Use MCP context manager to handle connection lifecycle
        with mcp_tools_context(
            authorization_context=self.authorization_context,
            forwarded_headers=self.forwarded_headers,
        ) as mcp_tools:
            # Set MCP tools for all agents if MCP is not configured this is effectively a no-op
            self.set_mcp_tools(mcp_tools)

            with crewai_event_bus.scoped_handlers():
                ragas_event_listener = CrewAIRagasEventListener()
                ragas_event_listener.setup_listeners(crewai_event_bus)
                streaming_event_listener = CrewAIStreamingEventListener()
                streaming_event_listener.setup_listeners(crewai_event_bus)

                crew = self.crew()

                kickoff_inputs = self.make_kickoff_inputs(str(user_prompt_content))
                # Chat history is opt-in: only populate it if the agent/template
                # declares a `chat_history` kickoff input (i.e. it uses `{chat_history}`
                # in prompts).
                if "chat_history" in kickoff_inputs:
                    history_summary = self.build_history_summary(run_agent_input)
                    existing_history_text = str(kickoff_inputs.get("chat_history") or "")

                    if history_summary and not existing_history_text.strip():
                        kickoff_inputs["chat_history"] = (
                            f"\n\nPrior conversation:\n{history_summary}"
                        )
                message_id = str(uuid.uuid4())
                crew_output = await crew.kickoff_async(inputs=kickoff_inputs)

                if isinstance(crew_output, CrewStreamingOutput):
                    current_task = ""
                    reasoning_started = False
                    step_started = False
                    text_started = False
                    async for chunk in crew_output:
                        # Show task transitions
                        if chunk.task_name != current_task:
                            current_task = chunk.task_name
                            print(f"\n[{chunk.agent_role}] Working on: {chunk.task_name}")
                            print("-" * 60)

                        if streaming_event_listener.reasoning_event:
                            if not reasoning_started:
                                yield (
                                    ReasoningStartEvent(
                                        type=EventType.REASONING_START,
                                        message_id=message_id,
                                    ),
                                    None,
                                    zero_metrics,
                                )
                                reasoning_started = True
                        elif reasoning_started:
                            yield (
                                ReasoningEndEvent(
                                    type=EventType.REASONING_END,
                                    message_id=message_id,
                                ),
                                None,
                                zero_metrics,
                            )
                            reasoning_started = False

                        if streaming_event_listener.step_event:
                            if not step_started:
                                yield (
                                    StepStartedEvent(
                                        type=EventType.STEP_STARTED,
                                        step_name=current_task,
                                    ),
                                    None,
                                    zero_metrics,
                                )
                                step_started = True
                        elif step_started:
                            yield (
                                StepFinishedEvent(
                                    type=EventType.STEP_FINISHED,
                                    step_name=current_task,
                                ),
                                None,
                                zero_metrics,
                            )
                            step_started = False

                        # Display text chunks
                        if chunk.chunk_type == StreamChunkType.TEXT:
                            if streaming_event_listener.reasoning_event:
                                yield (
                                    ReasoningMessageContentEvent(
                                        type=EventType.REASONING_MESSAGE_CONTENT,
                                        message_id=message_id,
                                        delta=chunk.content,
                                    ),
                                    None,
                                    zero_metrics,
                                )
                            else:
                                if not text_started:
                                    yield (
                                        TextMessageStartEvent(
                                            type=EventType.TEXT_MESSAGE_START,
                                            message_id=message_id,
                                        ),
                                        None,
                                        zero_metrics,
                                    )
                                text_started = True
                                yield (
                                    TextMessageContentEvent(
                                        type=EventType.TEXT_MESSAGE_CONTENT,
                                        message_id=message_id,
                                        delta=chunk.content,
                                    ),
                                    None,
                                    zero_metrics,
                                )
                        # Display tool calls
                        elif chunk.chunk_type == StreamChunkType.TOOL_CALL and chunk.tool_call:
                            print(f"\nUsing tool: {chunk.tool_call.tool_name}")
                    pipeline_interactions = self.create_pipeline_interactions_from_messages(
                        ragas_event_listener.messages
                    )
                    usage_metrics = self._extract_usage_metrics(crew_output)
                    if text_started:
                        yield (
                            TextMessageEndEvent(
                                type=EventType.TEXT_MESSAGE_END, message_id=message_id
                            ),
                            None,
                            zero_metrics,
                        )
                    if step_started:
                        yield (
                            StepFinishedEvent(
                                type=EventType.STEP_FINISHED,
                                step_name=current_task,
                            ),
                            None,
                            zero_metrics,
                        )
                else:
                    response_text = str(crew_output.raw)
                    pipeline_interactions = self.create_pipeline_interactions_from_messages(
                        ragas_event_listener.messages
                    )
                    usage_metrics = self._extract_usage_metrics(crew_output)

                    if response_text:
                        yield (
                            TextMessageChunkEvent(
                                type=EventType.TEXT_MESSAGE_CHUNK,
                                message_id=message_id,
                                delta=response_text,
                            ),
                            None,
                            zero_metrics,
                        )

                yield (
                    RunFinishedEvent(
                        type=EventType.RUN_FINISHED, thread_id=thread_id, run_id=run_id
                    ),
                    pipeline_interactions,
                    usage_metrics,
                )
