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
import logging
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING
from typing import Any

from ag_ui.core import EventType
from ag_ui.core import ReasoningEndEvent
from ag_ui.core import ReasoningMessageContentEvent
from ag_ui.core import ReasoningMessageEndEvent
from ag_ui.core import ReasoningMessageStartEvent
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
from crewai import Agent
from crewai import Crew
from crewai import Task
from crewai.events import crewai_event_bus
from crewai.llm import LLM
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

if TYPE_CHECKING:
    from ragas import MultiTurnSample
    from ragas.messages import AIMessage
    from ragas.messages import HumanMessage
    from ragas.messages import ToolMessage

logger = logging.getLogger(__name__)


class CrewAIAgent(BaseAgent[BaseTool], abc.ABC):
    """Abstract base agent for CrewAI workflows.

    Subclasses should define the ``agents`` and ``tasks`` properties
    and may override ``crew`` to customize the workflow construction.
    """

    def set_llm(self, llm: LLM | None) -> None:
        super().set_llm(llm)
        if llm is None:
            return
        for agent in self.agents:
            agent.llm = llm
            agent.function_calling_llm = llm

    def set_tools(self, tools: list[BaseTool]) -> None:
        super().set_tools(tools)
        for agent in self.agents:
            agent.tools = tools

    def set_verbose(self, verbose: bool) -> None:
        super().set_verbose(verbose)
        for agent in self.agents:
            agent.verbose = verbose

        self.crew.verbose = verbose

    @property
    @abc.abstractmethod
    def agents(self) -> list[Agent]:  # CrewAI Agent list
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def tasks(self) -> list[Task]:  # CrewAI Task list
        raise NotImplementedError

    @property
    def crew(self) -> Crew:
        """Create a CrewAI workflow instance.

        Default implementation constructs a Crew with provided agents and tasks.
        Subclasses can override to customize Crew options.
        """
        return Crew(agents=self.agents, tasks=self.tasks, verbose=self.verbose, stream=True)

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

        - ``memory`` (optional): Include this key with an empty string value
          (``""``) to opt into automatic memory retrieval. When present, the
          base class will populate it with relevant long-term memory before
          kickoff and store the user turn after a successful run. Use
          ``{memory}`` in agent goals/backstories or task descriptions to
          reference it.

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
        logger.info(
            f"Running agent with user prompt: {user_prompt_content}",
        )

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

        with crewai_event_bus.scoped_handlers():
            ragas_event_listener = CrewAIRagasEventListener()
            ragas_event_listener.setup_listeners(crewai_event_bus)
            streaming_event_listener = CrewAIStreamingEventListener()
            streaming_event_listener.setup_listeners(crewai_event_bus)

            crew = self.crew

            kickoff_inputs = self.make_kickoff_inputs(str(user_prompt_content))
            # Chat history is opt-in: only populate it if the agent/template
            # declares a `chat_history` kickoff input (i.e. it uses `{chat_history}`
            # in prompts).
            if "chat_history" in kickoff_inputs:
                history_summary = self.build_history_summary(run_agent_input)
                existing_history_text = str(kickoff_inputs.get("chat_history") or "")

                if history_summary and not existing_history_text.strip():
                    kickoff_inputs["chat_history"] = f"\n\nPrior conversation:\n{history_summary}"
            if "memory" in kickoff_inputs:
                existing_memory_text = str(kickoff_inputs.get("memory") or "")
                if not existing_memory_text.strip():
                    try:
                        kickoff_inputs["memory"] = await self.retrieve_memory_for_run(
                            user_prompt_content,
                            run_agent_input,
                        )
                    except Exception as exc:
                        logger.warning("CrewAI memory retrieval failed: %s", exc)
            message_id = str(uuid.uuid4())
            crew_output = await crew.kickoff_async(inputs=kickoff_inputs)
            current_agent_role = ""

            if isinstance(crew_output, CrewStreamingOutput):
                reasoning_started = False
                text_started = False
                async for chunk in crew_output:
                    # Show task transitions
                    logger.debug(f"CrewAI chunk: {chunk.model_dump_json()}")
                    if chunk.agent_role != current_agent_role:
                        logger.info(f"[{chunk.agent_role}] Working on task: {chunk.task_name}")
                        if current_agent_role:
                            yield (
                                StepFinishedEvent(
                                    type=EventType.STEP_FINISHED,
                                    step_name=current_agent_role,
                                ),
                                None,
                                zero_metrics,
                            )
                        yield (
                            StepStartedEvent(
                                type=EventType.STEP_STARTED,
                                step_name=chunk.agent_role,
                            ),
                            None,
                            zero_metrics,
                        )
                        current_agent_role = chunk.agent_role

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
                            yield (
                                ReasoningMessageStartEvent(
                                    type=EventType.REASONING_MESSAGE_START,
                                    message_id=message_id,
                                ),
                                None,
                                zero_metrics,
                            )
                            reasoning_started = True
                    elif reasoning_started:
                        yield (
                            ReasoningMessageEndEvent(
                                type=EventType.REASONING_MESSAGE_END,
                                message_id=message_id,
                            ),
                            None,
                            zero_metrics,
                        )
                        yield (
                            ReasoningEndEvent(
                                type=EventType.REASONING_END,
                                message_id=message_id,
                            ),
                            None,
                            zero_metrics,
                        )
                        reasoning_started = False

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
                        logger.info(f"Using tool: {chunk.tool_call.tool_name}")
                pipeline_interactions = self.create_pipeline_interactions_from_messages(
                    ragas_event_listener.messages
                )
                usage_metrics = self._extract_usage_metrics(crew_output.result)
                if text_started:
                    yield (
                        TextMessageEndEvent(type=EventType.TEXT_MESSAGE_END, message_id=message_id),
                        None,
                        usage_metrics,
                    )
                if reasoning_started:
                    yield (
                        ReasoningMessageEndEvent(
                            type=EventType.REASONING_MESSAGE_END,
                            message_id=message_id,
                        ),
                        None,
                        usage_metrics,
                    )
                    yield (
                        ReasoningEndEvent(
                            type=EventType.REASONING_END,
                            message_id=message_id,
                        ),
                        None,
                        usage_metrics,
                    )
                if not text_started and not reasoning_started:
                    # Non-streaming LLM (e.g. RouterLitellmOnlyLLM): the streaming
                    # loop produced no TEXT chunks. Fall back to crew_output.result.raw.
                    result_raw = str(getattr(crew_output.result, "raw", "") or "")
                    if result_raw:
                        yield (
                            TextMessageStartEvent(
                                type=EventType.TEXT_MESSAGE_START,
                                message_id=message_id,
                                role="assistant",
                            ),
                            None,
                            usage_metrics,
                        )
                        yield (
                            TextMessageContentEvent(
                                type=EventType.TEXT_MESSAGE_CONTENT,
                                message_id=message_id,
                                delta=result_raw,
                            ),
                            None,
                            usage_metrics,
                        )
                        yield (
                            TextMessageEndEvent(
                                type=EventType.TEXT_MESSAGE_END,
                                message_id=message_id,
                            ),
                            None,
                            usage_metrics,
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
                        usage_metrics,
                    )

            if current_agent_role:
                yield (
                    StepFinishedEvent(
                        type=EventType.STEP_FINISHED,
                        step_name=current_agent_role,
                    ),
                    None,
                    usage_metrics,
                )
            if "memory" in kickoff_inputs:
                try:
                    await self.store_memory_for_run(user_prompt_content, run_agent_input)
                except Exception as exc:
                    logger.warning("CrewAI memory storage failed: %s", exc)
            yield (
                RunFinishedEvent(type=EventType.RUN_FINISHED, thread_id=thread_id, run_id=run_id),
                pipeline_interactions,
                usage_metrics,
            )


def datarobot_agent_class_from_crew(
    crew: Crew,
    agents: list[Agent],
    tasks: list[Task],
    kickoff_inputs: Callable[[str], dict[str, Any]],
) -> type[CrewAIAgent]:
    """Create a CrewAI agent class from a pre-built crew, agents, and tasks.

    This is a convenience helper that dynamically builds a concrete
    :class:`CrewAIAgent` subclass so that callers can define an agent
    entirely from existing CrewAI objects without writing a class by hand.

    When the returned class is instantiated, calling ``set_tools``
    propagates the platform-injected tools to every agent in *agents*
    while preserving each agent's originally configured tools.

    Parameters
    ----------
    crew : Crew
        A fully configured CrewAI :class:`Crew` instance that orchestrates
        the provided agents and tasks.
    agents : list[Agent]
        The list of CrewAI :class:`Agent` instances participating in the crew.
        Their ``llm`` and ``tools`` attributes are updated at runtime when the
        DataRobot platform injects the LLM and MCP tools.
    tasks : list[Task]
        The list of CrewAI :class:`Task` instances that define the work to be
        performed during kickoff.
    kickoff_inputs : Callable[[str], dict[str, Any]]
        A callable that receives the raw user prompt string and returns a
        dictionary of inputs for ``Crew.kickoff()``. The dictionary keys
        should match placeholder names used in task descriptions and agent
        configurations (e.g. ``{topic}``). Include a ``"chat_history"`` key
        with an empty string value to opt into automatic history injection.

    Returns
    -------
    type[CrewAIAgent]
        A new :class:`CrewAIAgent` subclass wired to the provided crew,
        agents, tasks, and kickoff-input builder.
    """

    class DataRobotAgent(CrewAIAgent):
        def __init__(
            self,
            *args: Any,
            **kwargs: Any,
        ) -> None:
            self._original_agent_tools = {agent: agent.tools for agent in agents}
            super().__init__(*args, **kwargs)

        def set_tools(self, tools: list[BaseTool]) -> None:
            super().set_tools(tools)
            for agent in agents:
                # make sure we don't overwrite the original tools
                agent.tools = self._original_agent_tools[agent] + tools

        @property
        def crew(self) -> Crew:
            return crew

        @property
        def agents(self) -> list[Agent]:
            return agents

        @property
        def tasks(self) -> list[Task]:
            return tasks

        def make_kickoff_inputs(self, user_prompt_content: str) -> dict[str, Any]:
            return kickoff_inputs(user_prompt_content)

    return DataRobotAgent
