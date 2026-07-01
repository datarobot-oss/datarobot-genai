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
from collections.abc import Sequence
from typing import TYPE_CHECKING
from typing import Any

from ag_ui.core import EventType
from ag_ui.core import RunAgentInput
from ag_ui.core import RunFinishedEvent
from ag_ui.core import RunStartedEvent
from ag_ui.core import TextMessageChunkEvent
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
from datarobot_genai.crewai.agui_stream import AGUIStreamEmitter
from datarobot_genai.crewai.kickoff_storage import neutralize_kickoff_storage
from datarobot_genai.crewai.logging_events import CrewAILoggingEventListener
from datarobot_genai.crewai.ragas_events import CrewAIRagasEventListener
from datarobot_genai.crewai.streaming_events import CrewAIStreamingEventListener

if TYPE_CHECKING:
    from ragas import MultiTurnSample
    from ragas.messages import AIMessage
    from ragas.messages import HumanMessage
    from ragas.messages import ToolMessage

logger = logging.getLogger(__name__)


def _dedupe_tools_by_name(tools: list[BaseTool]) -> list[BaseTool]:
    """Keep the first tool per name, so an injected tool sharing a name with an agent's own
    tool isn't passed to CrewAI as a duplicate (CrewAI mangles dupes into ``name_2``/``name_3``,
    which bedrock then rejects).
    """
    seen: set[str] = set()
    unique: list[BaseTool] = []
    for tool in tools:
        if tool.name in seen:
            continue
        seen.add(tool.name)
        unique.append(tool)
    return unique


class CrewAIAgent(BaseAgent[BaseTool], abc.ABC):
    """Abstract base agent for CrewAI workflows.

    Subclasses should define the ``agents`` and ``tasks`` properties
    and may override ``crew`` to customize the workflow construction.

    Framework-specific parameters:

    - ``roles``: One role string per crew agent (length ``len(agents)``); sets
      each agent's ``role``.
    - ``goals``: One goal string per agent; sets each agent's ``goal``.
    - ``backstories``: One backstory string per agent; sets each agent's
      ``backstory``.
    - ``max_iter``: Upper bound on iterations (tool / act loops) per agent.
    - ``max_rpm``: Requests-per-minute limit per agent.
    - ``max_execution_time``: Wall-clock seconds cap per agent (only applied
      when not ``None``).
    - ``allow_delegation``: Whether agents may delegate work to other crew
      members.
    - ``max_retry_limit``: Retries per agent after recoverable failures.
    - ``reasoning``: Turns CrewAI structured reasoning on or off per agent.
    - ``max_reasoning_attempts``: Maximum reasoning passes per agent (only
      applied when not ``None``).
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        llm: Any | None = None,
        tools: list[BaseTool] | None = None,
        verbose: bool = True,
        timeout: int = 90,
        forwarded_headers: dict[str, str] | None = None,
        max_history_messages: int | None = None,
        model: str | None = None,
        roles: Sequence[str] | None = None,
        goals: Sequence[str] | None = None,
        backstories: Sequence[str] | None = None,
        max_iter: int | None = None,
        max_rpm: int | None = None,
        max_execution_time: int | None = None,
        allow_delegation: bool | None = None,
        max_retry_limit: int | None = None,
        reasoning: bool | None = None,
        max_reasoning_attempts: int | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            api_base=api_base,
            llm=llm,
            tools=tools,
            verbose=verbose,
            timeout=timeout,
            forwarded_headers=forwarded_headers,
            max_history_messages=max_history_messages,
            model=model,
        )
        if roles is not None:
            self.set_roles(roles)
        if goals is not None:
            self.set_goals(goals)
        if backstories is not None:
            self.set_backstories(backstories)
        if max_iter is not None:
            self.set_max_iter(max_iter)
        if max_rpm is not None:
            self.set_max_rpm(max_rpm)
        if max_execution_time is not None:
            self.set_max_execution_time(max_execution_time)
        if allow_delegation is not None:
            self.set_allow_delegation(allow_delegation)
        if max_retry_limit is not None:
            self.set_max_retry_limit(max_retry_limit)
        if reasoning is not None:
            self.set_reasoning(reasoning)
        if max_reasoning_attempts is not None:
            self.set_max_reasoning_attempts(max_reasoning_attempts)

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
        self._propagate_tools_to_tasks()

    def _propagate_tools_to_tasks(self) -> None:
        """Re-sync each task's tools to its agent's tools.

        CrewAI snapshots ``agent.tools`` into ``task.tools`` when the Crew is built,
        and runs tasks off that snapshot. Without this, platform-injected tools (e.g.
        MCP) reach ``agent.tools`` but never the model — the stale snapshot wins.

        Only tasks with an explicit ``agent`` are re-synced; a hierarchical/manager crew
        that assigns agents at runtime would keep its build-time snapshot.
        """
        for task in self.tasks:
            agent = getattr(task, "agent", None)
            if agent is not None:
                task.tools = agent.tools

    def set_verbose(self, verbose: bool) -> None:
        super().set_verbose(verbose)
        for agent in self.agents:
            agent.verbose = verbose

        self.crew.verbose = verbose

    def set_roles(self, roles: Sequence[str]) -> None:
        """Set each agent's ``role``. Length must match ``len(self.agents)``."""
        agents = self.agents
        if len(roles) != len(agents):
            raise ValueError(
                f"roles length ({len(roles)}) must match number of agents ({len(agents)})"
            )
        for agent, role in zip(agents, roles, strict=True):
            agent.role = role

    def set_goals(self, goals: Sequence[str]) -> None:
        """Set each agent's ``goal``. Length must match ``len(self.agents)``."""
        agents = self.agents
        if len(goals) != len(agents):
            raise ValueError(
                f"goals length ({len(goals)}) must match number of agents ({len(agents)})"
            )
        for agent, goal in zip(agents, goals, strict=True):
            agent.goal = goal

    def set_backstories(self, backstories: Sequence[str]) -> None:
        """Set each agent's ``backstory``. Length must match ``len(self.agents)``."""
        agents = self.agents
        if len(backstories) != len(agents):
            raise ValueError(
                f"backstories ({len(backstories)}) must match number of agents ({len(agents)})"
            )
        for agent, backstory in zip(agents, backstories, strict=True):
            agent.backstory = backstory

    def set_role(self, role: str, *, agent_index: int | None = None) -> None:
        """Set ``role`` on one agent.

        With multiple agents, pass ``agent_index`` (0-based). If omitted and there is
        exactly one agent, that agent is updated. With multiple agents and no index,
        use :meth:`set_roles` instead.
        """
        agents = self.agents
        if agent_index is None:
            if len(agents) != 1:
                raise ValueError(
                    "set_role(role) without agent_index requires exactly one agent; "
                    "use set_roles() or pass agent_index="
                )
            idx = 0
        else:
            idx = agent_index
            if not 0 <= idx < len(agents):
                raise IndexError(f"agent_index {idx} out of range for {len(agents)} agents")
        agents[idx].role = role

    def set_goal(self, goal: str, *, agent_index: int | None = None) -> None:
        """Set ``goal`` on one agent. See :meth:`set_role`."""
        agents = self.agents
        if agent_index is None:
            if len(agents) != 1:
                raise ValueError(
                    "set_goal(goal) without agent_index requires exactly one agent; "
                    "use set_goals() or pass agent_index="
                )
            idx = 0
        else:
            idx = agent_index
            if not 0 <= idx < len(agents):
                raise IndexError(f"agent_index {idx} out of range for {len(agents)} agents")
        agents[idx].goal = goal

    def set_backstory(self, backstory: str, *, agent_index: int | None = None) -> None:
        """Set ``backstory`` on one agent. See :meth:`set_role`."""
        agents = self.agents
        if agent_index is None:
            if len(agents) != 1:
                raise ValueError(
                    "set_backstory(backstory) without agent_index requires exactly one agent; "
                    "use set_backstories() or pass agent_index="
                )
            idx = 0
        else:
            idx = agent_index
            if not 0 <= idx < len(agents):
                raise IndexError(f"agent_index {idx} out of range for {len(agents)} agents")
        agents[idx].backstory = backstory

    def set_max_iter(self, max_iter: int) -> None:
        for agent in self.agents:
            agent.max_iter = max_iter

    def set_max_rpm(self, max_rpm: int) -> None:
        for agent in self.agents:
            agent.max_rpm = max_rpm

    def set_max_execution_time(self, max_execution_time: int | None) -> None:
        for agent in self.agents:
            agent.max_execution_time = max_execution_time

    def set_allow_delegation(self, allow_delegation: bool) -> None:
        for agent in self.agents:
            agent.allow_delegation = allow_delegation

    def set_max_retry_limit(self, max_retry_limit: int) -> None:
        for agent in self.agents:
            agent.max_retry_limit = max_retry_limit

    def set_reasoning(self, reasoning: bool) -> None:
        for agent in self.agents:
            agent.reasoning = reasoning

    def set_max_reasoning_attempts(self, max_reasoning_attempts: int | None) -> None:
        for agent in self.agents:
            agent.max_reasoning_attempts = max_reasoning_attempts

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
            logging_event_listener = CrewAILoggingEventListener()
            logging_event_listener.setup_listeners(crewai_event_bus)

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
            # Reused crew: CrewAI caches each agent's executor and never resets its accumulated
            # messages/iterations, leaking prior-request state (bedrock "tool calling without
            # tools=" errors; tool calls leaked as text). Force a fresh executor per run.
            for agent in self.agents:
                agent.agent_executor = None
            crew_output = await crew.akickoff(inputs=kickoff_inputs)

            if isinstance(crew_output, CrewStreamingOutput):
                emitter = AGUIStreamEmitter()

                def out(events: Any, metrics: UsageMetrics = zero_metrics) -> Any:
                    """Wrap bare events into the (event, interactions, metrics) tuple."""
                    for ev in events:
                        yield (ev, None, metrics)

                def drain(metrics: UsageMetrics = zero_metrics) -> Any:
                    """Emit the tool-call events the bus has queued so far."""
                    pending = streaming_event_listener.tool_call_events
                    while not pending.empty():
                        yield from out(emitter.tool_call(pending.get_nowait()), metrics)

                def events_for(chunk: Any) -> Any:
                    """Translate one CrewAI chunk into AG-UI events via the emitter."""
                    yield from drain()  # tool calls the bus queued since the last chunk
                    # Gateway chunks carry an empty agent_role; fall back to the bus-tracked role.
                    role = chunk.agent_role or streaming_event_listener.active_agent_role
                    yield from out(emitter.step(role))
                    yield from out(emitter.reasoning(streaming_event_listener.reasoning_event))
                    if chunk.chunk_type == StreamChunkType.TEXT and chunk.content:
                        yield from out(emitter.text(chunk.content))

                try:
                    async for chunk in crew_output:
                        logger.debug(f"CrewAI chunk: {chunk.model_dump_json()}")
                        for item in events_for(chunk):
                            yield item

                    pipeline_interactions = self.create_pipeline_interactions_from_messages(
                        ragas_event_listener.messages
                    )
                    usage_metrics = self._extract_usage_metrics(crew_output.result)
                    # Stream done: close the open message, flush tool calls that fired after the
                    # last chunk (now detached, parent=""), then close the open step.
                    for item in out(emitter.close_messages(), usage_metrics):
                        yield item
                    for item in drain(usage_metrics):
                        yield item
                    for item in out(emitter.finish(), usage_metrics):
                        yield item
                except Exception:
                    # Aborted mid-stream (dropped connection, bad chunk): close the open message and
                    # step so the partial AG-UI stream stays well-formed -- an orphaned STEP_STARTED
                    # rejects any terminal event the caller appends -- then let the error propagate.
                    for item in out(emitter.finish(), usage_metrics):
                        yield item
                    raise
            else:
                message_id = str(uuid.uuid4())
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

    The returned class accepts the same parameters as :class:`CrewAIAgent`
    (including optional ``roles``, ``goals``, and ``backstories``).

    Returns
    -------
    type[CrewAIAgent]
        A new :class:`CrewAIAgent` subclass wired to the provided crew,
        agents, tasks, and kickoff-input builder.
    """
    # Disable crewai's kickoff-outputs SQLite handler on the supplied crew so a
    # long-lived serve process can't leak file descriptors (see
    # :mod:`datarobot_genai.crewai.kickoff_storage`).
    crew = neutralize_kickoff_storage(crew)

    # Capture each agent's original tools ONCE here, not per-instance: the crew/agents are reused
    # across requests, so re-deriving per request would snapshot the *previous* request's injected
    # MCP tools (bound to a now-closed event loop) -> "Event loop is closed".
    original_agent_tools = {agent: list(agent.tools) for agent in agents}

    class DataRobotAgent(CrewAIAgent):
        def set_tools(self, tools: list[BaseTool]) -> None:
            super().set_tools(tools)
            for agent in agents:
                # Original tools + freshly-injected; dedupe so platform tools don't accumulate
                # across requests on the reused crew (see _dedupe_tools_by_name).
                agent.tools = _dedupe_tools_by_name(original_agent_tools[agent] + tools)
            # Re-sync tasks now that agent.tools carries original + injected tools
            # (super() synced them to the injected-only set).
            self._propagate_tools_to_tasks()

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
