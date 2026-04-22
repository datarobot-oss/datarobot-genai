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

import abc
import inspect
import json
import logging
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from ag_ui.core import EventType
from ag_ui.core import RunAgentInput
from ag_ui.core import RunFinishedEvent
from ag_ui.core import RunStartedEvent
from ag_ui.core import StepFinishedEvent
from ag_ui.core import StepStartedEvent
from ag_ui.core import TextMessageContentEvent
from ag_ui.core import TextMessageEndEvent
from ag_ui.core import TextMessageStartEvent
from ag_ui.core import ToolCallArgsEvent
from ag_ui.core import ToolCallEndEvent
from ag_ui.core import ToolCallResultEvent
from ag_ui.core import ToolCallStartEvent
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.agent.workflow import BaseWorkflowAgent
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.base.llms.types import LLMMetadata
from llama_index.core.tools import BaseTool
from llama_index.core.workflow import Event
from llama_index.llms.litellm import LiteLLM

from datarobot_genai.core.agents.base import BaseAgent
from datarobot_genai.core.agents.base import InvokeReturn
from datarobot_genai.core.agents.base import UsageMetrics
from datarobot_genai.core.agents.base import default_usage_metrics
from datarobot_genai.core.agents.base import extract_user_prompt_content
from datarobot_genai.core.memory.base import BaseMemoryClient

if TYPE_CHECKING:
    from ragas import MultiTurnSample

logger = logging.getLogger(__name__)


class DataRobotLiteLLM(LiteLLM):
    """LiteLLM wrapper providing chat/function capability metadata for LlamaIndex."""

    @property
    def metadata(self) -> LLMMetadata:
        """Return LLM metadata."""
        return LLMMetadata(
            context_window=128000,
            num_output=self.max_tokens or -1,
            is_chat_model=True,
            is_function_calling_model=True,
            model_name=self.model,
        )


class LlamaIndexAgent(BaseAgent[BaseTool], abc.ABC):
    """Abstract base agent for LlamaIndex workflows."""

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
        memory_client: BaseMemoryClient | None = None,
        allow_parallel_tool_calls: bool = True,
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
            memory_client=memory_client,
        )
        self.allow_parallel_tool_calls = allow_parallel_tool_calls

    @abc.abstractmethod
    async def build_workflow(self) -> Any:
        """Return an AgentWorkflow instance ready to run."""
        raise NotImplementedError

    @abc.abstractmethod
    def extract_response_text(self, result_state: Any, events: list[Any]) -> str:
        """Extract final response text from workflow state and/or events."""
        raise NotImplementedError

    async def invoke(self, run_agent_input: RunAgentInput) -> InvokeReturn:
        """Run the LlamaIndex workflow with the provided completion parameters."""
        user_prompt_content = extract_user_prompt_content(run_agent_input)
        input_message = str(user_prompt_content)
        uses_memory = "{memory}" in input_message

        # Handle {chat_history} placeholder replacement for subclass templates
        if "{chat_history}" in input_message:
            history_summary = self.build_history_summary(run_agent_input)
            formatted_history = (
                f"\n\nPrior conversation:\n{history_summary}" if history_summary else ""
            )
            input_message = input_message.replace("{chat_history}", formatted_history)
        if uses_memory:
            memory = ""
            try:
                memory = await self.retrieve_memory_for_run(user_prompt_content, run_agent_input)
            except Exception as exc:
                logger.warning("LlamaIndex memory retrieval failed: %s", exc)
            input_message = input_message.replace("{memory}", memory)

        logger.info(f"Running agent with user prompt: {input_message}")

        thread_id = run_agent_input.thread_id
        run_id = run_agent_input.run_id
        usage_metrics: UsageMetrics = default_usage_metrics()

        # Partial AG-UI: lifecycle + text + tool calls + steps
        yield (
            RunStartedEvent(type=EventType.RUN_STARTED, thread_id=thread_id, run_id=run_id),
            None,
            usage_metrics,
        )

        # Subclasses may implement build_workflow as async or sync; support both.
        built: Any = self.build_workflow()
        workflow = await built if inspect.isawaitable(built) else built
        handler = workflow.run(user_msg=input_message)

        events: list[Any] = []
        current_agent_name: str | None = None
        message_id = str(uuid.uuid4())
        text_started = False
        agent: str | None = None

        async for event in handler.stream_events():
            events.append(event)
            # Best-effort extraction of incremental text from LlamaIndex events
            delta: str | None = None

            try:
                if hasattr(event, "delta") and isinstance(getattr(event, "delta"), str):
                    delta = getattr(event, "delta")
                # Some event types may carry incremental text under "text" or similar
                elif hasattr(event, "text") and isinstance(getattr(event, "text"), str):
                    delta = getattr(event, "text")
            except Exception:
                # Ignore malformed events and continue
                delta = None

            if delta:
                if not text_started:
                    yield (
                        TextMessageStartEvent(
                            type=EventType.TEXT_MESSAGE_START, message_id=message_id
                        ),
                        None,
                        usage_metrics,
                    )
                    text_started = True
                yield (
                    TextMessageContentEvent(
                        type=EventType.TEXT_MESSAGE_CONTENT, message_id=message_id, delta=delta
                    ),
                    None,
                    usage_metrics,
                )

                # Agent switch banner if available on event
            if hasattr(event, "current_agent_name"):
                agent = getattr(event, "current_agent_name", None)
                if agent is not None and agent != current_agent_name:
                    if current_agent_name is not None:
                        yield (
                            StepFinishedEvent(step_name=current_agent_name),
                            None,
                            usage_metrics,
                        )

                    yield (
                        StepStartedEvent(step_name=agent),
                        None,
                        usage_metrics,
                    )
                    current_agent_name = agent
                    agent = None
                    logger.info(f"Agent: {current_agent_name}")

            event_type = type(event).__name__
            if event_type == "AgentInput" and hasattr(event, "input"):
                logger.info(f"Input: {getattr(event, 'input')}")
            elif event_type == "AgentOutput":
                # Output content
                resp = getattr(event, "response", None)
                if resp is not None and hasattr(resp, "content") and getattr(resp, "content"):
                    logger.info(f"Output: {getattr(resp, 'content')}")
                # Planned tool calls
                tcalls = getattr(event, "tool_calls", None)
                if isinstance(tcalls, list) and tcalls:
                    names = []
                    for c in tcalls:
                        try:
                            nm = getattr(c, "tool_name", None) or (
                                c.get("tool_name") if isinstance(c, dict) else None
                            )
                            if nm:
                                names.append(str(nm))
                        except Exception:
                            pass
                    if names:
                        logger.info(f"Planning to use tools: {names}")
            elif event_type == "ToolCallResult":
                tname = getattr(event, "tool_name", None)
                tid = getattr(event, "tool_id", None)
                tkwargs = getattr(event, "tool_kwargs", None)
                tout = getattr(event, "tool_output", None)
                logger.info(f"Tool Result: {tname}")
                logger.debug(f"Arguments: {tkwargs}")
                logger.debug(f"Output: {tout}")
                yield (
                    ToolCallEndEvent(
                        type=EventType.TOOL_CALL_END,
                        tool_call_id=tid,
                    ),
                    None,
                    usage_metrics,
                )
                yield (
                    ToolCallResultEvent(
                        type=EventType.TOOL_CALL_RESULT,
                        message_id=message_id,
                        tool_call_id=tid,
                        content=json.dumps(tout, default=str),
                        role="tool",
                    ),
                    None,
                    usage_metrics,
                )
            elif event_type == "ToolCall":
                tname = getattr(event, "tool_name", None)
                tkwargs = getattr(event, "tool_kwargs", None)
                tid = getattr(event, "tool_id", None)
                logger.info(f"Calling Tool: {tname}")
                logger.debug(f"With arguments: {tkwargs}")
                yield (
                    ToolCallStartEvent(
                        type=EventType.TOOL_CALL_START,
                        tool_call_id=tid,
                        tool_call_name=tname,
                    ),
                    None,
                    usage_metrics,
                )
                yield (
                    ToolCallArgsEvent(
                        type=EventType.TOOL_CALL_ARGS,
                        tool_call_id=tid,
                        delta=json.dumps(tkwargs, default=str),
                    ),
                    None,
                    usage_metrics,
                )

        if agent is not None:
            yield (
                StepFinishedEvent(step_name=agent),
                None,
                usage_metrics,
            )
            agent = None

        if text_started:
            yield (
                TextMessageEndEvent(
                    type=EventType.TEXT_MESSAGE_END,
                    message_id=message_id,
                ),
                None,
                usage_metrics,
            )

        # After streaming completes, build final interactions and finish chunk
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

        # Run subclass-defined response extraction (not streamed) for completeness
        _ = self.extract_response_text(state, events)

        pipeline_interactions = self.create_pipeline_interactions_from_events(events)
        if uses_memory:
            try:
                await self.store_memory_for_run(user_prompt_content, run_agent_input)
            except Exception as exc:
                logger.warning("LlamaIndex memory storage failed: %s", exc)
        # TODO: find a way to count usage (LlamaIndex does not report it)
        yield (
            RunFinishedEvent(type=EventType.RUN_FINISHED, thread_id=thread_id, run_id=run_id),
            pipeline_interactions,
            usage_metrics,
        )

    @classmethod
    def create_pipeline_interactions_from_events(
        cls,
        events: list[Event] | None,
    ) -> MultiTurnSample | None:
        if not events:
            return None
        # Lazy import to reduce memory overhead when ragas is not used
        from ragas import MultiTurnSample
        from ragas.integrations.llama_index import convert_to_ragas_messages
        from ragas.messages import AIMessage
        from ragas.messages import HumanMessage
        from ragas.messages import ToolMessage

        # convert_to_ragas_messages expects a list[Event]
        ragas_trace = convert_to_ragas_messages(list(events))
        ragas_messages = cast(list[HumanMessage | AIMessage | ToolMessage], ragas_trace)
        return MultiTurnSample(user_input=ragas_messages)


def datarobot_agent_class_from_llamaindex(
    workflow: AgentWorkflow,
    agents: list[BaseWorkflowAgent],
    extract_response_text: Callable[[Any, list[Any]], str],
) -> type[LlamaIndexAgent]:
    """Create a LlamaIndex agent class from a pre-built workflow and agents.

    This is a convenience helper that dynamically builds a concrete
    :class:`LlamaIndexAgent` subclass so that callers can define an agent
    entirely from an existing :class:`AgentWorkflow` and its constituent
    agents without writing a class by hand.

    When the returned class is instantiated, calling ``set_llm`` or
    ``set_tools`` propagates the LLM / tools to every agent in *agents*
    while preserving each agent's originally configured tools.

    Parameters
    ----------
    workflow : AgentWorkflow
        A fully configured LlamaIndex :class:`AgentWorkflow` instance that
        orchestrates the provided agents.
    agents : list[BaseWorkflowAgent]
        The list of LlamaIndex workflow agents participating in the workflow.
        Their ``llm`` and ``tools`` attributes are updated at runtime when the
        DataRobot platform injects the LLM and MCP tools.
    extract_response_text : Callable[[Any, list[Any]], str]
        A callback that extracts the final human-readable response from the
        workflow result state and the list of streamed events.  Receives
        ``(result_state, events)`` and must return a ``str``.

    Returns
    -------
    type[LlamaIndexAgent]
        A new :class:`LlamaIndexAgent` subclass wired to the provided
        workflow, agents, and response extractor.
    """
    original_agent_tools = {agent.name: agent.tools for agent in agents}

    class DataRobotLlamaIndexAgent(LlamaIndexAgent):
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
            memory_client: BaseMemoryClient | None = None,
            allow_parallel_tool_calls: bool = True,
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
                memory_client=memory_client,
                allow_parallel_tool_calls=allow_parallel_tool_calls,
            )
            for agent in agents:
                if isinstance(agent, FunctionAgent):
                    agent.allow_parallel_tool_calls = allow_parallel_tool_calls

        def set_llm(self, llm: Any) -> None:
            super().set_llm(llm)
            for agent in agents:
                agent.llm = llm

        def set_tools(self, tools: list[BaseTool]) -> None:
            super().set_tools(tools)
            for agent in agents:
                agent.tools = original_agent_tools[agent.name] + tools

        def set_allow_parallel_tool_calls(self, allow_parallel_tool_calls: bool) -> None:
            self.allow_parallel_tool_calls = allow_parallel_tool_calls
            for agent in agents:
                if isinstance(agent, FunctionAgent):
                    agent.allow_parallel_tool_calls = allow_parallel_tool_calls

        async def build_workflow(self) -> AgentWorkflow:
            return workflow

        def extract_response_text(self, result_state: Any, events: list[Any]) -> str:
            return extract_response_text(result_state, events)

    return DataRobotLlamaIndexAgent
