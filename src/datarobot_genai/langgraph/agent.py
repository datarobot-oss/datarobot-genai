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
import logging
from collections.abc import AsyncGenerator
from collections.abc import Callable
from collections.abc import Mapping
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from ag_ui.core import CustomEvent
from ag_ui.core import EventType
from ag_ui.core import RunAgentInput
from ag_ui.core import RunFinishedEvent
from ag_ui.core import RunStartedEvent
from ag_ui.core import TextMessageContentEvent
from ag_ui.core import TextMessageEndEvent
from ag_ui.core import TextMessageStartEvent
from ag_ui.core import ToolCallArgsEvent
from ag_ui.core import ToolCallEndEvent
from ag_ui.core import ToolCallResultEvent
from ag_ui.core import ToolCallStartEvent
from langchain.chat_models import BaseChatModel
from langchain.tools import BaseTool
from langchain_core.messages import AIMessageChunk
from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph
from langgraph.types import Command
from nat.plugins.langchain.callback_handler import LangchainProfilerHandler

from datarobot_genai.core.agents.base import BaseAgent
from datarobot_genai.core.agents.base import InvokeReturn
from datarobot_genai.core.agents.base import UsageMetrics
from datarobot_genai.core.agents.base import extract_user_prompt_content

if TYPE_CHECKING:
    from ragas import MultiTurnSample

logger = logging.getLogger(__name__)

# RunAgentInput.state / forwarded_props key for Command(resume=...).
LANGGRAPH_RESUME_STATE_KEY = "langgraph_resume"


class LangGraphAgent(BaseAgent[BaseTool], abc.ABC):
    """Base class for LangGraph-powered agents.

    This class wires LangGraph workflows into the generic `BaseAgent` interface
    and provides a default implementation for turning OpenAI-style chat inputs
    into a LangGraph `Command`.

    History is opt-in: prior turns are only injected when the prompt template
    declares and uses a `{chat_history}` input variable. If the template does
    not use `{chat_history}`, no chat history is passed to the model.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Forward ``*args`` / ``**kwargs`` to :class:`BaseAgent` after setting LangGraph fields.

        The following keyword arguments are handled here and are not passed to
        :meth:`BaseAgent.__init__`: ``checkpointer``, ``interrupt_before``,
        ``interrupt_after``, ``debug``, ``name``.

        - ``checkpointer``: if set, :attr:`langgraph_checkpointer` returns it;
          otherwise a :class:`~langgraph.checkpoint.memory.InMemorySaver` is used.
        - ``interrupt_before`` / ``interrupt_after``: passed to
          :meth:`langgraph.graph.state.StateGraph.compile`.
        - ``debug``: if provided, passed to ``compile(debug=...)``; for ``astream``,
          when omitted, ``debug`` follows ``verbose`` (previous behavior).
        - ``name``: optional graph name for ``compile(name=...)``.
        """
        self.checkpointer = kwargs.pop("checkpointer", None)
        self.interrupt_before = kwargs.pop("interrupt_before", None)
        self.interrupt_after = kwargs.pop("interrupt_after", None)
        self.debug = kwargs.pop("debug", False)
        self.name = kwargs.pop("name", None)
        super().__init__(*args, **kwargs)

    @property
    @abc.abstractmethod
    def workflow(self) -> StateGraph[MessagesState]:
        raise NotImplementedError("Not implemented")

    @property
    @abc.abstractmethod
    def prompt_template(self) -> ChatPromptTemplate:
        raise NotImplementedError("Not implemented")

    @property
    def langgraph_config(self) -> dict[str, Any]:
        return {
            "recursion_limit": 150,  # Maximum number of steps to take in the graph
            "callbacks": [LangchainProfilerHandler()],
        }

    def build_langgraph_runnable_config(self, run_agent_input: RunAgentInput) -> dict[str, Any]:
        """Merge `langgraph_config` with per-run `thread_id` for checkpointing."""
        cfg: dict[str, Any] = dict(self.langgraph_config)
        existing = cfg.get("configurable")
        if isinstance(existing, dict):
            configurable = {**existing, "thread_id": run_agent_input.thread_id}
        else:
            configurable = {"thread_id": run_agent_input.thread_id}
        cfg["configurable"] = configurable
        return cfg

    def _compile_workflow(self) -> Any:
        """Compile the workflow graph, attaching `langgraph_checkpointer` when set."""
        return self.workflow.compile(
            checkpointer=self.checkpointer,
            interrupt_before=self.interrupt_before,
            interrupt_after=self.interrupt_after,
            debug=self.debug,
            name=self.name,
        )

    async def _command_for_pending_interrupt(
        self,
        compiled_graph: Any,
        run_agent_input: RunAgentInput,
    ) -> Command | None:
        """When the thread is paused on `interrupt()`, map the user message to `resume`."""
        if self.checkpointer is None:
            return None
        config = cast(Any, self.build_langgraph_runnable_config(run_agent_input))
        ag = getattr(compiled_graph, "aget_state", None)
        if ag is None:
            return None
        try:
            snap = await ag(config)
        except (TypeError, ValueError):
            return None
        interrupts = getattr(snap, "interrupts", None)
        if not interrupts:
            return None
        user_reply = extract_user_prompt_content(run_agent_input)
        if len(interrupts) == 1:
            return Command(resume=user_reply)
        return Command(resume={intr.id: user_reply for intr in interrupts})

    async def _build_input_command(
        self,
        run_agent_input: RunAgentInput,
        compiled_graph: Any,
    ) -> Command:
        """Resolve LangGraph input: explicit resume, pending interrupt, or normal prompt."""
        state = run_agent_input.state
        resume_payload: Any | None = None
        if isinstance(state, Mapping) and LANGGRAPH_RESUME_STATE_KEY in state:
            resume_payload = state[LANGGRAPH_RESUME_STATE_KEY]
        else:
            forwarded = run_agent_input.forwarded_props
            if isinstance(forwarded, Mapping) and LANGGRAPH_RESUME_STATE_KEY in forwarded:
                resume_payload = forwarded[LANGGRAPH_RESUME_STATE_KEY]
        if resume_payload is not None:
            return Command(resume=resume_payload)

        pending = await self._command_for_pending_interrupt(compiled_graph, run_agent_input)
        if pending is not None:
            return pending

        return await self.convert_input_message(run_agent_input)

    async def convert_input_message(self, run_agent_input: RunAgentInput) -> Command:
        """Convert AG-UI input into a LangGraph `Command`.

        By default this:
        - Extracts the last user message content via `extract_user_prompt_content`
          and feeds it through `prompt_template` to build the current turn.
        - Includes prior turns only when the prompt template opts in via a
          `{chat_history}` variable.
        """
        user_prompt = extract_user_prompt_content(run_agent_input)

        # Chat history is opt-in: the model only sees history when the prompt
        # template declares/uses the `{chat_history}` variable.
        input_vars = getattr(self.prompt_template, "input_variables", [])
        try:
            vars_list = list(input_vars)
        except TypeError:
            vars_list = []
        uses_chat_history = "chat_history" in vars_list
        history_summary = self.build_history_summary(run_agent_input) if uses_chat_history else ""

        memory = ""
        if "memory" in vars_list:
            try:
                memory = await self.retrieve_memory_for_run(user_prompt, run_agent_input)
            except Exception as exc:
                logger.warning("LangGraph memory retrieval failed: %s", exc)

        # Prefer structured dict input when available so templates can access both
        # the original fields (e.g. {topic}) and a plain-text {chat_history}.
        if isinstance(user_prompt, Mapping):
            template_input: Any = dict(user_prompt)
            template_input.setdefault("chat_history", history_summary)
            template_input.setdefault("memory", memory)
        elif vars_list:
            # When the prompt is a bare value, best-effort map it into the declared
            # input variables. Known variable "chat_history" always receives the
            # history summary; all other variables receive the raw user prompt.
            template_input = {}
            for name in vars_list:
                if name == "chat_history":
                    template_input[name] = history_summary
                elif name == "memory":
                    template_input[name] = memory
                else:
                    template_input[name] = user_prompt
        else:
            # No declared variables: preserve pre-history and pre-memory behaviour.
            template_input = user_prompt

        current_messages = self.prompt_template.invoke(template_input).to_messages()
        command = Command(  # type: ignore[var-annotated]
            update={
                "messages": current_messages,
            },
        )
        return command

    async def invoke(self, run_agent_input: RunAgentInput) -> InvokeReturn:
        """Run the agent with the provided input.

        Args:
            run_agent_input: The agent run input.

        Returns
        -------
            Returns a generator yielding tuples of (response_text,
            pipeline_interactions, usage_metrics).
        """
        try:
            result = await self._invoke(run_agent_input)

            # Yield all items from the result generator
            # The context will be closed when this generator is exhausted
            # Cast to async generator since we know stream=True means it's a generator
            async for item in result:
                yield item
        except RuntimeError as e:
            error_message = str(e).lower()
            if "different task" in error_message and "cancel scope" in error_message:
                # Due to anyio task group constraints when consuming async generators
                # across task boundaries, we cannot always clean up properly.
                # The underlying HTTP client/connection pool should handle resource cleanup
                # via timeouts and connection pooling, but this
                # may lead to delayed resource release.
                logger.debug(
                    "MCP context cleanup attempted in different task. "
                    "This is a limitation when consuming async generators "
                    "across task boundaries."
                )
            else:
                # Re-raise if it's a different RuntimeError
                raise

    async def _invoke(self, run_agent_input: RunAgentInput) -> InvokeReturn:
        langgraph_execution_graph = self._compile_workflow()
        input_command = await self._build_input_command(run_agent_input, langgraph_execution_graph)
        logger.info(
            f"Running a langgraph agent with a command: {input_command}",
        )

        graph_stream = cast(
            AsyncGenerator[tuple[Any, str, Any], None],
            langgraph_execution_graph.astream(
                input=input_command,
                # LangGraph expects a RunnableConfig, but our config is a plain dict.
                # Cast to Any to avoid leaking LangGraph internals into this interface.
                config=cast(Any, self.build_langgraph_runnable_config(run_agent_input)),
                debug=self.verbose,
                # Streaming updates and messages from all the nodes
                stream_mode=["updates", "messages"],
                subgraphs=True,
            ),
        )

        usage_metrics: UsageMetrics = {
            "completion_tokens": 0,
            "prompt_tokens": 0,
            "total_tokens": 0,
        }

        return self._stream_generator(graph_stream, usage_metrics, run_agent_input)

    async def _stream_generator(
        self,
        graph_stream: AsyncGenerator[tuple[Any, str, Any]],
        usage_metrics: UsageMetrics,
        run_agent_input: RunAgentInput,
    ) -> InvokeReturn:
        # Partial AG-UI: workflow lifecycle events
        thread_id = run_agent_input.thread_id
        run_id = run_agent_input.run_id
        yield RunStartedEvent(thread_id=thread_id, run_id=run_id), None, usage_metrics

        # Iterate over the graph stream. For message events, yield the content.
        # For update events, accumulate the usage metrics.
        events = []
        current_message_id = None
        tool_call_id = ""
        async for _, mode, event in graph_stream:
            if mode == "messages":
                message_event: tuple[AIMessageChunk | ToolMessage, dict[str, Any]] = event  # type: ignore[assignment]
                message = message_event[0]
                if isinstance(message, ToolMessage):
                    tool_message_id = (
                        str(message.id)
                        if getattr(message, "id", None) is not None
                        else message.tool_call_id
                    )
                    yield (
                        ToolCallEndEvent(
                            type=EventType.TOOL_CALL_END, tool_call_id=message.tool_call_id
                        ),
                        None,
                        usage_metrics,
                    )
                    yield (
                        ToolCallResultEvent(
                            type=EventType.TOOL_CALL_RESULT,
                            message_id=tool_message_id,
                            tool_call_id=message.tool_call_id,
                            content=str(message.content),
                            role="tool",
                        ),
                        None,
                        usage_metrics,
                    )
                    tool_call_id = ""
                elif isinstance(message, AIMessageChunk):
                    if message.tool_call_chunks:
                        # This is a tool call message
                        for tool_call_chunk in message.tool_call_chunks:
                            if name := tool_call_chunk.get("name"):
                                # Its a tool call start message
                                tcid = tool_call_chunk.get("id")
                                if tcid:
                                    tool_call_id = str(tcid)
                                yield (
                                    ToolCallStartEvent(
                                        type=EventType.TOOL_CALL_START,
                                        tool_call_id=tool_call_id,
                                        tool_call_name=name,
                                        parent_message_id=str(message.id or ""),
                                    ),
                                    None,
                                    usage_metrics,
                                )
                            elif args := tool_call_chunk.get("args"):
                                # Its a tool call args message
                                yield (
                                    ToolCallArgsEvent(
                                        type=EventType.TOOL_CALL_ARGS,
                                        # Its empty when the tool chunk is not a start message
                                        # So we use the tool call id from a previous start message
                                        tool_call_id=tool_call_id,
                                        delta=args,
                                    ),
                                    None,
                                    usage_metrics,
                                )
                    elif message.content:
                        # Its a text message
                        # Handle the start and end of the text message
                        if message.id != current_message_id:
                            if current_message_id:
                                yield (
                                    TextMessageEndEvent(
                                        type=EventType.TEXT_MESSAGE_END,
                                        message_id=current_message_id,
                                    ),
                                    None,
                                    usage_metrics,
                                )
                            current_message_id = str(message.id or "")
                            yield (
                                TextMessageStartEvent(
                                    type=EventType.TEXT_MESSAGE_START,
                                    message_id=current_message_id,
                                    role="assistant",
                                ),
                                None,
                                usage_metrics,
                            )
                        yield (
                            TextMessageContentEvent(
                                type=EventType.TEXT_MESSAGE_CONTENT,
                                message_id=current_message_id,
                                delta=str(message.content),
                            ),
                            None,
                            usage_metrics,
                        )
                else:
                    raise ValueError(f"Invalid message event: {message_event}")
            elif mode == "updates":
                update_event: dict[str, Any] = event  # type: ignore[assignment]
                if "__interrupt__" in update_event:
                    intr_tuple = update_event["__interrupt__"]
                    serialized: list[dict[str, Any]] = []
                    for intr in intr_tuple:
                        serialized.append({"id": intr.id, "value": intr.value})
                    custom_value = {
                        "kind": "langgraph_interrupt",
                        "interrupts": serialized,
                    }
                    yield (
                        CustomEvent(
                            type=EventType.CUSTOM,
                            name="langgraph.interrupt",
                            value=custom_value,
                        ),
                        None,
                        usage_metrics,
                    )
                    events.append(update_event)
                    yield (
                        RunFinishedEvent(
                            thread_id=thread_id,
                            run_id=run_id,
                            result={"langgraph": {"interrupted": True, **custom_value}},
                        ),
                        None,
                        usage_metrics,
                    )
                    return
                events.append(update_event)
                current_node = next(iter(update_event))
                node_data = update_event[current_node]
                current_usage = node_data.get("usage", {}) if node_data is not None else {}
                if current_usage:
                    usage_metrics["total_tokens"] += current_usage.get("total_tokens", 0)
                    usage_metrics["prompt_tokens"] += current_usage.get("prompt_tokens", 0)
                    usage_metrics["completion_tokens"] += current_usage.get("completion_tokens", 0)
                if current_message_id:
                    yield (
                        TextMessageEndEvent(
                            type=EventType.TEXT_MESSAGE_END,
                            message_id=current_message_id,
                        ),
                        None,
                        usage_metrics,
                    )
                    current_message_id = None

        # Create a list of events from the event listener
        pipeline_interactions = self.create_pipeline_interactions_from_events(events)

        vars_list = getattr(self.prompt_template, "input_variables", [])
        if "memory" in vars_list:
            user_prompt = extract_user_prompt_content(run_agent_input)
            try:
                await self.store_memory_for_run(user_prompt, run_agent_input)
            except Exception as exc:
                logger.warning("LangGraph memory storage failed: %s", exc)

        yield (
            RunFinishedEvent(thread_id=thread_id, run_id=run_id),
            pipeline_interactions,
            usage_metrics,
        )

    @classmethod
    def create_pipeline_interactions_from_events(
        cls,
        events: list[dict[str, Any]] | None,
    ) -> MultiTurnSample | None:
        """Convert a list of LangGraph events into Ragas MultiTurnSample."""
        if not events:
            return None
        messages = []
        for e in events:
            for _, v in e.items():
                if v is not None:
                    messages.extend(v.get("messages", []))
        messages = [m for m in messages if not isinstance(m, ToolMessage)]
        # Lazy import to reduce memory overhead when ragas is not used
        from ragas import MultiTurnSample
        from ragas.integrations.langgraph import convert_to_ragas_messages

        ragas_trace = convert_to_ragas_messages(messages)
        return MultiTurnSample(user_input=ragas_trace)


def datarobot_agent_class_from_langgraph(
    graph_factory: Callable[[BaseChatModel, list[BaseTool], bool], StateGraph[MessagesState]],
    prompt_template: ChatPromptTemplate,
) -> type[LangGraphAgent]:
    """Create a LangGraph agent class from a graph factory and prompt template.

    This is a convenience helper that dynamically builds a concrete
    :class:`LangGraphAgent` subclass so that callers can define an agent
    entirely from a graph-building function and a prompt template without
    writing a class by hand.

    Parameters
    ----------
    graph_factory : Callable[[BaseChatModel, list[BaseTool], bool], StateGraph[MessagesState]]
        A callable that receives the LLM client, the list of tools bound to
        the agent, and a ``verbose`` flag, and returns a compiled LangGraph
        :class:`StateGraph` ready for execution.
    prompt_template : ChatPromptTemplate
        The LangChain prompt template used to format user input before it is
        fed into the graph. If the template declares a ``{chat_history}``
        input variable, prior conversation turns are automatically injected.

    Returns
    -------
    type[LangGraphAgent]
        A new :class:`LangGraphAgent` subclass whose ``workflow`` and
        ``prompt_template`` properties are wired to the provided arguments.

    Human-in-the-loop
    ------------------
    Checkpointing defaults to :class:`~langgraph.checkpoint.memory.InMemorySaver`
    per agent instance. Override :attr:`LangGraphAgent.langgraph_checkpointer` to
    use another saver. Use ``interrupt()`` in graph nodes and pass resume payloads
    via ``run_agent_input.state["langgraph_resume"]``.
    """

    class DataRobotLangAgent(LangGraphAgent):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)

        @property
        def workflow(self) -> StateGraph[MessagesState]:
            return graph_factory(self.llm, self.tools, self.verbose)

        @property
        def prompt_template(self) -> ChatPromptTemplate:
            return prompt_template

    return DataRobotLangAgent
