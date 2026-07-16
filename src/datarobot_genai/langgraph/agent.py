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
import contextlib
import json
import logging
import uuid
from collections.abc import AsyncGenerator
from collections.abc import Callable
from collections.abc import Mapping
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from ag_ui.core import EventType
from ag_ui.core import ReasoningMessageChunkEvent
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
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph
from langgraph.types import Command
from nat.plugins.langchain.callback_handler import LangchainProfilerHandler

from datarobot_genai.core.agents.base import BaseAgent
from datarobot_genai.core.agents.base import InvokeReturn
from datarobot_genai.core.agents.base import UsageMetrics
from datarobot_genai.core.agents.base import extract_user_prompt_content
from datarobot_genai.core.agents.reasoning import flatten_to_text
from datarobot_genai.langgraph.history import ag_ui_history_to_langchain
from datarobot_genai.langgraph.moderations_events import convert_langchain_messages
from datarobot_genai.langgraph.reasoning import iter_message_blocks

if TYPE_CHECKING:
    from datarobot_genai.core.pipeline_interactions import MultiTurnSample

logger = logging.getLogger(__name__)

# Must match the AG-UI client: useAgUiTool({ name: "confirmation" }) registers "ui-confirmation"
# (e.g. ConfirmationWidget with { message: string }).
INTERRUPT_CONFIRMATION_AGUI_TOOL_NAME = "ui-confirmation"


def _confirmation_args_from_interrupt_value(value: Any) -> dict[str, str]:
    """Map a LangGraph interrupt value to the confirmation tool args."""
    if isinstance(value, dict):
        m = value.get("message")
        if isinstance(m, str):
            return {"message": m}
        if "e2e_prompt" in value:
            return {"message": str(value["e2e_prompt"])}
        for key in ("prompt", "text", "question"):
            if key in value:
                return {"message": str(value[key])}
    return {"message": str(value)}


# RunAgentInput.state / forwarded_props key for Command(resume=..., goto=START).
LANGGRAPH_RESUME_STATE_KEY = "langgraph_resume"


class LangGraphAgent(BaseAgent[BaseTool], abc.ABC):
    """Base class for LangGraph-powered agents.

    This class wires LangGraph workflows into the generic `BaseAgent` interface
    and provides a default implementation for turning OpenAI-style chat inputs
    into LangGraph input.

    History is opt-in: prior turns are only injected when the prompt template
    declares and uses a `{chat_history}` input variable. If the template does
    not use `{chat_history}`, no chat history is passed to the model.

    Framework-specific parameters:

    - ``checkpointer``: checkpoint store for ``interrupt()`` and thread resume. If omitted,
      use  no checkpointer is installed.
    - ``interrupt_before`` / ``interrupt_after``: compile-time interrupt node lists.
    - ``debug``: forwarded to ``StateGraph.compile(debug=...)``.
    - ``name``: optional name for the compiled graph.
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
        structured_history: bool = True,
        checkpointer: Any | None = None,
        interrupt_before: Any | None = None,
        interrupt_after: Any | None = None,
        debug: bool = False,
        name: str | None = None,
    ) -> None:
        self.checkpointer = checkpointer
        self.interrupt_before = interrupt_before
        self.interrupt_after = interrupt_after
        self.debug = debug
        self.name = name
        self._structured_history = structured_history
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

    @property
    def structured_history(self) -> bool:
        """When true, prior turns are fed to the model as structured native messages
        (tool calls preserved) instead of the text ``{chat_history}`` summary. Only
        applies when the prompt template does not declare ``{chat_history}``.

        Defaults to ``True`` so multi-turn history is replayed by default; pass
        ``structured_history=False`` to opt out.
        """
        return self._structured_history

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

    def _command_for_interrupt_resume(self, resume_payload: Any) -> Command:
        """Return the LangGraph input for continuing after ``interrupt()``."""
        return Command(resume=resume_payload, goto=START)

    async def _command_for_pending_interrupt(
        self,
        compiled_graph: Any,
        run_agent_input: RunAgentInput,
    ) -> Command | None:
        """When the thread is paused on `interrupt()`, map the user message to `resume`."""
        if self.checkpointer is None:
            return None
        config = cast(Any, self.build_langgraph_runnable_config(run_agent_input))
        snap = await compiled_graph.aget_state(config)
        interrupts = getattr(snap, "interrupts", None)
        if not interrupts:
            return None
        user_reply = extract_user_prompt_content(run_agent_input)
        if len(interrupts) == 1:
            return self._command_for_interrupt_resume(user_reply)
        return self._command_for_interrupt_resume({intr.id: user_reply for intr in interrupts})

    async def _build_input_command(
        self,
        run_agent_input: RunAgentInput,
        compiled_graph: Any,
    ) -> dict[str, Any] | Command:
        """Resolve LangGraph input: explicit resume, pending interrupt, or normal prompt."""
        state = run_agent_input.state
        resume_present = False
        resume_payload: Any = None
        if isinstance(state, Mapping) and LANGGRAPH_RESUME_STATE_KEY in state:
            resume_present, resume_payload = True, state[LANGGRAPH_RESUME_STATE_KEY]
        elif (
            isinstance(run_agent_input.forwarded_props, Mapping)
            and LANGGRAPH_RESUME_STATE_KEY in run_agent_input.forwarded_props
        ):
            resume_present, resume_payload = (
                True,
                run_agent_input.forwarded_props[LANGGRAPH_RESUME_STATE_KEY],
            )
        if resume_present:
            return self._command_for_interrupt_resume(resume_payload)

        pending = await self._command_for_pending_interrupt(compiled_graph, run_agent_input)
        if pending is not None:
            return pending

        return await self.convert_input_message(run_agent_input)

    async def convert_input_message(self, run_agent_input: RunAgentInput) -> dict[str, Any]:
        """Convert AG-UI input into a LangGraph state input.

        By default this:
        - Extracts the last user message content via `extract_user_prompt_content`
          and feeds it through `prompt_template` to build the current turn.
        - Includes prior turns as a plain-text `{chat_history}` variable when the
          template declares it, or as structured native messages (tool calls
          preserved) when `structured_history` is set.
        """
        user_prompt = extract_user_prompt_content(run_agent_input)

        # Chat history is opt-in: the model only sees a text summary when the prompt
        # template declares/uses the `{chat_history}` variable.
        input_vars = getattr(self.prompt_template, "input_variables", [])
        try:
            vars_list = list(input_vars)
        except TypeError:
            vars_list = []
        uses_chat_history = "chat_history" in vars_list
        history_summary = self.build_history_summary(run_agent_input) if uses_chat_history else ""

        # Prefer structured dict input when available so templates can access both
        # the original fields (e.g. {topic}) and a plain-text {chat_history}.
        if isinstance(user_prompt, Mapping):
            template_input: Any = dict(user_prompt)
            template_input.setdefault("chat_history", history_summary)
        elif vars_list:
            # When the prompt is a bare value, best-effort map it into the declared
            # input variables. Known variable "chat_history" always receives the
            # history summary; all other variables receive the raw user prompt.
            template_input = {}
            for name in vars_list:
                if name == "chat_history":
                    template_input[name] = history_summary
                else:
                    template_input[name] = user_prompt
        else:
            # No declared variables: pass the prompt through unchanged.
            template_input = user_prompt

        current_messages = self.prompt_template.invoke(template_input).to_messages()
        # Structured native history is opt-in via `structured_history`, and only when
        # the template does not already take a text `{chat_history}`.
        if self.structured_history and not uses_chat_history:
            current_messages = self._with_structured_history(current_messages, run_agent_input)
        return {"messages": current_messages}

    def _with_structured_history(
        self,
        current_messages: list[BaseMessage],
        run_agent_input: RunAgentInput,
    ) -> list[BaseMessage]:
        """Splice structured prior turns (tool calls preserved) before the current
        turn's first human message, after any leading system prompt.
        """
        history = ag_ui_history_to_langchain(self.history_messages(run_agent_input))
        if not history:
            return current_messages
        # Insert history before the current turn's first human message (so it sits
        # after any leading system prompt). Don't assume the user turn is last — some
        # templates emit trailing messages after it.
        insert_at = next(
            (i for i, m in enumerate(current_messages) if isinstance(m, HumanMessage)),
            len(current_messages),
        )
        return current_messages[:insert_at] + history + current_messages[insert_at:]

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
        graph_input = await self._build_input_command(run_agent_input, langgraph_execution_graph)
        logger.info(
            f"Running a langgraph agent with input: {graph_input}",
        )
        graph_stream = cast(
            AsyncGenerator[tuple[Any, str, Any], None],
            langgraph_execution_graph.astream(
                input=graph_input,
                # LangGraph expects a RunnableConfig, but our config is a plain dict.
                # Cast to Any to avoid leaking LangGraph internals into this interface.
                config=cast(Any, self.build_langgraph_runnable_config(run_agent_input)),
                debug=self.debug or self.verbose,
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
        # ``current_text_key`` tracks whether a text message is open and, if so, which
        # one: (langgraph_node, message_id). ``None`` means no text message is open.
        # Keying on the node (not the message id alone) is what keeps sequential nodes
        # separate when their messages share or omit an id (see the text branch below).
        # ``current_message_id`` holds the id emitted for the open message; it is always
        # overwritten with a non-empty value before any event is emitted for it.
        current_message_id: str = ""
        current_text_key: tuple[str | None, str] | None = None
        tool_call_id = ""
        async with contextlib.aclosing(graph_stream):
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
                                            # So we use the tool call id from a previous start
                                            # message
                                            tool_call_id=tool_call_id,
                                            delta=args,
                                        ),
                                        None,
                                        usage_metrics,
                                    )
                        elif message.content or message.additional_kwargs.get("reasoning_content"):
                            for kind, delta in iter_message_blocks(message):
                                if kind == "thinking":
                                    yield (
                                        ReasoningMessageChunkEvent(
                                            type=EventType.REASONING_MESSAGE_CHUNK,
                                            # Its own message id, distinct from the assistant text
                                            # so a frontend grouping by id renders reasoning as its
                                            # own block instead of folding it into the text bubble.
                                            # Derived (uuid5) from the text id: a valid UUID that is
                                            # stable across this message's chunks (so they group)
                                            # without extra state.
                                            message_id=str(
                                                uuid.uuid5(
                                                    uuid.NAMESPACE_OID,
                                                    f"{message.id or ''}-reasoning",
                                                )
                                            ),
                                            delta=delta,
                                        ),
                                        None,
                                        usage_metrics,
                                    )
                                    continue
                                # kind == "text": open a new AG-UI text message at each
                                # node/message boundary. The boundary key pairs the
                                # LangGraph node name with the message id. The id alone is
                                # not enough: sequential nodes that each call ``.invoke()``
                                # can surface messages with a shared or empty id, and keying
                                # on the id alone would fuse both nodes' text into one bubble
                                # (and, for non-streaming callers, into one concatenated
                                # response) because no boundary event is ever emitted.
                                node_meta = message_event[1]
                                node_name = (
                                    node_meta.get("langgraph_node")
                                    if isinstance(node_meta, Mapping)
                                    else None
                                )
                                text_key = (node_name, str(message.id or ""))
                                if text_key != current_text_key:
                                    if current_text_key is not None:
                                        yield (
                                            TextMessageEndEvent(
                                                type=EventType.TEXT_MESSAGE_END,
                                                message_id=current_message_id,
                                            ),
                                            None,
                                            usage_metrics,
                                        )
                                    current_text_key = text_key
                                    # Prefer the real id; mint a stable one when the node
                                    # emitted a message without an id so the stream stays
                                    # well-formed (START before CONTENT).
                                    current_message_id = (
                                        str(message.id) if message.id else uuid.uuid4().hex
                                    )
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
                                        delta=delta,
                                    ),
                                    None,
                                    usage_metrics,
                                )
                    elif isinstance(message, HumanMessage):
                        # Intermediate relay nodes (e.g. planner-to-writer handoffs) may emit
                        # HumanMessages as state updates. These are not streamed to the caller.
                        pass
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
                            "kind": "on_interrupt",
                            "interrupts": serialized,
                        }
                        for intr in intr_tuple:
                            tool_call_id = (
                                str(intr.id) if getattr(intr, "id", None) else uuid.uuid4().hex
                            )
                            args_dict = _confirmation_args_from_interrupt_value(intr.value)
                            args_json = json.dumps(args_dict)
                            yield (
                                ToolCallStartEvent(
                                    type=EventType.TOOL_CALL_START,
                                    tool_call_id=tool_call_id,
                                    tool_call_name=INTERRUPT_CONFIRMATION_AGUI_TOOL_NAME,
                                    parent_message_id="",
                                ),
                                None,
                                usage_metrics,
                            )
                            yield (
                                ToolCallArgsEvent(
                                    type=EventType.TOOL_CALL_ARGS,
                                    tool_call_id=tool_call_id,
                                    delta=args_json,
                                ),
                                None,
                                usage_metrics,
                            )
                            yield (
                                ToolCallEndEvent(
                                    type=EventType.TOOL_CALL_END,
                                    tool_call_id=tool_call_id,
                                ),
                                None,
                                usage_metrics,
                            )
                            yield (
                                ToolCallResultEvent(
                                    type=EventType.TOOL_CALL_RESULT,
                                    message_id=tool_call_id,
                                    tool_call_id=tool_call_id,
                                    content=json.dumps(
                                        {
                                            "langgraphInterrupt": True,
                                            "interruptId": tool_call_id,
                                        }
                                    ),
                                    role="tool",
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
                        usage_metrics["completion_tokens"] += current_usage.get(
                            "completion_tokens", 0
                        )
                    if current_text_key is not None:
                        yield (
                            TextMessageEndEvent(
                                type=EventType.TEXT_MESSAGE_END,
                                message_id=current_message_id,
                            ),
                            None,
                            usage_metrics,
                        )
                        current_text_key = None

        # Create a list of events from the event listener
        pipeline_interactions = self.create_pipeline_interactions_from_events(events)

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
        """Convert a list of LangGraph events into a pipeline-interactions sample."""
        if not events:
            return None
        messages = []
        for e in events:
            for _, v in e.items():
                if v is not None:
                    messages.extend(v.get("messages", []))
        messages = [m for m in messages if not isinstance(m, ToolMessage)]
        # Flatten list-form content (reasoning models emit blocks) so the
        # string-only message schema does not raise. Thinking blocks are dropped
        # from the evaluation trace; the AG-UI stream still carries them as
        # structured Thinking* events.
        flattened: list[Any] = []
        for m in messages:
            if isinstance(m.content, list):
                flattened.append(m.model_copy(update={"content": flatten_to_text(m.content)}))
            else:
                flattened.append(m)
        # Lazy import so the moderations-backed primitives load only when a run
        # actually records pipeline interactions.
        from datarobot_genai.core.pipeline_interactions import MultiTurnSample

        return MultiTurnSample(user_input=convert_langchain_messages(flattened))


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
    Pass any explicit ``checkpointer=``.
    You can also pass resume payloads via ``run_agent_input.state["langgraph_resume"]``.
    """

    class DataRobotLangAgent(LangGraphAgent):
        @property
        def workflow(self) -> StateGraph[MessagesState]:
            return graph_factory(self.llm, self.tools, self.verbose)

        @property
        def prompt_template(self) -> ChatPromptTemplate:
            return prompt_template

    return DataRobotLangAgent
