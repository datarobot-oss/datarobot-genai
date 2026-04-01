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
import abc
import logging
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING
from typing import Any
from typing import Optional
from typing import cast

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
from langchain.tools import BaseTool
from langchain_core.messages import AIMessageChunk
from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph
from langgraph.types import Command

from datarobot_genai.core.agents.base import BaseAgent
from datarobot_genai.core.agents.base import InvokeReturn
from datarobot_genai.core.agents.base import UsageMetrics
from datarobot_genai.core.agents.base import extract_user_prompt_content
from datarobot_genai.core.agents.message_converters import to_langchain_messages
from datarobot_genai.core.agents.message_converters import truncate_messages

if TYPE_CHECKING:
    from ragas import MultiTurnSample

logger = logging.getLogger(__name__)


class LangGraphAgent(BaseAgent[BaseTool], abc.ABC):
    """Base class for LangGraph-powered agents.

    This class wires LangGraph workflows into the generic `BaseAgent` interface
    and provides a default implementation for turning OpenAI-style chat inputs
    into a LangGraph `Command`.

    For multi-turn conversations (more than one message), all messages are
    converted to native LangChain types preserving tool_call structure.
    The system prompt from ``prompt_template`` is prepended when no system
    message is present in the request.  For single-message input, the
    prompt template is used for formatting.
    """

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
        }

    def convert_input_message(self, run_agent_input: RunAgentInput) -> Command:
        """Convert AG-UI input into a LangGraph `Command`.

        When multi-turn history is present (more than one message) and
        ``max_history_messages > 0``, converts all messages to native LangChain
        types preserving tool_call structure.  If ``max_history_messages == 0``
        or only a single message is present, falls back to the template-based
        approach.
        """
        from collections.abc import Mapping

        messages = list(run_agent_input.messages)

        # Multi-turn: convert to native LangChain types (truncated to max_history_messages)
        if len(messages) > 1 and self.max_history_messages > 0:
            from langchain_core.messages import SystemMessage

            truncated = truncate_messages(messages, self.max_history_messages)
            lc_messages = to_langchain_messages(truncated)

            # Preserve system prompt from the template when not already in messages
            if not any(isinstance(m, SystemMessage) for m in lc_messages):
                input_vars = getattr(self.prompt_template, "input_variables", [])
                try:
                    empty_vars = {name: "" for name in list(input_vars)}
                except TypeError:
                    empty_vars = {}
                template_msgs = self.prompt_template.invoke(empty_vars).to_messages()
                sys_msgs = [m for m in template_msgs if isinstance(m, SystemMessage)]
                if sys_msgs:
                    lc_messages = sys_msgs + lc_messages

            return Command(update={"messages": lc_messages})

        # Single-turn: use prompt template for formatting
        user_prompt = extract_user_prompt_content(run_agent_input)

        input_vars = getattr(self.prompt_template, "input_variables", [])
        try:
            vars_list = list(input_vars)
        except TypeError:
            vars_list = []

        if isinstance(user_prompt, Mapping):
            template_input: Any = dict(user_prompt)
            for name in vars_list:
                template_input.setdefault(name, "")
        elif vars_list:
            # Map the first declared variable to user_prompt, default the rest to ""
            template_input = {}
            for i, name in enumerate(vars_list):
                template_input[name] = user_prompt if i == 0 else ""
        else:
            template_input = user_prompt

        current_messages = self.prompt_template.invoke(template_input).to_messages()
        return Command(update={"messages": current_messages})

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
        input_command = self.convert_input_message(run_agent_input)
        logger.info(
            f"Running a langgraph agent with a command: {input_command}",
        )

        # Create and invoke the Langgraph Agentic Workflow with the inputs
        langgraph_execution_graph = self.workflow.compile()

        graph_stream = cast(
            AsyncGenerator[tuple[Any, str, Any], None],
            langgraph_execution_graph.astream(
                input=input_command,
                # LangGraph expects a RunnableConfig, but our config is a plain dict.
                # Cast to Any to avoid leaking LangGraph internals into this interface.
                config=cast(Any, self.langgraph_config),
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
        graph_stream: AsyncGenerator[tuple[Any, str, Any], None],
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

        yield (
            RunFinishedEvent(thread_id=thread_id, run_id=run_id),
            pipeline_interactions,
            usage_metrics,
        )

    @classmethod
    def create_pipeline_interactions_from_events(
        cls,
        events: list[dict[str, Any]] | None,
    ) -> Optional["MultiTurnSample"]:
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
