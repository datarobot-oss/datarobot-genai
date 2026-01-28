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
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from ag_ui.core import Event
from ag_ui.core import EventType
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
from openai.types.chat import CompletionCreateParams

from datarobot_genai.core.agents.base import BaseAgent
from datarobot_genai.core.agents.base import InvokeReturn
from datarobot_genai.core.agents.base import UsageMetrics
from datarobot_genai.core.agents.base import extract_user_prompt_content
from datarobot_genai.core.agents.base import is_streaming
from datarobot_genai.langgraph.mcp import mcp_tools_context

if TYPE_CHECKING:
    from ragas import MultiTurnSample

logger = logging.getLogger(__name__)


class LangGraphAgent(BaseAgent[BaseTool], abc.ABC):
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

    def convert_input_message(self, completion_create_params: CompletionCreateParams) -> Command:
        user_prompt = extract_user_prompt_content(completion_create_params)
        command = Command(  # type: ignore[var-annotated]
            update={
                "messages": self.prompt_template.invoke(user_prompt).to_messages(),
            },
        )
        return command

    async def invoke(self, completion_create_params: CompletionCreateParams) -> InvokeReturn:
        """Run the agent with the provided completion parameters.

        Args:
            completion_create_params: The completion request parameters including input topic and
            settings.

        Returns
        -------
            For streaming requests, returns a generator yielding tuples of (response_text,
            pipeline_interactions, usage_metrics).
            For non-streaming requests, returns a single tuple of (response_text,
            pipeline_interactions, usage_metrics).
        """
        # For streaming, we need to keep the context alive until the generator is exhausted
        if completion_create_params.get("stream"):

            async def wrapped_generator() -> AsyncGenerator[
                tuple[str, Any | None, UsageMetrics], None
            ]:
                try:
                    async with mcp_tools_context(
                        authorization_context=self._authorization_context,
                        forwarded_headers=self.forwarded_headers,
                    ) as mcp_tools:
                        self.set_mcp_tools(mcp_tools)
                        result = await self._invoke(completion_create_params)

                        # Yield all items from the result generator
                        # The context will be closed when this generator is exhausted
                        # Cast to async generator since we know stream=True means it's a generator
                        result_generator = cast(
                            AsyncGenerator[tuple[str, Any | None, UsageMetrics], None], result
                        )
                        async for item in result_generator:
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

            return wrapped_generator()
        else:
            # For non-streaming, use async with directly
            async with mcp_tools_context(
                authorization_context=self._authorization_context,
                forwarded_headers=self.forwarded_headers,
            ) as mcp_tools:
                self.set_mcp_tools(mcp_tools)
                result = await self._invoke(completion_create_params)

            return result

    async def _invoke(self, completion_create_params: CompletionCreateParams) -> InvokeReturn:
        input_command = self.convert_input_message(completion_create_params)
        logger.info(
            f"Running a langgraph agent with a command: {input_command}",
        )

        # Create and invoke the Langgraph Agentic Workflow with the inputs
        langgraph_execution_graph = self.workflow.compile()

        graph_stream = langgraph_execution_graph.astream(
            input=input_command,
            config=self.langgraph_config,
            debug=self.verbose,
            # Streaming updates and messages from all the nodes
            stream_mode=["updates", "messages"],
            subgraphs=True,
        )

        usage_metrics: UsageMetrics = {
            "completion_tokens": 0,
            "prompt_tokens": 0,
            "total_tokens": 0,
        }

        # The following code demonstrate both a synchronous and streaming response.
        # You can choose one or the other based on your use case, they function the same.
        # The main difference is returning a generator for streaming or a final response for sync.
        if is_streaming(completion_create_params):
            # Streaming response: yield each message as it is generated
            return self._stream_generator(graph_stream, usage_metrics)
        else:
            # Synchronous response: collect all events and return the final message
            events: list[dict[str, Any]] = [
                event  # type: ignore[misc]
                async for _, mode, event in graph_stream
                if mode == "updates"
            ]

            # Accumulate the usage metrics from the updates
            for update in events:
                current_node = next(iter(update))
                node_data = update[current_node]
                current_usage = node_data.get("usage", {}) if node_data is not None else {}
                if current_usage:
                    usage_metrics["total_tokens"] += current_usage.get("total_tokens", 0)
                    usage_metrics["prompt_tokens"] += current_usage.get("prompt_tokens", 0)
                    usage_metrics["completion_tokens"] += current_usage.get("completion_tokens", 0)

            pipeline_interactions = self.create_pipeline_interactions_from_events(events)

            # Extract the final event from the graph stream as the synchronous response
            last_event = events[-1]
            node_name = next(iter(last_event))
            node_data = last_event[node_name]
            response_text = (
                str(node_data["messages"][-1].content)
                if node_data is not None and "messages" in node_data
                else ""
            )

            return response_text, pipeline_interactions, usage_metrics

    async def _stream_generator(
        self, graph_stream: AsyncGenerator[tuple[Any, str, Any], None], usage_metrics: UsageMetrics
    ) -> AsyncGenerator[tuple[str | Event, MultiTurnSample | None, UsageMetrics], None]:
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
                            message_id=message.id,
                            tool_call_id=message.tool_call_id,
                            content=message.content,
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
                                tool_call_id = tool_call_chunk["id"]
                                yield (
                                    ToolCallStartEvent(
                                        type=EventType.TOOL_CALL_START,
                                        tool_call_id=tool_call_id,
                                        tool_call_name=name,
                                        parent_message_id=message.id,
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
                            current_message_id = message.id
                            yield (
                                TextMessageStartEvent(
                                    type=EventType.TEXT_MESSAGE_START,
                                    message_id=message.id,
                                    role="assistant",
                                ),
                                None,
                                usage_metrics,
                            )
                        yield (
                            TextMessageContentEvent(
                                type=EventType.TEXT_MESSAGE_CONTENT,
                                message_id=message.id,
                                delta=message.content,
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

        # yield the final response indicating completion
        yield "", pipeline_interactions, usage_metrics

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
