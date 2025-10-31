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
from typing import Any

from langchain_core.messages import AIMessageChunk
from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph
from langgraph.types import Command
from openai.types.chat import CompletionCreateParams
from ragas import MultiTurnSample
from ragas.integrations.langgraph import convert_to_ragas_messages

from datarobot_genai.core.agents.base import BaseAgent
from datarobot_genai.core.agents.base import InvokeReturn
from datarobot_genai.core.agents.base import UsageMetrics
from datarobot_genai.core.agents.base import extract_user_prompt_content

logger = logging.getLogger(__name__)


class LangGraphAgent(BaseAgent, abc.ABC):
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
        if completion_create_params.get("stream"):
            # Streaming response: yield each message as it is generated
            async def stream_generator() -> AsyncGenerator[
                tuple[str, Any | None, UsageMetrics], None
            ]:
                # Iterate over the graph stream. For message events, yield the content.
                # For update events, accumulate the usage metrics.
                events = []
                async for _, mode, event in graph_stream:
                    if mode == "messages":
                        message_event: tuple[AIMessageChunk, dict[str, Any]] = event  # type: ignore[assignment]
                        llm_token, _ = message_event
                        yield (
                            str(llm_token.content),
                            None,
                            usage_metrics,
                        )
                    elif mode == "updates":
                        update_event: dict[str, Any] = event  # type: ignore[assignment]
                        events.append(update_event)
                        current_node = next(iter(update_event))
                        current_usage = update_event[current_node].get("usage", {})
                        if current_usage:
                            usage_metrics["total_tokens"] += current_usage.get("total_tokens", 0)
                            usage_metrics["prompt_tokens"] += current_usage.get("prompt_tokens", 0)
                            usage_metrics["completion_tokens"] += current_usage.get(
                                "completion_tokens", 0
                            )
                    else:
                        raise ValueError(f"Invalid mode: {mode}")

                # Create a list of events from the event listener
                pipeline_interactions = self.create_pipeline_interactions_from_events(events)

                # yield the final response indicating completion
                yield "", pipeline_interactions, usage_metrics

            return stream_generator()
        else:
            # Synchronous response: collect all events and return the final message
            events: list[dict[str, Any]] = [
                event  # type: ignore[misc]
                async for _, mode, event in graph_stream
                if mode == "updates"
            ]

            pipeline_interactions = self.create_pipeline_interactions_from_events(events)

            # Extract the final event from the graph stream as the synchronous response
            last_event = events[-1]
            node_name = next(iter(last_event))
            response_text = str(last_event[node_name]["messages"][-1].content)
            current_usage = last_event[node_name].get("usage", {})
            if current_usage:
                usage_metrics["total_tokens"] += current_usage.get("total_tokens", 0)
                usage_metrics["prompt_tokens"] += current_usage.get("prompt_tokens", 0)
                usage_metrics["completion_tokens"] += current_usage.get("completion_tokens", 0)

            return response_text, pipeline_interactions, usage_metrics

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
                messages.extend(v.get("messages", []))
        messages = [m for m in messages if not isinstance(m, ToolMessage)]
        ragas_trace = convert_to_ragas_messages(messages)
        return MultiTurnSample(user_input=ragas_trace)
