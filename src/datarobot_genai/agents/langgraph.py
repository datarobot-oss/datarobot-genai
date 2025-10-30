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
from collections.abc import AsyncGenerator
from typing import Any
from typing import Union

from langchain_core.messages import HumanMessage
from langchain_core.messages import ToolMessage
from langgraph.graph import START
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph
from langgraph.types import Command
from openai.types.chat import CompletionCreateParams
from ragas import MultiTurnSample
from ragas.integrations.langgraph import convert_to_ragas_messages

from .base import BaseAgent
from .base import extract_user_prompt_content


class LangGraphAgent(BaseAgent, abc.ABC):
    @abc.abstractmethod
    @property
    def workflow(self) -> StateGraph[MessagesState]:
        raise NotImplementedError("Not implemented")

    @abc.abstractmethod
    @property
    def prompt_template(self) -> str:
        raise NotImplementedError("Not implemented")

    def convert_input_message(self, completion_create_params: CompletionCreateParams) -> Command:
        user_prompt_content = extract_user_prompt_content(completion_create_params)
        command = Command(
            update={
                "messages": [
                    HumanMessage(content=self.prompt_template.format(user_prompt_content))
                ],
            },
            goto=START,
        )
        return command

    async def invoke(
        self, completion_create_params: CompletionCreateParams
    ) -> Union[  # noqa: UP007
        AsyncGenerator[tuple[str, Any | None, dict[str, int]], None],
        tuple[str, Any | None, dict[str, int]],
    ]:
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
        # Print commands may need flush=True to ensure they are displayed in real-time.
        print(
            f"Running agent with user prompt: {input_command.update['messages'][0].content}",  # type: ignore[index]
            flush=True,
        )

        # Create and invoke the Langgraph Agentic Workflow with the inputs
        langgraph_execution_graph = self.workflow.compile()

        graph_stream = langgraph_execution_graph.astream(
            input=input_command,
            config={"recursion_limit": 150},  # Maximum number of steps to take in the graph
            debug=True,
        )

        usage_metrics: dict[str, int] = {
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
                tuple[str, Any | None, dict[str, int]], None
            ]:
                # For each event in the graph stream, yield the latest message content
                # along with updated usage metrics.
                events = []
                async for event in graph_stream:
                    events.append(event)
                    current_node = next(iter(event))
                    yield (
                        str(event[current_node]["messages"][-1].content),
                        None,
                        usage_metrics,
                    )
                    current_usage = event[current_node].get("usage", {})
                    if current_usage:
                        usage_metrics["total_tokens"] += current_usage.get("total_tokens", 0)
                        usage_metrics["prompt_tokens"] += current_usage.get("prompt_tokens", 0)
                        usage_metrics["completion_tokens"] += current_usage.get(
                            "completion_tokens", 0
                        )

                # Create a list of events from the event listener
                pipeline_interactions = self.create_pipeline_interactions_from_events(events)

                # yield the final response indicating completion
                yield "", pipeline_interactions, usage_metrics

            return stream_generator()
        else:
            # Synchronous response: collect all events and return the final message
            events = [event async for event in graph_stream]
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
