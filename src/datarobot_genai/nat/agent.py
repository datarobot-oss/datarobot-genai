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
import asyncio
import logging
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from ag_ui.core import Event
from ag_ui.core import EventType
from ag_ui.core import TextMessageChunkEvent
from ag_ui.core import TextMessageEndEvent
from ag_ui.core import TextMessageStartEvent
from nat.builder.context import Context
from nat.data_models.api_server import ChatRequest
from nat.data_models.api_server import ChatResponse
from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.intermediate_step import IntermediateStepType
from nat.utils.type_utils import StrPath
from openai.types.chat import CompletionCreateParams
from ragas import MultiTurnSample
from ragas.messages import AIMessage
from ragas.messages import HumanMessage
from ragas.messages import ToolMessage

from datarobot_genai.core.agents.base import BaseAgent
from datarobot_genai.core.agents.base import InvokeReturn
from datarobot_genai.core.agents.base import UsageMetrics
from datarobot_genai.core.agents.base import extract_user_prompt_content
from datarobot_genai.core.agents.base import is_streaming
from datarobot_genai.core.mcp.common import MCPConfig
from datarobot_genai.nat.helpers import load_workflow

logger = logging.getLogger(__name__)


def convert_to_ragas_messages(
    steps: list[IntermediateStep],
) -> list[HumanMessage | AIMessage | ToolMessage]:
    def _to_ragas(step: IntermediateStep) -> HumanMessage | AIMessage | ToolMessage:
        if step.event_type == IntermediateStepType.LLM_START:
            return HumanMessage(content=_parse(step.data.input))
        elif step.event_type == IntermediateStepType.LLM_END:
            return AIMessage(content=_parse(step.data.output))
        else:
            raise ValueError(f"Unknown event type {step.event_type}")

    def _include_step(step: IntermediateStep) -> bool:
        return step.event_type in {
            IntermediateStepType.LLM_END,
            IntermediateStepType.LLM_START,
        }

    def _parse(messages: Any) -> str:
        if isinstance(messages, list):
            last_message = messages[-1]
        else:
            last_message = messages

        if isinstance(last_message, dict):
            content = last_message.get("content") or last_message
        elif hasattr(last_message, "content"):
            content = getattr(last_message, "content") or last_message
        else:
            content = last_message
        return str(content)

    return [_to_ragas(step) for step in steps if _include_step(step)]


def create_pipeline_interactions_from_steps(
    steps: list[IntermediateStep],
) -> MultiTurnSample | None:
    if not steps:
        return None
    ragas_trace = convert_to_ragas_messages(steps)
    return MultiTurnSample(user_input=ragas_trace)


def pull_intermediate_structured() -> asyncio.Future[list[IntermediateStep]]:
    """
    Subscribe to the runner's event stream using callbacks.
    Intermediate steps are collected and, when complete, the future is set
    with the list of dumped intermediate steps.
    """
    future: asyncio.Future[list[IntermediateStep]] = asyncio.Future()
    intermediate_steps = []  # We'll store the dumped steps here.
    context = Context.get()

    def on_next_cb(item: IntermediateStep) -> None:
        # Append each new intermediate step to the list.
        intermediate_steps.append(item)

    def on_error_cb(exc: Exception) -> None:
        logger.error("Hit on_error: %s", exc)
        if not future.done():
            future.set_exception(exc)

    def on_complete_cb() -> None:
        logger.debug("Completed reading intermediate steps")
        if not future.done():
            future.set_result(intermediate_steps)

    # Subscribe with our callbacks.
    context.intermediate_step_manager.subscribe(
        on_next=on_next_cb, on_error=on_error_cb, on_complete=on_complete_cb
    )

    return future


async def stream_intermediate_steps() -> AsyncGenerator[IntermediateStep, None]:
    """
    Stream intermediate steps as they arrive from the NAT workflow.
    Uses an async queue to bridge the callback-based subscription to an async generator.
    """
    queue: asyncio.Queue[IntermediateStep | Exception | None] = asyncio.Queue()
    context = Context.get()
    completed = False

    def on_next_cb(item: IntermediateStep) -> None:
        if not completed:
            queue.put_nowait(item)

    def on_error_cb(exc: Exception) -> None:
        logger.error("Hit on_error in stream_intermediate_steps: %s", exc)
        if not completed:
            queue.put_nowait(exc)

    def on_complete_cb() -> None:
        logger.debug("Completed reading intermediate steps stream")
        nonlocal completed
        completed = True
        queue.put_nowait(None)  # Sentinel to signal completion

    # Subscribe with our callbacks.
    context.intermediate_step_manager.subscribe(
        on_next=on_next_cb, on_error=on_error_cb, on_complete=on_complete_cb
    )

    # Yield steps as they arrive
    while True:
        item = await queue.get()
        if item is None:  # Sentinel indicating completion
            break
        if isinstance(item, Exception):
            raise item
        yield item


def convert_intermediate_step_to_event(
    step: IntermediateStep, message_id: str | None = None
) -> Event | None:
    """
    Convert a NAT IntermediateStep to an ag_ui.core Event.

    Args:
        step: The intermediate step from NAT
        message_id: Optional message ID to use for text message events

    Returns
    -------
        An Event object or None if the step type doesn't map to an event
    """
    if step.event_type == IntermediateStepType.LLM_START:
        # Generate a message ID if not provided
        msg_id = message_id or str(uuid.uuid4())
        return TextMessageStartEvent(
            type=EventType.TEXT_MESSAGE_START,
            message_id=msg_id,
            role="assistant",
        )
    elif step.event_type == IntermediateStepType.LLM_NEW_TOKEN:
        # For new tokens, extract the delta content if available
        # Note: LLM_NEW_TOKEN may not always have data, as the actual content
        # comes from result_stream. We emit the event to signal token arrival.
        msg_id = message_id or str(uuid.uuid4())
        delta = ""
        if step.data:
            if hasattr(step.data, "output") and step.data.output:
                delta = str(step.data.output)
            elif hasattr(step.data, "content") and step.data.content:
                delta = str(step.data.content)
            elif hasattr(step.data, "delta") and step.data.delta:
                delta = str(step.data.delta)
            elif hasattr(step.data, "chunk") and step.data.chunk:
                delta = str(step.data.chunk)
        # Emit event even if delta is empty to signal token arrival
        return TextMessageChunkEvent(
            type=EventType.TEXT_MESSAGE_CHUNK,
            message_id=msg_id,
            delta=delta,
        )
    elif step.event_type == IntermediateStepType.LLM_END:
        msg_id = message_id or str(uuid.uuid4())
        return TextMessageEndEvent(
            type=EventType.TEXT_MESSAGE_END,
            message_id=msg_id,
        )
    # Other event types (tool calls, etc.) can be added here as needed
    return None


class NatAgent(BaseAgent[None]):
    def __init__(
        self,
        *,
        workflow_path: StrPath,
        api_key: str | None = None,
        api_base: str | None = None,
        model: str | None = None,
        verbose: bool | str | None = True,
        timeout: int | None = 90,
        authorization_context: dict[str, Any] | None = None,
        forwarded_headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            api_key=api_key,
            api_base=api_base,
            model=model,
            verbose=verbose,
            timeout=timeout,
            authorization_context=authorization_context,
            forwarded_headers=forwarded_headers,
            **kwargs,
        )
        self.workflow_path = workflow_path

    def make_chat_request(self, completion_create_params: CompletionCreateParams) -> ChatRequest:
        user_prompt_content = str(extract_user_prompt_content(completion_create_params))
        return ChatRequest.from_string(user_prompt_content)

    async def invoke(self, completion_create_params: CompletionCreateParams) -> InvokeReturn:
        """Run the agent with the provided completion parameters.

        [THIS METHOD IS REQUIRED FOR THE AGENT TO WORK WITH DRUM SERVER]

        Args:
            completion_create_params: The completion request parameters including input topic
            and settings.

        Returns
        -------
            For streaming requests, returns a generator yielding tuples of (response_text,
            pipeline_interactions, usage_metrics).
            For non-streaming requests, returns a single tuple of (response_text,
            pipeline_interactions, usage_metrics).

        """
        # Retrieve the starting chat request from the CompletionCreateParams
        chat_request = self.make_chat_request(completion_create_params)

        # Print commands may need flush=True to ensure they are displayed in real-time.
        print("Running agent with user prompt:", chat_request.messages[0].content, flush=True)

        mcp_config = MCPConfig(
            authorization_context=self.authorization_context,
            forwarded_headers=self.forwarded_headers,
        )
        server_config = mcp_config.server_config
        headers = server_config["headers"] if server_config else None

        if is_streaming(completion_create_params):

            async def stream_generator() -> AsyncGenerator[
                tuple[str | Event, MultiTurnSample | None, UsageMetrics], None
            ]:
                usage_metrics: UsageMetrics = {
                    "completion_tokens": 0,
                    "prompt_tokens": 0,
                    "total_tokens": 0,
                }
                steps: list[IntermediateStep] = []
                message_id_map: dict[str, str] = {}  # Maps function_id to message_id
                # Queue to collect events from intermediate steps
                event_queue: asyncio.Queue[Event | None] = asyncio.Queue()

                async def process_intermediate_steps() -> None:
                    """Process intermediate steps and put events in the queue."""
                    intermediate_stream = stream_intermediate_steps()
                    try:
                        async for step in intermediate_stream:
                            steps.append(step)

                            # Get or create message_id for this function
                            function_id = (
                                step.function_ancestry.function_id
                                if step.function_ancestry
                                else None
                            )
                            if function_id and function_id not in message_id_map:
                                message_id_map[function_id] = str(uuid.uuid4())

                            message_id = message_id_map.get(function_id) if function_id else None

                            # Convert step to event
                            event = convert_intermediate_step_to_event(step, message_id)
                            if event:
                                await event_queue.put(event)
                    finally:
                        # Signal completion
                        await event_queue.put(None)

                async with load_workflow(self.workflow_path, headers=headers) as workflow:
                    async with workflow.run(chat_request) as runner:
                        # Start processing intermediate steps in background
                        intermediate_task = asyncio.create_task(process_intermediate_steps())

                        # Stream result chunks (text content) and events concurrently
                        result_stream_done = False
                        try:
                            async for result in runner.result_stream():
                                # Yield any available events first
                                while True:
                                    try:
                                        event = event_queue.get_nowait()
                                        if event is None:
                                            # Stream completed, but continue processing results
                                            result_stream_done = True
                                            break
                                        if hasattr(event, "delta"):
                                            yield (event.delta, None, usage_metrics)
                                    except asyncio.QueueEmpty:
                                        break

                                if result_stream_done:
                                    break

                                if isinstance(result, ChatResponse):
                                    result_text = result.choices[0].message.content
                                else:
                                    result_text = str(result)

                                if result_text:
                                    yield (result_text, None, usage_metrics)
                        except Exception:
                            # If result stream fails, continue to process events
                            pass

                        # Wait for intermediate steps processing to complete
                        await intermediate_task

                        # Drain any remaining events
                        while True:
                            try:
                                event = event_queue.get_nowait()
                                if event is None:
                                    break
                                yield (event, None, usage_metrics)
                            except asyncio.QueueEmpty:
                                break

                        # Calculate final usage metrics from all LLM_END steps (for accuracy)
                        llm_end_steps = [
                            step
                            for step in steps
                            if step.event_type == IntermediateStepType.LLM_END
                        ]
                        final_usage_metrics: UsageMetrics = {
                            "completion_tokens": 0,
                            "prompt_tokens": 0,
                            "total_tokens": 0,
                        }
                        for step in llm_end_steps:
                            if step.usage_info:
                                token_usage = step.usage_info.token_usage
                                final_usage_metrics["total_tokens"] += token_usage.total_tokens
                                final_usage_metrics["prompt_tokens"] += token_usage.prompt_tokens
                                final_usage_metrics["completion_tokens"] += (
                                    token_usage.completion_tokens
                                )

                        pipeline_interactions = create_pipeline_interactions_from_steps(steps)
                        yield "", pipeline_interactions, final_usage_metrics

            return stream_generator()

        # Create and invoke the NAT (Nemo Agent Toolkit) Agentic Workflow with the inputs
        result, steps = await self.run_nat_workflow(self.workflow_path, chat_request, headers)

        llm_end_steps = [step for step in steps if step.event_type == IntermediateStepType.LLM_END]
        usage_metrics: UsageMetrics = {
            "completion_tokens": 0,
            "prompt_tokens": 0,
            "total_tokens": 0,
        }
        for step in llm_end_steps:
            if step.usage_info:
                token_usage = step.usage_info.token_usage
                usage_metrics["total_tokens"] += token_usage.total_tokens
                usage_metrics["prompt_tokens"] += token_usage.prompt_tokens
                usage_metrics["completion_tokens"] += token_usage.completion_tokens

        if isinstance(result, ChatResponse):
            result_text = result.choices[0].message.content
        else:
            result_text = str(result)
        pipeline_interactions = create_pipeline_interactions_from_steps(steps)

        return result_text, pipeline_interactions, usage_metrics

    async def run_nat_workflow(
        self, workflow_path: StrPath, chat_request: ChatRequest, headers: dict[str, str] | None
    ) -> tuple[ChatResponse | str, list[IntermediateStep]]:
        """Run the NAT workflow with the provided config file and input string.

        Args:
            workflow_path: Path to the NAT workflow configuration file
            input_str: Input string to process through the workflow

        Returns
        -------
            ChatResponse | str: The result from the NAT workflow
            list[IntermediateStep]: The list of intermediate steps
        """
        async with load_workflow(workflow_path, headers=headers) as workflow:
            async with workflow.run(chat_request) as runner:
                intermediate_future = pull_intermediate_structured()
                runner_outputs = await runner.result()
                steps = await intermediate_future

        line = f"{'-' * 50}"
        prefix = f"{line}\nWorkflow Result:\n"
        suffix = f"\n{line}"

        print(f"{prefix}{runner_outputs}{suffix}")

        return runner_outputs, steps
