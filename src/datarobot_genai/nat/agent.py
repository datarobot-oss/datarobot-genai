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

import asyncio
import logging
import uuid
from typing import TYPE_CHECKING
from typing import Any

from ag_ui.core import RunAgentInput
from ag_ui.core import RunFinishedEvent
from ag_ui.core import RunStartedEvent
from ag_ui.core import TextMessageContentEvent
from ag_ui.core import TextMessageEndEvent
from ag_ui.core import TextMessageStartEvent
from nat.builder.context import Context
from nat.data_models.api_server import ChatRequest
from nat.data_models.api_server import ChatResponse
from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.intermediate_step import IntermediateStepType
from nat.utils.type_utils import StrPath

from datarobot_genai.core.agents.base import BaseAgent
from datarobot_genai.core.agents.base import InvokeReturn
from datarobot_genai.core.agents.base import UsageMetrics
from datarobot_genai.core.agents.base import extract_user_prompt_content
from datarobot_genai.core.mcp.common import MCPConfig
from datarobot_genai.nat.helpers import load_workflow

if TYPE_CHECKING:
    from ragas import MultiTurnSample
    from ragas.messages import AIMessage
    from ragas.messages import HumanMessage
    from ragas.messages import ToolMessage

logger = logging.getLogger(__name__)


def convert_to_ragas_messages(
    steps: list[IntermediateStep],
) -> list[HumanMessage | AIMessage | ToolMessage]:
    # Lazy import to reduce memory overhead when ragas is not used
    from ragas.messages import AIMessage
    from ragas.messages import HumanMessage

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

    def make_user_prompt(self, run_agent_input: RunAgentInput) -> str:
        """Create the user prompt text. Override to customize formatting.

        Chat history is automatically appended by `invoke` when
        max_history_messages > 0 (controlled via DATAROBOT_GENAI_MAX_HISTORY_MESSAGES env var).

        Default implementation returns the raw user message content.
        """
        user_prompt_content = extract_user_prompt_content(run_agent_input)
        return str(user_prompt_content)

    def make_chat_request(self, user_prompt: str) -> ChatRequest:
        """Create a NAT ChatRequest from the processed user prompt."""
        return ChatRequest.from_string(user_prompt)

    async def invoke(self, run_agent_input: RunAgentInput) -> InvokeReturn:
        """Run the agent with the provided input.

        Args:
            run_agent_input: The agent run input including messages, tools, and context.

        Returns
        -------
            Returns a generator yielding tuples of (event, pipeline_interactions, usage_metrics).

        """
        # Build the user prompt from the template
        user_prompt = self.make_user_prompt(run_agent_input)

        # Automatically inject chat history when enabled (max_history_messages > 0)
        history_summary = self.build_history_summary(run_agent_input)
        if history_summary:
            user_prompt = f"{user_prompt}\n\nPrior conversation:\n{history_summary}"

        # Create the chat request from the processed prompt
        chat_request = self.make_chat_request(user_prompt)

        # Print commands may need flush=True to ensure they are displayed in real-time.
        print("Running agent with user prompt:", chat_request.messages[0].content, flush=True)

        mcp_config = MCPConfig(
            authorization_context=self.authorization_context,
            forwarded_headers=self.forwarded_headers,
        )
        server_config = mcp_config.server_config
        headers = server_config["headers"] if server_config else None

        thread_id = run_agent_input.thread_id
        run_id = run_agent_input.run_id
        zero_metrics: UsageMetrics = {
            "completion_tokens": 0,
            "prompt_tokens": 0,
            "total_tokens": 0,
        }

        # Partial AG-UI: workflow lifecycle + text message events
        yield RunStartedEvent(thread_id=thread_id, run_id=run_id), None, zero_metrics

        message_id = str(uuid.uuid4())
        text_started = False

        async with load_workflow(self.workflow_path, headers=headers) as workflow:
            async with workflow.run(chat_request) as runner:
                intermediate_future = pull_intermediate_structured()
                async for result in runner.result_stream():
                    if isinstance(result, ChatResponse):
                        result_text = result.choices[0].message.content
                    else:
                        result_text = str(result)

                    if result_text:
                        if not text_started:
                            yield (
                                TextMessageStartEvent(message_id=message_id),
                                None,
                                zero_metrics,
                            )
                            text_started = True
                        yield (
                            TextMessageContentEvent(message_id=message_id, delta=result_text),
                            None,
                            zero_metrics,
                        )

                if text_started:
                    yield TextMessageEndEvent(message_id=message_id), None, zero_metrics

                steps = await intermediate_future
                llm_end_steps = [
                    step for step in steps if step.event_type == IntermediateStepType.LLM_END
                ]
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

                pipeline_interactions = self.create_pipeline_interactions_from_steps(steps)
                yield (
                    RunFinishedEvent(thread_id=thread_id, run_id=run_id),
                    pipeline_interactions,
                    usage_metrics,
                )

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

    @classmethod
    def create_pipeline_interactions_from_steps(
        cls,
        steps: list[IntermediateStep],
    ) -> MultiTurnSample | None:
        if not steps:
            return None
        # Lazy import to reduce memory overhead when ragas is not used
        from ragas import MultiTurnSample

        ragas_trace = convert_to_ragas_messages(steps)
        return MultiTurnSample(user_input=ragas_trace)
