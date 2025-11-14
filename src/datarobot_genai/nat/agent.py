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
import logging
from typing import Any

from nat.data_models.api_server import ChatRequest
from nat.data_models.api_server import ChatResponse
from nat.data_models.intermediate_step import IntermediateStepType
from nat.eval.runtime_event_subscriber import pull_intermediate
from nat.runtime.loader import load_workflow
from nat.utils.type_utils import StrPath
from openai.types.chat import CompletionCreateParams

from datarobot_genai.core.agents.base import BaseAgent
from datarobot_genai.core.agents.base import InvokeReturn
from datarobot_genai.core.agents.base import UsageMetrics
from datarobot_genai.core.agents.base import extract_user_prompt_content

logger = logging.getLogger(__name__)


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
        **kwargs: Any,
    ) -> None:
        super().__init__(
            api_key=api_key,
            api_base=api_base,
            model=model,
            verbose=verbose,
            timeout=timeout,
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

        # Create and invoke the NAT (Nemo Agent Toolkit) Agentic Workflow with the inputs
        result, steps = await self.run_nat_workflow(self.workflow_path, chat_request)

        llm_end_steps = [
            step for step in steps if step["payload"]["event_type"] == IntermediateStepType.LLM_END
        ]
        usage_metrics: UsageMetrics = {
            "completion_tokens": 0,
            "prompt_tokens": 0,
            "total_tokens": 0,
        }
        for step in llm_end_steps:
            if step["payload"]["usage_info"]:
                token_usage = step["payload"]["usage_info"]["token_usage"]
                usage_metrics["total_tokens"] += token_usage["total_tokens"]
                usage_metrics["prompt_tokens"] += token_usage["prompt_tokens"]
                usage_metrics["completion_tokens"] += token_usage["completion_tokens"]

        # Create a list of events from the event listener
        events: list[Any]
        events = []  # This should be populated with the agent's events/messages

        if isinstance(result, ChatResponse):
            result_text = result.choices[0].message.content
        else:
            result_text = str(result)
        pipeline_interactions = self.create_pipeline_interactions_from_events(events)

        return result_text, pipeline_interactions, usage_metrics

    async def run_nat_workflow(
        self, workflow_path: StrPath, chat_request: ChatRequest
    ) -> tuple[ChatResponse | str, list[dict]]:
        """Run the NAT workflow with the provided config file and input string.

        Args:
            workflow_path: Path to the NAT workflow configuration file
            input_str: Input string to process through the workflow

        Returns
        -------
            str: The result from the NAT workflow
        """
        async with load_workflow(workflow_path) as workflow:
            async with workflow.run(chat_request) as runner:
                intermediate_future = pull_intermediate()
                runner_outputs = await runner.result()
                steps = await intermediate_future

        line = f"{'-' * 50}"
        prefix = f"{line}\nWorkflow Result:\n"
        suffix = f"\n{line}"

        print(f"{prefix}{runner_outputs}{suffix}")

        return runner_outputs, steps
