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
from collections.abc import Mapping
from typing import Any
from typing import cast
from uuid import uuid4

from ag_ui.core import AssistantMessage
from ag_ui.core import Message
from ag_ui.core import RunAgentInput
from ag_ui.core import RunFinishedEvent
from ag_ui.core import SystemMessage
from ag_ui.core import TextMessageChunkEvent
from ag_ui.core import TextMessageContentEvent
from ag_ui.core import TextMessageStartEvent
from ag_ui.core import Tool
from ag_ui.core import ToolMessage
from ag_ui.core import UserMessage
from openai.types.chat import CompletionCreateParams
from ragas import MultiTurnSample

from datarobot_genai.core.agents import InvokeReturn
from datarobot_genai.core.agents import default_usage_metrics
from datarobot_genai.core.agents.base import BaseAgent
from datarobot_genai.core.agents.base import UsageMetrics

logger = logging.getLogger(__name__)


def is_streaming(completion_create_params: CompletionCreateParams | Mapping[str, Any]) -> bool:
    """Return True when the request asks for streaming, False otherwise.

    Accepts both pydantic types and plain dictionaries.
    """
    params = cast(Mapping[str, Any], completion_create_params)
    value = params.get("stream", False)
    # Handle non-bool truthy values defensively (e.g., "true")
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


def convert_chat_completion_params_to_run_agent_input(
    chat_completion_params: CompletionCreateParams | Mapping[str, Any],
) -> RunAgentInput:
    """Convert a chat completion parameters to a run agent input."""
    tools = [
        Tool(
            name=tool.get("function").get("name"),
            description=tool.get("function").get("description"),
            parameters=tool.get("function").get("parameters"),
        )
        for tool in chat_completion_params.get("tools", []) or []
        if tool.get("type") == "function"  # type: ignore[union-attr]
    ]
    messages: list[Message] = []
    for i, message in enumerate(chat_completion_params.get("messages", [])):  # type: ignore[arg-type]
        id = f"message_{i}"
        if message.get("role") == "user":
            messages.append(UserMessage(id=id, content=message.get("content")))
        elif message.get("role") == "assistant":
            messages.append(AssistantMessage(id=id, content=message.get("content")))
        elif message.get("role") == "tool":
            messages.append(
                ToolMessage(
                    id=id,
                    content=message.get("content"),
                    tool_call_id=message.get("tool_call_id"),
                    error=message.get("error"),
                )
            )
        elif message.get("role") == "system":
            messages.append(SystemMessage(id=id, content=message.get("content")))

    forwarded_props: dict[str, Any] = {
        "model": chat_completion_params.get("model"),
        "authorization_context": chat_completion_params.get("authorization_context"),
        "forwarded_headers": chat_completion_params.get("forwarded_headers"),
    }

    return RunAgentInput(
        messages=messages,
        tools=tools,
        forwarded_props=forwarded_props,
        thread_id=str(uuid4()),
        run_id=str(uuid4()),
        state={},
        context=[],
    )


async def agent_chat_completion_wrapper(
    agent: BaseAgent, chat_completion_params: CompletionCreateParams | Mapping[str, Any]
) -> InvokeReturn | tuple[str, MultiTurnSample | None, UsageMetrics]:
    """Wrap the agent's invoke method in a chat completion wrapper.

    Returns
    -------
    InvokeReturn
        When streaming is requested — the raw async event generator.
    tuple[str, MultiTurnSample | None, UsageMetrics]
        When non-streaming — the reassembled final text, pipeline
        interactions, and accumulated usage metrics.
    """
    run_agent_input = convert_chat_completion_params_to_run_agent_input(chat_completion_params)

    if is_streaming(chat_completion_params):
        return agent.invoke(run_agent_input)
    else:
        final_response = ""
        pipeline_interactions = None
        usage_metrics = default_usage_metrics()
        received_run_finished = False

        async for event, iter_interactions, iter_metrics in agent.invoke(run_agent_input):
            # When we work in non-streaming mode, we only send back the final message.
            # It is because of limitation of completions interface we can not send back the
            # intermediate messages.
            if isinstance(event, TextMessageStartEvent):
                final_response = ""
            elif isinstance(event, (TextMessageContentEvent, TextMessageChunkEvent)):
                final_response += event.delta
            elif isinstance(event, RunFinishedEvent):
                received_run_finished = True
                pipeline_interactions = iter_interactions
                usage_metrics = iter_metrics

        if not received_run_finished:
            logger.warning("Agent stream ended without RunFinishedEvent")

        return final_response, pipeline_interactions, usage_metrics
