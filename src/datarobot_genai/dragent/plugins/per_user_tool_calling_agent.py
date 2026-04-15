# Copyright 2026 DataRobot, Inc. and its affiliates.
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

"""Per-user variant of the NAT tool_calling_agent workflow.

The built-in ``tool_calling_agent`` is registered as a *shared* workflow, which
means NAT's dependency validator forbids it from referencing per-user function
groups such as ``a2a_client``.  This module registers an identical workflow under
the name ``per_user_tool_calling_agent`` using ``register_per_user_function`` so
that per-user function groups can be used while still benefiting from OpenAI-style
structured tool calling (``bind_tools``).

NAT 1.6 added a ``stream_fn`` that yields ``ChatResponseChunk``.  We wrap it
using ``DRAgentNestedReasoningStepAdaptor.process_chunks()`` to produce
``DRAgentEventResponse`` with valid AG-UI event sequences.
"""

from collections.abc import AsyncGenerator
from typing import Any

from nat.builder.framework_enum import LLMFrameworkEnum
from nat.cli.register_workflow import register_per_user_function
from nat.data_models.api_server import ChatRequest
from nat.data_models.api_server import ChatRequestOrMessage
from nat.data_models.api_server import ChatResponse
from nat.data_models.step_adaptor import StepAdaptorConfig
from nat.plugins.langchain.agent.tool_calling_agent.register import ToolCallAgentWorkflowConfig
from nat.plugins.langchain.agent.tool_calling_agent.register import tool_calling_agent_workflow

from datarobot_genai.dragent.frontends.response import DRAgentEventResponse
from datarobot_genai.dragent.frontends.step_adaptor import DRAgentNestedReasoningStepAdaptor


class PerUserToolCallAgentWorkflowConfig(
    ToolCallAgentWorkflowConfig,
    name="per_user_tool_calling_agent",  # type: ignore[call-arg]
):
    """Per-user version of tool_calling_agent."""

    pass


async def _per_user_tool_calling_agent(
    config: PerUserToolCallAgentWorkflowConfig, builder: Any
) -> AsyncGenerator[Any, None]:
    """Wrap the original tool_calling_agent with AG-UI stream conversion."""
    from nat.builder.function_info import FunctionInfo  # noqa: PLC0415

    original_gen = tool_calling_agent_workflow.__wrapped__(config, builder)
    try:
        fn_info: FunctionInfo = await original_gen.__anext__()

        if fn_info.stream_fn is None:
            yield fn_info
            return

        original_stream_fn = fn_info.stream_fn

        async def wrapped_stream(
            chat_request_or_message: ChatRequestOrMessage,
        ) -> AsyncGenerator[DRAgentEventResponse, None]:
            adaptor = DRAgentNestedReasoningStepAdaptor(StepAdaptorConfig())
            async for event in adaptor.process_chunks(original_stream_fn(chat_request_or_message)):
                yield event

        yield FunctionInfo.create(
            single_fn=fn_info.single_fn,
            stream_fn=wrapped_stream,
            description=fn_info.description,
        )
    finally:
        await original_gen.aclose()


register_per_user_function(
    config_type=PerUserToolCallAgentWorkflowConfig,
    input_type=ChatRequest,
    single_output_type=ChatResponse,
    framework_wrappers=[LLMFrameworkEnum.LANGCHAIN],
)(_per_user_tool_calling_agent)
