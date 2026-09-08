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

from collections.abc import AsyncGenerator
from typing import Annotated

from ag_ui.core import RunAgentInput
from datarobot_genai.core.telemetry.agent import instrument
from datarobot_genai.dragent.frontends.response import DRAgentEventResponse
from datarobot_genai.langgraph.telemetry import instrument as langgraph_instrument
from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.cli.register_workflow import register_per_user_function
from nat.data_models.agent import AgentBaseConfig
from nat.data_models.component_ref import FunctionGroupRef
from nat.data_models.component_ref import FunctionRef
from pydantic import Field

# INSTRUMENTATION CALL IS REQUIRED TO SETUP TRACING AND TELEMETRY FOR AGENTS
instrument()
langgraph_instrument()


class LanggraphAgentConfig(AgentBaseConfig, name="langgraph_agent"):
    """NAT config for the LangGraph agent.

    Extends AgentBaseConfig which provides: llm_name, description, verbose.
    The LLM is managed by NAT and accessed via builder.get_llm().
    """

    tool_names: list[FunctionRef | FunctionGroupRef] = Field(
        default_factory=list,
        description=(
            "Tools and function groups to give the agent, by name. An MCP server is a "
            "`function_groups` entry, so listing its name here attaches every tool it "
            "exposes. NAT builds those groups once and keeps the connections open."
        ),
    )


@register_per_user_function(
    config_type=LanggraphAgentConfig,
    input_type=RunAgentInput,  # noqa: F821
    single_output_type=DRAgentEventResponse,
    streaming_output_type=DRAgentEventResponse,
    framework_wrappers=[LLMFrameworkEnum.LANGCHAIN],
)
async def langgraph_agent(config: LanggraphAgentConfig, builder: Builder) -> AsyncGenerator:
    from datarobot_genai.dragent.context import extract_datarobot_headers_from_context
    from datarobot_genai.dragent.frontends.converters import aggregate_dragent_event_responses
    from nat.builder.function_info import FunctionInfo
    from nat.data_models.streaming import Streaming

    from dragent.langgraph.myagent import HITL_E2E_CHECKPOINTER
    from dragent.langgraph.myagent import MyAgent

    # Built ONCE, here, not per request. NAT owns the MCP connections: each
    # `function_groups` entry named in `tool_names` is built at workflow build and its
    # client stays open, while `DataRobotAuthAdapter` recomputes the credentials on
    # every HTTP request from the request context. Connecting per prompt instead --
    # which is what an AsyncExitStack inside the response function does -- pays a
    # connect and a tool-discovery round trip per server per prompt, and buys nothing,
    # because the headers were the only per-request part.
    tools = await builder.get_tools(
        tool_names=config.tool_names, wrapper_type=LLMFrameworkEnum.LANGCHAIN
    )

    async def _response_fn(
        input_message: RunAgentInput,
    ) -> Annotated[
        AsyncGenerator[DRAgentEventResponse, None],
        # Streaming tells NAT how to go from a list of streaming events to a single response
        # object for non-streaming routes.
        Streaming(convert=aggregate_dragent_event_responses),
    ]:
        # LLM might contain user-specific headers
        llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

        # Agent contains user-specific headers
        forwarded_headers = extract_datarobot_headers_from_context()

        agent = MyAgent(
            llm=llm,
            forwarded_headers=forwarded_headers,
            tools=tools,
            verbose=config.verbose,
            checkpointer=HITL_E2E_CHECKPOINTER,
        )

        async for event, pipeline_interactions, usage_metrics in agent.invoke(input_message):
            yield DRAgentEventResponse(
                events=[event],
                usage_metrics=usage_metrics,
                pipeline_interactions=pipeline_interactions,
            )

    yield FunctionInfo.from_fn(
        _response_fn,
        description=config.description,
    )
