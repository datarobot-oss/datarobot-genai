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
from datarobot_genai.dragent.frontends.response import DRAgentEventResponse
from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.cli.register_workflow import register_per_user_function
from nat.data_models.agent import AgentBaseConfig


class LanggraphHitlAgentConfig(AgentBaseConfig, name="langgraph_hitl_agent"):
    """NAT config for the LangGraph interrupt/resume E2E agent."""


@register_per_user_function(
    config_type=LanggraphHitlAgentConfig,
    input_type=RunAgentInput,  # noqa: F821
    single_output_type=DRAgentEventResponse,
    streaming_output_type=DRAgentEventResponse,
    framework_wrappers=[LLMFrameworkEnum.LANGCHAIN],
)
async def langgraph_hitl_agent(
    config: LanggraphHitlAgentConfig,
    builder: Builder,
) -> AsyncGenerator:
    from datarobot_genai.core.mcp import MCPConfig
    from datarobot_genai.dragent.frontends.converters import aggregate_dragent_event_responses
    from datarobot_genai.langgraph.mcp import mcp_tools_context
    from datarobot_genai.nat.helpers import extract_authorization_from_context
    from datarobot_genai.nat.helpers import extract_datarobot_headers_from_context
    from nat.builder.function_info import FunctionInfo
    from nat.data_models.streaming import Streaming

    from dragent.langgraph_hitl.myagent import HitlMyAgent

    async def _response_fn(
        input_message: RunAgentInput,
    ) -> Annotated[
        AsyncGenerator[DRAgentEventResponse, None],
        Streaming(convert=aggregate_dragent_event_responses),
    ]:
        llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

        forwarded_headers = extract_datarobot_headers_from_context()
        authorization_context = extract_authorization_from_context()
        mcp_config = MCPConfig(
            forwarded_headers=forwarded_headers, authorization_context=authorization_context
        )
        async with mcp_tools_context(mcp_config) as tools:
            agent = HitlMyAgent(
                llm=llm,
                forwarded_headers=forwarded_headers,
                tools=tools,
                verbose=config.verbose,
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
