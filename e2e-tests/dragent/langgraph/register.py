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

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.cli.register_workflow import register_function
from nat.data_models.agent import AgentBaseConfig


class LanggraphAgentConfig(AgentBaseConfig, name="langgraph_agent"):
    """NAT config for the LangGraph agent.

    Extends AgentBaseConfig which provides: llm_name, description, verbose.
    The LLM is managed by NAT and accessed via builder.get_llm().
    """


@register_function(
    config_type=LanggraphAgentConfig,
    framework_wrappers=[LLMFrameworkEnum.LANGCHAIN],
)
async def langgraph_agent(config: LanggraphAgentConfig, builder: Builder) -> AsyncGenerator:
    from ag_ui.core import Event  # noqa: PLC0415
    from ag_ui.core import RunAgentInput  # noqa: PLC0415
    from datarobot_genai.core.agents import default_usage_metrics  # noqa: PLC0415
    from datarobot_genai.dragent.response import DRAgentEventResponse  # noqa: PLC0415
    from nat.builder.function_info import FunctionInfo  # noqa: PLC0415

    from dragent.langgraph.myagent import MyAgent  # noqa: PLC0415

    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    agent = MyAgent(llm=llm)

    async def _response_fn(input_message: RunAgentInput) -> DRAgentEventResponse:
        """Invoke the LangGraph agent and return a DRAgentEventResponse."""
        events: list[Event] = []
        metrics = default_usage_metrics()
        async for event, _, iteration_metrics in agent.invoke(input_message):
            metrics = iteration_metrics
            events.append(event)

        return DRAgentEventResponse(
            events=events,
            usage_metrics=metrics,
        )

    yield FunctionInfo.from_fn(
        _response_fn,
        description=config.description,
    )
