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

from datarobot_genai.dragent.response import DRAgentEventResponse
from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.cli.register_workflow import register_function
from nat.data_models.agent import AgentBaseConfig
from nat.data_models.api_server import GlobalTypeConverter


def _convert_event_response_to_str(response: DRAgentEventResponse) -> str:
    return response.get_delta()


GlobalTypeConverter.register_converter(_convert_event_response_to_str)


class CrewaiAgentConfig(AgentBaseConfig, name="crewai_agent"):
    """NAT config for the CrewAI agent.

    Extends AgentBaseConfig which provides: llm_name, description, verbose.
    The LLM is managed by NAT and accessed via builder.get_llm().
    """


# Workaround: NAT's profiler auto-detects frameworks by scanning source for
# LLMFrameworkEnum member names (e.g. \bCREWAI\b). When detected, it tries to
# import nat.plugins.crewai which doesn't exist yet (no try/except guard unlike
# ADK/Strands/AutoGen). Constructing from the string value avoids the regex match.
_crewai_wrapper = LLMFrameworkEnum("crewai")


@register_function(
    config_type=CrewaiAgentConfig,
)
async def crewai_agent(config: CrewaiAgentConfig, builder: Builder) -> AsyncGenerator:
    from ag_ui.core import RunAgentInput
    from datarobot_genai.dragent.response import DRAgentEventResponse
    from nat.builder.function_info import FunctionInfo

    from dragent.crewai.myagent import MyAgent

    llm = await builder.get_llm(config.llm_name, wrapper_type=_crewai_wrapper)

    agent = MyAgent(llm=llm)

    async def _response_fn(input_message: RunAgentInput) -> DRAgentEventResponse:
        """Invoke the CrewAI agent and return a DRAgentEventResponse."""
        response_text = ""
        metrics: dict = {}
        async for text, _, iteration_metrics in agent.invoke(input_message):
            metrics = iteration_metrics
            if isinstance(text, str):
                response_text += text

        return DRAgentEventResponse(
            delta=response_text if response_text else None,
            usage_metrics=metrics,
        )

    yield FunctionInfo.from_fn(
        _response_fn,
        description=config.description,
    )
