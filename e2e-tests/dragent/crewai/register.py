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

from ag_ui.core.types import RunAgentInput
from datarobot_genai.dragent.frontends.converters import aggregate_dragent_event_responses
from datarobot_genai.dragent.frontends.response import DRAgentEventResponse
from datarobot_genai.nat.helpers import extract_authorization_from_context
from datarobot_genai.nat.helpers import extract_datarobot_headers_from_context
from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import Streaming
from nat.cli.register_workflow import register_per_user_function
from nat.data_models.agent import AgentBaseConfig


class CrewaiAgentConfig(AgentBaseConfig, name="crewai_agent"):
    """NAT config for the CrewAI agent.

    Extends AgentBaseConfig which provides: llm_name, description, verbose.
    The LLM is managed by NAT and accessed via builder.get_llm().
    """


@register_per_user_function(
    config_type=CrewaiAgentConfig,
    input_type=RunAgentInput,  # noqa: F821
    single_output_type=DRAgentEventResponse,  # noqa: F821
    framework_wrappers=[LLMFrameworkEnum.CREWAI],
)
async def crewai_agent(config: CrewaiAgentConfig, builder: Builder) -> AsyncGenerator:
    from nat.builder.function_info import FunctionInfo  # noqa: PLC0415

    from dragent.crewai.myagent import MyAgent  # noqa: PLC0415

    async def _response_fn(
        input_message: RunAgentInput,
    ) -> Annotated[
        AsyncGenerator[DRAgentEventResponse, None],
        # Streaming tells NAT how to go from a list of streaming events to a single response
        # object for non-streaming routes.
        Streaming(convert=aggregate_dragent_event_responses),
    ]:
        # LLM might contain user-specific headers
        llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.CREWAI)

        # Agent contains user-specific headers and authorization context
        forwarded_headers = extract_datarobot_headers_from_context()
        authorization_context = extract_authorization_from_context()
        agent = MyAgent(
            llm=llm,
            forwarded_headers=forwarded_headers,
            authorization_context=authorization_context,
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
