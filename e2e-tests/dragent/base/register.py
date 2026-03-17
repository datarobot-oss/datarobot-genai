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
from nat.cli.register_workflow import register_function
from nat.data_models.agent import AgentBaseConfig


class BaseAgentConfig(AgentBaseConfig, name="base_agent"):
    """NAT config for the base agent."""


@register_function(
    config_type=BaseAgentConfig,
)
async def base_agent(config: BaseAgentConfig, builder: Builder) -> AsyncGenerator:
    from ag_ui.core import RunAgentInput  # noqa: PLC0415
    from datarobot_genai.dragent.response import DRAgentEventResponse  # noqa: PLC0415
    from nat.builder.function_info import FunctionInfo  # noqa: PLC0415

    from dragent.base.myagent import MyAgent  # noqa: PLC0415

    async def _response_fn(
        input_message: RunAgentInput,
    ) -> AsyncGenerator[DRAgentEventResponse, None]:
        agent = MyAgent()

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
