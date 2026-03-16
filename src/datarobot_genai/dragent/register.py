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

import typing
from collections.abc import AsyncGenerator

from a2a.types import AgentSkill
from nat.builder.builder import Builder
from nat.cli.register_workflow import register_front_end
from nat.cli.register_workflow import register_per_user_function_group
from nat.data_models.api_server import GlobalTypeConverter
from nat.data_models.config import Config
from nat.front_ends.fastapi.fastapi_front_end_config import FastApiFrontEndConfig
from nat.plugins.a2a.client.client_impl import A2AClientFunctionGroup
from nat.plugins.a2a.server.front_end_config import A2AFrontEndConfig
from pydantic import BaseModel
from pydantic import Field

import datarobot_genai.dragent.per_user_tool_calling_agent  # noqa: F401 — registers per_user_tool_calling_agent
from datarobot_genai.dragent.converters import convert_chat_request_to_run_agent_input
from datarobot_genai.dragent.converters import convert_dragent_event_response_to_str
from datarobot_genai.dragent.converters import convert_dragent_run_agent_input_to_chat_request
from datarobot_genai.dragent.converters import (
    convert_dragent_run_agent_input_to_chat_request_or_message,
)
from datarobot_genai.dragent.converters import convert_str_to_dragent_event_response
from datarobot_genai.dragent.converters import convert_tool_message_to_str
from datarobot_genai.dragent.datarobot_auth_provider import AuthenticatedA2AClientConfig
from datarobot_genai.dragent.datarobot_auth_provider import _AuthCardA2AClientFunctionGroup


@register_per_user_function_group(config_type=AuthenticatedA2AClientConfig)  # type: ignore[untyped-decorator]
async def authenticated_a2a_client(
    config: AuthenticatedA2AClientConfig, _builder: Builder
) -> AsyncGenerator[A2AClientFunctionGroup, None]:
    """Drop-in replacement for the upstream ``a2a_client_function_group`` that
    authenticates the agent-card fetch request.
    """
    async with _AuthCardA2AClientFunctionGroup(config, _builder) as group:
        yield group


class DRAgentA2AConfig(BaseModel):
    """DR-owned wrapper around NAT's A2AFrontEndConfig with optional skill definitions."""

    server: A2AFrontEndConfig = Field(description="NAT A2A server configuration.")
    skills: list[AgentSkill] = Field(
        default=[],
        description="Skills to advertise in the A2A agent card. "
        "If empty, a single default skill is generated from the agent name and description.",
    )


# Register frontend
class DRAgentFastApiFrontEndConfig(FastApiFrontEndConfig, name="dragent_fastapi"):  # type: ignore
    a2a: DRAgentA2AConfig | None = Field(
        default=None,
        description="Expose this agent via the Agent2Agent protocol. "
        "A2A server endpoints are mounted under /a2a/.",
    )


@register_front_end(config_type=DRAgentFastApiFrontEndConfig)
async def dragent_fastapi_front_end(
    config: DRAgentFastApiFrontEndConfig, full_config: Config
) -> AsyncGenerator[typing.Any, None]:
    from datarobot_genai.dragent.frontserver import DRAgentFastApiFrontEndPlugin

    yield DRAgentFastApiFrontEndPlugin(full_config=full_config)


# Register converters
GlobalTypeConverter.register_converter(convert_dragent_run_agent_input_to_chat_request)
GlobalTypeConverter.register_converter(convert_chat_request_to_run_agent_input)
GlobalTypeConverter.register_converter(convert_dragent_run_agent_input_to_chat_request_or_message)
GlobalTypeConverter.register_converter(convert_tool_message_to_str)
GlobalTypeConverter.register_converter(convert_str_to_dragent_event_response)
GlobalTypeConverter.register_converter(convert_dragent_event_response_to_str)
