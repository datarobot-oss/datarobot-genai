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
import typing
import warnings
from collections.abc import AsyncGenerator

from a2a.types import AgentSkill
from nat.cli.register_workflow import register_front_end
from nat.data_models.api_server import GlobalTypeConverter
from nat.data_models.config import Config
from nat.front_ends.fastapi.fastapi_front_end_config import FastApiFrontEndConfig
from nat.plugins.a2a.server.front_end_config import A2AFrontEndConfig
from pydantic import BaseModel
from pydantic import Field

from .converters import convert_chat_request_to_run_agent_input
from .converters import convert_dragent_event_response_to_str
from .converters import convert_dragent_run_agent_input_to_chat_request
from .converters import convert_dragent_run_agent_input_to_chat_request_or_message
from .converters import convert_str_to_dragent_event_response
from .converters import convert_tool_message_to_str
from .patches import patch_crewai_callback_handler

# Patch nvidia-nat-crewai callback handler for crewai >= 1.1.0 compatibility.
# Must run before NAT's instrument() is called. Safe no-op if crewai not installed.
patch_crewai_callback_handler()

# Suppress specific non-actionable NAT warning messages by content.
# Patch Handler.handle (inherited by all subclasses - they only override emit)
# because root-logger filters are skipped during log propagation.
_SUPPRESSED_NAT_MESSAGES = [
    "StepAdaptor is disabled",
    "Dask is not installed",
    "Dask is not available",
    "feature is experimental",
    "Using provided input_schema for multi-argument function",
    "not found in outstanding start steps",
]
_orig_handler_handle = logging.Handler.handle


def _filtered_handle(self: logging.Handler, record: logging.LogRecord) -> bool | None:
    if any(s in record.getMessage() for s in _SUPPRESSED_NAT_MESSAGES):
        return None
    return _orig_handler_handle(self, record)


logging.Handler.handle = _filtered_handle  # type: ignore[assignment]

# Suppress UserWarning from langchain about non-default parameters (uses warnings.warn, not logging)
warnings.filterwarnings("ignore", message=".*stream_options is not default parameter.*")


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
    from .fastapi import DRAgentFastApiFrontEndPlugin

    yield DRAgentFastApiFrontEndPlugin(full_config=full_config)


# Register converters
GlobalTypeConverter.register_converter(convert_dragent_run_agent_input_to_chat_request)
GlobalTypeConverter.register_converter(convert_chat_request_to_run_agent_input)
GlobalTypeConverter.register_converter(convert_dragent_run_agent_input_to_chat_request_or_message)
GlobalTypeConverter.register_converter(convert_tool_message_to_str)
GlobalTypeConverter.register_converter(convert_str_to_dragent_event_response)
GlobalTypeConverter.register_converter(convert_dragent_event_response_to_str)
