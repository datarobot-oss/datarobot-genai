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
import warnings
from collections.abc import AsyncGenerator

from nat.cli.register_workflow import register_front_end
from nat.data_models.api_server import GlobalTypeConverter
from nat.data_models.config import Config

from .config import DRAgentA2AConfig
from .config import DRAgentA2AExternalConfig
from .config import DRAgentFastApiFrontEndConfig
from .converters import convert_chat_request_to_run_agent_input
from .converters import convert_dragent_event_response_to_chat_response_chunk
from .converters import convert_dragent_event_response_to_str
from .converters import convert_dragent_run_agent_input_to_chat_request
from .converters import convert_dragent_run_agent_input_to_chat_request_or_message
from .converters import convert_run_agent_input_to_chat_request_or_message
from .converters import convert_str_to_chat_response
from .converters import convert_str_to_dragent_event_response
from .converters import convert_tool_message_to_str
from .logging import logging_handler_setup

# Re-export the config models so the public import path
# ``datarobot_genai.dragent.frontends.register.*`` keeps working after they moved to config.py.
__all__ = [
    "DRAgentA2AConfig",
    "DRAgentA2AExternalConfig",
    "DRAgentFastApiFrontEndConfig",
]

# Suppress specific non-actionable NAT warning messages by content.
# Patch Handler.handle (inherited by all subclasses - they only override emit)
# because root-logger filters are skipped during log propagation.
logging_handler_setup()

# Suppress UserWarning from langchain about non-default parameters (uses warnings.warn, not logging)
warnings.filterwarnings("ignore", message=".*stream_options is not default parameter.*")


@register_front_end(config_type=DRAgentFastApiFrontEndConfig)
async def dragent_fastapi_front_end(
    config: DRAgentFastApiFrontEndConfig, full_config: Config
) -> AsyncGenerator[typing.Any, None]:
    from .fastapi import DRAgentFastApiFrontEndPlugin

    yield DRAgentFastApiFrontEndPlugin(full_config=full_config)


# Register console frontend for `nat dragent run`
from .console import DRAgentConsoleFrontEndConfig  # noqa: E402


@register_front_end(config_type=DRAgentConsoleFrontEndConfig)
async def dragent_console_front_end(
    config: DRAgentConsoleFrontEndConfig, full_config: Config
) -> AsyncGenerator[typing.Any, None]:
    from .console import DRAgentConsoleFrontEndPlugin

    yield DRAgentConsoleFrontEndPlugin(full_config=full_config)


# Register converters
GlobalTypeConverter.register_converter(convert_dragent_run_agent_input_to_chat_request)
GlobalTypeConverter.register_converter(convert_chat_request_to_run_agent_input)
GlobalTypeConverter.register_converter(convert_dragent_run_agent_input_to_chat_request_or_message)
GlobalTypeConverter.register_converter(convert_run_agent_input_to_chat_request_or_message)
GlobalTypeConverter.register_converter(convert_tool_message_to_str)
GlobalTypeConverter.register_converter(convert_str_to_dragent_event_response)
GlobalTypeConverter.register_converter(convert_dragent_event_response_to_str)
GlobalTypeConverter.register_converter(convert_dragent_event_response_to_chat_response_chunk)
# Overrides NAT's built-in str -> ChatResponse so the non-streaming /chat/completions
# response reports the agent's configured model instead of NAT's "unknown-model" default.
GlobalTypeConverter.register_converter(convert_str_to_chat_response)
