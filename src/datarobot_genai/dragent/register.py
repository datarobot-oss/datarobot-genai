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

from nat.cli.register_workflow import register_front_end
from nat.data_models.api_server import GlobalTypeConverter
from nat.data_models.config import Config
from nat.front_ends.fastapi.fastapi_front_end_config import FastApiFrontEndConfig

from datarobot_genai.dragent.converters import convert_chat_request_to_run_agent_input
from datarobot_genai.dragent.converters import convert_dragent_run_agent_input_to_chat_request
from datarobot_genai.dragent.converters import (
    convert_dragent_run_agent_input_to_chat_request_or_message,
)
from datarobot_genai.dragent.converters import convert_event_response_to_str
from datarobot_genai.dragent.converters import convert_str_to_dragent_event_response
from datarobot_genai.dragent.converters import convert_tool_message_to_str


# Register frontend
class DRAgentFastApiFrontEndConfig(FastApiFrontEndConfig, name="dragent_fastapi"):  # type: ignore
    pass


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
GlobalTypeConverter.register_converter(convert_event_response_to_str)
