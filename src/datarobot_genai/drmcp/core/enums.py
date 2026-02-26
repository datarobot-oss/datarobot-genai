# Copyright 2026 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from enum import Enum
from enum import auto


class DataRobotMCPToolCategory(Enum):
    USER_TOOL = auto()  # tools created by users
    BUILD_IN_TOOL = auto()  # tools as a wrapper of external service
    USER_TOOL_DEPLOYMENT = auto()  # tools dynamically loaded after MCP server is up
    UNKNOWN = auto()  # tools without category

    @staticmethod
    def from_string(enum_str: str) -> "DataRobotMCPToolCategory":
        enum_str_map = {
            "USER_TOOL": DataRobotMCPToolCategory.USER_TOOL,
            "BUILD_IN_TOOL": DataRobotMCPToolCategory.BUILD_IN_TOOL,
            "USER_TOOL_DEPLOYMENT": DataRobotMCPToolCategory.USER_TOOL_DEPLOYMENT,
            "UNKNOWN": DataRobotMCPToolCategory.UNKNOWN,
        }
        if enum_str not in enum_str_map:
            error_msg = f"Enum string should be one of {', '.join(enum_str_map.keys())}"
            raise ValueError(error_msg)

        return enum_str_map[enum_str]


class DataRobotMCPPromptCategory(Enum):
    USER_PROMPT = auto()  # prompt created by users
    USER_PROMPT_TEMPLATE_VERSION = auto()  # prompts dynamically loaded after MCP server is up
    UNKNOWN = auto()  # prompts without category

    @staticmethod
    def from_string(enum_str: str) -> "DataRobotMCPPromptCategory":
        enum_str_map = {
            "USER_PROMPT": DataRobotMCPPromptCategory.USER_PROMPT,
            "USER_PROMPT_TEMPLATE_VERSION": DataRobotMCPPromptCategory.USER_PROMPT_TEMPLATE_VERSION,
            "UNKNOWN": DataRobotMCPPromptCategory.UNKNOWN,
        }
        if enum_str not in enum_str_map:
            error_msg = f"Enum string should be one of {', '.join(enum_str_map.keys())}"
            raise ValueError(error_msg)

        return enum_str_map[enum_str]


class DataRobotMCPResourceCategory(Enum):
    USER_RESOURCE = auto()  # resource created by users
    UNKNOWN = auto()  # resources without category

    @staticmethod
    def from_string(enum_str: str) -> "DataRobotMCPResourceCategory":
        enum_str_map = {
            "USER_RESOURCE": DataRobotMCPResourceCategory.USER_RESOURCE,
            "UNKNOWN": DataRobotMCPResourceCategory.UNKNOWN,
        }
        if enum_str not in enum_str_map:
            error_msg = f"Enum string should be one of {', '.join(enum_str_map.keys())}"
            raise ValueError(error_msg)

        return enum_str_map[enum_str]
