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
    BUILT_IN_TOOL = auto()  # tools as a wrapper of external service
    USER_TOOL_DEPLOYMENT = auto()  # DataRobot deployment tools loaded dynamically per user
    PROXIED_USER_MCP = auto()  # tools proxied from a user's own deployed MCP server

    @staticmethod
    def from_string(enum_str: str) -> "DataRobotMCPToolCategory":
        enum_str_map = {
            "USER_TOOL": DataRobotMCPToolCategory.USER_TOOL,
            "BUILT_IN_TOOL": DataRobotMCPToolCategory.BUILT_IN_TOOL,
            "USER_TOOL_DEPLOYMENT": DataRobotMCPToolCategory.USER_TOOL_DEPLOYMENT,
            "PROXIED_USER_MCP": DataRobotMCPToolCategory.PROXIED_USER_MCP,
        }
        if enum_str not in enum_str_map:
            error_msg = f"Enum string should be one of {', '.join(enum_str_map.keys())}"
            raise ValueError(error_msg)

        return enum_str_map[enum_str]
