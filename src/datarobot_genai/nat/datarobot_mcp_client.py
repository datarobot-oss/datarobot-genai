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

from typing import Literal

from nat.plugins.mcp.client_config import MCPServerConfig as NATMCPServerConfig
from nat.plugins.mcp.client_impl import MCPClientConfig
from pydantic import Field
from pydantic import HttpUrl

from datarobot_genai.drmcp.core.config import MCPServerConfig

config = MCPServerConfig()


class DataRobotMCPServerConfig(NATMCPServerConfig):
    transport: Literal["streamable-http"] = Field(
        default="streamable-http",
        description="Transport type to connect to the MCP server (streamable-http)",
    )
    url: HttpUrl = Field(
        default=config.mcp_url, description="URL of the MCP server (for streamable-http transport)"
    )


class DataRobotMCPClientConfig(MCPClientConfig, name="datarobot_mcp_client"):  # type: ignore[call-arg]
    server: DataRobotMCPServerConfig = Field(
        default=DataRobotMCPServerConfig(), description="DataRobot MCP Server configuration"
    )
