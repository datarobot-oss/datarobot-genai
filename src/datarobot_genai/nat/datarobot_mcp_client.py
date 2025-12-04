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

from nat.plugins.mcp.client_config import MCPServerConfig
from nat.plugins.mcp.client_impl import MCPClientConfig
from pydantic import Field
from pydantic import HttpUrl

from datarobot_genai.core.mcp.common import MCPConfig

config = MCPConfig().server_config


class DataRobotMCPServerConfig(MCPServerConfig):
    transport: Literal["streamable-http", "sse"] = Field(
        default=config["transport"] if config else "streamable-http",
        description="Transport type to connect to the MCP server (sse or streamable-http)",
    )
    url: HttpUrl | None = Field(
        default=config["url"] if config else None,
        description="URL of the MCP server (for sse or streamable-http transport)",
    )
    headers: dict[str, str] | None = Field(default=config["headers"] if config else None)


class DataRobotMCPClientConfig(MCPClientConfig, name="datarobot_mcp_client"):  # type: ignore[call-arg]
    server: DataRobotMCPServerConfig = Field(
        default=DataRobotMCPServerConfig(), description="DataRobot MCP Server configuration"
    )
