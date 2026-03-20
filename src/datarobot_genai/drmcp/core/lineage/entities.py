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
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import TypeVar

from datarobot._experimental.models.user_mcp_server_deployment import (
    PromptInUserMCPServerDeployment,
)
from datarobot._experimental.models.user_mcp_server_deployment import (
    ResourceInUserMCPServerDeployment,
)
from datarobot._experimental.models.user_mcp_server_deployment import ToolInUserMCPServerDeployment
from fastmcp.prompts import Prompt as FastMCPPrompt
from fastmcp.resources import Resource as FastMCPResource
from fastmcp.tools import Tool as FastMCPTool


@dataclass
class BaseMCPMetadata(ABC):
    name: str
    type: str

    @classmethod
    @abstractmethod
    def from_fastmcp_item(
        cls, fastmcp_item: FastMCPTool | FastMCPPrompt | FastMCPResource
    ) -> "BaseMCPMetadata":
        pass

    @classmethod
    @abstractmethod
    def from_datarobot_mcp_server_deployment_item(
        cls,
        datarobot_mcp_item: ToolInUserMCPServerDeployment
        | PromptInUserMCPServerDeployment
        | ResourceInUserMCPServerDeployment,
    ) -> "BaseMCPMetadata":
        pass

    @abstractmethod
    def to_datarobot_mcp_server_deployment_item(
        self,
    ) -> (
        ToolInUserMCPServerDeployment
        | PromptInUserMCPServerDeployment
        | ResourceInUserMCPServerDeployment
    ):
        pass


BASE_MCP_METADATA_TYPE = TypeVar("BASE_MCP_METADATA_TYPE", bound=BaseMCPMetadata)


@dataclass
class MCPToolMetadata(BaseMCPMetadata):
    id: str | None = None
    created_at: str | None = None
    user_id: str | None = None
    user_name: str | None = None
    mcp_server_deployment_id: str | None = None

    @classmethod
    def from_fastmcp_item(cls, fastmcp_item: FastMCPTool) -> "MCPToolMetadata":
        return cls(
            name=fastmcp_item.name,
            type=fastmcp_item.meta["tool_category"],  # type: ignore[index]
        )

    @classmethod
    def from_datarobot_mcp_server_deployment_item(
        cls, datarobot_mcp_item: ToolInUserMCPServerDeployment
    ) -> "MCPToolMetadata":
        return cls(
            name=datarobot_mcp_item.name,
            type=datarobot_mcp_item.type.to_api_representation(),
            id=datarobot_mcp_item.id,
            created_at=datarobot_mcp_item.created_at,
            user_id=datarobot_mcp_item.user_id,
            user_name=datarobot_mcp_item.user_name,
            mcp_server_deployment_id=datarobot_mcp_item.mcp_server_deployment_id,
        )

    def to_datarobot_mcp_server_deployment_item(
        self,
    ) -> ToolInUserMCPServerDeployment:
        return ToolInUserMCPServerDeployment(
            id=self.id,
            name=self.name,
            type=self.type,
            created_at=self.created_at,
            user_id=self.user_id,
            user_name=self.user_name,
            mcp_server_deployment_id=self.mcp_server_deployment_id,
        )


@dataclass
class MCPPromptMetadata(BaseMCPMetadata):
    id: str | None = None
    created_at: str | None = None
    user_id: str | None = None
    user_name: str | None = None
    mcp_server_deployment_id: str | None = None

    @classmethod
    def from_fastmcp_item(cls, fastmcp_item: FastMCPPrompt) -> "MCPPromptMetadata":
        return cls(
            name=fastmcp_item.name,
            type=fastmcp_item.meta["prompt_category"],  # type: ignore[index]
        )

    @classmethod
    def from_datarobot_mcp_server_deployment_item(
        cls, datarobot_mcp_item: PromptInUserMCPServerDeployment
    ) -> "MCPPromptMetadata":
        return cls(
            name=datarobot_mcp_item.name,
            type=datarobot_mcp_item.type.to_api_representation(),
            id=datarobot_mcp_item.id,
            created_at=datarobot_mcp_item.created_at,
            user_id=datarobot_mcp_item.user_id,
            user_name=datarobot_mcp_item.user_name,
            mcp_server_deployment_id=datarobot_mcp_item.mcp_server_deployment_id,
        )

    def to_datarobot_mcp_server_deployment_item(
        self,
    ) -> PromptInUserMCPServerDeployment:
        return PromptInUserMCPServerDeployment(
            id=self.id,
            name=self.name,
            type=self.type,
            created_at=self.created_at,
            user_id=self.user_id,
            user_name=self.user_name,
            mcp_server_deployment_id=self.mcp_server_deployment_id,
        )


@dataclass
class MCPResourceMetadata(BaseMCPMetadata):
    uri: str
    id: str | None = None
    created_at: str | None = None
    user_id: str | None = None
    user_name: str | None = None
    mcp_server_deployment_id: str | None = None

    @classmethod
    def from_fastmcp_item(cls, fastmcp_item: FastMCPResource) -> "MCPResourceMetadata":
        return cls(
            name=fastmcp_item.name,
            type=fastmcp_item.meta["resource_category"],  # type: ignore[index]
            uri=str(fastmcp_item.uri),
        )

    @classmethod
    def from_datarobot_mcp_server_deployment_item(
        cls, datarobot_mcp_item: ResourceInUserMCPServerDeployment
    ) -> "MCPResourceMetadata":
        return cls(
            name=datarobot_mcp_item.name,
            type=datarobot_mcp_item.type.to_api_representation(),
            uri=datarobot_mcp_item.uri,
            id=datarobot_mcp_item.id,
            created_at=datarobot_mcp_item.created_at,
            user_id=datarobot_mcp_item.user_id,
            user_name=datarobot_mcp_item.user_name,
            mcp_server_deployment_id=datarobot_mcp_item.mcp_server_deployment_id,
        )

    def to_datarobot_mcp_server_deployment_item(
        self,
    ) -> ResourceInUserMCPServerDeployment:
        return ResourceInUserMCPServerDeployment(
            id=self.id,
            name=self.name,
            uri=self.uri,
            type=self.type,
            created_at=self.created_at,
            user_id=self.user_id,
            user_name=self.user_name,
            mcp_server_deployment_id=self.mcp_server_deployment_id,
        )
