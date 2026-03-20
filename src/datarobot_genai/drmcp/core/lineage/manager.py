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
import logging

from datarobot._experimental.models.user_mcp_server_deployment import (
    PromptInUserMCPServerDeployment,
)
from datarobot._experimental.models.user_mcp_server_deployment import (
    ResourceInUserMCPServerDeployment,
)
from datarobot._experimental.models.user_mcp_server_deployment import ToolInUserMCPServerDeployment
from datarobot._experimental.models.user_mcp_server_deployment import (
    TypeOfPromptInUserMCPServerDeployment,
)
from datarobot._experimental.models.user_mcp_server_deployment import (
    TypeOfResourceInUserMCPServerDeployment,
)
from datarobot._experimental.models.user_mcp_server_deployment import (
    TypeOfToolInUserMCPServerDeployment,
)
from datarobot.errors import ClientError as DataRobotAPIClientError
from fastmcp import FastMCP

from datarobot_genai.drmcp.core.clients import (
    setup_and_return_dr_api_client_with_static_config_in_container,
)
from datarobot_genai.drmcp.core.lineage.entities import BASE_MCP_METADATA_TYPE
from datarobot_genai.drmcp.core.lineage.entities import MCPPromptMetadata
from datarobot_genai.drmcp.core.lineage.entities import MCPResourceMetadata
from datarobot_genai.drmcp.core.lineage.entities import MCPToolMetadata
from datarobot_genai.drmcp.core.lineage.enums import LRSEnvVars

logger = logging.getLogger(__name__)


class LineageManager:
    def __init__(self, mcp_server_instance: FastMCP) -> None:
        setup_and_return_dr_api_client_with_static_config_in_container()
        self.mcp_server_deployment_id = LRSEnvVars.MLOPS_DEPLOYMENT_ID.get_os_env_value()
        self.mcp_server_instance = mcp_server_instance

    @staticmethod
    def get_mcp_items_to_associate_with_mcp_server_deployment(
        mcp_items_associated_with_mcp_server_deployment: list[BASE_MCP_METADATA_TYPE],
        mcp_items_in_mcp_server: list[BASE_MCP_METADATA_TYPE],
    ) -> list[BASE_MCP_METADATA_TYPE]:
        names_of_mcp_items_associate_with_mcp_server_deployment = {
            mcp_item.name for mcp_item in mcp_items_associated_with_mcp_server_deployment
        }
        return [
            mcp_item
            for mcp_item in mcp_items_in_mcp_server
            if mcp_item.name not in names_of_mcp_items_associate_with_mcp_server_deployment
        ]

    @staticmethod
    def get_mcp_items_to_dissociate_from_mcp_server_deployment(
        mcp_items_associated_with_mcp_server_deployment: list[BASE_MCP_METADATA_TYPE],
        mcp_items_in_mcp_server: list[BASE_MCP_METADATA_TYPE],
    ) -> list[BASE_MCP_METADATA_TYPE]:
        names_of_mcp_items_in_mcp_server = {mcp_item.name for mcp_item in mcp_items_in_mcp_server}
        return [
            mcp_item
            for mcp_item in mcp_items_associated_with_mcp_server_deployment
            if mcp_item.name not in names_of_mcp_items_in_mcp_server
        ]

    async def get_mcp_tools_associated_with_mcp_server_deployment(
        self,
    ) -> list[MCPToolMetadata]:
        mcp_tools = ToolInUserMCPServerDeployment.list(
            mcp_server_deployment_id=self.mcp_server_deployment_id,
            limit=0,
        )
        return [
            MCPToolMetadata.from_datarobot_mcp_server_deployment_item(mcp_tool)
            for mcp_tool in mcp_tools
        ]

    async def get_mcp_prompts_associated_with_mcp_server_deployment(
        self,
    ) -> list[MCPPromptMetadata]:
        mcp_prompts = PromptInUserMCPServerDeployment.list(
            mcp_server_deployment_id=self.mcp_server_deployment_id,
            limit=0,
        )
        return [
            MCPPromptMetadata.from_datarobot_mcp_server_deployment_item(mcp_prompt)
            for mcp_prompt in mcp_prompts
        ]

    async def get_mcp_resources_associated_with_mcp_server_deployment(
        self,
    ) -> list[MCPResourceMetadata]:
        mcp_resources = ResourceInUserMCPServerDeployment.list(
            mcp_server_deployment_id=self.mcp_server_deployment_id,
            limit=0,
        )
        return [
            MCPResourceMetadata.from_datarobot_mcp_server_deployment_item(mcp_resource)
            for mcp_resource in mcp_resources
        ]

    async def get_mcp_tools_in_mcp_server(self) -> list[MCPToolMetadata]:
        mcp_tools = await self.mcp_server_instance._list_tools_mcp()
        return [MCPToolMetadata.from_fastmcp_item(mcp_tool) for mcp_tool in mcp_tools]

    async def get_mcp_prompts_in_mcp_server(self) -> list[MCPPromptMetadata]:
        mcp_prompts = await self.mcp_server_instance._list_prompts_mcp()
        return [MCPPromptMetadata.from_fastmcp_item(mcp_prompt) for mcp_prompt in mcp_prompts]

    async def get_mcp_resources_in_mcp_server(self) -> list[MCPResourceMetadata]:
        mcp_resources = await self.mcp_server_instance._list_resources_mcp()
        return [
            MCPResourceMetadata.from_fastmcp_item(mcp_resource) for mcp_resource in mcp_resources
        ]

    async def associate_mcp_tools_with_mcp_server_deployment(
        self, mcp_tool_metadatas: list[MCPToolMetadata]
    ) -> None:
        for mcp_tool_metadata in mcp_tool_metadatas:
            try:
                ToolInUserMCPServerDeployment.create(
                    mcp_server_deployment_id=self.mcp_server_deployment_id,
                    name=mcp_tool_metadata.name,
                    type=TypeOfToolInUserMCPServerDeployment.from_string(mcp_tool_metadata.type),
                )
            except DataRobotAPIClientError:
                error_msg = (
                    f"Fail during associating one mcp tool (name: {mcp_tool_metadata.name})"
                    f" from mcp server deployment (ID: {self.mcp_server_deployment_id})"
                )
                logger.exception(error_msg)
                continue

    async def associate_mcp_prompts_with_mcp_server_deployment(
        self, mcp_prompt_metadatas: list[MCPPromptMetadata]
    ) -> None:
        for mcp_prompt_metadata in mcp_prompt_metadatas:
            try:
                PromptInUserMCPServerDeployment.create(
                    mcp_server_deployment_id=self.mcp_server_deployment_id,
                    name=mcp_prompt_metadata.name,
                    type=TypeOfPromptInUserMCPServerDeployment.from_string(
                        mcp_prompt_metadata.type
                    ),
                )
            except DataRobotAPIClientError:
                error_msg = (
                    f"Fail during associating one mcp prompt (name: {mcp_prompt_metadata.name})"
                    f" from mcp server deployment (ID: {self.mcp_server_deployment_id})"
                )
                logger.exception(error_msg)
                continue

    async def associate_mcp_resources_with_mcp_server_deployment(
        self, mcp_resource_metadatas: list[MCPResourceMetadata]
    ) -> None:
        for mcp_resource_metadata in mcp_resource_metadatas:
            try:
                ResourceInUserMCPServerDeployment.create(
                    mcp_server_deployment_id=self.mcp_server_deployment_id,
                    uri=mcp_resource_metadata.uri,
                    name=mcp_resource_metadata.name,
                    type=TypeOfResourceInUserMCPServerDeployment.from_string(
                        mcp_resource_metadata.type
                    ),
                )
            except DataRobotAPIClientError:
                error_msg = (
                    f"Fail during associating one mcp resource (name: {mcp_resource_metadata.name})"
                    f" from mcp server deployment (ID: {self.mcp_server_deployment_id})"
                )
                logger.exception(error_msg)
                continue

    @staticmethod
    async def dissociate_mcp_tools_from_mcp_server_deployment(
        mcp_tool_metadatas: list[MCPToolMetadata],
    ) -> None:
        for mcp_tool_metadata in mcp_tool_metadatas:
            mcp_tool = mcp_tool_metadata.to_datarobot_mcp_server_deployment_item()
            try:
                mcp_tool.delete()
            except DataRobotAPIClientError:
                error_msg = (
                    f"Fail during dissociating one mcp tool (ID: {mcp_tool.id})"
                    f"from mcp server deployment (ID: {mcp_tool.mcp_server_deployment_id})"
                )
                logger.exception(error_msg)
                continue

    @staticmethod
    async def dissociate_mcp_prompts_from_mcp_server_deployment(
        mcp_prompt_metadatas: list[MCPPromptMetadata],
    ) -> None:
        for mcp_prompt_metadata in mcp_prompt_metadatas:
            mcp_prompt = mcp_prompt_metadata.to_datarobot_mcp_server_deployment_item()
            try:
                mcp_prompt.delete()
            except DataRobotAPIClientError:
                error_msg = (
                    f"Fail during dissociating one mcp prompt (ID: {mcp_prompt.id})"
                    f"from mcp server deployment (ID: {mcp_prompt.mcp_server_deployment_id})"
                )
                logger.exception(error_msg)
                continue

    @staticmethod
    async def dissociate_mcp_resources_from_mcp_server_deployment(
        mcp_resource_metadatas: list[MCPResourceMetadata],
    ) -> None:
        for mcp_resource_metadata in mcp_resource_metadatas:
            mcp_resource = mcp_resource_metadata.to_datarobot_mcp_server_deployment_item()
            try:
                mcp_resource.delete()
            except DataRobotAPIClientError:
                error_msg = (
                    f"Fail during dissociating one mcp resource (ID: {mcp_resource.id})"
                    f"from mcp server deployment (ID: {mcp_resource.mcp_server_deployment_id})"
                )
                logger.exception(error_msg)
                continue

    async def sync_mcp_tools(self) -> None:
        mcp_tools_associated_with_deployment = (
            await self.get_mcp_tools_associated_with_mcp_server_deployment()
        )
        mcp_tools_in_server = await self.get_mcp_tools_in_mcp_server()

        mcp_tools_to_associated_with_deployment = (
            self.get_mcp_items_to_associate_with_mcp_server_deployment(
                mcp_tools_associated_with_deployment, mcp_tools_in_server
            )
        )
        mcp_tools_to_dissociate_from_deployment = (
            self.get_mcp_items_to_dissociate_from_mcp_server_deployment(
                mcp_tools_associated_with_deployment, mcp_tools_in_server
            )
        )

        await self.associate_mcp_tools_with_mcp_server_deployment(
            mcp_tools_to_associated_with_deployment
        )
        await self.dissociate_mcp_tools_from_mcp_server_deployment(
            mcp_tools_to_dissociate_from_deployment
        )

    async def sync_mcp_prompts(self) -> None:
        mcp_prompts_associated_with_deployment = (
            await self.get_mcp_prompts_associated_with_mcp_server_deployment()
        )
        mcp_prompts_in_server = await self.get_mcp_prompts_in_mcp_server()

        mcp_prompts_to_associated_with_deployment = (
            self.get_mcp_items_to_associate_with_mcp_server_deployment(
                mcp_prompts_associated_with_deployment, mcp_prompts_in_server
            )
        )
        mcp_prompts_to_dissociate_from_deployment = (
            self.get_mcp_items_to_dissociate_from_mcp_server_deployment(
                mcp_prompts_associated_with_deployment, mcp_prompts_in_server
            )
        )

        await self.associate_mcp_prompts_with_mcp_server_deployment(
            mcp_prompts_to_associated_with_deployment
        )
        await self.dissociate_mcp_prompts_from_mcp_server_deployment(
            mcp_prompts_to_dissociate_from_deployment
        )

    async def sync_mcp_resources(self) -> None:
        mcp_resources_associated_with_deployment = (
            await self.get_mcp_resources_associated_with_mcp_server_deployment()
        )
        mcp_resources_in_server = await self.get_mcp_resources_in_mcp_server()

        mcp_resources_to_associated_with_deployment = (
            self.get_mcp_items_to_associate_with_mcp_server_deployment(
                mcp_resources_associated_with_deployment, mcp_resources_in_server
            )
        )
        mcp_resources_to_dissociate_from_deployment = (
            self.get_mcp_items_to_dissociate_from_mcp_server_deployment(
                mcp_resources_associated_with_deployment, mcp_resources_in_server
            )
        )

        await self.associate_mcp_resources_with_mcp_server_deployment(
            mcp_resources_to_associated_with_deployment
        )
        await self.dissociate_mcp_resources_from_mcp_server_deployment(
            mcp_resources_to_dissociate_from_deployment
        )
