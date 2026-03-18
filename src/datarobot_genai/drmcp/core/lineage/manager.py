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

from datarobot._experimental.models.user_mcp_server_deployment import ToolInUserMCPServerDeployment
from datarobot._experimental.models.user_mcp_server_deployment import (
    TypeOfToolInUserMCPServerDeployment,
)
from datarobot.errors import ClientError as DataRobotAPIClientError
from fastmcp import FastMCP

from datarobot_genai.drmcp.core.clients import get_dr_api_client_with_static_config_in_container
from datarobot_genai.drmcp.core.feature_flags import FeatureFlag
from datarobot_genai.drmcp.core.lineage.entities import BASE_MCP_METADATA_TYPE
from datarobot_genai.drmcp.core.lineage.entities import MCPToolMetadata
from datarobot_genai.drmcp.core.lineage.enums import LRSEnvVars

logger = logging.getLogger(__name__)


class LineageManager:
    def __init__(self, mcp_server_instance: FastMCP) -> None:
        get_dr_api_client_with_static_config_in_container()
        self.feature_flag_enabled = FeatureFlag.create("ENABLE_MCP_TOOLS_GALLERY_SUPPORT").enabled
        self.mcp_server_deployment_id = LRSEnvVars.MLOPS_DEPLOYMENT_ID.get_os_env_value()
        self.mcp_server_version_id = LRSEnvVars.MLOPS_MODEL_ID.get_os_env_value()
        self.mcp_server_instance = mcp_server_instance

    async def get_mcp_tools_associated_with_mcp_server_deployment(
        self,
    ) -> list[MCPToolMetadata]:
        datarobot_public_api_tool_dtos = ToolInUserMCPServerDeployment.list(
            mcp_server_deployment_id=self.mcp_server_deployment_id,
            limit=0,
        )
        return [
            MCPToolMetadata.from_datarobot_mcp_item_in_mcp_server_deployment(datarobot_dto)
            for datarobot_dto in datarobot_public_api_tool_dtos
        ]

    async def get_mcp_tools_in_mcp_server(self) -> list[MCPToolMetadata]:
        mcp_tools = await self.mcp_server_instance._list_tools_mcp()
        return [MCPToolMetadata.from_fastmcp_item(mcp_tool) for mcp_tool in mcp_tools]

    @staticmethod
    def get_mcp_items_to_associate_with_mcp_server_deployment(
        items_already_associated_with_mcp_server_deployments: list[BASE_MCP_METADATA_TYPE],
        mcp_items_in_mcp_server: list[BASE_MCP_METADATA_TYPE],
    ) -> list[BASE_MCP_METADATA_TYPE]:
        names_of_mcp_items_associate_with_mcp_server_deployment = {
            mcp_item.name for mcp_item in items_already_associated_with_mcp_server_deployments
        }
        return [
            mcp_item
            for mcp_item in mcp_items_in_mcp_server
            if mcp_item.name not in names_of_mcp_items_associate_with_mcp_server_deployment
        ]

    @staticmethod
    def get_mcp_items_to_dissociate_from_mcp_server_deployment(
        items_already_associated_with_mcp_server_deployments: list[BASE_MCP_METADATA_TYPE],
        mcp_items_in_mcp_server: list[BASE_MCP_METADATA_TYPE],
    ) -> list[BASE_MCP_METADATA_TYPE]:
        names_of_mcp_items_in_mcp_server = {mcp_item.name for mcp_item in mcp_items_in_mcp_server}
        return [
            mcp_item
            for mcp_item in items_already_associated_with_mcp_server_deployments
            if mcp_item.name not in names_of_mcp_items_in_mcp_server
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
                    f"from mcp server deployment (ID: {self.mcp_server_deployment_id})"
                )
                logger.exception(error_msg)
                continue

    @staticmethod
    async def dissociate_mcp_tools_from_mcp_server_deployment(
        mcp_tool_metadatas: list[MCPToolMetadata],
    ) -> None:
        for mcp_tool_metadata in mcp_tool_metadatas:
            mcp_tool = mcp_tool_metadata.to_datarobot_mcp_item_in_mcp_server_deployment()
            try:
                mcp_tool.delete()
            except DataRobotAPIClientError:
                error_msg = (
                    f"Fail during dissociating one mcp tool (ID: {mcp_tool.id})"
                    f"from mcp server deployment (ID: {mcp_tool.mcp_server_deployment_id})"
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

    async def sync_mcp_item_metadata_with_mcp_items_in_server(self) -> None:
        pass
