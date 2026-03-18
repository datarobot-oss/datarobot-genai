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
from collections.abc import Iterator
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from datarobot._experimental.models.user_mcp_server_deployment import ToolInUserMCPServerDeployment
from datarobot._experimental.models.user_mcp_server_deployment import (
    TypeOfToolInUserMCPServerDeployment,
)

from datarobot_genai.drmcp.core.feature_flags import FeatureFlag
from datarobot_genai.drmcp.core.lineage.entities import MCPToolMetadata
from datarobot_genai.drmcp.core.lineage.manager import LineageManager


class TestLineageManager:
    @pytest.fixture
    def module_under_test(self) -> str:
        return "datarobot_genai.drmcp.core.lineage.manager"

    @pytest.fixture
    def mock_get_datarobot_client(self, module_under_test: str) -> Iterator[Mock]:
        with patch(f"{module_under_test}.get_datarobot_client") as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_feature_flag_create(self) -> Iterator[Mock]:
        with patch.object(FeatureFlag, "create") as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_lrs_env_var(self, module_under_test: str) -> Iterator[Mock]:
        with patch(f"{module_under_test}.LRSEnvVars") as mock_enum:
            yield mock_enum

    @pytest.fixture
    def mock_get_mcp_tools_associated_with_mcp_server_deployment(self) -> Iterator[Mock]:
        with patch.object(
            LineageManager,
            "get_mcp_tools_associated_with_mcp_server_deployment",
            new_callable=AsyncMock,
        ) as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_get_mcp_tools_in_mcp_server(self) -> Iterator[Mock]:
        with patch.object(
            LineageManager,
            "get_mcp_tools_in_mcp_server",
            new_callable=AsyncMock,
        ) as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_get_mcp_items_to_associate_with_mcp_server_deployment(self) -> Iterator[Mock]:
        with patch.object(
            LineageManager,
            "get_mcp_items_to_associate_with_mcp_server_deployment",
        ) as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_get_mcp_items_to_dissociate_from_mcp_server_deployment(self) -> Iterator[Mock]:
        with patch.object(
            LineageManager,
            "get_mcp_items_to_dissociate_from_mcp_server_deployment",
        ) as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_associate_mcp_tools_with_mcp_server_deployment(self) -> Iterator[Mock]:
        with patch.object(
            LineageManager,
            "associate_mcp_tools_with_mcp_server_deployment",
            new_callable=AsyncMock,
        ) as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_dissociate_mcp_tools_from_mcp_server_deployment(self) -> Iterator[Mock]:
        with patch.object(
            LineageManager,
            "dissociate_mcp_tools_from_mcp_server_deployment",
            new_callable=AsyncMock,
        ) as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_list_tools_in_user_mcp_server_deployment(self) -> Iterator[Mock]:
        with patch.object(ToolInUserMCPServerDeployment, "list") as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_create_tool_in_user_mcp_server_deployment(self) -> Iterator[Mock]:
        with patch.object(ToolInUserMCPServerDeployment, "create") as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_from_fastmcp_tool(self) -> Iterator[Mock]:
        with patch.object(MCPToolMetadata, "from_fastmcp_item") as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_from_datarobot_tool_in_mcp_server_deployment(self) -> Iterator[Mock]:
        with patch.object(
            MCPToolMetadata, "from_datarobot_mcp_item_in_mcp_server_deployment"
        ) as mock_func:
            yield mock_func

    def test_init(
        self,
        mock_feature_flag_create: Mock,
        mock_get_datarobot_client: Mock,
        mock_lrs_env_var: Mock,
    ) -> None:
        mock_mcp_server_instance = Mock()
        manager = LineageManager(mock_mcp_server_instance)

        mock_get_datarobot_client.assert_called_once_with()
        mock_feature_flag_create.assert_called_once_with("ENABLE_MCP_TOOLS_GALLERY_SUPPORT")
        mock_lrs_env_var.MLOPS_DEPLOYMENT_ID.get_os_env_value.assert_called_once_with()
        mock_lrs_env_var.MLOPS_MODEL_ID.get_os_env_value.assert_called_once_with()

        assert manager.datarobot_client == mock_get_datarobot_client.return_value
        assert manager.feature_flag_enabled == mock_feature_flag_create.return_value.enabled
        assert (
            manager.mcp_server_deployment_id
            == mock_lrs_env_var.MLOPS_DEPLOYMENT_ID.get_os_env_value.return_value
        )
        assert (
            manager.mcp_server_version_id
            == mock_lrs_env_var.MLOPS_MODEL_ID.get_os_env_value.return_value
        )
        assert manager.mcp_server_instance == mock_mcp_server_instance

    @pytest.mark.asyncio
    @pytest.mark.usefixtures(
        "mock_feature_flag_create",
        "mock_get_datarobot_client",
        "mock_lrs_env_var",
    )
    async def test_get_mcp_tools_associated_with_mcp_server_deployment(
        self,
        mock_from_datarobot_tool_in_mcp_server_deployment: Mock,
        mock_list_tools_in_user_mcp_server_deployment: Mock,
    ) -> None:
        mock_tool_in_user_mcp_server_deployment = Mock()
        mock_list_tools_in_user_mcp_server_deployment.return_value = [
            mock_tool_in_user_mcp_server_deployment
        ]

        manager = LineageManager(Mock())
        mcp_tools = await manager.get_mcp_tools_associated_with_mcp_server_deployment()

        mock_list_tools_in_user_mcp_server_deployment.assert_called_once_with(
            mcp_server_deployment_id=manager.mcp_server_deployment_id,
            limit=0,
        )
        mock_from_datarobot_tool_in_mcp_server_deployment.assert_called_once_with(
            mock_tool_in_user_mcp_server_deployment
        )
        assert mcp_tools == [mock_from_datarobot_tool_in_mcp_server_deployment.return_value]

    @pytest.mark.asyncio
    @pytest.mark.usefixtures(
        "mock_feature_flag_create",
        "mock_get_datarobot_client",
        "mock_lrs_env_var",
    )
    async def test_get_mcp_tools_in_mcp_server(
        self,
        mock_from_fastmcp_tool: Mock,
    ) -> None:
        mock_mcp_server = Mock()
        mock_fastmcp_tool = Mock()
        mock_mcp_server._list_tools_mcp = AsyncMock(return_value=[mock_fastmcp_tool])

        manager = LineageManager(mock_mcp_server)
        mcp_tools = await manager.get_mcp_tools_in_mcp_server()

        mock_from_fastmcp_tool.assert_called_once_with(mock_fastmcp_tool)
        assert mcp_tools == [mock_from_fastmcp_tool.return_value]

    def test_get_mcp_items_to_associate_with_mcp_server_deployment(self) -> None:
        share_item = Mock(name="adfa")
        diff_item_one = Mock(name="212rads")
        diff_item_two = Mock(name="32qerqew")
        items_already_associated_with_mcp_server_deployments = [share_item, diff_item_one]
        mcp_items_in_mcp_server = [share_item, diff_item_two]
        outputs = LineageManager.get_mcp_items_to_associate_with_mcp_server_deployment(
            items_already_associated_with_mcp_server_deployments,
            mcp_items_in_mcp_server,
        )

        assert outputs == [diff_item_two]

    def test_get_mcp_items_to_dissociate_from_mcp_server_deployment(self) -> None:
        share_item = Mock(name="adfa")
        diff_item_one = Mock(name="212rads")
        diff_item_two = Mock(name="32qerqew")
        items_already_associated_with_mcp_server_deployments = [share_item, diff_item_one]
        mcp_items_in_mcp_server = [share_item, diff_item_two]
        outputs = LineageManager.get_mcp_items_to_dissociate_from_mcp_server_deployment(
            items_already_associated_with_mcp_server_deployments,
            mcp_items_in_mcp_server,
        )

        assert outputs == [diff_item_one]

    @pytest.mark.asyncio
    @pytest.mark.usefixtures(
        "mock_feature_flag_create",
        "mock_get_datarobot_client",
        "mock_lrs_env_var",
    )
    async def test_associate_mcp_tools_with_mcp_server_deployment(
        self,
        mock_create_tool_in_user_mcp_server_deployment: Mock,
    ) -> None:
        manager = LineageManager(Mock())

        mcp_tool = Mock(type="USER_TOOL")
        await manager.associate_mcp_tools_with_mcp_server_deployment([mcp_tool])

        mock_create_tool_in_user_mcp_server_deployment.assert_called_once_with(
            mcp_server_deployment_id=manager.mcp_server_deployment_id,
            name=mcp_tool.name,
            type=TypeOfToolInUserMCPServerDeployment.from_api_representation(mcp_tool.type),
        )

    @pytest.mark.asyncio
    async def test_dissociate_mcp_tools_from_mcp_server_deployment(
        self,
    ) -> None:
        mcp_tool = Mock()
        await LineageManager.dissociate_mcp_tools_from_mcp_server_deployment([mcp_tool])

        mcp_tool.to_datarobot_mcp_item_in_mcp_server_deployment.assert_called_once_with()
        datarobot_mcp_tool_object = (
            mcp_tool.to_datarobot_mcp_item_in_mcp_server_deployment.return_value
        )
        datarobot_mcp_tool_object.delete.assert_called_once_with()

    @pytest.mark.asyncio
    @pytest.mark.usefixtures(
        "mock_feature_flag_create",
        "mock_get_datarobot_client",
        "mock_lrs_env_var",
    )
    async def test_sync_metadata_of_mcp_tools_in_server(
        self,
        mock_get_mcp_tools_associated_with_mcp_server_deployment: AsyncMock,
        mock_get_mcp_tools_in_mcp_server: AsyncMock,
        mock_get_mcp_items_to_associate_with_mcp_server_deployment: Mock,
        mock_get_mcp_items_to_dissociate_from_mcp_server_deployment: Mock,
        mock_associate_mcp_tools_with_mcp_server_deployment: AsyncMock,
        mock_dissociate_mcp_tools_from_mcp_server_deployment: AsyncMock,
    ) -> None:
        manager = LineageManager(Mock())

        await manager.sync_metadata_of_mcp_tools_in_server()

        mock_get_mcp_tools_associated_with_mcp_server_deployment.assert_called_once_with()
        mock_get_mcp_tools_in_mcp_server.assert_called_once_with()
        mcp_tools_associated_with_deployment = (
            mock_get_mcp_tools_associated_with_mcp_server_deployment.return_value
        )
        mcp_tools_in_server = mock_get_mcp_tools_in_mcp_server.return_value
        mock_get_mcp_items_to_associate_with_mcp_server_deployment.assert_called_once_with(
            mcp_tools_associated_with_deployment, mcp_tools_in_server
        )
        mock_get_mcp_items_to_dissociate_from_mcp_server_deployment.assert_called_once_with(
            mcp_tools_associated_with_deployment, mcp_tools_in_server
        )
        mock_associate_mcp_tools_with_mcp_server_deployment.assert_called_once_with(
            mock_get_mcp_items_to_associate_with_mcp_server_deployment.return_value
        )
        mock_dissociate_mcp_tools_from_mcp_server_deployment.assert_called_once_with(
            mock_get_mcp_items_to_dissociate_from_mcp_server_deployment.return_value
        )
