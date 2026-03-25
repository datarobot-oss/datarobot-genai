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

import datarobot as dr
import pytest

from datarobot_genai.drmcp.core.dynamic_tools.deployment.register import (
    get_datarobot_tool_deployments,
)
from datarobot_genai.drmcp.core.dynamic_tools.deployment.register import (
    register_tools_of_datarobot_deployments,
)


@pytest.fixture
def module_under_test() -> str:
    return "datarobot_genai.drmcp.core.dynamic_tools.deployment.register"


@pytest.fixture
def mock_get_datarobot_tool_deployments(module_under_test: str) -> Iterator[Mock]:
    with patch(f"{module_under_test}.get_datarobot_tool_deployments") as mock_func:
        yield mock_func


@pytest.fixture
def mock_register_tool_of_datarobot_deployment(module_under_test: str) -> Iterator[AsyncMock]:
    with patch(
        f"{module_under_test}.register_tool_of_datarobot_deployment",
        new_callable=AsyncMock,
    ) as mock_func:
        yield mock_func


@pytest.fixture
def mock_mcp_server(module_under_test: str) -> Iterator[Mock]:
    with patch(f"{module_under_test}.mcp") as mock_mcp:
        yield mock_mcp


@pytest.fixture
def mock_datarobot_deployment_get() -> Iterator[Mock]:
    with patch.object(dr.Deployment, "get") as mock_func:
        yield mock_func


@patch("datarobot_genai.drmcp.core.dynamic_tools.deployment.register.get_api_client")
@patch("datarobot_genai.drmcp.core.dynamic_tools.deployment.register.dr")
def test_get_datarobot_tool_deployments_filters_tags_correctly(mock_dr, mock_get_api_client):
    """Test get_datarobot_tool_deployments accurately filters tags with AND logic."""
    mock_deployments_data = [
        # Should be included (has name="tool" and value="tool")
        {
            "id": "deployment_1",
            "tags": [{"name": "tool", "value": "tool"}, {"name": "other", "value": "other"}],
        },
        # Should be excluded (has name="tool", but value="MCP")
        {
            "id": "deployment_2",
            "tags": [{"name": "tool", "value": "MCP"}],
        },
        # Should be excluded (no tags)
        {
            "id": "deployment_3",
            "tags": [],
        },
        # Should be excluded (no 'tags' key)
        {
            "id": "deployment_4",
        },
        # Should be included (has name="tool" and value="tool" in a different position)
        {
            "id": "deployment_5",
            "tags": [{"name": "something", "value": "else"}, {"name": "tool", "value": "tool"}],
        },
        # Should be excluded (name="other", value="tool")
        {
            "id": "deployment_6",
            "tags": [{"name": "other", "value": "tool"}],
        },
    ]

    mock_dr.utils.pagination.unpaginate.return_value = mock_deployments_data

    result = get_datarobot_tool_deployments()

    # It calls dr.utils.pagination.unpaginate with exactly these parameters
    mock_dr.utils.pagination.unpaginate.assert_called_once_with(
        initial_url="deployments/",
        initial_params={"tag_values": "tool", "tag_keys": "tool"},
        client=mock_get_api_client(),
    )

    assert result == ["deployment_1", "deployment_5"]


@pytest.mark.asyncio
async def test_sync_mcp_metadata_after_register_tools(
    mock_datarobot_deployment_get: Mock,
    mock_is_mcp_tools_gallery_support_enabled: Mock,
    mock_get_datarobot_tool_deployments: Mock,
    mock_lineage_manager_init: Mock,
    mock_sync_mcp_tools: Mock,
    mock_mcp_server: Mock,
    mock_register_tool_of_datarobot_deployment: AsyncMock,
) -> None:
    mock_deployment_id = Mock()
    mock_get_datarobot_tool_deployments.return_value = [mock_deployment_id]

    await register_tools_of_datarobot_deployments()

    mock_get_datarobot_tool_deployments.assert_called_once_with()
    mock_datarobot_deployment_get.assert_called_once_with(mock_deployment_id)
    mock_register_tool_of_datarobot_deployment.assert_called_once_with(
        mock_datarobot_deployment_get.return_value,
    )
    mock_is_mcp_tools_gallery_support_enabled.assert_called_once_with()
    mock_lineage_manager_init.assert_called_once_with(mock_mcp_server)
    mock_sync_mcp_tools.assert_called_once_with()


@pytest.mark.asyncio
async def test_not_run_sync_mcp_metadata_after_no_tool_is_registered(
    mock_is_mcp_tools_gallery_support_enabled: Mock,
    mock_get_datarobot_tool_deployments: Mock,
    mock_lineage_manager_init: Mock,
    mock_mcp_server: Mock,
    mock_sync_mcp_tools: Mock,
) -> None:
    mock_get_datarobot_tool_deployments.return_value = []

    await register_tools_of_datarobot_deployments()

    mock_get_datarobot_tool_deployments.assert_called_once_with()
    mock_is_mcp_tools_gallery_support_enabled.assert_not_called()
    mock_lineage_manager_init.assert_not_called()
    mock_sync_mcp_tools.assert_not_called()
