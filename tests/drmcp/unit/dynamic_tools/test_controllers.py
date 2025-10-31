# Copyright 2025 DataRobot, Inc.
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

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
import pytest_asyncio
from fastmcp.tools.tool import Tool

from datarobot_genai.drmcp.core.dynamic_tools.deployment.controllers import (
    delete_registered_tool_deployment,
)
from datarobot_genai.drmcp.core.dynamic_tools.deployment.controllers import (
    get_registered_tool_deployments,
)
from datarobot_genai.drmcp.core.dynamic_tools.deployment.controllers import (
    register_tool_for_deployment_id,
)
from datarobot_genai.drmcp.core.dynamic_tools.register import ExternalToolRegistrationConfig
from datarobot_genai.drmcp.core.mcp_instance import TaggedFastMCP


@pytest.fixture
def deployment_id():
    """Sample deployment ID for testing."""
    return "deployment-123"


@pytest.fixture
def valid_metadata():
    """Return valid external deployment metadata."""
    return


@pytest.fixture
def mock_config():
    """Mock tool metadata."""
    config = ExternalToolRegistrationConfig(
        name="weather_tool",
        description="Get weather for cities",
        base_url="https://api.weather.com",
        endpoint="/forecast",
        method="POST",
        headers={"x-custom-key": "value"},
        input_schema={
            "type": "object",
            "properties": {
                "json": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                }
            },
        },
        tags={"weather", "api"},
    )
    return config


@pytest_asyncio.fixture
async def mcp_server():
    """Create a separate MCP instance for testing."""
    test_mcp = TaggedFastMCP()

    # Patch the mcp import in controllers, register modules, and mcp_instance
    with (
        patch("datarobot_genai.drmcp.core.dynamic_tools.deployment.controllers.mcp", test_mcp),
        patch("datarobot_genai.drmcp.core.mcp_instance.mcp", test_mcp),
    ):
        yield test_mcp


@pytest.fixture
def mock_external_dependencies():
    """Set up all external dependency mocks."""
    with (
        patch(
            "datarobot_genai.drmcp.core.dynamic_tools.deployment.controllers.get_sdk_client"
        ) as mock_sdk,
        patch(
            "datarobot_genai.drmcp.core.dynamic_tools.deployment.register.create_deployment_tool_config"
        ) as mock_config,
    ):
        mock_sdk.return_value.Deployment.get = lambda x: MagicMock(id=x)
        yield {"sdk": mock_sdk, "mock_config": mock_config}


class TestToolRegistration:
    """Tests for tool registration functionality."""

    @pytest.mark.asyncio
    async def test_register_tool_with_deployment_description(
        self,
        deployment_id,
        mock_config,
        mcp_server,
        mock_external_dependencies,
    ):
        """Test tool registration using deployment description (primary path)."""
        # Setup external data sources
        mock_external_dependencies["mock_config"].return_value = mock_config

        result = await register_tool_for_deployment_id(deployment_id)

        # Assert tool creation and registration
        assert isinstance(result, Tool)
        assert result.name == "weather_tool"

        # Verify deployment mapping
        deployment_mappings = await mcp_server.get_deployment_mapping()
        assert deployment_mappings == {deployment_id: "weather_tool"}

        # Verify tool registration in MCP
        registered_tools = await mcp_server.list_tools()
        assert len(registered_tools) == 1
        assert registered_tools[0].name == "weather_tool"
        assert registered_tools[0].description == "Get weather for cities"

    @pytest.mark.asyncio
    async def test_register_multiple_tools(
        self,
        deployment_id,
        mock_config,
        mcp_server,
        mock_external_dependencies,
    ):
        """Test registering multiple tools and verify both are tracked correctly."""
        # Setup first deployment
        mock_external_dependencies["mock_config"].return_value = mock_config

        # Register first tool
        result1 = await register_tool_for_deployment_id(deployment_id)
        assert result1.name == "weather_tool"

        # Setup second deployment
        mock_config.description = "Another description"
        mock_config.name = "Another Tool"

        # Register second tool
        result2 = await register_tool_for_deployment_id("deployment-456")
        assert result2.name == "Another Tool"  # name should be as in config

        # Verify both deployments are tracked
        deployment_mappings = await mcp_server.get_deployment_mapping()
        assert len(deployment_mappings) == 2
        assert deployment_mappings[deployment_id] == "weather_tool"
        assert deployment_mappings["deployment-456"] == "Another Tool"

        # Verify both tools are registered in MCP
        registered_tools = await mcp_server.list_tools()
        assert len(registered_tools) == 2
        tool_names = {tool.name for tool in registered_tools}
        assert tool_names == {"weather_tool", "Another Tool"}


class TestDeploymentListing:
    """Tests for deployment listing functionality."""

    @pytest.mark.asyncio
    async def test_get_registered_deployments_with_data(self, mcp_server):
        """Test listing registered deployments when deployments exist."""
        # Setup - add test deployments directly to MCP
        await mcp_server.set_deployment_mapping("deployment-123", "weather_tool")
        await mcp_server.set_deployment_mapping("deployment-456", "another_tool")

        result = await get_registered_tool_deployments()

        expected_mappings = {
            "deployment-123": "weather_tool",
            "deployment-456": "another_tool",
        }
        assert result == expected_mappings

        # Verify MCP internal state consistency
        internal_mappings = await mcp_server.get_deployment_mapping()
        assert internal_mappings == expected_mappings

    @pytest.mark.asyncio
    async def test_get_registered_deployments_when_empty(self, mcp_server):
        """Test listing registered deployments when no deployments exist."""
        result = await get_registered_tool_deployments()

        assert result == {}

        # Verify MCP internal state consistency
        internal_mappings = await mcp_server.get_deployment_mapping()
        assert internal_mappings == {}


class TestDeploymentDeletion:
    """Tests for deployment deletion functionality."""

    @pytest.mark.asyncio
    async def test_delete_existing_deployment_with_logging(
        self,
        deployment_id,
        mock_config,
        mcp_server,
        mock_external_dependencies,
    ):
        """Test successful deletion of existing deployment with proper logging."""
        # Setup - register a real tool first (not just mapping)
        mock_external_dependencies["mock_config"].return_value = mock_config

        # Register the tool properly (this creates both the tool and the mapping)
        await register_tool_for_deployment_id(deployment_id)

        with patch(
            "datarobot_genai.drmcp.core.dynamic_tools.deployment.controllers.logger"
        ) as mock_logger:
            result = await delete_registered_tool_deployment(deployment_id)

        assert result is True

        # Verify deployment was removed from MCP
        remaining_mappings = await mcp_server.get_deployment_mapping()
        assert remaining_mappings == {}
        assert deployment_id not in remaining_mappings

        # Verify logging behavior
        expected_log_msg = f"Deleted tool weather_tool for deployment {deployment_id}"
        mock_logger.info.assert_called_once_with(expected_log_msg)

    @pytest.mark.asyncio
    async def test_delete_nonexistent_deployment(
        self,
        deployment_id,
        mock_config,
        mcp_server,
        mock_external_dependencies,
    ):
        """Test deletion of non-existent deployment returns False and logs appropriately."""
        # Setup - register one tool but try to delete different one
        mock_external_dependencies["mock_config"].return_value = mock_config

        # Register a real tool first
        await register_tool_for_deployment_id(deployment_id)

        with patch(
            "datarobot_genai.drmcp.core.dynamic_tools.deployment.controllers.logger"
        ) as mock_logger:
            result = await delete_registered_tool_deployment("deployment-999")

        assert result is False

        # Verify MCP state unchanged
        unchanged_mappings = await mcp_server.get_deployment_mapping()
        assert unchanged_mappings == {deployment_id: "weather_tool"}

        # Verify debug logging
        expected_debug_msg = "No tool registered for deployment deployment-999"
        mock_logger.debug.assert_called_once_with(expected_debug_msg)

    @pytest.mark.asyncio
    async def test_delete_from_empty_deployments(self, mcp_server):
        """Test deletion when no deployments are registered."""
        result = await delete_registered_tool_deployment("any-deployment")

        assert result is False

        # Verify MCP remains empty
        final_mappings = await mcp_server.get_deployment_mapping()
        assert final_mappings == {}

    @pytest.mark.asyncio
    async def test_delete_multiple_deployments_sequentially(
        self,
        deployment_id,
        mock_config,
        mcp_server,
        mock_external_dependencies,
    ):
        """Test deleting multiple deployments one by one."""
        # Setup external dependencies
        mock_external_dependencies["mock_config"].return_value = mock_config

        # Register first tool
        tool1 = await register_tool_for_deployment_id(deployment_id)
        assert tool1.name == "weather_tool"

        # Setup and register second tool
        mock_config.name = "Another Tool"
        mock_config.description = "Another description"

        tool2 = await register_tool_for_deployment_id("deployment-456")
        assert tool2.name == "Another Tool"

        # Verify initial state
        initial_mappings = await mcp_server.get_deployment_mapping()
        assert len(initial_mappings) == 2
        assert initial_mappings == {
            deployment_id: "weather_tool",
            "deployment-456": "Another Tool",
        }

        # Delete first deployment
        result1 = await delete_registered_tool_deployment(deployment_id)
        assert result1 is True

        # Verify first deletion
        after_first_delete = await mcp_server.get_deployment_mapping()
        assert after_first_delete == {"deployment-456": "Another Tool"}

        # Delete second deployment
        result2 = await delete_registered_tool_deployment("deployment-456")
        assert result2 is True

        # Verify all deployments deleted
        final_mappings = await mcp_server.get_deployment_mapping()
        assert final_mappings == {}


class TestIntegratedWorkflow:
    """Tests for complete workflow integration."""

    @pytest.mark.asyncio
    async def test_complete_deployment_lifecycle(
        self,
        deployment_id,
        mock_config,
        mcp_server,
        mock_external_dependencies,
    ):
        """Test complete workflow: register → list → delete → verify empty."""
        # Setup external dependencies
        mock_external_dependencies["mock_config"].return_value = mock_config

        # Step 1: Register a deployment
        result = await register_tool_for_deployment_id(deployment_id)
        assert isinstance(result, Tool)

        # Step 2: List deployments and verify registration
        deployments = await get_registered_tool_deployments()
        assert deployments == {deployment_id: "weather_tool"}

        # Step 3: Delete the deployment
        deleted = await delete_registered_tool_deployment(deployment_id)
        assert deleted is True

        # Step 4: Verify empty state
        final_deployments = await get_registered_tool_deployments()
        assert final_deployments == {}

        # Verify MCP internal state is clean
        internal_mappings = await mcp_server.get_deployment_mapping()
        assert internal_mappings == {}
