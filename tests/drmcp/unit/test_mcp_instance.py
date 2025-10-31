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

from unittest.mock import patch

import pytest
from fastmcp.exceptions import NotFoundError

from datarobot_genai.drmcp.core.mcp_instance import TaggedFastMCP


class TestDataRobotMCPInstanceAdditional:
    """Additional test cases for DataRobotMCP class."""

    @pytest.mark.asyncio
    async def test_set_deployment_mapping_updates_existing(self):
        """Test that set_deployment_mapping updates existing mapping and removes old tool."""
        mcp = TaggedFastMCP()
        mcp._deployments_map = {"deployment1": "old_tool"}

        with patch.object(mcp, "remove_tool") as mock_remove_tool:
            await mcp.set_deployment_mapping("deployment1", "new_tool")

            assert mcp._deployments_map["deployment1"] == "new_tool"
            mock_remove_tool.assert_called_once_with("old_tool")

    @pytest.mark.asyncio
    async def test_set_deployment_mapping_handles_remove_tool_not_found(self):
        """Test that set_deployment_mapping handles NotFoundError when removing old tool."""
        mcp = TaggedFastMCP()
        mcp._deployments_map = {"deployment1": "old_tool"}

        with patch.object(mcp, "remove_tool", side_effect=NotFoundError("Tool not found")):
            # Should not raise an exception
            await mcp.set_deployment_mapping("deployment1", "new_tool")

            assert mcp._deployments_map["deployment1"] == "new_tool"

    @pytest.mark.asyncio
    async def test_set_deployment_mapping_new_deployment(self):
        """Test that set_deployment_mapping works for new deployment."""
        mcp = TaggedFastMCP()
        mcp._deployments_map = {}

        await mcp.set_deployment_mapping("deployment1", "new_tool")

        assert mcp._deployments_map["deployment1"] == "new_tool"

    @pytest.mark.asyncio
    async def test_set_deployment_mapping_same_tool(self):
        """Test that set_deployment_mapping works when mapping to same tool."""
        mcp = TaggedFastMCP()
        mcp._deployments_map = {"deployment1": "existing_tool"}

        with patch.object(mcp, "remove_tool") as mock_remove_tool:
            await mcp.set_deployment_mapping("deployment1", "existing_tool")

            assert mcp._deployments_map["deployment1"] == "existing_tool"
            # Should not call remove_tool when mapping to same tool
            mock_remove_tool.assert_not_called()

    @pytest.mark.asyncio
    async def test_remove_deployment_mapping_existing(self):
        """Test that remove_deployment_mapping removes existing mapping."""
        mcp = TaggedFastMCP()
        mcp._deployments_map = {"deployment1": "tool1", "deployment2": "tool2"}

        await mcp.remove_deployment_mapping("deployment1")

        assert "deployment1" not in mcp._deployments_map
        assert "deployment2" in mcp._deployments_map

    @pytest.mark.asyncio
    async def test_remove_deployment_mapping_nonexistent(self):
        """Test that remove_deployment_mapping handles nonexistent deployment."""
        mcp = TaggedFastMCP()
        mcp._deployments_map = {"deployment1": "tool1"}

        # Should not raise an exception
        await mcp.remove_deployment_mapping("nonexistent")

        assert mcp._deployments_map == {"deployment1": "tool1"}

    @pytest.mark.asyncio
    async def test_remove_deployment_mapping_empty_map(self):
        """Test that remove_deployment_mapping works with empty map."""
        mcp = TaggedFastMCP()
        mcp._deployments_map = {}

        # Should not raise an exception
        await mcp.remove_deployment_mapping("deployment1")

        assert mcp._deployments_map == {}

    @pytest.mark.asyncio
    @patch("datarobot_genai.drmcp.core.mcp_instance.logger")
    async def test_set_deployment_mapping_logs_debug_message(self, mock_logger):
        """Test that set_deployment_mapping logs debug message when updating existing mapping."""
        mcp = TaggedFastMCP()
        mcp._deployments_map = {"deployment1": "old_tool"}

        with patch.object(mcp, "remove_tool"):
            await mcp.set_deployment_mapping("deployment1", "new_tool")

            mock_logger.debug.assert_called_with(
                "Deployment ID deployment1 already mapped to old_tool, updating to new_tool"
            )

    @pytest.mark.asyncio
    @patch("datarobot_genai.drmcp.core.mcp_instance.logger")
    async def test_set_deployment_mapping_logs_remove_tool_not_found(self, mock_logger):
        """Test that set_deployment_mapping logs debug message when remove_tool raises NotFoundError."""  # noqa: E501
        mcp = TaggedFastMCP()
        mcp._deployments_map = {"deployment1": "old_tool"}

        with patch.object(mcp, "remove_tool", side_effect=NotFoundError("Tool not found")):
            await mcp.set_deployment_mapping("deployment1", "new_tool")

            mock_logger.debug.assert_called_with(
                "Tool old_tool not found in registry, skipping removal"
            )
