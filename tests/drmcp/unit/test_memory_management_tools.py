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

import json
from datetime import datetime
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from datarobot_genai.drmcp.core.memory_management.manager import MemoryResource
from datarobot_genai.drmcp.core.memory_management.manager import ToolContext
from datarobot_genai.drmcp.core.memory_management.memory_tools import delete_resource
from datarobot_genai.drmcp.core.memory_management.memory_tools import get_resource
from datarobot_genai.drmcp.core.memory_management.memory_tools import list_resources
from datarobot_genai.drmcp.core.memory_management.memory_tools import store_resource


class TestMemoryManagementTools:
    """Test cases for memory management tools."""

    @pytest.mark.asyncio
    @patch("datarobot_genai.drmcp.core.memory_management.memory_tools.get_memory_manager")
    async def test_store_resource_success(self, mock_get_memory_manager):
        """Test store_resource with successful storage."""
        mock_memory_manager = Mock()
        mock_memory_manager.store_resource = AsyncMock(return_value="resource123")
        mock_get_memory_manager.return_value = mock_memory_manager

        result = await store_resource(
            data="test data",
            memory_storage_id="storage123",
            agent_identifier="agent123",
            prompt="test prompt",
            tool_name="test_tool",
            tool_parameters={"param1": "value1"},
            embedding_vector=[0.1, 0.2, 0.3],
        )

        assert result == "Resource stored with ID: resource123"
        mock_memory_manager.store_resource.assert_called_once_with(
            data="test data",
            memory_storage_id="storage123",
            agent_identifier="agent123",
            prompt="test prompt",
            tool_context=ToolContext(name="test_tool", parameters={"param1": "value1"}),
            embedding_vector=[0.1, 0.2, 0.3],
        )

    @pytest.mark.asyncio
    @patch("datarobot_genai.drmcp.core.memory_management.memory_tools.get_memory_manager")
    async def test_store_resource_no_tool_context(self, mock_get_memory_manager):
        """Test store_resource without tool context."""
        mock_memory_manager = Mock()
        mock_memory_manager.store_resource = AsyncMock(return_value="resource123")
        mock_get_memory_manager.return_value = mock_memory_manager

        result = await store_resource(
            data="test data", memory_storage_id="storage123", agent_identifier="agent123"
        )

        assert result == "Resource stored with ID: resource123"
        mock_memory_manager.store_resource.assert_called_once_with(
            data="test data",
            memory_storage_id="storage123",
            agent_identifier="agent123",
            prompt=None,
            tool_context=None,
            embedding_vector=None,
        )

    @pytest.mark.asyncio
    @patch("datarobot_genai.drmcp.core.memory_management.memory_tools.get_memory_manager")
    async def test_store_resource_no_memory_manager(self, mock_get_memory_manager):
        """Test store_resource when memory manager is not initialized."""
        mock_get_memory_manager.return_value = None

        result = await store_resource(data="test data")

        assert result == "Memory manager not initialized"

    @pytest.mark.asyncio
    @patch("datarobot_genai.drmcp.core.memory_management.memory_tools.get_memory_manager")
    async def test_get_resource_success_with_data(self, mock_get_memory_manager):
        """Test get_resource with successful retrieval including data."""
        mock_tool_context = ToolContext(name="test_tool", parameters={"param1": "value1"})
        mock_resource = MemoryResource(
            id="resource123",
            memory_storage_id="storage123",
            prompt="test prompt",
            tool_context=mock_tool_context,
            embedding_vector=[0.1, 0.2, 0.3],
            created_at=datetime.now(),
        )

        mock_memory_manager = Mock()
        mock_memory_manager.get_resource = AsyncMock(return_value=mock_resource)
        mock_memory_manager.get_resource_data = AsyncMock(return_value="test data")
        mock_get_memory_manager.return_value = mock_memory_manager

        result = await get_resource(
            resource_id="resource123",
            memory_storage_id="storage123",
            agent_identifier="agent123",
            include_data=True,
        )

        result_data = json.loads(result)
        assert result_data["id"] == "resource123"
        assert result_data["memory_storage_id"] == "storage123"
        assert result_data["prompt"] == "test prompt"
        assert result_data["tool_context"]["name"] == "test_tool"
        assert result_data["tool_context"]["parameters"]["param1"] == "value1"
        assert result_data["embedding_vector"] == [0.1, 0.2, 0.3]
        assert result_data["data"] == "test data"
        assert "created_at" in result_data

    @pytest.mark.asyncio
    @patch("datarobot_genai.drmcp.core.memory_management.memory_tools.get_memory_manager")
    async def test_get_resource_success_without_data(self, mock_get_memory_manager):
        """Test get_resource with successful retrieval without data."""
        mock_tool_context = ToolContext(name="test_tool", parameters={"param1": "value1"})
        mock_resource = MemoryResource(
            id="resource123",
            memory_storage_id="storage123",
            prompt="test prompt",
            tool_context=mock_tool_context,
            embedding_vector=[0.1, 0.2, 0.3],
            created_at=datetime.now(),
        )

        mock_memory_manager = Mock()
        mock_memory_manager.get_resource = AsyncMock(return_value=mock_resource)
        mock_get_memory_manager.return_value = mock_memory_manager

        result = await get_resource(
            resource_id="resource123",
            memory_storage_id="storage123",
            agent_identifier="agent123",
            include_data=False,
        )

        result_data = json.loads(result)
        assert result_data["id"] == "resource123"
        assert result_data["memory_storage_id"] == "storage123"
        assert result_data["prompt"] == "test prompt"
        assert result_data["tool_context"]["name"] == "test_tool"
        assert result_data["tool_context"]["parameters"]["param1"] == "value1"
        assert result_data["embedding_vector"] == [0.1, 0.2, 0.3]
        assert "data" not in result_data
        assert "created_at" in result_data

    @pytest.mark.asyncio
    @patch("datarobot_genai.drmcp.core.memory_management.memory_tools.get_memory_manager")
    async def test_get_resource_with_binary_data(self, mock_get_memory_manager):
        """Test get_resource with binary data."""
        mock_tool_context = ToolContext(name="test_tool", parameters={"param1": "value1"})
        mock_resource = MemoryResource(
            id="resource123",
            memory_storage_id="storage123",
            prompt="test prompt",
            tool_context=mock_tool_context,
            embedding_vector=[0.1, 0.2, 0.3],
            created_at=datetime.now(),
        )

        mock_memory_manager = Mock()
        mock_memory_manager.get_resource = AsyncMock(return_value=mock_resource)
        mock_memory_manager.get_resource_data = AsyncMock(return_value=b"binary data")
        mock_get_memory_manager.return_value = mock_memory_manager

        result = await get_resource(
            resource_id="resource123",
            memory_storage_id="storage123",
            agent_identifier="agent123",
            include_data=True,
        )

        result_data = json.loads(result)
        assert result_data["data"] == "binary data"  # JSON converts bytes to string

    @pytest.mark.asyncio
    @patch("datarobot_genai.drmcp.core.memory_management.memory_tools.get_memory_manager")
    async def test_get_resource_with_unicode_decode_error(self, mock_get_memory_manager):
        """Test get_resource with unicode decode error for binary data."""
        mock_tool_context = ToolContext(name="test_tool", parameters={"param1": "value1"})
        mock_resource = MemoryResource(
            id="resource123",
            memory_storage_id="storage123",
            prompt="test prompt",
            tool_context=mock_tool_context,
            embedding_vector=[0.1, 0.2, 0.3],
            created_at=datetime.now(),
        )

        mock_memory_manager = Mock()
        mock_memory_manager.get_resource = AsyncMock(return_value=mock_resource)
        mock_memory_manager.get_resource_data = AsyncMock(return_value=b"\x00\x01\x02\x03")
        mock_get_memory_manager.return_value = mock_memory_manager

        result = await get_resource(
            resource_id="resource123",
            memory_storage_id="storage123",
            agent_identifier="agent123",
            include_data=True,
        )

        result_data = json.loads(result)
        # For binary data that can't be decoded, it should be returned as-is
        # JSON serialization will convert it to a string representation
        assert result_data["data"] == "\x00\x01\x02\x03"

    @pytest.mark.asyncio
    @patch("datarobot_genai.drmcp.core.memory_management.memory_tools.get_memory_manager")
    async def test_get_resource_not_found(self, mock_get_memory_manager):
        """Test get_resource when resource is not found."""
        mock_memory_manager = Mock()
        mock_memory_manager.get_resource = AsyncMock(return_value=None)
        mock_get_memory_manager.return_value = mock_memory_manager

        result = await get_resource(
            resource_id="nonexistent", memory_storage_id="storage123", agent_identifier="agent123"
        )

        assert result == "Resource not found"

    @pytest.mark.asyncio
    @patch("datarobot_genai.drmcp.core.memory_management.memory_tools.get_memory_manager")
    async def test_get_resource_no_memory_manager(self, mock_get_memory_manager):
        """Test get_resource when memory manager is not initialized."""
        mock_get_memory_manager.return_value = None

        result = await get_resource(resource_id="resource123")

        assert result == "Memory manager not initialized"

    @pytest.mark.asyncio
    @patch("datarobot_genai.drmcp.core.memory_management.memory_tools.get_memory_manager")
    async def test_list_resources_success(self, mock_get_memory_manager):
        """Test list_resources with successful retrieval."""
        mock_tool_context = ToolContext(name="test_tool", parameters={"param1": "value1"})
        mock_resource1 = MemoryResource(
            id="resource1",
            memory_storage_id="storage123",
            prompt="prompt1",
            tool_context=mock_tool_context,
            embedding_vector=[0.1, 0.2, 0.3],
            created_at=datetime.now(),
        )
        mock_resource2 = MemoryResource(
            id="resource2",
            memory_storage_id="storage123",
            prompt="prompt2",
            tool_context=None,
            embedding_vector=None,
            created_at=datetime.now(),
        )

        mock_memory_manager = Mock()
        mock_memory_manager.list_resources = AsyncMock(
            return_value=[mock_resource1, mock_resource2]
        )
        mock_get_memory_manager.return_value = mock_memory_manager

        result = await list_resources(agent_identifier="agent123", memory_storage_id="storage123")

        result_data = json.loads(result)
        assert len(result_data) == 2

        # Check first resource
        assert result_data[0]["id"] == "resource1"
        assert result_data[0]["memory_storage_id"] == "storage123"
        assert result_data[0]["prompt"] == "prompt1"
        assert result_data[0]["tool_context"]["name"] == "test_tool"
        assert "created_at" in result_data[0]

        # Check second resource
        assert result_data[1]["id"] == "resource2"
        assert result_data[1]["memory_storage_id"] == "storage123"
        assert result_data[1]["prompt"] == "prompt2"
        assert result_data[1]["tool_context"] is None
        assert "created_at" in result_data[1]

    @pytest.mark.asyncio
    @patch("datarobot_genai.drmcp.core.memory_management.memory_tools.get_memory_manager")
    async def test_list_resources_no_resources(self, mock_get_memory_manager):
        """Test list_resources when no resources are found."""
        mock_memory_manager = Mock()
        mock_memory_manager.list_resources = AsyncMock(return_value=[])
        mock_get_memory_manager.return_value = mock_memory_manager

        result = await list_resources(agent_identifier="agent123")

        assert result == "No resources found"

    @pytest.mark.asyncio
    @patch("datarobot_genai.drmcp.core.memory_management.memory_tools.get_memory_manager")
    async def test_list_resources_no_memory_manager(self, mock_get_memory_manager):
        """Test list_resources when memory manager is not initialized."""
        mock_get_memory_manager.return_value = None

        result = await list_resources(agent_identifier="agent123")

        assert result == "Memory manager not initialized"

    @pytest.mark.asyncio
    @patch("datarobot_genai.drmcp.core.memory_management.memory_tools.get_memory_manager")
    async def test_delete_resource_success(self, mock_get_memory_manager):
        """Test delete_resource with successful deletion."""
        mock_memory_manager = Mock()
        mock_memory_manager.delete_resource = AsyncMock(return_value=True)
        mock_get_memory_manager.return_value = mock_memory_manager

        result = await delete_resource(
            resource_id="resource123", memory_storage_id="storage123", agent_identifier="agent123"
        )

        assert result == "Resource resource123 deleted successfully"
        mock_memory_manager.delete_resource.assert_called_once_with(
            resource_id="resource123", memory_storage_id="storage123", agent_identifier="agent123"
        )

    @pytest.mark.asyncio
    @patch("datarobot_genai.drmcp.core.memory_management.memory_tools.get_memory_manager")
    async def test_delete_resource_failure(self, mock_get_memory_manager):
        """Test delete_resource with failed deletion."""
        mock_memory_manager = Mock()
        mock_memory_manager.delete_resource = AsyncMock(return_value=False)
        mock_get_memory_manager.return_value = mock_memory_manager

        result = await delete_resource(
            resource_id="resource123", memory_storage_id="storage123", agent_identifier="agent123"
        )

        assert result == "Failed to delete resource resource123"

    @pytest.mark.asyncio
    @patch("datarobot_genai.drmcp.core.memory_management.memory_tools.get_memory_manager")
    async def test_delete_resource_no_memory_manager(self, mock_get_memory_manager):
        """Test delete_resource when memory manager is not initialized."""
        mock_get_memory_manager.return_value = None

        result = await delete_resource(resource_id="resource123")

        assert result == "Memory manager not initialized"

    @pytest.mark.asyncio
    @patch("datarobot_genai.drmcp.core.memory_management.memory_tools.get_memory_manager")
    async def test_store_resource_with_tool_name_only(self, mock_get_memory_manager):
        """Test store_resource with tool_name but no tool_parameters."""
        mock_memory_manager = Mock()
        mock_memory_manager.store_resource = AsyncMock(return_value="resource123")
        mock_get_memory_manager.return_value = mock_memory_manager

        result = await store_resource(data="test data", tool_name="test_tool")

        assert result == "Resource stored with ID: resource123"
        mock_memory_manager.store_resource.assert_called_once_with(
            data="test data",
            memory_storage_id=None,
            agent_identifier=None,
            prompt=None,
            tool_context=None,  # Should be None when no tool_parameters
            embedding_vector=None,
        )

    @pytest.mark.asyncio
    @patch("datarobot_genai.drmcp.core.memory_management.memory_tools.get_memory_manager")
    async def test_get_resource_with_non_string_data(self, mock_get_memory_manager):
        """Test get_resource with non-string data."""
        mock_tool_context = ToolContext(name="test_tool", parameters={"param1": "value1"})
        mock_resource = MemoryResource(
            id="resource123",
            memory_storage_id="storage123",
            prompt="test prompt",
            tool_context=mock_tool_context,
            embedding_vector=[0.1, 0.2, 0.3],
            created_at=datetime.now(),
        )

        mock_memory_manager = Mock()
        mock_memory_manager.get_resource = AsyncMock(return_value=mock_resource)
        mock_memory_manager.get_resource_data = AsyncMock(return_value={"json": "data"})
        mock_get_memory_manager.return_value = mock_memory_manager

        result = await get_resource(
            resource_id="resource123",
            memory_storage_id="storage123",
            agent_identifier="agent123",
            include_data=True,
        )

        result_data = json.loads(result)
        assert result_data["data"] == {"json": "data"}

    def test_tool_context_creation(self):
        """Test ToolContext creation for edge cases."""
        # Test with empty parameters
        tool_context = ToolContext(name="test_tool", parameters={})
        assert tool_context.name == "test_tool"
        assert tool_context.parameters == {}

        # Test with complex parameters
        complex_params = {
            "string_param": "value",
            "int_param": 123,
            "list_param": [1, 2, 3],
            "dict_param": {"nested": "value"},
        }
        tool_context = ToolContext(name="complex_tool", parameters=complex_params)
        assert tool_context.name == "complex_tool"
        assert tool_context.parameters == complex_params
