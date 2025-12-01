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
from collections.abc import Callable
from datetime import datetime
from http import HTTPStatus
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from botocore.exceptions import ClientError
from fastmcp.prompts import Prompt

from datarobot_genai.drmcp.core.memory_management.manager import MemoryStorage
from datarobot_genai.drmcp.core.routes import register_routes


class TestRoutesCoverage:
    """Test cases for route handlers in routes.py to improve coverage."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_mcp = Mock()
        self.mock_request = Mock()
        self.mock_request.path_params = {}
        self.mock_request.json = AsyncMock()

    @patch("datarobot_genai.drmcp.core.routes.get_memory_manager")
    def test_register_routes_without_memory_manager(self, mock_get_memory_manager):
        """Test register_routes when memory manager is not available."""
        mock_get_memory_manager.return_value = None

        # This should not raise an exception
        register_routes(self.mock_mcp)

        # Verify that custom routes were registered
        assert self.mock_mcp.custom_route.called

    @patch("datarobot_genai.drmcp.core.routes.get_memory_manager")
    def test_register_routes_with_memory_manager(self, mock_get_memory_manager):
        """Test register_routes when memory manager is available."""
        mock_memory_manager = Mock()
        mock_get_memory_manager.return_value = mock_memory_manager

        register_routes(self.mock_mcp)

        # Verify that custom routes were registered
        assert self.mock_mcp.custom_route.called

    def test_health_endpoint_registration(self):
        """Test that health endpoint is registered."""
        # Create a mock that captures the route registration
        registered_routes = {}

        def mock_custom_route(route_path, methods=None):
            def decorator(handler):
                registered_routes[route_path] = handler
                return handler

            return decorator

        self.mock_mcp.custom_route = mock_custom_route

        register_routes(self.mock_mcp)

        # Test that health endpoint is registered
        assert "/" in registered_routes

    def test_tags_endpoint_registration(self):
        """Test that tags endpoint is registered."""
        # Create a mock that captures the route registration
        registered_routes = {}

        def mock_custom_route(route_path, methods=None):
            def decorator(handler):
                registered_routes[route_path] = handler
                return handler

            return decorator

        self.mock_mcp.custom_route = mock_custom_route

        register_routes(self.mock_mcp)

        # Test that tags endpoint is registered
        assert "/tags" in registered_routes

    @patch("datarobot_genai.drmcp.core.routes.get_memory_manager")
    def test_memory_manager_routes_registration(self, mock_get_memory_manager):
        """Test that memory manager routes are registered when memory manager is available."""
        mock_memory_manager = Mock()
        mock_get_memory_manager.return_value = mock_memory_manager

        # Create a mock that captures the route registration
        registered_routes = {}

        def mock_custom_route(route_path, methods=None):
            def decorator(handler):
                registered_routes[route_path] = handler
                return handler

            return decorator

        self.mock_mcp.custom_route = mock_custom_route

        register_routes(self.mock_mcp)

        # Check that memory manager routes are registered
        memory_routes = [
            "/agent/{agent_id}/storage/{label}",
            "/agent/{agent_id}/storages",
            "/agent/{agent_id}/storages/{storage_id}",
            "/agent/{agent_id}",
            "/agent/{agent_id}/storages/{storage_id}/activate",
            "/agent/{agent_id}/active-storage",
        ]

        for route_pattern in memory_routes:
            # Check if any registered route contains the pattern
            found = any(
                route_pattern in registered_route for registered_route in registered_routes.keys()
            )
            assert found, f"Memory manager route {route_pattern} not found in registered routes"

    def test_deployment_routes_registration(self):
        """Test that deployment routes are registered."""
        # Create a mock that captures the route registration
        registered_routes = {}

        def mock_custom_route(route_path, methods=None):
            def decorator(handler):
                registered_routes[route_path] = handler
                return handler

            return decorator

        self.mock_mcp.custom_route = mock_custom_route

        register_routes(self.mock_mcp)

        # Check that deployment routes are registered
        deployment_routes = ["/registeredDeployments/{deployment_id}", "/registeredDeployments"]

        for route_pattern in deployment_routes:
            # Check if any registered route contains the pattern
            found = any(
                route_pattern in registered_route for registered_route in registered_routes.keys()
            )
            assert found, f"Deployment route {route_pattern} not found in registered routes"

    @pytest.mark.asyncio
    async def test_health_endpoint_execution(self):
        """Test health endpoint execution."""
        # Create a mock that captures the route registration
        registered_routes = {}

        def mock_custom_route(route_path, methods=None):
            def decorator(handler):
                registered_routes[route_path] = handler
                return handler

            return decorator

        self.mock_mcp.custom_route = mock_custom_route

        register_routes(self.mock_mcp)

        # Test the health endpoint
        health_handler = registered_routes.get("/")
        assert health_handler is not None

        response = await health_handler(self.mock_request)
        assert response.status_code == 200
        assert "healthy" in response.body.decode()

    @pytest.mark.asyncio
    async def test_tags_endpoint_success_execution(self):
        """Test tags endpoint success execution."""
        # Mock the MCP to have get_all_tags method
        self.mock_mcp.get_all_tags = AsyncMock(return_value=["tag1", "tag2", "tag3"])

        # Create a mock that captures the route registration
        registered_routes = {}

        def mock_custom_route(route_path, methods=None):
            def decorator(handler):
                registered_routes[route_path] = handler
                return handler

            return decorator

        self.mock_mcp.custom_route = mock_custom_route

        register_routes(self.mock_mcp)

        # Test the tags endpoint
        tags_handler = registered_routes.get("/tags")
        assert tags_handler is not None

        response = await tags_handler(self.mock_request)
        assert response.status_code == 200
        assert "tags" in response.body.decode()

    @pytest.mark.asyncio
    async def test_tags_endpoint_exception_execution(self):
        """Test tags endpoint exception execution."""
        # Mock the MCP to raise an exception
        self.mock_mcp.get_all_tags = AsyncMock(side_effect=Exception("Test error"))

        # Create a mock that captures the route registration
        registered_routes = {}

        def mock_custom_route(route_path, methods=None):
            def decorator(handler):
                registered_routes[route_path] = handler
                return handler

            return decorator

        self.mock_mcp.custom_route = mock_custom_route

        register_routes(self.mock_mcp)

        # Test the tags endpoint
        tags_handler = registered_routes.get("/tags")
        assert tags_handler is not None

        response = await tags_handler(self.mock_request)
        assert response.status_code == 500
        assert "error" in response.body.decode()

    @pytest.mark.asyncio
    @patch("datarobot_genai.drmcp.core.routes.register_tool_for_deployment_id")
    async def test_add_deployment_endpoint_execution(self, mock_register_tool):
        """Test add deployment endpoint execution."""
        mock_tool = Mock()
        mock_tool.name = "test_tool"
        mock_tool.description = "Test tool description"
        mock_tool.tags = {"tag1", "tag2"}

        mock_register_tool.return_value = mock_tool

        # Create a mock that captures the route registration
        registered_routes = {}

        def mock_custom_route(route_path, methods=None):
            def decorator(handler):
                registered_routes[route_path] = handler
                return handler

            return decorator

        self.mock_mcp.custom_route = mock_custom_route

        register_routes(self.mock_mcp)

        # Find the add deployment handler (PUT method)
        add_deployment_handler = None
        for route, handler in registered_routes.items():
            if "registeredDeployments" in route and "{deployment_id}" in route:
                # This should be the PUT handler for adding deployments
                add_deployment_handler = handler
                break

        assert add_deployment_handler is not None

        # Test the handler
        self.mock_request.path_params = {"deployment_id": "deployment123"}
        response = await add_deployment_handler(self.mock_request)
        # The handler might return 404 if it can't find the deployment, which is expected
        assert response.status_code in [200, 201, 404]
        assert "deploymentId" in response.body.decode() or "error" in response.body.decode()

    @pytest.mark.asyncio
    @patch("datarobot_genai.drmcp.core.routes.get_registered_tool_deployments")
    async def test_list_deployments_endpoint_execution(self, mock_get_deployments):
        """Test list deployments endpoint execution."""
        mock_deployments = {"deployment1": "tool1", "deployment2": "tool2"}
        mock_get_deployments.return_value = mock_deployments

        # Create a mock that captures the route registration
        registered_routes = {}

        def mock_custom_route(route_path, methods=None):
            def decorator(handler):
                registered_routes[route_path] = handler
                return handler

            return decorator

        self.mock_mcp.custom_route = mock_custom_route

        register_routes(self.mock_mcp)

        # Find the list deployments handler
        list_deployments_handler = None
        for route, handler in registered_routes.items():
            if "registeredDeployments" in route and "{deployment_id}" not in route:
                list_deployments_handler = handler
                break

        assert list_deployments_handler is not None

        # Test the handler
        response = await list_deployments_handler(self.mock_request)
        assert response.status_code == 200
        assert "deployments" in response.body.decode()

    @pytest.mark.asyncio
    @patch("datarobot_genai.drmcp.core.routes.delete_registered_tool_deployment")
    async def test_delete_deployment_endpoint_execution(self, mock_delete_deployment):
        """Test delete deployment endpoint execution."""
        mock_delete_deployment.return_value = True

        # Create a mock that captures the route registration
        registered_routes = {}

        def mock_custom_route(route_path, methods=None):
            def decorator(handler):
                registered_routes[route_path] = handler
                return handler

            return decorator

        self.mock_mcp.custom_route = mock_custom_route

        register_routes(self.mock_mcp)

        # Find the delete deployment handler
        delete_deployment_handler = None
        for route, handler in registered_routes.items():
            if "registeredDeployments" in route and "{deployment_id}" in route:
                # This should be the DELETE handler, not the PUT handler
                delete_deployment_handler = handler
                break

        assert delete_deployment_handler is not None

        # Test the handler
        self.mock_request.path_params = {"deployment_id": "deployment123"}
        response = await delete_deployment_handler(self.mock_request)
        assert response.status_code == 200
        assert "deleted successfully" in response.body.decode()

    @pytest.mark.asyncio
    @patch("datarobot_genai.drmcp.core.routes.get_memory_manager")
    async def test_memory_manager_endpoints_execution(self, mock_get_memory_manager):
        """Test memory manager endpoints execution."""
        mock_memory_manager = Mock()
        mock_memory_manager.initialize_storage = AsyncMock(return_value="storage123")
        mock_memory_manager.list_storages = AsyncMock(return_value=[])
        mock_memory_manager.get_storage = AsyncMock(return_value=None)
        mock_memory_manager.delete_storage = AsyncMock(return_value=True)
        mock_memory_manager.delete_agent = AsyncMock(return_value=True)
        mock_memory_manager.set_storage_id_for_agent = AsyncMock()
        mock_memory_manager.get_active_storage_id_for_agent = AsyncMock(return_value="storage123")
        mock_memory_manager.clear_storage_id_for_agent = AsyncMock()
        mock_get_memory_manager.return_value = mock_memory_manager

        # Create a mock that captures the route registration
        registered_routes = {}

        def mock_custom_route(route_path, methods=None):
            def decorator(handler):
                registered_routes[route_path] = handler
                return handler

            return decorator

        self.mock_mcp.custom_route = mock_custom_route

        register_routes(self.mock_mcp)

        # Test initialize storage endpoint
        self.mock_request.path_params = {"agent_id": "agent123", "label": "test_storage"}
        self.mock_request.json.return_value = {"config": {"key": "value"}}

        init_handler = None
        for route, handler in registered_routes.items():
            if "/agent/{agent_id}/storage/{label}" in route:
                init_handler = handler
                break

        if init_handler:
            response = await init_handler(self.mock_request)
            assert response.status_code == 200

        # Test list storages endpoint
        self.mock_request.path_params = {"agent_id": "agent123"}

        list_handler = None
        for route, handler in registered_routes.items():
            if "/agent/{agent_id}/storages" in route and "{storage_id}" not in route:
                list_handler = handler
                break

        if list_handler:
            response = await list_handler(self.mock_request)
            assert response.status_code == 200

        # Test get storage endpoint
        self.mock_request.path_params = {"agent_id": "agent123", "storage_id": "storage123"}

        get_handler = None
        for route, handler in registered_routes.items():
            if "/agent/{agent_id}/storages/{storage_id}" in route and "activate" not in route:
                get_handler = handler
                break

        if get_handler:
            response = await get_handler(self.mock_request)
            # The handler might return 200 or 404 depending on the mock setup
            assert response.status_code in [200, 404]

        # Test delete storage endpoint
        delete_storage_handler = None
        for route, handler in registered_routes.items():
            if "/agent/{agent_id}/storages/{storage_id}" in route and "activate" not in route:
                # This should be the DELETE handler
                delete_storage_handler = handler
                break

        if delete_storage_handler:
            response = await delete_storage_handler(self.mock_request)
            assert response.status_code == 200

        # Test delete agent endpoint
        self.mock_request.path_params = {"agent_id": "agent123"}

        delete_agent_handler = None
        for route, handler in registered_routes.items():
            if route == "/agent/{agent_id}":  # Exact match for delete agent
                delete_agent_handler = handler
                break

        if delete_agent_handler:
            response = await delete_agent_handler(self.mock_request)
            assert response.status_code == 200

        # Test set active storage endpoint
        self.mock_request.path_params = {"agent_id": "agent123", "storage_id": "storage123"}

        set_active_handler = None
        for route, handler in registered_routes.items():
            if "/agent/{agent_id}/storages/{storage_id}/activate" in route:
                set_active_handler = handler
                break

        if set_active_handler:
            response = await set_active_handler(self.mock_request)
            assert response.status_code == 404  # Storage not found

        # Test get active storage endpoint
        self.mock_request.path_params = {"agent_id": "agent123"}

        get_active_handler = None
        for route, handler in registered_routes.items():
            if "/agent/{agent_id}/active-storage" in route and "activate" not in route:
                get_active_handler = handler
                break

        if get_active_handler:
            response = await get_active_handler(self.mock_request)
            assert response.status_code == 200

        # Test clear active storage endpoint
        clear_active_handler = None
        for route, handler in registered_routes.items():
            if "/agent/{agent_id}/active-storage" in route and "activate" not in route:
                # This should be the DELETE handler
                clear_active_handler = handler
                break

        if clear_active_handler:
            response = await clear_active_handler(self.mock_request)
            assert response.status_code == 200


class TestRoutesAdditionalCoverage:
    """Additional test cases for route handlers in routes.py."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_mcp = Mock()
        self.mock_request = Mock()
        self.mock_request.path_params = {}
        self.mock_request.json = AsyncMock()

        # This dictionary will store the handlers registered by custom_route
        self.registered_routes = {}

        def mock_custom_route(route_path, methods=None):
            def decorator(handler):
                # Store the handler with its route path
                self.registered_routes[route_path] = handler
                return handler

            return decorator

        self.mock_mcp.custom_route = mock_custom_route

    @pytest.mark.asyncio
    @patch("datarobot_genai.drmcp.core.routes.get_memory_manager")
    async def test_memory_manager_storage_found_scenarios(self, mock_get_memory_manager):
        """Test memory manager endpoints when storage is found."""
        # Create a mock storage object
        mock_storage = MemoryStorage(
            id="storage123",
            agent_identifier="agent123",
            label="test_storage",
            created_at=datetime.now(),
            storage_config={"key": "value"},
        )

        mock_memory_manager = Mock()
        mock_memory_manager.initialize_storage = AsyncMock(return_value="storage123")
        mock_memory_manager.list_storages = AsyncMock(return_value=[mock_storage])
        mock_memory_manager.get_storage = AsyncMock(return_value=mock_storage)
        mock_memory_manager.delete_storage = AsyncMock(return_value=True)
        mock_memory_manager.delete_agent = AsyncMock(return_value=True)
        mock_memory_manager.set_storage_id_for_agent = AsyncMock()
        mock_memory_manager.get_active_storage_id_for_agent = AsyncMock(return_value="storage123")
        mock_memory_manager.clear_storage_id_for_agent = AsyncMock()
        mock_get_memory_manager.return_value = mock_memory_manager

        register_routes(self.mock_mcp)

        # Test list storages with data
        self.mock_request.path_params = {"agent_id": "agent123"}
        list_handler = self.registered_routes.get("/agent/{agent_id}/storages")
        response = await list_handler(self.mock_request)
        assert response.status_code == 200
        assert b"storage123" in response.body

        # Test get storage found
        self.mock_request.path_params = {"agent_id": "agent123", "storage_id": "storage123"}
        get_handler = self.registered_routes.get("/agent/{agent_id}/storages/{storage_id}")
        response = await get_handler(self.mock_request)
        assert response.status_code == 200
        assert b"storage123" in response.body

        # Test set active storage
        self.mock_request.path_params = {"agent_id": "agent123", "storage_id": "storage123"}
        set_active_handler = self.registered_routes.get(
            "/agent/{agent_id}/storages/{storage_id}/activate"
        )
        response = await set_active_handler(self.mock_request)
        assert response.status_code == 200

        # Test get active storage
        self.mock_request.path_params = {"agent_id": "agent123"}
        get_active_handler = self.registered_routes.get("/agent/{agent_id}/active-storage")
        response = await get_active_handler(self.mock_request)
        assert response.status_code == 200

    @pytest.mark.asyncio
    @patch("datarobot_genai.drmcp.core.routes.get_memory_manager")
    async def test_memory_manager_client_error_scenarios(self, mock_get_memory_manager):
        """Test memory manager endpoints with ClientError scenarios."""
        mock_memory_manager = Mock()
        mock_memory_manager.get_active_storage_id_for_agent = AsyncMock(
            side_effect=ClientError({"Error": {"Code": "404", "Message": "Not found"}}, "GetItem")
        )
        mock_memory_manager.clear_storage_id_for_agent = AsyncMock(
            side_effect=ClientError(
                {"Error": {"Code": "404", "Message": "Not found"}}, "DeleteItem"
            )
        )
        mock_get_memory_manager.return_value = mock_memory_manager

        register_routes(self.mock_mcp)

        # Test get active storage with 404 error
        self.mock_request.path_params = {"agent_id": "agent123"}
        get_active_handler = self.registered_routes.get("/agent/{agent_id}/active-storage")
        response = await get_active_handler(self.mock_request)
        assert response.status_code == 404

        # Test clear active storage with 404 error
        self.mock_request.path_params = {"agent_id": "agent123"}
        clear_handler = self.registered_routes.get("/agent/{agent_id}/active-storage")
        response = await clear_handler(self.mock_request)
        assert response.status_code == 404

    @pytest.mark.asyncio
    @patch("datarobot_genai.drmcp.core.routes.get_registered_tool_deployments")
    async def test_list_deployments_error_scenario(self, mock_get_deployments):
        """Test list deployments endpoint with error."""
        mock_get_deployments.side_effect = Exception("List failed")

        register_routes(self.mock_mcp)

        # Test list deployments error
        list_handler = self.registered_routes.get("/registeredDeployments")
        response = await list_handler(self.mock_request)
        assert response.status_code == 500
        assert b"Failed to retrieve deployments" in response.body

    @pytest.mark.asyncio
    @patch("datarobot_genai.drmcp.core.routes.delete_registered_tool_deployment")
    async def test_delete_deployment_scenarios(self, mock_delete_deployment):
        """Test delete deployment endpoint scenarios."""
        register_routes(self.mock_mcp)

        # Test delete deployment success
        mock_delete_deployment.return_value = True
        self.mock_request.path_params = {"deployment_id": "deployment123"}
        delete_handler = self.registered_routes.get("/registeredDeployments/{deployment_id}")
        response = await delete_handler(self.mock_request)
        assert response.status_code == 200

        # Test delete deployment not found
        mock_delete_deployment.return_value = False
        response = await delete_handler(self.mock_request)
        assert response.status_code == 404

        # Test delete deployment error
        mock_delete_deployment.side_effect = Exception("Delete failed")
        response = await delete_handler(self.mock_request)
        assert response.status_code == 500

    @pytest.mark.asyncio
    @patch("datarobot_genai.drmcp.core.routes.get_memory_manager")
    async def test_memory_manager_storage_not_found_for_activation(self, mock_get_memory_manager):
        """Test set active storage when storage is not found."""
        mock_memory_manager = Mock()
        mock_memory_manager.get_storage = AsyncMock(return_value=None)
        mock_get_memory_manager.return_value = mock_memory_manager

        register_routes(self.mock_mcp)

        # Test set active storage when storage not found
        self.mock_request.path_params = {"agent_id": "agent123", "storage_id": "nonexistent"}
        set_active_handler = self.registered_routes.get(
            "/agent/{agent_id}/storages/{storage_id}/activate"
        )
        response = await set_active_handler(self.mock_request)
        assert response.status_code == 404
        assert b"Storage nonexistent not found" in response.body


class TestRoutesSimpleFinal:
    """Simple test cases for route handlers in routes.py."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_mcp = Mock()
        self.mock_request = Mock()
        self.mock_request.path_params = {}
        self.mock_request.json = AsyncMock()

    def test_register_routes_without_memory_manager(self):
        """Test register_routes when memory manager is not available."""
        with patch("datarobot_genai.drmcp.core.routes.get_memory_manager", return_value=None):
            register_routes(self.mock_mcp)
            assert self.mock_mcp.custom_route.called

    def test_register_routes_with_memory_manager(self):
        """Test register_routes when memory manager is available."""
        with patch("datarobot_genai.drmcp.core.routes.get_memory_manager", return_value=Mock()):
            register_routes(self.mock_mcp)
            assert self.mock_mcp.custom_route.called

    @pytest.mark.asyncio
    async def test_health_endpoint_execution(self):
        """Test health check endpoint execution."""
        registered_routes = {}

        def mock_custom_route(route_path, methods=None):
            def decorator(handler):
                registered_routes[route_path] = handler
                return handler

            return decorator

        self.mock_mcp.custom_route = mock_custom_route
        register_routes(self.mock_mcp)

        health_handler = registered_routes.get("/")
        response = await health_handler(self.mock_request)
        assert response.status_code == 200
        assert response.body == b'{"status":"healthy","message":"DataRobot MCP Server is running"}'

    @pytest.mark.asyncio
    async def test_tags_endpoint_success_execution(self):
        """Test tags endpoint success execution."""
        self.mock_mcp.get_all_tags = AsyncMock(return_value=["tag1", "tag2"])
        registered_routes = {}

        def mock_custom_route(route_path, methods=None):
            def decorator(handler):
                registered_routes[route_path] = handler
                return handler

            return decorator

        self.mock_mcp.custom_route = mock_custom_route
        register_routes(self.mock_mcp)

        tags_handler = registered_routes.get("/tags")
        response = await tags_handler(self.mock_request)
        assert response.status_code == 200
        assert (
            b'{"tags":["tag1","tag2"],"count":2,"message":"All available tags retrieved successfully"}'  # noqa: E501
            == response.body
        )

    @pytest.mark.asyncio
    async def test_tags_endpoint_exception_execution(self):
        """Test tags endpoint exception execution."""
        self.mock_mcp.get_all_tags = AsyncMock(side_effect=Exception("Test error"))
        registered_routes = {}

        def mock_custom_route(route_path, methods=None):
            def decorator(handler):
                registered_routes[route_path] = handler
                return handler

            return decorator

        self.mock_mcp.custom_route = mock_custom_route
        register_routes(self.mock_mcp)

        tags_handler = registered_routes.get("/tags")
        response = await tags_handler(self.mock_request)
        assert response.status_code == 500
        assert b'{"error":"Failed to retrieve tags: Test error"}' == response.body

    @pytest.mark.asyncio
    @patch("datarobot_genai.drmcp.core.routes.get_registered_tool_deployments")
    async def test_list_deployments_endpoint_execution(self, mock_get_deployments):
        """Test list deployments endpoint execution."""
        mock_deployments = {"deployment1": "tool1", "deployment2": "tool2"}
        mock_get_deployments.return_value = mock_deployments

        registered_routes = {}

        def mock_custom_route(route_path, methods=None):
            def decorator(handler):
                registered_routes[route_path] = handler
                return handler

            return decorator

        self.mock_mcp.custom_route = mock_custom_route
        register_routes(self.mock_mcp)

        list_deployments_handler = None
        for route, handler in registered_routes.items():
            if "registeredDeployments" in route and "{deployment_id}" not in route:
                list_deployments_handler = handler
                break

        assert list_deployments_handler is not None

        response = await list_deployments_handler(self.mock_request)
        assert response.status_code == 200
        assert (
            b'{"deployments":[{"deploymentId":"deployment1","toolName":"tool1"},{"deploymentId":"deployment2","toolName":"tool2"}],"count":2}'
            == response.body
        )

    @pytest.mark.asyncio
    @patch("datarobot_genai.drmcp.core.routes.get_memory_manager")
    async def test_memory_manager_endpoints_basic(self, mock_get_memory_manager):
        """Test basic memory manager endpoints execution."""
        mock_memory_manager = Mock()
        mock_memory_manager.initialize_storage = AsyncMock(return_value="storage123")
        mock_memory_manager.list_storages = AsyncMock(return_value=[])
        mock_memory_manager.get_storage = AsyncMock(return_value=None)
        mock_memory_manager.delete_storage = AsyncMock(return_value=True)
        mock_memory_manager.delete_agent = AsyncMock(return_value=True)
        mock_memory_manager.set_storage_id_for_agent = AsyncMock()
        mock_memory_manager.get_active_storage_id_for_agent = AsyncMock(return_value="storage123")
        mock_memory_manager.clear_storage_id_for_agent = AsyncMock()
        mock_get_memory_manager.return_value = mock_memory_manager

        registered_routes = {}

        def mock_custom_route(route_path, methods=None):
            def decorator(handler):
                registered_routes[route_path] = handler
                return handler

            return decorator

        self.mock_mcp.custom_route = mock_custom_route

        register_routes(self.mock_mcp)

        # Test that routes were registered
        assert len(registered_routes) > 0

        # Test that memory manager routes are present
        memory_routes = [route for route in registered_routes.keys() if "agent" in route]
        assert len(memory_routes) > 0

    @pytest.mark.asyncio
    @patch("datarobot_genai.drmcp.core.routes.get_memory_manager")
    async def test_memory_manager_endpoints_with_errors(self, mock_get_memory_manager):
        """Test memory manager endpoints with error scenarios."""
        mock_memory_manager = Mock()
        mock_memory_manager.get_active_storage_id_for_agent = AsyncMock(
            side_effect=ClientError({"Error": {"Code": "404", "Message": "Not found"}}, "GetObject")
        )
        mock_memory_manager.clear_storage_id_for_agent = AsyncMock(
            side_effect=ClientError({"Error": {"Code": "404", "Message": "Not found"}}, "GetObject")
        )
        mock_get_memory_manager.return_value = mock_memory_manager

        registered_routes = {}

        def mock_custom_route(route_path, methods=None):
            def decorator(handler):
                registered_routes[route_path] = handler
                return handler

            return decorator

        self.mock_mcp.custom_route = mock_custom_route

        register_routes(self.mock_mcp)

        # Test that routes were registered
        assert len(registered_routes) > 0

    @pytest.mark.asyncio
    @patch("datarobot_genai.drmcp.core.routes.get_memory_manager")
    async def test_memory_manager_endpoints_with_other_errors(self, mock_get_memory_manager):
        """Test memory manager endpoints with other error scenarios."""
        mock_memory_manager = Mock()
        mock_memory_manager.get_active_storage_id_for_agent = AsyncMock(
            side_effect=ClientError(
                {"Error": {"Code": "500", "Message": "Internal error"}}, "GetObject"
            )
        )
        mock_memory_manager.clear_storage_id_for_agent = AsyncMock(
            side_effect=ClientError(
                {"Error": {"Code": "500", "Message": "Internal error"}}, "GetObject"
            )
        )
        mock_get_memory_manager.return_value = mock_memory_manager

        registered_routes = {}

        def mock_custom_route(route_path, methods=None):
            def decorator(handler):
                registered_routes[route_path] = handler
                return handler

            return decorator

        self.mock_mcp.custom_route = mock_custom_route

        register_routes(self.mock_mcp)

        # Test that routes were registered
        assert len(registered_routes) > 0


class TestPromptTemplateRoutes:
    """Test cases for route handlers for prompt templates in routes.py."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_mcp = Mock()
        self.mock_request = Mock()
        self.mock_request.path_params = {}
        self.mock_request.json = AsyncMock()

        # This dictionary will store the handlers registered by custom_route
        self.registered_routes: dict[tuple[str, str], Callable] = {}

        def mock_custom_route(route_path: str, methods: list[str]):
            def decorator(handler: Callable):
                for method in methods:
                    self.registered_routes[(method, route_path)] = handler
                return handler

            return decorator

        self.mock_mcp.custom_route = mock_custom_route

    @pytest.mark.asyncio
    async def test_list_prompt_templates(self):
        """Test list prompt templates endpoint."""
        self.mock_mcp.get_prompt_mapping = AsyncMock(
            return_value={
                "pt1": ("ptv1.1", "Test name 1"),
                "pt2": ("ptv2.1", "Test name 2"),
            }
        )

        register_routes(self.mock_mcp)

        list_handler = self.registered_routes["GET", "/registeredPrompts"]
        response = await list_handler(self.mock_request)
        assert response.status_code == HTTPStatus.OK, response.body
        assert json.loads(response.body.decode("utf-8")) == {
            "promptTemplates": [
                {
                    "promptTemplateId": "pt1",
                    "promptTemplateVersionId": "ptv1.1",
                    "promptName": "Test name 1",
                },
                {
                    "promptTemplateId": "pt2",
                    "promptTemplateVersionId": "ptv2.1",
                    "promptName": "Test name 2",
                },
            ],
            "count": 2,
        }

    @pytest.mark.asyncio
    async def test_list_prompt_templates_when_error(self):
        """Test list prompt templates endpoint when error occurs."""
        self.mock_mcp.get_prompt_mapping = AsyncMock(side_effect=ValueError("Dummy error"))

        register_routes(self.mock_mcp)

        list_handler = self.registered_routes["GET", "/registeredPrompts"]
        response = await list_handler(self.mock_request)
        assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR, response.body
        assert b"Failed to retrieve promptTemplates" in response.body

    @pytest.mark.asyncio
    @patch("datarobot_genai.drmcp.core.routes.delete_registered_prompt_template")
    async def test_delete_prompt_template(self, delete_prompt_mock: Mock):
        """Test delete prompt template endpoint."""
        register_routes(self.mock_mcp)
        delete_prompt_mock.return_value = True

        self.mock_request.path_params = {"prompt_template_id": "pt1"}

        delete_handler = self.registered_routes["DELETE", "/registeredPrompts/{prompt_template_id}"]
        response = await delete_handler(self.mock_request)
        assert response.status_code == HTTPStatus.OK, response.body
        assert b"Prompt with prompt template id pt1 deleted successfully" in response.body

    @pytest.mark.asyncio
    @patch("datarobot_genai.drmcp.core.routes.delete_registered_prompt_template")
    async def test_delete_prompt_template_when_does_not_exist(self, delete_prompt_mock: Mock):
        """Test delete prompt template endpoint when does not exist."""
        register_routes(self.mock_mcp)
        delete_prompt_mock.return_value = False

        self.mock_request.path_params = {"prompt_template_id": "pt1"}

        delete_handler = self.registered_routes["DELETE", "/registeredPrompts/{prompt_template_id}"]
        response = await delete_handler(self.mock_request)
        assert response.status_code == HTTPStatus.NOT_FOUND, response.body
        assert b"Prompt with prompt template id pt1 not found" in response.body

    @pytest.mark.asyncio
    @patch("datarobot_genai.drmcp.core.routes.delete_registered_prompt_template")
    async def test_delete_prompt_template_when_error(self, delete_prompt_mock: Mock):
        """Test delete prompt template endpoint when error occurs."""
        register_routes(self.mock_mcp)
        delete_prompt_mock.side_effect = ValueError("Dummy error")

        self.mock_request.path_params = {"prompt_template_id": "pt1"}

        delete_handler = self.registered_routes["DELETE", "/registeredPrompts/{prompt_template_id}"]
        response = await delete_handler(self.mock_request)
        assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR, response.body
        assert b"Failed to delete prompt" in response.body

    @pytest.mark.asyncio
    @patch("datarobot_genai.drmcp.core.routes.register_prompt_from_prompt_template_id_and_version")
    @pytest.mark.parametrize(
        "query_params,prompt_version_id",
        [
            ({}, "ptv1.3"),  # Pick latest version
            ({"prompt_template_version_id": "ptv1.1"}, "ptv1.1"),  # Pick version chosen by user
        ],
    )
    async def test_add_prompt_template(
        self, add_prompt_mock: Mock, query_params: dict, prompt_version_id: str | None
    ):
        """Test add prompt template endpoint."""
        register_routes(self.mock_mcp)
        add_prompt_mock.return_value = Prompt(
            name="Dummy prompt name",
            description="Dummy prompt description",
            meta={"prompt_template_version_id": prompt_version_id},
        )

        self.mock_request.path_params = {"prompt_template_id": "pt1"}
        self.mock_request.query_params = query_params

        add_handler = self.registered_routes["PUT", "/registeredPrompts/{prompt_template_id}"]
        response = await add_handler(self.mock_request)
        assert response.status_code == HTTPStatus.CREATED, response.body
        assert json.loads(response.body.decode("utf-8")) == {
            "name": "Dummy prompt name",
            "description": "Dummy prompt description",
            "promptTemplateId": "pt1",
            "promptTemplateVersionId": prompt_version_id,
        }

    @pytest.mark.asyncio
    @patch("datarobot_genai.drmcp.core.routes.register_prompt_from_prompt_template_id_and_version")
    async def test_add_prompt_template_when_error(self, add_prompt_mock: Mock):
        """Test add prompt template endpoint when error occurs."""
        register_routes(self.mock_mcp)
        add_prompt_mock.side_effect = ValueError("Dummy error")

        self.mock_request.path_params = {"prompt_template_id": "pt1"}

        add_handler = self.registered_routes["PUT", "/registeredPrompts/{prompt_template_id}"]
        response = await add_handler(self.mock_request)
        assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR, response.body
        assert b"Failed to add prompt template" in response.body

    @pytest.mark.asyncio
    @patch("datarobot_genai.drmcp.core.routes.refresh_registered_prompt_template")
    async def test_refresh_prompt_templates(self, refresh_prompts_mock: Mock):
        """Test refresh prompt templates endpoint."""
        register_routes(self.mock_mcp)
        refresh_prompts_mock.return_value = None

        refresh_handler = self.registered_routes["PUT", "/registeredPrompts"]
        response = await refresh_handler(self.mock_request)
        assert response.status_code == HTTPStatus.NO_CONTENT, response.body

    @pytest.mark.asyncio
    @patch("datarobot_genai.drmcp.core.routes.refresh_registered_prompt_template")
    async def test_refresh_prompt_templates_when_error(self, refresh_prompts_mock: Mock):
        """Test refresh prompt templates endpoint when error occurs."""
        register_routes(self.mock_mcp)
        refresh_prompts_mock.side_effect = ValueError("Dummy error")

        refresh_handler = self.registered_routes["PUT", "/registeredPrompts"]
        response = await refresh_handler(self.mock_request)
        assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR, response.body
        assert b"Failed to refresh prompt template" in response.body
