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

"""Tests for external tool registration."""

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from fastmcp.tools.tool import Tool

from datarobot_genai.drmcp.core.dynamic_tools.register import ExternalToolRegistrationConfig
from datarobot_genai.drmcp.core.dynamic_tools.register import register_external_tool


@pytest.fixture
def mock_tool():
    """Mock Tool object."""
    tool = MagicMock(spec=Tool)
    tool.name = "test_tool"
    return tool


@pytest.fixture
def basic_tool_config():
    """Return basic external tool configuration."""
    return ExternalToolRegistrationConfig(
        name="weather_api",
        title="Weather API",
        description="Get weather forecast",
        method="GET",
        base_url="https://api.weather.com",
        endpoint="forecast",
        headers={"X-API-Key": "secret"},
        input_schema={
            "type": "object",
            "properties": {
                "query_params": {
                    "type": "object",
                    "properties": {"city": {"type": "string", "description": "City name"}},
                    "required": ["city"],
                }
            },
            "required": ["query_params"],
        },
    )


@pytest.fixture
def post_tool_config():
    """POST method tool configuration."""
    return ExternalToolRegistrationConfig(
        name="create_user",
        title="Create User",
        description="Create a new user",
        method="POST",
        base_url="https://api.example.com",
        endpoint="users",
        input_schema={
            "type": "object",
            "properties": {
                "json": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "email": {"type": "string"},
                    },
                    "required": ["name", "email"],
                }
            },
            "required": ["json"],
        },
    )


@pytest.fixture
def tool_config_with_path_params():
    """Tool configuration with path parameters."""
    return ExternalToolRegistrationConfig(
        name="get_user",
        title="Get User",
        description="Get user by ID",
        method="GET",
        base_url="https://api.example.com",
        endpoint="users/{user_id}",
        input_schema={
            "type": "object",
            "properties": {
                "path_params": {
                    "type": "object",
                    "properties": {"user_id": {"type": "string"}},
                    "required": ["user_id"],
                }
            },
            "required": ["path_params"],
        },
    )


class TestRegisterExternalTool:
    """Tests for register_external_tool function - happy path."""

    @pytest.mark.asyncio
    @patch("datarobot_genai.drmcp.core.dynamic_tools.register.register_tools")
    async def test_register_tool_basic_config(
        self, mock_register_tools, basic_tool_config, mock_tool
    ):
        """Test registering a basic external tool."""
        mock_tool.name = "weather_api"
        mock_register_tools.return_value = mock_tool

        result = await register_external_tool(config=basic_tool_config)

        mock_register_tools.assert_called_once()
        assert mock_register_tools.call_args.kwargs["name"] == "weather_api"
        assert mock_register_tools.call_args.kwargs["description"] == "Get weather forecast"
        assert result == mock_tool

    @pytest.mark.asyncio
    @patch("datarobot_genai.drmcp.core.dynamic_tools.register.register_tools")
    async def test_register_tool_with_tags(self, mock_register_tools, basic_tool_config, mock_tool):
        """Test registering a tool with custom tags."""
        test_tags = {"weather", "external", "api"}
        basic_tool_config.tags = test_tags
        mock_register_tools.return_value = mock_tool

        await register_external_tool(config=basic_tool_config)

        assert mock_register_tools.call_args.kwargs["tags"] == test_tags

    @pytest.mark.asyncio
    @patch("datarobot_genai.drmcp.core.dynamic_tools.register.register_tools")
    async def test_register_tool_with_additional_kwargs(
        self, mock_register_tools, basic_tool_config, mock_tool
    ):
        """Test registering a tool with additional kwargs."""
        mock_register_tools.return_value = mock_tool

        await register_external_tool(
            config=basic_tool_config,
            deployment_id="test-123",
            custom_field="value",
        )

        kwargs = mock_register_tools.call_args.kwargs
        assert kwargs["deployment_id"] == "test-123"
        assert kwargs["custom_field"] == "value"

    @pytest.mark.asyncio
    @patch("datarobot_genai.drmcp.core.dynamic_tools.register.register_tools")
    async def test_register_post_tool(self, mock_register_tools, post_tool_config, mock_tool):
        """Test registering a POST method tool."""
        mock_tool.name = "create_user"
        mock_register_tools.return_value = mock_tool

        result = await register_external_tool(config=post_tool_config)

        mock_register_tools.assert_called_once()
        assert mock_register_tools.call_args.kwargs["name"] == "create_user"
        assert result.name == "create_user"

    @pytest.mark.asyncio
    @patch("datarobot_genai.drmcp.core.dynamic_tools.register.register_tools")
    async def test_register_tool_with_path_params(
        self, mock_register_tools, tool_config_with_path_params, mock_tool
    ):
        """Test registering a tool with path parameters."""
        mock_register_tools.return_value = mock_tool

        await register_external_tool(config=tool_config_with_path_params)

        mock_register_tools.assert_called_once()
        assert mock_register_tools.call_args.kwargs["name"] == "get_user"

    @pytest.mark.asyncio
    @patch("datarobot_genai.drmcp.core.dynamic_tools.register.register_tools")
    async def test_register_tool_callable_is_created(
        self, mock_register_tools, basic_tool_config, mock_tool
    ):
        """Test that a callable function is created and passed to register_tools."""
        mock_register_tools.return_value = mock_tool

        await register_external_tool(config=basic_tool_config)

        mock_register_tools.assert_called_once()
        assert callable(mock_register_tools.call_args.kwargs["fn"])

    @pytest.mark.asyncio
    @patch("datarobot_genai.drmcp.core.dynamic_tools.register.register_tools")
    async def test_register_tool_uses_title_when_provided(
        self, mock_register_tools, basic_tool_config, mock_tool
    ):
        """Test that tool uses title when provided in config."""
        mock_register_tools.return_value = mock_tool

        await register_external_tool(config=basic_tool_config)

        assert mock_register_tools.call_args.kwargs["title"] == "Weather API"


class TestExternalToolRegistrationConfig:
    """Tests for ExternalToolRegistrationConfig model."""

    def test_config_creation_with_all_fields(self):
        """Test creating config with all fields."""
        config = ExternalToolRegistrationConfig(
            name="test_tool",
            title="Test Tool",
            description="A test tool",
            method="POST",
            base_url="https://api.test.com",
            endpoint="test",
            headers={"Authorization": "Bearer token"},
            input_schema={"type": "object", "properties": {}},
        )

        assert config.name == "test_tool"
        assert config.title == "Test Tool"
        assert config.description == "A test tool"
        assert config.method == "POST"
        assert config.base_url == "https://api.test.com"
        assert config.endpoint == "test"
        assert config.headers == {"Authorization": "Bearer token"}
        assert config.input_schema == {"type": "object", "properties": {}}

    def test_config_creation_with_optional_fields_none(self):
        """Test creating config with optional fields as None."""
        config = ExternalToolRegistrationConfig(
            name="test_tool",
            method="GET",
            base_url="https://api.test.com",
            endpoint="test",
            input_schema={"type": "object", "properties": {}},
        )

        assert config.title is None
        assert config.description is None
        assert config.headers is None

    @pytest.mark.parametrize("method", ["GET", "POST", "PATCH", "PUT", "DELETE"])
    def test_config_accepts_valid_methods(self, method):
        """Test that config accepts valid HTTP methods."""
        config = ExternalToolRegistrationConfig(
            name="test_tool",
            method=method,
            base_url="https://api.test.com",
            endpoint="test",
            input_schema={"type": "object", "properties": {}},
        )

        assert config.method == method

    def test_config_with_complex_input_schema(self):
        """Test config with complex nested input schema."""
        complex_schema = {
            "type": "object",
            "properties": {
                "path_params": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}},
                },
                "query_params": {
                    "type": "object",
                    "properties": {"filter": {"type": "string"}},
                },
                "json": {
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "object",
                            "properties": {"key": {"type": "string"}},
                        }
                    },
                },
            },
        }

        config = ExternalToolRegistrationConfig(
            name="complex_tool",
            method="POST",
            base_url="https://api.test.com",
            endpoint="complex/{id}",
            input_schema=complex_schema,
        )

        assert config.input_schema == complex_schema
        properties = config.input_schema["properties"]
        assert "path_params" in properties
        assert "query_params" in properties
        assert "json" in properties

    def test_config_with_empty_headers(self):
        """Test config with empty headers dict."""
        config = ExternalToolRegistrationConfig(
            name="test_tool",
            method="GET",
            base_url="https://api.test.com",
            endpoint="test",
            headers={},
            input_schema={"type": "object", "properties": {}},
        )

        assert config.headers == {}

    def test_config_base_url_without_trailing_slash(self):
        """Test that base_url works without trailing slash."""
        config = ExternalToolRegistrationConfig(
            name="test_tool",
            method="GET",
            base_url="https://api.test.com",
            endpoint="test",
            input_schema={"type": "object", "properties": {}},
        )

        assert config.base_url == "https://api.test.com"
        assert not config.base_url.endswith("/")

    def test_config_endpoint_without_leading_slash(self):
        """Test that endpoint works without leading slash."""
        config = ExternalToolRegistrationConfig(
            name="test_tool",
            method="GET",
            base_url="https://api.test.com",
            endpoint="test/path",
            input_schema={"type": "object", "properties": {}},
        )

        assert config.endpoint == "test/path"
        assert not config.endpoint.startswith("/")

    def test_config_endpoint_with_path_params_template(self):
        """Test endpoint with path parameter templates."""
        config = ExternalToolRegistrationConfig(
            name="test_tool",
            method="GET",
            base_url="https://api.test.com",
            endpoint="users/{user_id}/posts/{post_id}",
            input_schema={"type": "object", "properties": {}},
        )

        assert "{user_id}" in config.endpoint
        assert "{post_id}" in config.endpoint
