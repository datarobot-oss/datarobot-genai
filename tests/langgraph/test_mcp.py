# Copyright 2025 DataRobot, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest
from datarobot.models.genai.agent.auth import set_authorization_context
from pydantic import ValidationError

from datarobot_genai.core.mcp import MCPConfig
from datarobot_genai.langgraph.mcp import mcp_tools_context


@pytest.fixture
def mock_session(mock_session_instance):
    """Mock create_session to prevent actual network connections."""
    with patch("datarobot_genai.langgraph.mcp.create_session") as mock:
        # Configure mock to return the async context manager instance
        mock.return_value = mock_session_instance
        yield mock


@pytest.fixture
def mock_load_mcp_tools():
    with patch("datarobot_genai.langgraph.mcp.load_mcp_tools") as mock:
        yield mock


@pytest.fixture
def mock_tools():
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel
    from pydantic import Field

    # Dummy argument schema for a test tool
    class DummyToolInput(BaseModel):
        foo: str = Field(..., description="A foo string argument")
        bar: int = Field(42, description="A bar integer argument")

    async def dummy_tool_func(foo: str, bar: int = 42, runtime=None):
        """Do something."""
        # Runtime arg to mimic MCP runtime-injected parameter
        return ({"content": f"foo={foo}; bar={bar}", "artifact": None}, [])

    # Dummy 1: tool using args_schema and coroutine
    dummy_tool_1 = StructuredTool(
        name="dummy-tool-1",
        description="A dummy MCP-like tool for testing.",
        args_schema=DummyToolInput,
        coroutine=dummy_tool_func,
        response_format="content_and_artifact",
        metadata={"_meta": {"example": 1}},
    )

    # Dummy 2: Minimal version, no response_format or metadata
    class Tool2Input(BaseModel):
        val: int = Field(...)

    async def tool2_func(val: int, runtime=None):
        return ({"content": str(val * 10), "artifact": None}, [])

    dummy_tool_2 = StructuredTool(
        name="tool-2",
        description="A second dummy MCP tool.",
        args_schema=Tool2Input,
        coroutine=tool2_func,
    )

    return [dummy_tool_1, dummy_tool_2]


@pytest.fixture
def mock_session_instance():
    session_instance = AsyncMock()
    session_instance.__aenter__.return_value = session_instance
    session_instance.__aexit__.return_value = None
    return session_instance


@pytest.fixture
def assert_mock_tools_expected(mock_tools):
    tool_names = [t.name for t in mock_tools]
    tool_descriptions = [t.description for t in mock_tools]
    tool_args_schemas = [t.args_schema for t in mock_tools]

    def assert_mock_tools_expected(tools):
        assert [t.name for t in tools] == tool_names
        assert [t.description for t in tools] == tool_descriptions
        assert [t.args_schema for t in tools] == tool_args_schemas

    return assert_mock_tools_expected


@pytest.fixture
def setup_session_and_tools(mock_session, mock_load_mcp_tools, mock_session_instance, mock_tools):
    # mock_session is already configured in its fixture to return mock_session_instance
    mock_load_mcp_tools.return_value = mock_tools
    return {
        "session": mock_session,
        "session_instance": mock_session_instance,
        "load_tools": mock_load_mcp_tools,
        "tools": mock_tools,
    }


@pytest.fixture(autouse=True)
def clear_environment_variables():
    with patch.dict(os.environ, {}, clear=True):
        yield


class TestMCPToolsContext:
    async def test_mcp_tools_context_no_configuration(self):
        with patch.dict(os.environ, {}, clear=True):
            mcp_config = MCPConfig()
            async with mcp_tools_context(mcp_config) as tools:
                assert tools == []

    async def test_mcp_tools_context_with_external_url(
        self, setup_session_and_tools, assert_mock_tools_expected
    ):
        test_headers = '{"X-API-Key": "test-key", "Content-Type": "application/json"}'
        test_transport = "sse"
        external_url = "https://mcp-server.example.com/mcp"

        mcp_config = MCPConfig(
            external_mcp_url=external_url,
            external_mcp_headers=test_headers,
            external_mcp_transport=test_transport,
        )
        async with mcp_tools_context(mcp_config) as tools:
            assert_mock_tools_expected(tools)

            # Verify create_session was called with correct connection config
            setup_session_and_tools["session"].assert_called_once()
            call_args = setup_session_and_tools["session"].call_args
            connection_config = call_args[1]["connection"]
            assert connection_config["url"] == external_url.rstrip("/")
            expected_headers = {"X-API-Key": "test-key", "Content-Type": "application/json"}
            assert connection_config["headers"] == expected_headers
            # SSEConnection uses transport="sse"
            assert connection_config["transport"] == "sse"

            # Verify tools were loaded
            setup_session_and_tools["load_tools"].assert_called_once_with(
                session=setup_session_and_tools["session_instance"]
            )

    async def test_mcp_tools_context_with_external_url_default_transport(
        self, setup_session_and_tools, assert_mock_tools_expected
    ):
        external_url = "https://mcp-server.example.com/mcp"

        mcp_config = MCPConfig(external_mcp_url=external_url)
        async with mcp_tools_context(mcp_config) as tools:
            assert_mock_tools_expected(tools)

            # Verify create_session was called with correct connection config
            # (default transport)
            setup_session_and_tools["session"].assert_called_once()
            call_args = setup_session_and_tools["session"].call_args
            connection_config = call_args[1]["connection"]
            assert connection_config["url"] == external_url.rstrip("/")
            assert connection_config["headers"] == {}  # No custom headers
            # StreamableHttpConnection uses transport="streamable_http" (underscore)
            assert connection_config["transport"] == "streamable_http"

            setup_session_and_tools["load_tools"].assert_called_once_with(
                session=setup_session_and_tools["session_instance"]
            )

    async def test_mcp_tools_context_with_datarobot_deployment(
        self, setup_session_and_tools, agent_auth_context_data, assert_mock_tools_expected
    ):
        deployment_id = "abc123def456789012345678"
        api_base = "https://app.datarobot.com/api/v2"
        api_key = "test-api-key"

        # When the agent is initialized, it sets the authorization context for the
        # process, so subsequent tools and MCP calls receive it via a dedicated header.
        set_authorization_context(agent_auth_context_data)

        mcp_config = MCPConfig(
            mcp_deployment_id=deployment_id,
            datarobot_endpoint=api_base,
            datarobot_api_token=api_key,
        )
        async with mcp_tools_context(mcp_config) as tools:
            assert_mock_tools_expected(tools)
            # Check that create_session was called with correct connection config
            setup_session_and_tools["session"].assert_called_once()
            call_args = setup_session_and_tools["session"].call_args
            connection_config = call_args[1]["connection"]
            expected_url = f"{api_base}/deployments/{deployment_id}/directAccess/mcp"
            assert connection_config["url"] == expected_url
            assert connection_config["headers"]["Authorization"] == f"Bearer {api_key}"
            setup_session_and_tools["load_tools"].assert_called_once_with(
                session=setup_session_and_tools["session_instance"]
            )

    async def test_mcp_tools_context_with_parameters(
        self, setup_session_and_tools, agent_auth_context_data, assert_mock_tools_expected
    ):
        deployment_id = "abc123def456789012345678"
        custom_api_base = "https://custom.datarobot.com/api/v2"
        custom_api_key = "custom-key"

        # When the agent is initialized, it sets the authorization context for the
        # process, so subsequent tools and MCP calls receive it via a dedicated header.
        set_authorization_context(agent_auth_context_data)

        mcp_config = MCPConfig(
            mcp_deployment_id=deployment_id,
            datarobot_endpoint=custom_api_base,
            datarobot_api_token=custom_api_key,
        )
        async with mcp_tools_context(mcp_config) as tools:
            assert_mock_tools_expected(tools)
            # Check that create_session was called with custom parameters
            setup_session_and_tools["session"].assert_called_once()
            call_args = setup_session_and_tools["session"].call_args
            connection_config = call_args[1]["connection"]
            expected_url = f"{custom_api_base}/deployments/{deployment_id}/directAccess/mcp"
            assert connection_config["url"] == expected_url
            assert connection_config["headers"]["Authorization"] == f"Bearer {custom_api_key}"

    async def test_mcp_tools_context_with_sse_transport(
        self, setup_session_and_tools, assert_mock_tools_expected
    ):
        external_url = "https://mcp-server.example.com/mcp"

        mcp_config = MCPConfig(external_mcp_url=external_url, external_mcp_transport="sse")
        async with mcp_tools_context(mcp_config) as tools:
            assert_mock_tools_expected(tools)
            # Verify create_session was called with SSE transport
            setup_session_and_tools["session"].assert_called_once()
            call_args = setup_session_and_tools["session"].call_args
            connection_config = call_args[1]["connection"]
            # SSEConnection uses transport="sse"
            assert connection_config["transport"] == "sse"
            setup_session_and_tools["load_tools"].assert_called_once_with(
                session=setup_session_and_tools["session_instance"]
            )

    @pytest.mark.usefixtures("setup_session_and_tools")
    async def test_mcp_tools_context_exception_is_propagated(self):
        external_url = "https://mcp-server.example.com/mcp"

        mcp_config = MCPConfig(external_mcp_url=external_url, external_mcp_transport="sse")
        with pytest.raises(RuntimeError):
            async with mcp_tools_context(mcp_config):
                raise RuntimeError("Connection failed")

    async def test_mcp_tools_context_unsupported_transport(self):
        external_url = "https://mcp-server.example.com/mcp"

        # mcp_tools_context will raise ValidationError for unsupported transport
        with pytest.raises(ValidationError):
            mcp_config = MCPConfig(
                external_mcp_url=external_url, external_mcp_transport="invalid-transport"
            )
            async with mcp_tools_context(mcp_config):
                pass
