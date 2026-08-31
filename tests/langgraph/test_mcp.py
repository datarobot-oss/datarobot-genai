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
from unittest.mock import patch

import pytest
from datarobot.models.genai.agent.auth import set_authorization_context

from datarobot_genai.core.mcp import MCPServerRef
from datarobot_genai.core.mcp import build_target
from datarobot_genai.langgraph.mcp import mcp_tools_context


@pytest.fixture
def mock_load_mcp_tools():
    # ``mcp_tools_context`` loads tools via ``load_mcp_tools(session=None, connection=...)``
    # (no persistent session held across the stream), so this is the only thing to mock.
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
def setup_session_and_tools(mock_load_mcp_tools, mock_tools):
    mock_load_mcp_tools.return_value = mock_tools
    return {
        "load_tools": mock_load_mcp_tools,
        "tools": mock_tools,
    }


def _loaded_connection(setup):
    """Return the connection config passed to ``load_mcp_tools`` (session-less loading)."""
    call_args = setup["load_tools"].call_args
    assert call_args.kwargs["session"] is None
    return call_args.kwargs["connection"]


@pytest.fixture(autouse=True)
def clear_environment_variables():
    with patch.dict(os.environ, {}, clear=True):
        yield


EXTERNAL_URL = "https://mcp-server.example.com/mcp"
DEPLOYMENT_ID = "abc123def456789012345678"


def external_target(name="partner", url=EXTERNAL_URL, **kwargs):
    return build_target(MCPServerRef(name=name, url=url, **kwargs))


def deployment_target(deployment_id=DEPLOYMENT_ID, endpoint=None, token="test-api-key"):
    return build_target(
        MCPServerRef(name="analytics", deployment_id=deployment_id),
        datarobot_endpoint=endpoint or "https://app.datarobot.com/api/v2",
        datarobot_api_token=token,
    )


class TestMCPToolsContext:
    async def test_a_third_party_server_is_connected_with_its_own_headers(
        self, setup_session_and_tools, assert_mock_tools_expected
    ):
        target = external_target(
            headers={"X-API-Key": "test-key", "Content-Type": "application/json"},
            transport="sse",
        )
        async with mcp_tools_context(target, prefix="") as tools:
            assert_mock_tools_expected(tools)

            setup_session_and_tools["load_tools"].assert_called_once()
            connection_config = _loaded_connection(setup_session_and_tools)
            assert connection_config["url"] == EXTERNAL_URL
            assert connection_config["headers"] == {
                "X-API-Key": "test-key",
                "Content-Type": "application/json",
            }
            # SSEConnection uses transport="sse"
            assert connection_config["transport"] == "sse"

    async def test_the_default_transport_is_streamable_http(
        self, setup_session_and_tools, assert_mock_tools_expected
    ):
        async with mcp_tools_context(external_target(), prefix="") as tools:
            assert_mock_tools_expected(tools)

            setup_session_and_tools["load_tools"].assert_called_once()
            connection_config = _loaded_connection(setup_session_and_tools)
            assert connection_config["url"] == EXTERNAL_URL
            assert connection_config["headers"] == {}
            # StreamableHttpConnection uses transport="streamable_http" (underscore)
            assert connection_config["transport"] == "streamable_http"

    async def test_a_deployment_is_connected_with_datarobot_credentials(
        self, setup_session_and_tools, agent_auth_context_data, assert_mock_tools_expected
    ):
        # When the agent is initialized, it sets the authorization context for the
        # process, so subsequent tools and MCP calls receive it via a dedicated header.
        set_authorization_context(agent_auth_context_data)

        async with mcp_tools_context(deployment_target(), prefix="") as tools:
            assert_mock_tools_expected(tools)
            setup_session_and_tools["load_tools"].assert_called_once()
            connection_config = _loaded_connection(setup_session_and_tools)
            assert connection_config["url"] == (
                f"https://app.datarobot.com/api/v2/deployments/{DEPLOYMENT_ID}/directAccess/mcp"
            )
            assert connection_config["headers"]["Authorization"] == "Bearer test-api-key"

    async def test_the_endpoint_and_token_come_from_the_target(
        self, setup_session_and_tools, agent_auth_context_data, assert_mock_tools_expected
    ):
        set_authorization_context(agent_auth_context_data)
        target = deployment_target(endpoint="https://custom.datarobot.com/api/v2", token="custom")

        async with mcp_tools_context(target, prefix="") as tools:
            assert_mock_tools_expected(tools)
            connection_config = _loaded_connection(setup_session_and_tools)
            assert connection_config["url"] == (
                f"https://custom.datarobot.com/api/v2/deployments/{DEPLOYMENT_ID}/directAccess/mcp"
            )
            assert connection_config["headers"]["Authorization"] == "Bearer custom"

    async def test_forwarded_headers_and_auth_context_are_arguments_not_config(
        self, setup_session_and_tools
    ):
        # They are request scope. As fields on a shared config object they were what
        # made it unsafe to copy between requests.
        async with mcp_tools_context(
            deployment_target(), prefix="", forwarded={"x-datarobot-entity-id": "e1"}
        ):
            headers = _loaded_connection(setup_session_and_tools)["headers"]
        assert headers["x-datarobot-entity-id"] == "e1"

    async def test_tools_are_namespaced_by_server_name_so_a_fleet_cannot_collide(
        self, setup_session_and_tools
    ):
        # Two servers each exposing `search` would otherwise collide and one would
        # silently shadow the other. `__` is the separator NAT uses for function groups.
        async with mcp_tools_context(external_target(name="analytics")) as tools:
            assert [t.name for t in tools] == ["analytics__dummy-tool-1", "analytics__tool-2"]

    async def test_an_explicit_prefix_overrides_the_server_name(self, setup_session_and_tools):
        async with mcp_tools_context(external_target(name="analytics"), prefix="reports") as tools:
            assert [t.name for t in tools] == ["reports__dummy-tool-1", "reports__tool-2"]

    @pytest.mark.usefixtures("setup_session_and_tools")
    async def test_a_consumer_exception_is_propagated(self):
        with pytest.raises(RuntimeError):
            async with mcp_tools_context(external_target()):
                raise RuntimeError("Connection failed")

    async def test_an_unreachable_server_raises_by_default(self):
        # A server that was configured and cannot be reached is a failure. Yielding no
        # tools makes it indistinguishable from a server that was never configured.
        with patch(
            "datarobot_genai.langgraph.mcp.load_mcp_tools", side_effect=ConnectionError("refused")
        ):
            with pytest.raises(ConnectionError):
                async with mcp_tools_context(external_target()):
                    pass

    async def test_strict_false_restores_the_degrade_quietly_behaviour(self):
        with patch(
            "datarobot_genai.langgraph.mcp.load_mcp_tools", side_effect=ConnectionError("refused")
        ):
            async with mcp_tools_context(external_target(), strict=False) as tools:
                assert tools == []

    async def test_an_unreachable_local_server_is_reported_before_connecting(self):
        # A local server that is not running is routine when developing against a fleet;
        # probing is cheaper than waiting out the connect timeout on every build.
        target = build_target(MCPServerRef(name="docs", local_port=9), datarobot_api_token="tok")
        with patch("datarobot_genai.langgraph.mcp._local_server_reachable", return_value=False):
            with pytest.raises(ConnectionError, match="docs"):
                async with mcp_tools_context(target):
                    pass

            async with mcp_tools_context(target, strict=False) as tools:
                assert tools == []

    @pytest.mark.usefixtures("setup_session_and_tools")
    async def test_a_consumer_connection_error_propagates(self):
        """A connection-type exception raised by the consumer must propagate, not trigger
        the setup-phase fallback.  Before the `connected` guard this would hit `yield []`
        as a second yield and raise RuntimeError: generator didn't stop after athrow().
        """
        with pytest.raises(ConnectionError, match="downstream failure"):
            async with mcp_tools_context(external_target(transport="sse"), strict=False):
                raise ConnectionError("downstream failure")
