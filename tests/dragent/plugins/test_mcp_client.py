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
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any
from unittest.mock import patch

import pytest
from nat.builder.workflow_builder import WorkflowBuilder
from nat.plugins.mcp.client.client_base import MCPBaseClient
from nat.plugins.mcp.client.client_impl import MCPFunctionGroup
from pydantic import BaseModel
from pydantic import create_model

from datarobot_genai.core.mcp import MCPServerRef
from datarobot_genai.core.mcp import build_headers
from datarobot_genai.core.mcp import build_target
from datarobot_genai.dragent.plugins.datarobot_mcp_client import DataRobotMCPClientConfig
from datarobot_genai.dragent.plugins.datarobot_mcp_client import DataRobotMCPFunctionGroup
from datarobot_genai.dragent.plugins.datarobot_mcp_client import DataRobotMCPServerConfig
from datarobot_genai.dragent.plugins.datarobot_mcp_client import _make_input_schema_enum_safe
from datarobot_genai.dragent.plugins.datarobot_mcp_client import resolve_auth_provider_name

DEPLOYMENT_ID = "69331f1f30548f83b668d9dc"
WORKLOAD_ID = "6a72dd6d4417b3136f64fef0"
API_ENDPOINT = "https://app.datarobot.com/api/v2"


class _InputSchema(BaseModel):
    """Input schema for fake tools used in testing."""

    param: str


class _FakeTool:
    """Fake tool class for testing MCP tool functionality."""

    def __init__(self, name: str, description: str = "desc") -> None:
        self.name = name
        self.description = description
        self.input_schema = _InputSchema

    async def acall(self, args: dict[str, Any]) -> str:
        """Simulate tool execution by returning a formatted response."""
        return f"ok {args['param']}"

    def set_description(self, description: str) -> None:
        """Allow description to be updated for testing purposes."""
        if description is not None:
            self.description = description


class _FakeMCPClient(MCPBaseClient):
    """Fake MCP client for testing client-server interactions."""

    def __init__(
        self,
        *,
        tools: dict[str, _FakeTool],
        url: str | None = None,
    ) -> None:
        super().__init__("streamable-http")
        self._tools = tools
        self.url = url

    async def get_tool(self, name: str) -> _FakeTool:
        """Retrieve a tool by name."""
        return self._tools[name]

    async def get_tools(self) -> dict[str, _FakeTool]:
        """Retrieve all tools."""
        return self._tools

    @asynccontextmanager
    async def connect_to_server(self):
        """Support async context manager for testing."""
        yield self


@pytest.fixture
def one_configured_server():
    """Configure the `default` server the way an existing .env addresses it."""
    with patch.dict(
        os.environ,
        {
            "MCP_DEPLOYMENT_ID": DEPLOYMENT_ID,
            "DATAROBOT_ENDPOINT": API_ENDPOINT,
            "DATAROBOT_API_TOKEN": "tok",
        },
        clear=True,
    ):
        yield


async def test_datarobot_mcp_client(one_configured_server):
    with patch(
        "datarobot_genai.dragent.plugins.datarobot_mcp_client.DataRobotMCPStreamableHTTPClient"
    ) as mock_client:
        fake_tools = {"a": _FakeTool("a", "da"), "b": _FakeTool("b", "db")}

        def make_fake_client(url, *args, **kwargs):
            return _FakeMCPClient(tools=fake_tools, url=url)

        mock_client.side_effect = make_fake_client
        server_config = DataRobotMCPServerConfig(auth_provider=None)
        config = DataRobotMCPClientConfig(server=server_config)
        async with WorkflowBuilder() as builder:
            await builder.add_function_group("datarobot_mcp_tools", config)
            function_group = await builder.get_function_group("datarobot_mcp_tools")
            assert isinstance(function_group, MCPFunctionGroup)
            # Verify the happy path: fake client was used and tools were registered
            all_functions = await function_group.get_all_functions()
            # Function names are prefixed with the group name (e.g. datarobot_mcp_tools__a)
            assert "datarobot_mcp_tools__a" in all_functions
            assert "datarobot_mcp_tools__b" in all_functions
        # The address came from the environment, never from the block
        assert mock_client.call_args[0][0] == (
            f"{API_ENDPOINT}/deployments/{DEPLOYMENT_ID}/directAccess/mcp"
        )


async def test_the_block_carries_a_name_and_the_address_comes_from_the_fleet():
    fleet = (
        f'[{{"name":"analytics","deployment_id":"{DEPLOYMENT_ID}"}},'
        f'{{"name":"docs","local_port":9001}}]'
    )
    with patch.dict(
        os.environ,
        {
            "MCP_SERVERS": fleet,
            "DATAROBOT_ENDPOINT": API_ENDPOINT,
            "DATAROBOT_API_TOKEN": "tok",
        },
        clear=True,
    ):
        with patch(
            "datarobot_genai.dragent.plugins.datarobot_mcp_client.DataRobotMCPStreamableHTTPClient"
        ) as mock_client:
            mock_client.side_effect = lambda url, *a, **kw: _FakeMCPClient(tools={}, url=url)
            config = DataRobotMCPClientConfig(
                server=DataRobotMCPServerConfig(name="docs", auth_provider=None)
            )
            async with WorkflowBuilder() as builder:
                await builder.add_function_group("docs_tools", config)
                await builder.get_function_group("docs_tools")
    assert mock_client.call_args[0][0] == "http://localhost:9001/mcp"


async def test_an_unknown_server_name_fails_the_build():
    # A typo is otherwise indistinguishable from a working server, and this is also the
    # only thing that catches a config provider registered too late.
    with patch.dict(os.environ, {}, clear=True):
        config = DataRobotMCPClientConfig(
            server=DataRobotMCPServerConfig(name="analytics", auth_provider=None)
        )
        async with WorkflowBuilder() as builder:
            with pytest.raises(LookupError, match="analytics"):
                await builder.add_function_group("analytics_tools", config)
                await builder.get_function_group("analytics_tools")


@pytest.mark.parametrize(
    "inline",
    [
        pytest.param({"url": "https://elsewhere.example.com/mcp"}, id="url"),
        pytest.param({"transport": "sse"}, id="transport"),
        pytest.param({"custom_headers": {"x-key": "v"}}, id="custom_headers"),
    ],
)
def test_an_inline_address_in_the_block_is_rejected(inline):
    # These are inherited from NAT's server config and ignored by this client. A field
    # that is silently ignored is worse than one that raises.
    with pytest.raises(ValueError, match="MCP_SERVERS"):
        DataRobotMCPServerConfig(name="analytics", **inline)


class TestAMixedFleetShareOneAuthProvider:
    """NAT returns ONE auth provider instance per name, shared by every block using it.

    So the provider cannot hold the target: the last block to build would win and every
    server would receive that block's credentials. A single-server test passes either
    way, which is why this one exists.
    """

    async def test_each_block_gets_the_credentials_for_its_own_kind(self):
        targets = {
            "analytics": build_target(
                MCPServerRef(name="analytics", deployment_id=DEPLOYMENT_ID),
                datarobot_endpoint=API_ENDPOINT,
                datarobot_api_token="tok",
            ),
            "docs": build_target(
                MCPServerRef(name="docs", local_port=9001), datarobot_api_token="tok"
            ),
            "partner": build_target(
                MCPServerRef(name="partner", url="https://partner.example.com/mcp")
            ),
        }

        from datarobot_genai.dragent.plugins.datarobot_auth_provider import DataRobotMCPAuthProvider
        from datarobot_genai.dragent.plugins.datarobot_auth_provider import (
            DataRobotMCPAuthProviderConfig,
        )

        # ONE provider instance, as `builder.get_auth_provider` would hand back
        provider = DataRobotMCPAuthProvider(config=DataRobotMCPAuthProviderConfig())

        headers = {}
        for name, target in targets.items():
            result = await provider.authenticate(user_id=None, target=target)
            headers[name] = {c.name: c.value.get_secret_value() for c in result.credentials}

        assert headers["analytics"]["Authorization"] == "Bearer tok"
        assert headers["docs"]["Authorization"] == "Bearer tok"
        # A third-party server gets no DataRobot identity, even from a shared provider
        assert headers["partner"] == {}

    async def test_a_workload_and_a_deployment_do_not_share_an_api_key_header(self):
        # `x-datarobot-api-key` is attached only for workloads. Under one globally
        # resolved config a mixed fleet is wrong for at least one server either way.
        workload = build_headers(_workload_target())
        deployment = build_headers(
            build_target(
                MCPServerRef(name="analytics", deployment_id=DEPLOYMENT_ID),
                datarobot_endpoint=API_ENDPOINT,
                datarobot_api_token="tok",
            )
        )
        assert "x-datarobot-api-key" in workload
        assert "x-datarobot-api-key" not in deployment


def _workload_target():
    from datarobot_genai.core.mcp import MCPTarget

    return MCPTarget(
        ref=MCPServerRef(name="search", workload_id=WORKLOAD_ID),
        url="https://app.datarobot.com/workloads/x/mcp",
        api_token="tok",
    )


async def test_the_auth_provider_refuses_to_guess_which_server_is_asking():
    # Falling back to an environment read here would silently restore exactly the
    # behaviour this replaces: one server's credentials sent to all of them.
    from datarobot_genai.dragent.plugins.datarobot_auth_provider import DataRobotMCPAuthProvider
    from datarobot_genai.dragent.plugins.datarobot_auth_provider import (
        DataRobotMCPAuthProviderConfig,
    )

    provider = DataRobotMCPAuthProvider(config=DataRobotMCPAuthProviderConfig())
    with pytest.raises(ValueError, match="MCPTarget"):
        await provider.authenticate(user_id=None)


# Tests for the BUZZOK-30556 enum-safe input schema patch.


def test_make_input_schema_enum_safe_stores_string_values():
    """After the patch, validating a dict produces a model whose enum field
    stores the plain string. NAT's ``_convert_input_pydantic`` then extracts
    strings via ``getattr`` and downstream ``model_validate(kwargs)`` never
    sees a cross-class Enum instance.
    """
    topic_enum = Enum("TopicEnum", {"general": "general", "news": "news"})
    schema = create_model("Schema", topic=(topic_enum, ...), query=(str, ...))

    class FakeFnInfo:
        description = "fake"
        input_schema = schema
        converters: list[Any] = []
        single_fn = None

    _make_input_schema_enum_safe(FakeFnInfo)

    instance = schema.model_validate({"topic": "news", "query": "q"})
    assert instance.topic == "news"
    assert not isinstance(instance.topic, Enum)
    # NAT-style unpacking yields a plain string, which downstream
    # model_validate calls accept regardless of enum class identity.
    kwargs = {k: getattr(instance, k) for k in type(instance).model_fields}
    assert kwargs == {"topic": "news", "query": "q"}


def test_make_input_schema_enum_safe_returns_input_unchanged_when_no_schema():
    """If the upstream ``FunctionInfo`` has no usable input schema we return
    it untouched rather than mutating ``None``.
    """

    class FakeFnInfo:
        description = "fake"
        input_schema = None
        converters: list[Any] = []
        single_fn = None

    assert _make_input_schema_enum_safe(FakeFnInfo) is FakeFnInfo


class TestPerUserSessionClientsAreRefused:
    """The session-client factory is a guard, not an implementation.

    Tool calls reach the build-time client because NAT derives a session id only from a
    ``nat-session`` cookie, NAT issues that cookie only from its WebSocket route, and
    ``dragent`` registers no ``websocket_path``. That makes the factory unreachable
    today -- but it is a property of the front-end configuration, not of MCP, so the
    guard has to keep failing loudly if that ever changes.
    """

    @pytest.mark.asyncio
    async def test_it_raises_and_names_the_fix(self):
        group = DataRobotMCPFunctionGroup(
            config=DataRobotMCPClientConfig(server=DataRobotMCPServerConfig(name="analytics"))
        )
        with pytest.raises(RuntimeError, match="session_aware_tools"):
            await group._create_session_client("some-user")

    @pytest.mark.asyncio
    async def test_it_raises_even_with_a_resolved_target(self):
        """The guard is unconditional. A target does not make the path supported: NAT's
        own adapter still could not carry it.
        """
        group = DataRobotMCPFunctionGroup(
            config=DataRobotMCPClientConfig(server=DataRobotMCPServerConfig(name="analytics"))
        )
        group._target = build_target(
            MCPServerRef(name="analytics", deployment_id=DEPLOYMENT_ID),
            datarobot_endpoint=API_ENDPOINT,
            datarobot_api_token="token",
        )
        with pytest.raises(RuntimeError, match="does not support per-user MCP session"):
            await group._create_session_client("some-user")

    def test_nat_still_declares_the_method_we_override(self):
        """Pins the coupling. If a NAT upgrade renames or removes this, the override
        silently stops being an override and the base implementation runs instead --
        building a client with no URL and no credentials.
        """
        assert hasattr(MCPFunctionGroup, "_create_session_client")


class TestResolveAuthProviderName:
    """Which identity, when it can be declared in two places (#26).

    `workflow.yaml`'s `server.auth_provider` and the ref's `auth_provider` (from
    `<name>_mcp_auth_provider`) are two sources for one value. `model_fields_set` is
    what distinguishes an explicit YAML value from the field's non-None default.
    """

    def test_yaml_only(self):
        s = DataRobotMCPServerConfig(name="w", auth_provider="okta_auth_of_mcp")
        r = MCPServerRef(name="w", deployment_id="a" * 24)
        assert resolve_auth_provider_name(s, r) == "okta_auth_of_mcp"

    def test_env_only_when_yaml_left_at_default(self):
        s = DataRobotMCPServerConfig(name="w")
        r = MCPServerRef(name="w", deployment_id="a" * 24, auth_provider="okta_auth_of_mcp")
        assert resolve_auth_provider_name(s, r) == "okta_auth_of_mcp"

    def test_neither_falls_back_to_the_yaml_default(self):
        s = DataRobotMCPServerConfig(name="w")
        r = MCPServerRef(name="w", deployment_id="a" * 24)
        assert resolve_auth_provider_name(s, r) == "datarobot_mcp_auth"

    def test_none_means_no_provider(self):
        s = DataRobotMCPServerConfig(name="p")
        r = MCPServerRef(name="p", url="https://p.example.com/mcp", auth_provider="none")
        assert resolve_auth_provider_name(s, r) is None

    def test_both_set_and_conflicting_raises_naming_both(self):
        s = DataRobotMCPServerConfig(name="w", auth_provider="datarobot_mcp_auth")
        r = MCPServerRef(name="w", deployment_id="a" * 24, auth_provider="okta_auth_of_mcp")
        with pytest.raises(ValueError, match="auth provider in two places"):
            resolve_auth_provider_name(s, r)

    def test_both_set_and_agreeing_is_fine(self):
        s = DataRobotMCPServerConfig(name="w", auth_provider="datarobot_mcp_auth")
        r = MCPServerRef(name="w", deployment_id="a" * 24, auth_provider="datarobot_mcp_auth")
        assert resolve_auth_provider_name(s, r) == "datarobot_mcp_auth"
