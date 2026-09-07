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

import asyncio
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

from datarobot_genai.core.config import Config
from datarobot_genai.core.mcp import MCPServerRef
from datarobot_genai.core.mcp import build_headers
from datarobot_genai.core.mcp import build_target
from datarobot_genai.core.mcp.target import build_datarobot_mcp_headers
from datarobot_genai.dragent.plugins.datarobot_mcp_client import DataRobotMCPClientConfig
from datarobot_genai.dragent.plugins.datarobot_mcp_client import DataRobotMCPFunctionGroup
from datarobot_genai.dragent.plugins.datarobot_mcp_client import DataRobotMCPServerConfig
from datarobot_genai.dragent.plugins.datarobot_mcp_client import _make_input_schema_enum_safe
from datarobot_genai.dragent.plugins.datarobot_mcp_client import resolve_auth_provider_name
from datarobot_genai.dragent.plugins.datarobot_mcp_client import resolve_server_ref

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
        "datarobot_genai.dragent.plugins.datarobot_mcp_client.MCPStreamableHTTPClient"
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
            "datarobot_genai.dragent.plugins.datarobot_mcp_client.MCPStreamableHTTPClient"
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
        pytest.param({"command": "uvx"}, id="command"),
        pytest.param({"args": ["serve"]}, id="args"),
        pytest.param({"env": {"K": "v"}}, id="env"),
    ],
)
def test_stdio_fields_are_rejected(inline):
    # This client builds sse and streamable-http clients only, so it cannot honour a
    # stdio server. A field that is silently ignored is worse than one that raises.
    with pytest.raises(ValueError, match="stdio"):
        DataRobotMCPServerConfig(name="analytics", **inline)


class TestInlineUrlIsSupported:
    """`url` is optional, not forbidden -- NAT's documented form still works.

    The rule is one *source* per server, not "no addresses in YAML": an inline `url`
    makes the block self-contained, and a `name` resolves against the environment.
    """

    def test_an_inline_url_needs_no_configured_fleet(self):
        with patch.dict(os.environ, {}, clear=True):
            ref = resolve_server_ref(
                DataRobotMCPServerConfig(url="https://mcp.example.com/mcp/"),
                Config(),
            )
        assert ref.url == "https://mcp.example.com/mcp"  # trailing slash trimmed
        # `url` means third-party, so no DataRobot identity unless asked for.
        assert not ref.sends_datarobot_credentials

    def test_transport_and_custom_headers_travel_with_an_inline_url(self):
        with patch.dict(os.environ, {}, clear=True):
            ref = resolve_server_ref(
                DataRobotMCPServerConfig(
                    url="https://mcp.example.com/mcp",
                    transport="sse",
                    custom_headers={"x-api-key": "abc"},
                ),
                Config(),
            )
        assert ref.transport == "sse"
        assert ref.headers == {"x-api-key": "abc"}

    def test_a_name_with_no_url_still_resolves_from_the_environment(self):
        with patch.dict(os.environ, {"ANALYTICS_MCP_DEPLOYMENT_ID": DEPLOYMENT_ID}, clear=True):
            ref = resolve_server_ref(DataRobotMCPServerConfig(name="analytics"), Config())
        assert ref.deployment_id == DEPLOYMENT_ID

    def test_defining_one_named_server_in_both_places_raises(self):
        with patch.dict(os.environ, {"ANALYTICS_MCP_DEPLOYMENT_ID": DEPLOYMENT_ID}, clear=True):
            with pytest.raises(ValueError, match="defined twice"):
                resolve_server_ref(
                    DataRobotMCPServerConfig(
                        name="analytics", url="https://elsewhere.example.com/mcp"
                    ),
                    Config(),
                )

    def test_an_unnamed_inline_block_does_not_collide_with_the_synthesised_default(self):
        """MCP_SERVER_PORT synthesises `default`, and templates set it unconditionally.

        Treating that as a competing definition would break a stock template the moment
        it added an inline URL, so the conflict check only fires on an explicit `name`.
        """
        with patch.dict(os.environ, {"MCP_SERVER_PORT": "9000"}, clear=True):
            ref = resolve_server_ref(
                DataRobotMCPServerConfig(url="https://mcp.example.com/mcp"), Config()
            )
        assert ref.url == "https://mcp.example.com/mcp"


class TestOneSharedAuthProviderServesTheWholeFleet:
    """NAT returns ONE auth provider instance per name, shared by every block using it.

    That is safe here because nothing the provider produces varies per server, which is
    the property these tests pin. It was not always so: `x-datarobot-api-key` used to be
    attached for workloads only, which is what forced a per-block target to travel into
    `authenticate()` and, with it, four NAT subclasses. Sending the header to every
    DataRobot-hosted server -- a deployment or a local process ignores it -- removed the
    variation and all of that machinery. If someone reintroduces per-server behaviour
    here, these fail.
    """

    async def test_every_datarobot_hosted_server_gets_identical_credentials(self):
        from datarobot_genai.dragent.plugins.datarobot_auth_provider import DataRobotMCPAuthProvider
        from datarobot_genai.dragent.plugins.datarobot_auth_provider import (
            DataRobotMCPAuthProviderConfig,
        )

        # ONE provider instance, as `builder.get_auth_provider` would hand back.
        provider = DataRobotMCPAuthProvider(config=DataRobotMCPAuthProviderConfig())

        with patch.dict(
            os.environ,
            {"DATAROBOT_ENDPOINT": API_ENDPOINT, "DATAROBOT_API_TOKEN": "tok"},
            clear=True,
        ):
            first = await provider.authenticate(user_id=None)
            second = await provider.authenticate(user_id=None)

        as_headers = {c.name: c.value.get_secret_value() for c in first.credentials}
        assert as_headers["Authorization"] == "Bearer tok"
        # Sent unconditionally: the workload gateway needs it, everything else ignores it.
        assert as_headers["x-datarobot-api-key"] == "tok"
        # No per-server state means two calls cannot disagree.
        assert as_headers == {c.name: c.value.get_secret_value() for c in second.credentials}

    def test_a_forwarded_api_key_still_outranks_the_service_one(self):
        """The one guard in the header builder that is not about servers."""
        headers = build_datarobot_mcp_headers(
            api_token="service", forwarded={"x-datarobot-api-key": "callers-own"}
        )
        assert headers["x-datarobot-api-key"] == "callers-own"

    def test_a_third_party_server_gets_no_datarobot_identity(self):
        """`url` defaults to `auth_provider: none`, so the provider never runs for it."""
        partner = build_target(MCPServerRef(name="partner", url="https://mcp.example.com/mcp"))
        assert not partner.ref.sends_datarobot_credentials
        assert build_headers(partner) == {}


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

    def test_the_resolved_url_is_written_back_onto_the_block(self):
        """NAT's per-user session factory reads `config.server.url`.

        An address that came from the environment has to land there for NAT's own code
        to work unchanged -- that write-back is what lets us use NAT's session factory
        instead of overriding it.
        """
        with patch.dict(
            os.environ,
            {
                "ANALYTICS_MCP_DEPLOYMENT_ID": DEPLOYMENT_ID,
                "DATAROBOT_ENDPOINT": API_ENDPOINT,
                "DATAROBOT_API_TOKEN": "tok",
            },
            clear=True,
        ):
            config = DataRobotMCPClientConfig(
                server=DataRobotMCPServerConfig(name="analytics", auth_provider=None)
            )
            assert config.server.url is None
            with patch(
                "datarobot_genai.dragent.plugins.datarobot_mcp_client.MCPStreamableHTTPClient"
            ) as mock_client:
                mock_client.side_effect = lambda url, *a, **kw: _FakeMCPClient(tools={}, url=url)

                async def _build():
                    async with WorkflowBuilder() as builder:
                        await builder.add_function_group("analytics_tools", config)
                        await builder.get_function_group("analytics_tools")

                asyncio.run(_build())
            assert str(config.server.url) == (
                f"{API_ENDPOINT}/deployments/{DEPLOYMENT_ID}/directAccess/mcp"
            )

    def test_we_no_longer_override_nats_session_factory(self):
        """We use NAT's own per-user session factory rather than reimplementing it.

        It works because the provider needs no per-server state and `config.server.url`
        is written back at build. If someone reintroduces an override, this fails and
        they have to justify it.
        """
        assert (
            DataRobotMCPFunctionGroup._create_session_client
            is MCPFunctionGroup._create_session_client
        )


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


class TestStdioServers:
    """A local MCP process, as NAT documents it: `command` + `args` + `transport: stdio`.

    None of the address or credential machinery applies -- a child process has no URL
    and no identity -- so these pin that it stays out of the way rather than being
    threaded through with nulls.
    """

    def test_a_stdio_block_is_valid_with_no_url_and_no_fleet(self):
        server = DataRobotMCPServerConfig(
            transport="stdio", command="python", args=["-m", "mcp_server_time"]
        )
        assert server.is_stdio
        assert server.url is None

    def test_transport_stdio_without_a_command_raises(self):
        with pytest.raises(ValueError, match="no `command`"):
            DataRobotMCPServerConfig(transport="stdio")

    def test_a_command_without_stdio_transport_raises(self):
        with pytest.raises(ValueError, match="transport: stdio"):
            DataRobotMCPServerConfig(command="python", transport="streamable-http")

    def test_a_stdio_block_cannot_also_carry_a_url(self):
        with pytest.raises(ValueError, match="has no URL"):
            DataRobotMCPServerConfig(
                transport="stdio", command="python", url="https://x.example.com/mcp"
            )

    def test_a_stdio_block_cannot_name_an_auth_provider(self):
        # NAT supports auth_provider for streamable-http only, and a child process has
        # no identity to present.
        with pytest.raises(ValueError, match="streamable-http"):
            DataRobotMCPServerConfig(
                transport="stdio", command="python", auth_provider="datarobot_mcp_auth"
            )

    async def test_it_launches_the_process_and_never_touches_the_fleet(self):
        """No DATAROBOT_* in the environment and no configured server: still builds."""
        with patch.dict(os.environ, {}, clear=True):
            with patch(
                "datarobot_genai.dragent.plugins.datarobot_mcp_client.MCPStdioClient"
            ) as mock_stdio:
                mock_stdio.side_effect = lambda **kw: _FakeMCPClient(tools={}, url=None)
                config = DataRobotMCPClientConfig(
                    server=DataRobotMCPServerConfig(
                        transport="stdio", command="python", args=["-m", "mcp_server_time"]
                    )
                )
                async with WorkflowBuilder() as builder:
                    await builder.add_function_group("time_tools", config)
                    await builder.get_function_group("time_tools")

        assert mock_stdio.call_args.kwargs["command"] == "python"
        assert mock_stdio.call_args.kwargs["args"] == ["-m", "mcp_server_time"]


class TestCrossApplicationAccessNeedsNoServiceToken:
    """An XAA server presents an exchanged per-user token, not the service one.

    Which means it typically runs where no DATAROBOT_API_TOKEN exists at all -- that is
    much of the point of using XAA. Requiring one to *resolve* such a server conflated
    "has an identity" with "uses the DataRobot service token"; composing a URL needs the
    endpoint and never the token.
    """

    ENDPOINT = "https://app.datarobot.com/api/v2"

    def test_a_deployment_on_xaa_resolves_with_no_token(self):
        target = build_target(
            MCPServerRef(
                name="dr_user_mcp",
                deployment_id=DEPLOYMENT_ID,
                auth_provider="okta_auth_of_mcp",
            ),
            datarobot_endpoint=self.ENDPOINT,
            datarobot_api_token=None,
        )
        assert target.url.endswith(f"/deployments/{DEPLOYMENT_ID}/directAccess/mcp")
        assert target.api_token is None
        # ...and no service bearer is attached on the way out.
        assert build_headers(target) == {}

    def test_a_deployment_on_the_service_token_still_demands_one(self):
        with pytest.raises(ValueError, match="DATAROBOT_API_TOKEN"):
            build_target(
                MCPServerRef(name="analytics", deployment_id=DEPLOYMENT_ID),
                datarobot_endpoint=self.ENDPOINT,
                datarobot_api_token=None,
            )

    def test_the_endpoint_is_still_required_because_it_composes_the_url(self):
        with pytest.raises(ValueError, match="DATAROBOT_ENDPOINT"):
            build_target(
                MCPServerRef(
                    name="dr_user_mcp",
                    deployment_id=DEPLOYMENT_ID,
                    auth_provider="okta_auth_of_mcp",
                ),
                datarobot_endpoint=None,
                datarobot_api_token=None,
            )
