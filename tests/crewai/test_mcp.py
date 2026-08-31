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
import socket
from types import SimpleNamespace
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from crewai.utilities.agent_utils import convert_tools_to_openai_schema
from mcp.types import Tool
from pydantic import BaseModel

from datarobot_genai.core.mcp import MCPServerRef
from datarobot_genai.core.mcp import build_target
from datarobot_genai.crewai.mcp import _EMPTY_OBJECT_SCHEMA
from datarobot_genai.crewai.mcp import _local_server_reachable
from datarobot_genai.crewai.mcp import _RawSchemaCrewAIAdapter
from datarobot_genai.crewai.mcp import mcp_tools_context


@pytest.fixture
def mock_tools():
    return [MagicMock(), MagicMock()]


@pytest.fixture
def mock_adapter(mock_tools):
    """Fixture for mocking mcpadapt's MCPAdapt."""
    with patch("datarobot_genai.crewai.mcp.MCPAdapt") as mock:
        mock_adapter_instance = MagicMock()
        mock_adapter_instance.__enter__.return_value = mock_tools
        mock_adapter_instance.__exit__.return_value = None
        mock.return_value = mock_adapter_instance
        yield mock


@pytest.fixture(autouse=True)
def clear_environment_variables():
    with patch.dict(os.environ, {}, clear=True):
        yield


class TestRawSchemaAdapter:
    """The adapter keeps the MCP server's raw inputSchema as the tool schema, instead of
    mcpadapt's lossy pydantic round-trip (which drops property types -> azure rejects).
    """

    def test_preserves_raw_input_schema(self):
        raw = {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "default": 3},
            },
            "required": ["query"],
        }

        class _Base(BaseModel):
            query: str
            max_results: int = 3

        tool = MagicMock()
        tool.args_schema = _Base
        out = _RawSchemaCrewAIAdapter._keep_raw_schema(tool, SimpleNamespace(inputSchema=raw))
        # the LLM-facing schema is the raw one (every property typed), not the lossy model's
        assert out.args_schema.model_json_schema() == raw

    def test_missing_input_schema_falls_back_to_empty_object(self):
        tool = MagicMock()
        tool.args_schema = None
        out = _RawSchemaCrewAIAdapter._keep_raw_schema(tool, SimpleNamespace(inputSchema=None))
        assert out.args_schema.model_json_schema() == _EMPTY_OBJECT_SCHEMA

    def test_native_tool_call_schema_keeps_types_and_drops_null_keys(self):
        # End-to-end through CrewAI's native converter (not just model_json_schema()): the
        # LLM-facing function parameters must keep every property's ``type`` and carry no
        # null-valued keys (``enum``/``items``/``default``: null) that azure rejects. Without the
        # raw-schema override the stock pydantic round-trip drops ``type`` and injects those null
        # keys -- so this pins the regression the adapter fixes, below model_json_schema()'s reach.
        raw = {
            "type": "object",
            "properties": {
                "max_results": {"type": "integer", "default": 3},
                "filters": {"type": "object", "properties": {"lang": {"type": "string"}}},
            },
            "required": [],
        }
        tool = _RawSchemaCrewAIAdapter().adapt(
            lambda _args=None: "ok",
            Tool(name="searcher", description="s", inputSchema=raw),
        )
        fn_schemas, _fns, _lookup = convert_tools_to_openai_schema([tool])
        params = fn_schemas[0]["function"]["parameters"]

        assert params["properties"]["max_results"]["type"] == "integer"
        assert params["properties"]["filters"]["properties"]["lang"]["type"] == "string"

        def null_valued_keys(node):
            if isinstance(node, dict):
                here = [k for k, v in node.items() if v is None]
                return here + [p for v in node.values() for p in null_valued_keys(v)]
            if isinstance(node, list):
                return [p for v in node for p in null_valued_keys(v)]
            return []

        assert null_valued_keys(params) == []


class TestMCPToolsContext:
    """Test MCP tools context manager."""

    EXTERNAL_URL = "https://mcp-server.example.com/mcp"
    DEPLOYMENT_ID = "abc123def456789012345678"
    API_BASE = "https://app.datarobot.com/api/v2"
    API_KEY = "test-api-key"

    def external_target(self, name="partner"):
        return build_target(MCPServerRef(name=name, url=self.EXTERNAL_URL))

    def deployment_target(self):
        return build_target(
            MCPServerRef(name="analytics", deployment_id=self.DEPLOYMENT_ID),
            datarobot_endpoint=self.API_BASE,
            datarobot_api_token=self.API_KEY,
        )

    async def test_a_third_party_server_is_connected(self, mock_adapter, mock_tools):
        async with mcp_tools_context(self.external_target(), prefix="") as tools:
            assert tools == mock_tools
            mock_adapter.assert_called_once()
            # Check that the server config was passed correctly (first positional arg)
            call_args = mock_adapter.call_args[0][0]
            assert call_args["url"] == self.EXTERNAL_URL
            assert call_args["transport"] == "streamable-http"

    async def test_a_deployment_is_connected_with_datarobot_credentials(
        self, mock_adapter, agent_auth_context_data, mock_tools
    ):
        async with mcp_tools_context(
            self.deployment_target(), prefix="", auth_context=agent_auth_context_data
        ) as tools:
            assert tools == mock_tools
            mock_adapter.assert_called_once()
            call_args = mock_adapter.call_args[0][0]
            assert call_args["url"] == (
                f"{self.API_BASE}/deployments/{self.DEPLOYMENT_ID}/directAccess/mcp"
            )
            assert call_args["transport"] == "streamable-http"
            assert call_args["headers"]["Authorization"] == f"Bearer {self.API_KEY}"
            assert call_args["headers"]["X-DataRobot-Authorization-Context"] is not None

    async def test_a_forwarded_scoped_token_is_sent(
        self, mock_adapter, agent_auth_context_data, mock_tools
    ):
        async with mcp_tools_context(
            self.deployment_target(),
            prefix="",
            auth_context=agent_auth_context_data,
            forwarded={"x-datarobot-api-key": "scoped-token-123"},
        ) as tools:
            assert tools == mock_tools
            call_args = mock_adapter.call_args[0][0]
            assert call_args["headers"]["x-datarobot-api-key"] == "scoped-token-123"
            assert call_args["headers"]["Authorization"] == f"Bearer {self.API_KEY}"

    async def test_tools_are_namespaced_by_server_name(self, mock_adapter, mock_tools):
        # Two servers each exposing the same tool would otherwise collide.
        for tool, name in zip(mock_tools, ["search", "fetch"], strict=True):
            tool.name = name
        async with mcp_tools_context(self.external_target(name="analytics")) as tools:
            assert [t.name for t in tools] == ["analytics__search", "analytics__fetch"]

    @pytest.mark.usefixtures("mock_adapter")
    async def test_a_consumer_exception_is_propagated(self):
        """Exceptions raised inside the context body propagate (not swallowed)."""
        with pytest.raises(RuntimeError):
            async with mcp_tools_context(self.external_target()):
                raise RuntimeError("Connection failed")

    async def test_an_unreachable_server_raises_by_default(self):
        with patch("datarobot_genai.crewai.mcp.MCPAdapt", side_effect=ConnectionError("refused")):
            with pytest.raises(ConnectionError):
                async with mcp_tools_context(self.external_target()):
                    pass

    async def test_strict_false_restores_the_degrade_quietly_behaviour(self):
        with patch("datarobot_genai.crewai.mcp.MCPAdapt", side_effect=ConnectionError("refused")):
            async with mcp_tools_context(self.external_target(), strict=False) as tools:
                assert tools == []

    async def test_unreachable_local_server_skips_adapter(self):
        """A local MCP server that isn't running short-circuits before the adapter is
        built, avoiding the ~30s blocking connect and background-thread traceback.
        """
        # Bound-but-not-listening socket: connections are refused, and holding it open
        # reserves the port so the OS can't reuse it mid-test.
        bound = socket.socket()
        bound.bind(("127.0.0.1", 0))
        closed_port = bound.getsockname()[1]
        try:
            target = build_target(
                MCPServerRef(name="docs", local_port=closed_port), datarobot_api_token="tok"
            )
            with patch("datarobot_genai.crewai.mcp.MCPAdapt") as mock_adapter:
                with pytest.raises(ConnectionError, match="docs"):
                    async with mcp_tools_context(target):
                        pass
                async with mcp_tools_context(target, strict=False) as tools:
                    assert tools == []
                mock_adapter.assert_not_called()
        finally:
            bound.close()


class TestLocalServerReachable:
    def test_reachable_when_listening(self):
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(128)
        port = listener.getsockname()[1]
        try:
            assert _local_server_reachable(f"http://localhost:{port}/mcp") is True
        finally:
            listener.close()

    def test_unreachable_when_closed(self):
        # Bound-but-not-listening socket: connection refused; holding it open reserves
        # the port so the OS can't reuse it mid-test.
        bound = socket.socket()
        bound.bind(("127.0.0.1", 0))
        port = bound.getsockname()[1]
        try:
            assert _local_server_reachable(f"http://localhost:{port}/mcp") is False
        finally:
            bound.close()
