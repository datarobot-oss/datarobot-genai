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
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from datarobot_genai.core.mcp import MCPServerRef
from datarobot_genai.core.mcp import build_target
from datarobot_genai.llama_index.mcp import mcp_tools_context


@pytest.fixture(autouse=True)
def empty_agent_auth_context():
    with patch.dict(os.environ, {}, clear=True):
        yield


@pytest.fixture
def mock_aget():
    with patch(
        "datarobot_genai.llama_index.mcp.aget_tools_from_mcp_url", new_callable=AsyncMock
    ) as mock:
        yield mock


class TestLoadMCPTools:
    """Test async MCP tools loading."""

    EXTERNAL_URL = "https://mcp-server.example.com/mcp"
    DEPLOYMENT_ID = "abc123def456789012345678"
    API_BASE = "https://app.datarobot.com/api/v2"
    API_KEY = "test-api-key"

    def external_target(self, name="vendor"):
        return build_target(MCPServerRef(name=name, url=self.EXTERNAL_URL))

    def deployment_target(self, endpoint=None, token=None):
        return build_target(
            MCPServerRef(name="analytics", deployment_id=self.DEPLOYMENT_ID),
            datarobot_endpoint=endpoint or self.API_BASE,
            datarobot_api_token=token or self.API_KEY,
        )

    async def test_a_deployment_is_connected_with_datarobot_credentials(self, mock_aget):
        mock_tools = [MagicMock()]
        mock_aget.return_value = mock_tools

        async with mcp_tools_context(self.deployment_target(), prefix="") as tools:
            assert tools == mock_tools
            mock_aget.assert_awaited_once()
            call_args = mock_aget.await_args
            assert call_args[1]["command_or_url"] == (
                f"{self.API_BASE}/deployments/{self.DEPLOYMENT_ID}/directAccess/mcp"
            )
            assert call_args[1]["client"].headers["Authorization"] == f"Bearer {self.API_KEY}"

    async def test_no_tools_is_an_empty_list_not_none(self, mock_aget):
        mock_aget.return_value = None
        async with mcp_tools_context(self.external_target(), prefix="") as tools:
            assert tools == []

    async def test_the_endpoint_and_token_come_from_the_target(self, mock_aget):
        mock_tools = [MagicMock()]
        mock_aget.return_value = mock_tools

        target = self.deployment_target(
            endpoint="https://custom.datarobot.com/api/v2", token="custom-key"
        )
        async with mcp_tools_context(target, prefix="") as tools:
            assert tools == mock_tools
            call_args = mock_aget.await_args
            assert call_args[1]["command_or_url"] == (
                f"https://custom.datarobot.com/api/v2/deployments/{self.DEPLOYMENT_ID}"
                "/directAccess/mcp"
            )
            assert call_args[1]["client"].headers["Authorization"] == "Bearer custom-key"

    async def test_a_forwarded_scoped_token_is_sent(self, mock_aget):
        mock_aget.return_value = [MagicMock()]
        async with mcp_tools_context(
            self.deployment_target(),
            prefix="",
            forwarded={"x-datarobot-api-key": "scoped-token-123"},
        ):
            client_headers = mock_aget.await_args[1]["client"].headers
        assert client_headers["x-datarobot-api-key"] == "scoped-token-123"
        assert client_headers["Authorization"] == f"Bearer {self.API_KEY}"

    async def test_tools_are_namespaced_by_server_name(self, mock_aget):
        # Two servers each exposing the same tool would otherwise collide.
        tool = MagicMock()
        tool.metadata.name = "search"
        mock_aget.return_value = [tool]
        async with mcp_tools_context(self.external_target(name="analytics")) as tools:
            assert [t.metadata.name for t in tools] == ["analytics__search"]

    @pytest.mark.usefixtures("mock_aget")
    async def test_a_consumer_exception_is_propagated(self):
        with pytest.raises(RuntimeError):
            async with mcp_tools_context(self.external_target()):
                raise RuntimeError("Connection failed")

    async def test_an_unreachable_server_raises_by_default(self):
        with patch(
            "datarobot_genai.llama_index.mcp.BasicMCPClient", side_effect=ConnectionError("refused")
        ):
            with pytest.raises(ConnectionError):
                async with mcp_tools_context(self.external_target()):
                    pass

    async def test_strict_false_restores_the_degrade_quietly_behaviour(self):
        with patch(
            "datarobot_genai.llama_index.mcp.BasicMCPClient", side_effect=ConnectionError("refused")
        ):
            async with mcp_tools_context(self.external_target(), strict=False) as tools:
                assert tools == []
