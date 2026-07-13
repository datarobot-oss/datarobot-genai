# Copyright 2026 DataRobot, Inc.
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
from collections.abc import Iterator
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from fastmcp.tools.base import Tool
from fastmcp.tools.tool import ToolResult

from datarobot_genai.drmcpbase.auth.enums import DataRobotBearerHeaderEnum
from datarobot_genai.drmcpbase.auth.exceptions import (
    NoDataRobotBearerTokenFoundInRequestContextError,
)
from datarobot_genai.drmcpbase.dynamic_tools.deployment.discovery import is_tool_tagged
from datarobot_genai.drmcpbase.dynamic_tools.enums import DataRobotMCPToolCategory
from datarobot_genai.drmcpbase.mcp_providers.custom_model_tool_provider import (
    CustomModelToolProvider,
)
from datarobot_genai.drmcpbase.mcp_providers.custom_model_tool_provider import _cache_key

MODULE = "datarobot_genai.drmcpbase.mcp_providers.custom_model_tool_provider"
ENDPOINT = "https://dr.example.com/api/v2"
TOKEN = "user-token-1"
DEPLOYMENT_ID = "68aaaaaaaaaaaaaaaaaaaaaa"


class TestIsToolTagged:
    def test_matching_tag(self) -> None:
        assert is_tool_tagged([{"name": "tool", "value": "tool"}])

    def test_or_matched_results_are_rejected(self) -> None:
        # The deployments API ORs tag filters: these match server-side but
        # must be rejected by the AND predicate.
        assert not is_tool_tagged([{"name": "tool", "value": "other"}])
        assert not is_tool_tagged([{"name": "other", "value": "tool"}])

    def test_empty_and_none(self) -> None:
        assert not is_tool_tagged([])
        assert not is_tool_tagged(None)

    def test_match_among_many(self) -> None:
        assert is_tool_tagged([{"name": "env", "value": "prod"}, {"name": "tool", "value": "tool"}])


@pytest.fixture
def provider() -> CustomModelToolProvider:
    return CustomModelToolProvider(ENDPOINT)


@pytest.fixture
def mock_api_client(provider: CustomModelToolProvider) -> Mock:
    client = Mock()
    client._list_mcp_tool_custom_model_deployment_ids = AsyncMock(return_value=[DEPLOYMENT_ID])
    provider.datarobot_api_client = client
    return client


@pytest.fixture
def mock_gates() -> Iterator[tuple[Mock, AsyncMock]]:
    with (
        patch(f"{MODULE}.is_category_disabled_for_request", return_value=False) as mock_disabled,
        patch(
            f"{MODULE}.check_mcp_tools_gallery_support", new_callable=AsyncMock, return_value=True
        ) as mock_flag,
    ):
        yield mock_disabled, mock_flag


@pytest.fixture
def mock_token() -> Iterator[Mock]:
    with patch.object(
        DataRobotBearerHeaderEnum.X_DATAROBOT_AUTHORIZATION,
        "get_from_mcp_request",
        return_value=TOKEN,
    ) as mock_get:
        yield mock_get


def _make_tool(name: str = "demo_tool") -> Tool:
    async def fn() -> ToolResult:  # pragma: no cover - never called
        return ToolResult(content="ok")

    return Tool.from_function(
        fn=fn,
        name=name,
        meta={"tool_category": DataRobotMCPToolCategory.USER_TOOL_DEPLOYMENT.name},
    )


class TestListTools:
    @pytest.mark.asyncio
    async def test_lists_tools_for_tagged_deployments(
        self,
        provider: CustomModelToolProvider,
        mock_api_client: Mock,
        mock_gates: tuple[Mock, AsyncMock],
        mock_token: Mock,
    ) -> None:
        tool = _make_tool()
        with patch.object(
            provider, "_build_tool", new_callable=AsyncMock, return_value=tool
        ) as mock_build:
            tools = await provider._list_tools()

        assert tools == [tool]
        mock_build.assert_awaited_once_with(DEPLOYMENT_ID, TOKEN)
        assert tools[0].meta == {
            "tool_category": DataRobotMCPToolCategory.USER_TOOL_DEPLOYMENT.name
        }

    @pytest.mark.asyncio
    async def test_new_tag_appears_without_restart(
        self,
        provider: CustomModelToolProvider,
        mock_api_client: Mock,
        mock_gates: tuple[Mock, AsyncMock],
        mock_token: Mock,
    ) -> None:
        """The zero-restart contract: each listing re-reads tagged deployments."""
        second_id = "68bbbbbbbbbbbbbbbbbbbbbb"
        mock_api_client._list_mcp_tool_custom_model_deployment_ids = AsyncMock(
            side_effect=[[DEPLOYMENT_ID], [DEPLOYMENT_ID, second_id]]
        )
        with patch.object(
            provider,
            "_build_tool",
            new_callable=AsyncMock,
            side_effect=lambda dep_id, token: _make_tool(f"tool_{dep_id[-4:]}"),
        ):
            first = await provider._list_tools()
            second = await provider._list_tools()

        assert [t.name for t in first] == [f"tool_{DEPLOYMENT_ID[-4:]}"]
        assert [t.name for t in second] == [
            f"tool_{DEPLOYMENT_ID[-4:]}",
            f"tool_{second_id[-4:]}",
        ]

    @pytest.mark.asyncio
    async def test_category_disabled_for_request_skips_everything(
        self,
        provider: CustomModelToolProvider,
        mock_api_client: Mock,
        mock_token: Mock,
    ) -> None:
        with patch(f"{MODULE}.is_category_disabled_for_request", return_value=True):
            tools = await provider._list_tools()
        assert tools == []
        mock_api_client._list_mcp_tool_custom_model_deployment_ids.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_gallery_flag_disabled_returns_no_tools(
        self,
        provider: CustomModelToolProvider,
        mock_api_client: Mock,
        mock_token: Mock,
    ) -> None:
        with (
            patch(f"{MODULE}.is_category_disabled_for_request", return_value=False),
            patch(
                f"{MODULE}.check_mcp_tools_gallery_support",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            tools = await provider._list_tools()
        assert tools == []
        mock_api_client._list_mcp_tool_custom_model_deployment_ids.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_missing_token_returns_no_tools(
        self,
        provider: CustomModelToolProvider,
        mock_api_client: Mock,
        mock_gates: tuple[Mock, AsyncMock],
    ) -> None:
        with patch.object(
            DataRobotBearerHeaderEnum.X_DATAROBOT_AUTHORIZATION,
            "get_from_mcp_request",
            side_effect=NoDataRobotBearerTokenFoundInRequestContextError("no token"),
        ):
            tools = await provider._list_tools()
        assert tools == []

    @pytest.mark.asyncio
    async def test_one_broken_deployment_does_not_hide_the_rest(
        self,
        provider: CustomModelToolProvider,
        mock_api_client: Mock,
        mock_gates: tuple[Mock, AsyncMock],
        mock_token: Mock,
    ) -> None:
        broken_id = "68cccccccccccccccccccccc"
        mock_api_client._list_mcp_tool_custom_model_deployment_ids = AsyncMock(
            return_value=[broken_id, DEPLOYMENT_ID]
        )
        good_tool = _make_tool()

        async def build(dep_id: str, token: str) -> Tool:
            if dep_id == broken_id:
                raise ValueError("deployment has no usable inputSchema")
            return good_tool

        with patch.object(provider, "_build_tool", new_callable=AsyncMock, side_effect=build):
            tools = await provider._list_tools()
        assert tools == [good_tool]

    @pytest.mark.asyncio
    async def test_before_lifespan_returns_no_tools(
        self,
        provider: CustomModelToolProvider,
        mock_gates: tuple[Mock, AsyncMock],
        mock_token: Mock,
    ) -> None:
        assert provider.datarobot_api_client is None
        tools = await provider._list_tools()
        assert tools == []


class TestToolCache:
    @pytest.mark.asyncio
    async def test_cache_hit_skips_rebuild(
        self,
        provider: CustomModelToolProvider,
        mock_api_client: Mock,
        mock_gates: tuple[Mock, AsyncMock],
        mock_token: Mock,
    ) -> None:
        tool = _make_tool()
        with patch.object(
            provider, "_build_tool", new_callable=AsyncMock, return_value=tool
        ) as mock_build:
            await provider._list_tools()
            await provider._list_tools()
        mock_build.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_cache_isolates_users(
        self,
        provider: CustomModelToolProvider,
        mock_api_client: Mock,
        mock_gates: tuple[Mock, AsyncMock],
    ) -> None:
        """Tools carry per-user auth headers: another token must rebuild."""
        with (
            patch.object(
                provider, "_build_tool", new_callable=AsyncMock, return_value=_make_tool()
            ) as mock_build,
            patch.object(
                DataRobotBearerHeaderEnum.X_DATAROBOT_AUTHORIZATION,
                "get_from_mcp_request",
            ) as mock_get,
        ):
            mock_get.return_value = "token-of-user-a"
            await provider._list_tools()
            mock_get.return_value = "token-of-user-b"
            await provider._list_tools()
        assert mock_build.await_count == 2

    def test_cache_key_does_not_retain_raw_token(self) -> None:
        key = _cache_key(DEPLOYMENT_ID, TOKEN)
        assert TOKEN not in key
        assert key[0] == DEPLOYMENT_ID
        # Deterministic so lookups hit.
        assert key == _cache_key(DEPLOYMENT_ID, TOKEN)
        assert key != _cache_key(DEPLOYMENT_ID, "another-token")
