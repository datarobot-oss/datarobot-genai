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

import inspect
import os
from typing import Any

import pytest

from datarobot_genai.drmcp import ETETestExpectations
from datarobot_genai.drmcp import ToolBaseE2E
from datarobot_genai.drmcp import ToolCallTestExpectations
from datarobot_genai.drmcp import ete_test_mcp_session
from datarobot_genai.drmcp.test_utils.tool_base_ete import SHOULD_NOT_BE_EMPTY


@pytest.mark.skipif(not os.getenv("ENABLE_TAVILY_TOOLS"), reason="Tavily tools are not enabled")
@pytest.mark.asyncio
class TestTavilyToolsE2E(ToolBaseE2E):
    """End-to-end tests for Tavily search tools."""

    async def test_tavily_search_success(self, llm_client: Any) -> None:
        """Test basic Tavily search."""
        expectations = ETETestExpectations(
            tool_calls_expected=[
                ToolCallTestExpectations(
                    name="tavily_search",
                    parameters={"query": "DataRobot machine learning"},
                    result=SHOULD_NOT_BE_EMPTY,
                ),
            ],
            llm_response_content_contains_expectations=["DataRobot"],
        )

        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = frame.f_code.co_name if frame else "test_tavily_search_success"
            await self._run_test_with_expectations(
                "Search the web for 'DataRobot machine learning' and summarize what you find.",
                expectations,
                llm_client,
                session,
                test_name,
            )

    async def test_tavily_extract_success(self, llm_client: Any) -> None:
        """Test basic Tavily extract."""
        expectations = ETETestExpectations(
            tool_calls_expected=[
                ToolCallTestExpectations(
                    name="tavily_extract",
                    parameters={"urls": "https://docs.datarobot.com"},
                    result=SHOULD_NOT_BE_EMPTY,
                ),
            ],
            llm_response_content_contains_expectations=["DataRobot"],
        )

        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = frame.f_code.co_name if frame else "test_tavily_extract_success"
            await self._run_test_with_expectations(
                "Extract content from 'https://docs.datarobot.com' and summarize what you find.",
                expectations,
                llm_client,
                session,
                test_name,
            )

    async def test_tavily_map_success(self, llm_client: Any) -> None:
        """Test basic Tavily map."""
        expectations = ETETestExpectations(
            tool_calls_expected=[
                ToolCallTestExpectations(
                    name="tavily_map",
                    parameters={"url": "https://docs.datarobot.com", "limit": 5},
                    result=SHOULD_NOT_BE_EMPTY,
                ),
            ],
            llm_response_content_contains_expectations=["datarobot.com"],
        )

        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = frame.f_code.co_name if frame else "test_tavily_map_success"
            await self._run_test_with_expectations(
                "Map 'https://docs.datarobot.com' website. Limit to 5 results.",
                expectations,
                llm_client,
                session,
                test_name,
            )
