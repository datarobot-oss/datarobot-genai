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


@pytest.mark.skipif(not os.getenv("ENABLE_DR_DOCS_TOOLS"), reason="DR Docs tools are not enabled")
class TestDrDocsToolsE2E(ToolBaseE2E):
    """End-to-end tests for DataRobot Documentation search tools."""

    async def test_search_datarobot_docs_success(self, llm_client: Any) -> None:
        """Test basic DataRobot docs search."""
        expectations = ETETestExpectations(
            tool_calls_expected=[
                ToolCallTestExpectations(
                    name="search_datarobot_docs",
                    parameters={"query": "autopilot", "max_results": 3},
                    result=SHOULD_NOT_BE_EMPTY,
                ),
            ],
            llm_response_content_contains_expectations=["autopilot", "docs.datarobot.com"],
        )

        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = frame.f_code.co_name if frame else "test_search_datarobot_docs_success"
            await self._run_test_with_expectations(
                "Search the DataRobot documentation for 'autopilot' and list what you find.",
                expectations,
                llm_client,
                session,
                test_name,
            )

    async def test_fetch_datarobot_doc_page_success(self, llm_client: Any) -> None:
        """Test fetching a specific DataRobot documentation page."""
        expectations = ETETestExpectations(
            tool_calls_expected=[
                ToolCallTestExpectations(
                    name="fetch_datarobot_doc_page",
                    parameters={"url": "https://docs.datarobot.com/en/docs/modeling/"},
                    result=SHOULD_NOT_BE_EMPTY,
                ),
            ],
            llm_response_content_contains_expectations=["modeling", "DataRobot"],
        )

        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = frame.f_code.co_name if frame else "test_fetch_datarobot_doc_page_success"
            await self._run_test_with_expectations(
                "Fetch the content from 'https://docs.datarobot.com/en/docs/modeling/' "
                "and summarize what you find.",
                expectations,
                llm_client,
                session,
                test_name,
            )

    async def test_search_and_fetch_workflow(self, llm_client: Any) -> None:
        """Test the typical workflow: search for docs, then fetch a page."""
        expectations = ETETestExpectations(
            tool_calls_expected=[
                ToolCallTestExpectations(
                    name="search_datarobot_docs",
                    parameters={"query": "time series"},
                    result=SHOULD_NOT_BE_EMPTY,
                ),
                ToolCallTestExpectations(
                    name="fetch_datarobot_doc_page",
                    # Don't specify URL - let the LLM choose from search results
                    result=SHOULD_NOT_BE_EMPTY,
                ),
            ],
            llm_response_content_contains_expectations=["time series", "DataRobot"],
        )

        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = frame.f_code.co_name if frame else "test_search_and_fetch_workflow"
            await self._run_test_with_expectations(
                "Search the DataRobot documentation for 'time series' information, "
                "then fetch and summarize the most relevant page you find.",
                expectations,
                llm_client,
                session,
                test_name,
            )

    async def test_search_no_results(self, llm_client: Any) -> None:
        """Test search with a query that returns no results."""
        expectations = ETETestExpectations(
            tool_calls_expected=[
                ToolCallTestExpectations(
                    name="search_datarobot_docs",
                    parameters={"query": "xyznonexistentquery123"},
                    result=SHOULD_NOT_BE_EMPTY,
                ),
            ],
            llm_response_content_contains_expectations=["no", "found", "result"],
        )

        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = frame.f_code.co_name if frame else "test_search_no_results"
            await self._run_test_with_expectations(
                "Search the DataRobot documentation for 'xyznonexistentquery123'.",
                expectations,
                llm_client,
                session,
                test_name,
            )
