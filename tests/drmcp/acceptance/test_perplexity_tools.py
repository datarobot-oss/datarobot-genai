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


@pytest.fixture(scope="session")
def search_query() -> str:
    return "What is DataRobot?"


@pytest.fixture(scope="session")
def search_no_of_results() -> int:
    return 1


@pytest.fixture(scope="session")
def search_domain_filter() -> str:
    return "datarobot.com"


@pytest.fixture(scope="session")
def expectations_for_perplexity_search_success(
    search_query: str, search_no_of_results: int, search_domain_filter: str
) -> ETETestExpectations:
    return ETETestExpectations(
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="perplexity_search",
                parameters={
                    "query": search_query,
                    "search_domain_filter": [search_domain_filter],
                    "max_results": search_no_of_results,
                },
                result=SHOULD_NOT_BE_EMPTY,
            ),
        ],
        llm_response_content_contains_expectations=[
            f"Successfully executed search for '{search_query}'.",
            f"Found {search_no_of_results} result(s).",
            "DataRobot",
            "Platform",
        ],
    )


@pytest.mark.skipif(
    not os.getenv("ENABLE_PERPLEXITY_TOOLS"), reason="Perplexity tools are not enabled"
)
@pytest.mark.asyncio
class TestPerplexityToolsE2E(ToolBaseE2E):
    """End-to-end tests for Perplexity tools."""

    @pytest.mark.parametrize(
        "prompt_template",
        [
            "Please search web using perplexity."
            "I need to find out '{query}'. Search only in '{domain}' domain."
            "Give {number_of_results} of results."
        ],
    )
    async def test_perplexity_search_success(
        self,
        llm_client: Any,
        expectations_for_perplexity_search_success: ETETestExpectations,
        search_query: str,
        search_domain_filter: str,
        search_no_of_results: int,
        prompt_template: str,
    ) -> None:
        prompt = prompt_template.format(
            query=search_query,
            domain=search_domain_filter,
            number_of_results=search_no_of_results,
        )

        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = frame.f_code.co_name if frame else "test_perplexity_search_success"
            await self._run_test_with_expectations(
                prompt,
                expectations_for_perplexity_search_success,
                llm_client,
                session,
                test_name,
            )
