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
def expectations_for_microsoft_graph_search_content_success() -> ETETestExpectations:
    return ETETestExpectations(
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="microsoft_graph_search_content",
                parameters={
                    "search_query": "test query",
                },
                result=SHOULD_NOT_BE_EMPTY,
            ),
        ]
    )


@pytest.mark.skipif(
    not os.getenv("ENABLE_MICROSOFT_GRAPH_TOOLS"), reason="Microsoft Graph tools are not enabled"
)
@pytest.mark.asyncio
class TestMicrosoftGraphToolsE2E(ToolBaseE2E):
    """End-to-end tests for Microsoft Graph tools."""

    @pytest.mark.parametrize(
        "prompt_template",
        ["Please search for files in the Microsoft Graph with the query 'test query'."],
    )
    async def test_microsoft_graph_search_content_success(
        self,
        llm_client: Any,
        expectations_for_microsoft_graph_search_content_success: ETETestExpectations,
        prompt_template: str,
    ) -> None:
        prompt = prompt_template

        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = (
                frame.f_code.co_name if frame else "test_microsoft_graph_search_content_success"
            )
            await self._run_test_with_expectations(
                prompt,
                expectations_for_microsoft_graph_search_content_success,
                llm_client,
                session,
                test_name,
            )
