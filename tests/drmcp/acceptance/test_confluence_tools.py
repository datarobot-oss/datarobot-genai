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
from typing import Any

import pytest

from datarobot_genai.drmcp import ETETestExpectations
from datarobot_genai.drmcp import ToolBaseE2E
from datarobot_genai.drmcp import ToolCallTestExpectations
from datarobot_genai.drmcp import ete_test_mcp_session
from datarobot_genai.drmcp.test_utils.tool_base_ete import SHOULD_NOT_BE_EMPTY


@pytest.fixture(scope="session")
def confluence_page_id() -> str:
    return "856391684"


@pytest.fixture(scope="session")
def confluence_page_title() -> str:
    return "All check status"


@pytest.fixture(scope="session")
def confluence_space_key() -> str:
    return "ENGPROD"


@pytest.fixture(scope="session")
def expectations_for_confluence_get_page_by_id_success(
    confluence_page_id: str,
) -> ETETestExpectations:
    return ETETestExpectations(
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="confluence_get_page",
                parameters={"page_id_or_title": confluence_page_id},
                result=SHOULD_NOT_BE_EMPTY,
            ),
        ],
        llm_response_content_contains_expectations=[
            "All check status",
            "0-all-check-status",
        ],
    )


@pytest.fixture(scope="session")
def expectations_for_confluence_get_page_by_title_success(
    confluence_page_title: str,
    confluence_space_key: str,
) -> ETETestExpectations:
    return ETETestExpectations(
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="confluence_get_page",
                parameters={
                    "page_id_or_title": confluence_page_title,
                    "space_key": confluence_space_key,
                },
                result=SHOULD_NOT_BE_EMPTY,
            ),
        ],
        llm_response_content_contains_expectations=[
            "All check status",
            "0-all-check-status",
        ],
    )


@pytest.mark.asyncio
class TestConfluenceToolsE2E(ToolBaseE2E):
    """End-to-end tests for confluence tools."""

    @pytest.mark.parametrize(
        "prompt_template",
        ["Please retrieve the Confluence page with ID `{page_id}`."],
    )
    async def test_confluence_get_page_by_id_success(
        self,
        openai_llm_client: Any,
        expectations_for_confluence_get_page_by_id_success: ETETestExpectations,
        confluence_page_id: str,
        prompt_template: str,
    ) -> None:
        prompt = prompt_template.format(page_id=confluence_page_id)

        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = frame.f_code.co_name if frame else "test_confluence_get_page_by_id_success"
            await self._run_test_with_expectations(
                prompt,
                expectations_for_confluence_get_page_by_id_success,
                openai_llm_client,
                session,
                test_name,
            )

    @pytest.mark.parametrize(
        "prompt_template",
        ["Please retrieve the Confluence page titled `{page_title}` from space `{space_key}`."],
    )
    async def test_confluence_get_page_by_title_success(
        self,
        openai_llm_client: Any,
        expectations_for_confluence_get_page_by_title_success: ETETestExpectations,
        confluence_page_title: str,
        confluence_space_key: str,
        prompt_template: str,
    ) -> None:
        prompt = prompt_template.format(
            page_title=confluence_page_title,
            space_key=confluence_space_key,
        )

        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = (
                frame.f_code.co_name if frame else "test_confluence_get_page_by_title_success"
            )
            await self._run_test_with_expectations(
                prompt,
                expectations_for_confluence_get_page_by_title_success,
                openai_llm_client,
                session,
                test_name,
            )
