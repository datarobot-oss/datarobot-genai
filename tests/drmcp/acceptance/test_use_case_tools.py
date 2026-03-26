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

"""Acceptance tests for use case MCP tools (list_use_cases, list_use_case_assets)."""

import inspect
from typing import Any

import pytest

from datarobot_genai.drmcp import ETETestExpectations
from datarobot_genai.drmcp import ToolBaseE2E
from datarobot_genai.drmcp import ToolCallTestExpectations
from datarobot_genai.drmcp import ete_test_mcp_session
from datarobot_genai.drmcp.test_utils.tool_base_ete import SHOULD_NOT_BE_EMPTY


@pytest.fixture(scope="session")
def expectations_for_list_use_cases_success() -> ETETestExpectations:
    return ETETestExpectations(
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="list_use_cases",
                parameters={},
                result={"use_cases": {}},
            ),
        ],
        llm_response_content_contains_expectations=[
            "use case",
            "use cases",
            "cases",
        ],
    )


@pytest.fixture(scope="session")
def expectations_for_list_use_cases_with_search_success() -> ETETestExpectations:
    return ETETestExpectations(
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="list_use_cases",
                parameters={"search": "fraud"},
                result=SHOULD_NOT_BE_EMPTY,
            ),
        ],
        llm_response_content_contains_expectations=[
            "use case",
            "fraud",
            "search",
        ],
    )


@pytest.fixture(scope="session")
def expectations_for_list_use_case_assets_success(
    use_case_id: str,
) -> ETETestExpectations:
    return ETETestExpectations(
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="list_use_case_assets",
                parameters={"use_case_id": use_case_id},
                result=SHOULD_NOT_BE_EMPTY,
            ),
        ],
        llm_response_content_contains_expectations=[
            "use case",
            "asset",
            "dataset",
            "deployment",
            "experiment",
        ],
    )


@pytest.fixture(scope="session")
def use_case_id() -> str:
    """Use case ID for acceptance tests; can be overridden via environment or conftest."""
    import os

    return os.environ.get("TEST_USE_CASE_ID", "test_use_case_123")


@pytest.mark.skip(reason="MODEL-22978 - TODO: Fix tests")
@pytest.mark.asyncio
class TestUseCaseToolsE2E(ToolBaseE2E):
    """End-to-end tests for use case tools (list_use_cases, list_use_case_assets)."""

    @pytest.mark.parametrize(
        "prompt",
        [
            """
        I need to see all the DataRobot use cases I have access to. Can you list all the
        use cases available in DataRobot?
        """
        ],
    )
    async def test_list_use_cases(
        self,
        llm_client: Any,
        expectations_for_list_use_cases_success: ETETestExpectations,
        prompt: str,
    ) -> None:
        """E2E test: LLM lists all use cases via list_use_cases tool."""
        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = frame.f_code.co_name if frame else "test_list_use_cases"
            await self._run_test_with_expectations(
                prompt,
                expectations_for_list_use_cases_success,
                llm_client,
                session,
                test_name,
            )

    @pytest.mark.parametrize(
        "search_term",
        ["fraud"],
    )
    async def test_list_use_cases_with_search(
        self,
        llm_client: Any,
        expectations_for_list_use_cases_with_search_success: ETETestExpectations,
        search_term: str,
    ) -> None:
        """E2E test: LLM searches for use cases matching a keyword via list_use_cases tool."""
        prompt = (
            f"I need to find DataRobot use cases related to '{search_term}'. "
            f"Can you search for use cases containing '{search_term}' in their name?"
        )
        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = frame.f_code.co_name if frame else "test_list_use_cases_with_search"
            await self._run_test_with_expectations(
                prompt,
                expectations_for_list_use_cases_with_search_success,
                llm_client,
                session,
                test_name,
            )

    @pytest.mark.parametrize(
        "prompt_template",
        [
            """
        I have a DataRobot use case with ID '{use_case_id}'. Can you list all the assets
        (datasets, deployments, and experiments) belonging to that use case?
        """
        ],
    )
    async def test_list_use_case_assets(
        self,
        llm_client: Any,
        expectations_for_list_use_case_assets_success: ETETestExpectations,
        use_case_id: str,
        prompt_template: str,
    ) -> None:
        """E2E test: LLM lists assets in a use case via list_use_case_assets tool."""
        prompt = prompt_template.format(use_case_id=use_case_id)
        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = frame.f_code.co_name if frame else "test_list_use_case_assets"
            await self._run_test_with_expectations(
                prompt,
                expectations_for_list_use_case_assets_success,
                llm_client,
                session,
                test_name,
            )
