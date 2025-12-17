import inspect
from typing import Any

import pytest

from datarobot_genai.drmcp import (
    ToolBaseE2E,
    ETETestExpectations,
    ToolCallTestExpectations,
    ete_test_mcp_session,
)
from datarobot_genai.drmcp.test_utils.tool_base_ete import (
    SHOULD_NOT_BE_EMPTY,
)


@pytest.fixture(scope="session")
def jira_issue_key() -> str:
    return "MODEL-21631"


@pytest.fixture(scope="session")
def expectations_for_jira_get_issue_success(jira_issue_key: str) -> ETETestExpectations:
    return ETETestExpectations(
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="jira_get_issue",
                parameters={"issue_key": jira_issue_key},
                result=SHOULD_NOT_BE_EMPTY,
            ),
        ],
        llm_response_content_contains_expectations=[
            "summary:",
            "[Jira] Get Issue Tool",
        ],
    )


@pytest.mark.asyncio
class TestJiraToolsE2E(ToolBaseE2E):
    """End-to-end tests for jira tools."""

    @pytest.mark.parametrize(
        "prompt_template",
        ["Please give me information about jira ticket `{jira_issue_key}`."],
    )
    async def test_jira_issue_get_success(
        self,
        openai_llm_client: Any,
        expectations_for_jira_get_issue_success: ETETestExpectations,
        jira_issue_key: str,
        prompt_template: str,
    ) -> None:
        prompt = prompt_template.format(jira_issue_key=jira_issue_key)

        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = frame.f_code.co_name if frame else "test_jira_issue_get_success"
            await self._run_test_with_expectations(
                prompt,
                expectations_for_jira_get_issue_success,
                openai_llm_client,
                session,
                test_name,
            )
