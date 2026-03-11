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
def search_no_of_results() -> int:
    return 1


@pytest.fixture(scope="session")
def project_key() -> str:
    return "MODEL"


@pytest.fixture(scope="session")
def jira_issue_key(project_key: str) -> str:
    return f"{project_key}-21631"


@pytest.fixture(scope="session")
def jira_new_ticket_name() -> str:
    return "[ACCEPTANCE TEST] New ticket"


@pytest.fixture(scope="session")
def jira_updated_ticket_name() -> str:
    return "[ACCEPTANCE TEST] Updated ticket"


@pytest.fixture(scope="session")
def jira_updated_ticket_status() -> str:
    return "Closed"


@pytest.fixture(scope="session")
def expectations_for_jira_search_issues_success(
    project_key: str, search_no_of_results: int
) -> ETETestExpectations:
    return ETETestExpectations(
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="jira_search_issues",
                parameters={
                    "jql_query": f"project = {project_key} AND summary ~ 'get issue tool'",
                    "max_results": search_no_of_results,
                },
                result=SHOULD_NOT_BE_EMPTY,
            ),
        ],
        llm_response_content_contains_expectations=[
            f"Successfully executed JQL query and retrieved {search_no_of_results} issue(s).",
            "[Jira] Get Issue Tool",
        ],
    )


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


@pytest.fixture(scope="session")
def expectations_for_jira_create_issue_success(
    project_key: str, jira_new_ticket_name: str
) -> ETETestExpectations:
    return ETETestExpectations(
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="jira_create_issue",
                parameters={
                    "project_key": project_key,
                    "summary": jira_new_ticket_name,
                    "issue_type": "Bug",
                },
                result=SHOULD_NOT_BE_EMPTY,
            ),
        ],
        llm_response_content_contains_expectations=[
            "Successfully created issue",
            f"{project_key}-",  # Cannot check exact ticket number
        ],
    )


@pytest.fixture(scope="session")
def expectations_for_jira_update_issue_success(
    jira_issue_key: str, jira_updated_ticket_name: str
) -> ETETestExpectations:
    return ETETestExpectations(
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="jira_update_issue",
                parameters={
                    "issue_key": jira_issue_key,
                    "fields": {"summary": jira_updated_ticket_name},
                },
                result=SHOULD_NOT_BE_EMPTY,
            ),
        ],
        llm_response_content_contains_expectations=["Successfully updated issue"],
    )


@pytest.fixture(scope="session")
def expectations_for_jira_transition_issue_success(
    jira_issue_key: str, jira_updated_ticket_status: str
) -> ETETestExpectations:
    return ETETestExpectations(
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="jira_transition_issue",
                parameters={
                    "issue_key": jira_issue_key,
                    "transition_name": jira_updated_ticket_status,
                },
                result=SHOULD_NOT_BE_EMPTY,
            ),
        ],
        llm_response_content_contains_expectations=["Successfully transitioned issue"],
    )


@pytest.mark.skipif(not os.getenv("ENABLE_JIRA_TOOLS"), reason="Jira tools are not enabled")
@pytest.mark.asyncio
class TestJiraToolsE2E(ToolBaseE2E):
    """End-to-end tests for jira tools."""

    @pytest.mark.parametrize(
        "prompt_template",
        [
            "Please search jira tickets from {project_key} project "
            "and with summary like '{summary_like}'. "
            "Give {number_of_results} of results."
        ],
    )
    async def test_jira_issue_search_success(
        self,
        llm_client: Any,
        expectations_for_jira_search_issues_success: ETETestExpectations,
        project_key: str,
        search_no_of_results: int,
        prompt_template: str,
    ) -> None:
        prompt = prompt_template.format(
            project_key=project_key,
            summary_like="get issue tool",
            number_of_results=search_no_of_results,
        )

        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = frame.f_code.co_name if frame else "test_jira_issue_search_success"
            await self._run_test_with_expectations(
                prompt,
                expectations_for_jira_search_issues_success,
                llm_client,
                session,
                test_name,
            )

    @pytest.mark.parametrize(
        "prompt_template",
        ["Please give me information about jira ticket `{jira_issue_key}`."],
    )
    async def test_jira_issue_get_success(
        self,
        llm_client: Any,
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
                llm_client,
                session,
                test_name,
            )

    @pytest.mark.skip(
        reason="Need to implement delete also to not 'spam' production jira with dummy tickets."
    )
    @pytest.mark.parametrize(
        "prompt_template",
        ["Please create bug `{ticket_name}` in jira for project `{project_key}`."],
    )
    async def test_jira_issue_create_success(
        self,
        llm_client: Any,
        expectations_for_jira_create_issue_success: ETETestExpectations,
        project_key: str,
        jira_new_ticket_name: str,
        prompt_template: str,
    ) -> None:
        prompt = prompt_template.format(ticket_name=jira_new_ticket_name, project_key=project_key)

        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = frame.f_code.co_name if frame else "test_jira_issue_create_success"
            await self._run_test_with_expectations(
                prompt,
                expectations_for_jira_create_issue_success,
                llm_client,
                session,
                test_name,
            )

    @pytest.mark.skip(
        reason="Need to implement delete also to not 'spam' production jira with dummy tickets."
    )
    @pytest.mark.parametrize(
        "prompt_template",
        ["Please change summary for ticket `{jira_issue_key}` to `{new_ticket_name}`.`"],
    )
    async def test_jira_issue_update_success(
        self,
        llm_client: Any,
        expectations_for_jira_update_issue_success: ETETestExpectations,
        jira_issue_key: str,
        jira_updated_ticket_name: str,
        prompt_template: str,
    ) -> None:
        prompt = prompt_template.format(
            jira_issue_key=jira_issue_key, new_ticket_name=jira_updated_ticket_name
        )

        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = frame.f_code.co_name if frame else "test_jira_issue_update_success"
            await self._run_test_with_expectations(
                prompt,
                expectations_for_jira_update_issue_success,
                llm_client,
                session,
                test_name,
            )

    @pytest.mark.skip(
        reason="Need to implement delete also to not 'spam' production jira with dummy tickets."
    )
    @pytest.mark.parametrize(
        "prompt_template",
        ["Please change status for ticket `{jira_issue_key}` to `{new_ticket_status}`.`"],
    )
    async def test_jira_issue_transition_success(
        self,
        llm_client: Any,
        expectations_for_jira_transition_issue_success: ETETestExpectations,
        jira_issue_key: str,
        jira_updated_ticket_status: str,
        prompt_template: str,
    ) -> None:
        prompt = prompt_template.format(
            jira_issue_key=jira_issue_key, new_ticket_status=jira_updated_ticket_status
        )

        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = frame.f_code.co_name if frame else "test_jira_issue_transition_success"
            await self._run_test_with_expectations(
                prompt,
                expectations_for_jira_transition_issue_success,
                llm_client,
                session,
                test_name,
            )
