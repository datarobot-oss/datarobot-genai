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
from collections.abc import Iterator
from unittest.mock import patch

import pytest

from datarobot_genai.drmcp.core.exceptions import MCPError
from datarobot_genai.drmcp.tools.clients.jira import Issue
from datarobot_genai.drmcp.tools.jira.tools import jira_create_issue
from datarobot_genai.drmcp.tools.jira.tools import jira_get_issue
from datarobot_genai.drmcp.tools.jira.tools import jira_update_issue


@pytest.fixture
def get_atlassian_access_token_mock() -> Iterator[None]:
    with patch(
        "datarobot_genai.drmcp.tools.jira.tools.get_atlassian_access_token",
        return_value="token",
    ):
        yield


@pytest.fixture
def jira_client_get_issue_mock() -> Iterator[Issue]:
    with patch(
        "datarobot_genai.drmcp.tools.clients.jira.JiraClient.get_jira_issue"
    ) as jira_client_get_issue:
        issue = Issue(
            **{
                "id": "123",
                "key": "PROJ-123",
                "fields": {
                    "summary": "Dummy summary",
                    "status": {
                        "name": "In Progress",
                    },
                    "updated": "2025-12-15T07:47:19.176-0500",
                    "created": "2025-12-11T09:01:58.944-0500",
                    "reporter": {"emailAddress": "dummy@reporter.com"},
                    "assignee": {"emailAddress": "dummy@assignee.com"},
                },
            }
        )
        jira_client_get_issue.return_value = issue
        yield issue


@pytest.fixture
def jira_client_get_issue_error_mock() -> Iterator[None]:
    with patch(
        "datarobot_genai.drmcp.tools.clients.jira.JiraClient.get_jira_issue"
    ) as jira_client_get_issue:
        jira_client_get_issue.side_effect = ValueError("Dummy error")
        yield


@pytest.fixture
def jira_client_get_issue_types_mock() -> Iterator[dict[str, str]]:
    with patch(
        "datarobot_genai.drmcp.tools.clients.jira.JiraClient.get_jira_issue_types"
    ) as jira_client_get_issue_types:
        issue_types = {"Bug": "1", "Story": "2"}
        jira_client_get_issue_types.return_value = issue_types
        yield issue_types


@pytest.fixture
def jira_client_get_issue_types_error_mock() -> Iterator[None]:
    with patch(
        "datarobot_genai.drmcp.tools.clients.jira.JiraClient.get_jira_issue_types"
    ) as jira_client_get_issue_types:
        jira_client_get_issue_types.side_effect = ValueError("Dummy error")
        yield


@pytest.fixture
def jira_client_create_issue_mock() -> Iterator[str]:
    with patch(
        "datarobot_genai.drmcp.tools.clients.jira.JiraClient.create_jira_issue"
    ) as jira_client_create_issue:
        new_issue_key = "PROJ-123"
        jira_client_create_issue.return_value = new_issue_key
        yield new_issue_key


@pytest.fixture
def jira_client_create_issue_error_mock() -> Iterator[None]:
    with patch(
        "datarobot_genai.drmcp.tools.clients.jira.JiraClient.create_jira_issue"
    ) as jira_client_get_issue:
        jira_client_get_issue.side_effect = ValueError("Dummy error")
        yield


@pytest.fixture
def jira_client_update_issue_mock() -> Iterator[list[str]]:
    with patch(
        "datarobot_genai.drmcp.tools.clients.jira.JiraClient.update_jira_issue"
    ) as jira_client_update_issue:
        fields_list = ["summary"]
        jira_client_update_issue.return_value = fields_list
        yield fields_list


@pytest.fixture
def jira_client_update_issue_error_mock() -> Iterator[None]:
    with patch(
        "datarobot_genai.drmcp.tools.clients.jira.JiraClient.update_jira_issue"
    ) as jira_client_update_issue:
        jira_client_update_issue.side_effect = ValueError("Dummy error")
        yield


class TestJiraGetIssue:
    """Jira get issue tool test."""

    @pytest.mark.asyncio
    async def test_jira_get_issue_happy_path(
        self, get_atlassian_access_token_mock: None, jira_client_get_issue_mock: Issue
    ) -> None:
        """Jira get issue -- happy path."""
        issue_key = "PROJ-123"

        tool_result = await jira_get_issue(issue_key=issue_key)

        content, structured_content = tool_result.to_mcp_result()
        assert content[0].text == "Successfully retrieved details for issue 'PROJ-123'."
        assert structured_content == {
            "id": "123",
            "key": "PROJ-123",
            "status": "In Progress",
            "summary": "Dummy summary",
            "created": "2025-12-11T09:01:58.944-0500",
            "updated": "2025-12-15T07:47:19.176-0500",
            "assigneeEmailAddress": "dummy@assignee.com",
            "reporterEmailAddress": "dummy@reporter.com",
        }

    @pytest.mark.asyncio
    async def test_jira_get_issue_when_error_in_client(
        self, get_atlassian_access_token_mock: None, jira_client_get_issue_error_mock: None
    ) -> None:
        """Jira get issue -- error in client."""
        issue_key = "PROJ-123"

        with pytest.raises(MCPError):
            await jira_get_issue(issue_key=issue_key)


class TestJiraCreateIssue:
    """Jira create issue tool test."""

    @pytest.mark.asyncio
    async def test_jira_create_issue_happy_path(
        self,
        get_atlassian_access_token_mock: None,
        jira_client_get_issue_types_mock: dict[str, str],
        jira_client_create_issue_mock: str,
    ) -> None:
        """Jira get issue -- happy path."""
        project_key = "PROJ"
        summary = "Dummy summary"
        issue_type = "Bug"
        description = "Dummy description of bug"

        tool_result = await jira_create_issue(
            project_key=project_key, summary=summary, issue_type=issue_type, description=description
        )

        content, structured_content = tool_result.to_mcp_result()
        assert content[0].text == "Successfully created issue 'PROJ-123'."
        assert structured_content == {
            "newIssueKey": "PROJ-123",
            "projectKey": "PROJ",
        }

    @pytest.mark.asyncio
    async def test_jira_create_issue_when_not_existing_issue_type(
        self,
        get_atlassian_access_token_mock: None,
        jira_client_get_issue_types_mock: dict[str, str],
    ) -> None:
        """Jira create issue -- error in client."""
        project_key = "PROJ"
        summary = "Dummy summary"
        issue_type = "Not existing issue type"  # <- Main change here
        description = "Dummy description of bug"

        with pytest.raises(MCPError, match="Unexpected issue type"):
            await jira_create_issue(
                project_key=project_key,
                summary=summary,
                issue_type=issue_type,
                description=description,
            )

    @pytest.mark.asyncio
    async def test_jira_create_issue_when_error_in_client_while_getting_issue_types(
        self,
        get_atlassian_access_token_mock: None,
        jira_client_get_issue_types_error_mock: None,
    ) -> None:
        """Jira create issue -- error in client."""
        project_key = "PROJ"
        summary = "Dummy summary"
        issue_type = "Bug"
        description = "Dummy description of bug"

        with pytest.raises(MCPError):
            await jira_create_issue(
                project_key=project_key,
                summary=summary,
                issue_type=issue_type,
                description=description,
            )

    @pytest.mark.asyncio
    async def test_jira_create_issue_when_error_in_client_while_creating(
        self,
        get_atlassian_access_token_mock: None,
        jira_client_get_issue_types_mock: dict[str, str],
        jira_client_create_issue_error_mock: None,
    ) -> None:
        """Jira create issue -- error in client."""
        project_key = "PROJ"
        summary = "Dummy summary"
        issue_type = "Bug"
        description = "Dummy description of bug"

        with pytest.raises(MCPError):
            await jira_create_issue(
                project_key=project_key,
                summary=summary,
                issue_type=issue_type,
                description=description,
            )


class TestJiraUpdateIssue:
    """Jira update issue tool test."""

    @pytest.mark.asyncio
    async def test_jira_update_issue_happy_path(
        self, get_atlassian_access_token_mock: None, jira_client_update_issue_mock: list[str]
    ) -> None:
        """Jira update issue -- happy path."""
        issue_key = "PROJ-123"
        fields_to_update = {"summary": "New dummy summary"}

        tool_result = await jira_update_issue(
            issue_key=issue_key, fields_to_update=fields_to_update
        )

        content, structured_content = tool_result.to_mcp_result()
        assert content[0].text == "Successfully updated issue 'PROJ-123'. Fields modified: summary."
        assert structured_content == {
            "updatedIssueKey": "PROJ-123",
            "fields": jira_client_update_issue_mock,
        }

    @pytest.mark.asyncio
    async def test_jira_update_issue_when_error_in_client(
        self, get_atlassian_access_token_mock: None, jira_client_update_issue_error_mock: None
    ) -> None:
        """Jira update issue -- error in client."""
        issue_key = "PROJ-123"
        fields_to_update = {"summary": "New dummy summary"}

        with pytest.raises(MCPError):
            await jira_update_issue(issue_key=issue_key, fields_to_update=fields_to_update)
