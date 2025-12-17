from collections.abc import Iterator
from unittest.mock import patch

import pytest

from datarobot_genai.drmcp.core.exceptions import MCPError
from datarobot_genai.drmcp.tools.clients.jira import Issue
from datarobot_genai.drmcp.tools.jira.tools import jira_get_issue


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


class TestJiraGetIssue:
    """Jira get issue tool test."""

    @pytest.mark.asyncio
    async def test_jira_get_issue_happy_path(
        self, get_atlassian_access_token_mock: None, jira_client_get_issue_mock: Issue
    ) -> None:
        """Jira get issue -- happy path."""
        # GIVEN
        issue_key = "PROJ-123"

        # WHEN
        tool_result = await jira_get_issue(issue_key=issue_key)

        # THEN
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
        # GIVEN
        issue_key = "PROJ-123"

        # WHEN / THEN
        with pytest.raises(MCPError):
            await jira_get_issue(issue_key=issue_key)
