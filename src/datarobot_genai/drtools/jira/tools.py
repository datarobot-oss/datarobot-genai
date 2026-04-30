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

import logging
from typing import Annotated
from typing import Any

from datarobot_genai.drtools.core import tool_metadata
from datarobot_genai.drtools.core.clients.atlassian import get_atlassian_access_token
from datarobot_genai.drtools.core.clients.jira import JiraClient
from datarobot_genai.drtools.core.exceptions import ToolError
from datarobot_genai.drtools.core.exceptions import ToolErrorKind

logger = logging.getLogger(__name__)

_JQL_FUNCTIONS = "https://support.atlassian.com/jira-service-management-cloud/docs/jql-functions/"
_JQL_FIELDS = "https://support.atlassian.com/jira-service-management-cloud/docs/jql-fields/"
_JQL_KEYWORDS = (
    "https://support.atlassian.com/jira-service-management-cloud/docs/"
    "use-advanced-search-with-jira-query-language-jql/"
)
_JQL_OPERATORS = "https://support.atlassian.com/jira-service-management-cloud/docs/jql-operators/"
_JIRA_ISSUES_REST = "https://developer.atlassian.com/cloud/jira/platform/rest/v3/api-group-issues/"


@tool_metadata(
    tags={"jira", "search", "issues"},
    description=(
        "[Jira—search issues] Use when filtering many issues with JQL (project, status, text, "
        "dates, assignee, etc.). Returns matching issues as summaries. Not one known issue key "
        "(jira_get_issue), not Confluence (confluence_search).\n\n"
        'Example: jql_query="project = PROJ AND status = Open", max_results=50.\n\n'
        "JQL references: "
        f"{_JQL_FUNCTIONS} {_JQL_FIELDS} {_JQL_KEYWORDS} {_JQL_OPERATORS}"
    ),
)
async def jira_search_issues(
    *,
    jql_query: Annotated[
        str, "The JQL (Jira Query Language) string used to filter and search for issues."
    ],
    max_results: Annotated[int, "Maximum number of issues to return. Default is 50."] = 50,
) -> dict[str, Any]:
    if not jql_query:
        raise ToolError(
            "Argument validation error: 'jql_query' cannot be empty.", kind=ToolErrorKind.VALIDATION
        )

    access_token = await get_atlassian_access_token()
    if isinstance(access_token, ToolError):
        raise access_token

    async with JiraClient(access_token) as client:
        issues = await client.search_jira_issues(jql_query=jql_query, max_results=max_results)

    return {
        "data": [issue.as_flat_dict() for issue in issues],
        "count": len(issues),
    }


@tool_metadata(
    tags={"jira", "read", "get", "issue"},
    description=(
        "[Jira—get issue] Use when you already have an issue key (e.g. PROJ-123) and need full "
        "fields and current values. Read-only. Not JQL multi-issue search (jira_search_issues).\n\n"
        f'Example: issue_key="PROJ-123".\n\nReference: {_JIRA_ISSUES_REST}'
    ),
)
async def jira_get_issue(
    *, issue_key: Annotated[str, "The key (ID) of the Jira issue to retrieve, e.g., 'PROJ-123'."]
) -> dict[str, Any]:
    if not issue_key:
        raise ToolError(
            "Argument validation error: 'issue_key' cannot be empty.", kind=ToolErrorKind.VALIDATION
        )

    access_token = await get_atlassian_access_token()
    if isinstance(access_token, ToolError):
        raise access_token

    async with JiraClient(access_token) as client:
        issue = await client.get_jira_issue(issue_key)

    return issue.as_flat_dict()


@tool_metadata(
    tags={"jira", "create", "add", "issue"},
    description=(
        "[Jira—create issue] Use when opening a new work item: project key, summary, and issue "
        "type name (Task/Bug/Story as configured). Optional description. Not status moves "
        "(jira_transition_issue), not field patches on existing issues (jira_update_issue).\n\n"
        'Example: project_key="PROJ", summary="Fix login", issue_type="Bug", '
        'description="Steps...".\n\n'
        f"Reference: {_JIRA_ISSUES_REST}"
    ),
)
async def jira_create_issue(
    *,
    project_key: Annotated[str, "The key of the project where the issue should be created."],
    summary: Annotated[str, "A brief summary or title for the new issue."],
    issue_type: Annotated[str, "The type of issue to create (e.g., 'Task', 'Bug', 'Story')."],
    description: Annotated[str | None, "Detailed description of the issue."] = None,
) -> dict[str, Any]:
    if not all([project_key, summary, issue_type]):
        raise ToolError(
            "Argument validation error: project_key, summary, and issue_type are required fields.",
            kind=ToolErrorKind.VALIDATION,
        )

    access_token = await get_atlassian_access_token()
    if isinstance(access_token, ToolError):
        raise access_token

    async with JiraClient(access_token) as client:
        # Maybe we should cache it somehow?
        # It'll be probably constant through whole mcp server lifecycle...
        issue_types = await client.get_jira_issue_types(project_key=project_key)

        try:
            issue_type_id = issue_types[issue_type]
        except KeyError:
            possible_issue_types = ",".join(issue_types)
            msg = (
                f"Unexpected issue type `{issue_type}`. Possible values are {possible_issue_types}."
            )
            raise ToolError(msg, kind=ToolErrorKind.VALIDATION)

    async with JiraClient(access_token) as client:
        issue_key = await client.create_jira_issue(
            project_key=project_key,
            summary=summary,
            issue_type_id=issue_type_id,
            description=description,
        )

    return {"newIssueKey": issue_key, "projectKey": project_key}


@tool_metadata(
    tags={"jira", "update", "edit", "issue"},
    description=(
        "[Jira—update fields] Use when changing field values on an existing issue (summary, "
        "description, custom fields) by key; values must match Jira field formats. Not workflow "
        "status changes (jira_transition_issue), not create (jira_create_issue).\n\n"
        "Example fields_to_update for description (ADF-style JSON): "
        '{"description": {"type": "text", "version": 1, "text": [{"type": "paragraph", '
        '"content": [{"type": "text", "text": "<your text>"}]}]}}\n\n'
        f"Reference: {_JIRA_ISSUES_REST}"
    ),
)
async def jira_update_issue(
    *,
    issue_key: Annotated[str, "The key (ID) of the Jira issue to retrieve, e.g., 'PROJ-123'."],
    fields_to_update: Annotated[
        dict[str, Any],
        "A dictionary of field names and their new values (e.g., {'summary': 'New content'}).",
    ],
) -> dict[str, Any]:
    if not issue_key:
        raise ToolError(
            "Argument validation error: 'issue_key' cannot be empty.", kind=ToolErrorKind.VALIDATION
        )
    if not fields_to_update or not isinstance(fields_to_update, dict):
        raise ToolError(
            "Argument validation error: 'fields_to_update' must be a non-empty dictionary.",
            kind=ToolErrorKind.VALIDATION,
        )

    access_token = await get_atlassian_access_token()
    if isinstance(access_token, ToolError):
        raise access_token

    async with JiraClient(access_token) as client:
        updated_fields = await client.update_jira_issue(
            issue_key=issue_key, fields=fields_to_update
        )

    return {"updatedIssueKey": issue_key, "fields": updated_fields}


@tool_metadata(
    tags={"jira", "update", "transition", "issue"},
    description=(
        "[Jira—transition status] Use when moving an existing issue along its workflow by exact "
        "transition name (e.g. 'In Progress'). Not arbitrary field edits (jira_update_issue), "
        "not new issues (jira_create_issue).\n\n"
        'Example: issue_key="PROJ-123", transition_name="In Progress" (names come from '
        "the workflow; use the exact label returned for the issue).\n\n"
        f"Reference: {_JIRA_ISSUES_REST}"
    ),
)
async def jira_transition_issue(
    *,
    issue_key: Annotated[str, "The key (ID) of the Jira issue to transition, e.g. 'PROJ-123'."],
    transition_name: Annotated[
        str, "The exact name of the target status/transition (e.g., 'In Progress')."
    ],
) -> dict[str, Any]:
    if not all([issue_key, transition_name]):
        raise ToolError(
            "Argument validation error: issue_key and transition name/ID are required.",
            kind=ToolErrorKind.VALIDATION,
        )

    access_token = await get_atlassian_access_token()
    if isinstance(access_token, ToolError):
        raise access_token

    async with JiraClient(access_token) as client:
        available_transitions = await client.get_available_jira_transitions(issue_key=issue_key)

        try:
            transition_id = available_transitions[transition_name]
        except KeyError:
            available_transitions_str = ",".join(available_transitions)
            raise ToolError(
                f"Unexpected transition name `{transition_name}`. "
                f"Possible values are {available_transitions_str}.",
                kind=ToolErrorKind.VALIDATION,
            )

    async with JiraClient(access_token) as client:
        await client.transition_jira_issue(issue_key=issue_key, transition_id=transition_id)

    return {
        "transitionedIssueKey": issue_key,
        "newStatusName": transition_name,
        "newStatusId": transition_id,
    }
