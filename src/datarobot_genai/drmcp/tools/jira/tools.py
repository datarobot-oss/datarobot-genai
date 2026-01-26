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

from fastmcp.exceptions import ToolError
from fastmcp.tools.tool import ToolResult

from datarobot_genai.drmcp.core.mcp_instance import dr_mcp_tool
from datarobot_genai.drmcp.tools.clients.atlassian import get_atlassian_access_token
from datarobot_genai.drmcp.tools.clients.jira import JiraClient

logger = logging.getLogger(__name__)


@dr_mcp_tool(tags={"jira", "search", "issues"})
async def jira_search_issues(
    *,
    jql_query: Annotated[
        str, "The JQL (Jira Query Language) string used to filter and search for issues."
    ],
    max_results: Annotated[int, "Maximum number of issues to return. Default is 50."] = 50,
) -> ToolResult:
    """
    Search for Jira issues using a powerful JQL query string.

    Refer to JQL documentation for advanced query construction:
    JQL functions: https://support.atlassian.com/jira-service-management-cloud/docs/jql-functions/
    JQL fields: https://support.atlassian.com/jira-service-management-cloud/docs/jql-fields/
    JQL keywords: https://support.atlassian.com/jira-service-management-cloud/docs/use-advanced-search-with-jira-query-language-jql/
    JQL operators: https://support.atlassian.com/jira-service-management-cloud/docs/jql-operators/
    """
    if not jql_query:
        raise ToolError("Argument validation error: 'jql_query' cannot be empty.")

    access_token = await get_atlassian_access_token()
    if isinstance(access_token, ToolError):
        raise access_token

    async with JiraClient(access_token) as client:
        issues = await client.search_jira_issues(jql_query=jql_query, max_results=max_results)

    n = len(issues)
    return ToolResult(
        content=f"Successfully executed JQL query and retrieved {n} issue(s).",
        structured_content={"data": [issue.as_flat_dict() for issue in issues], "count": n},
    )


@dr_mcp_tool(tags={"jira", "read", "get", "issue"})
async def jira_get_issue(
    *, issue_key: Annotated[str, "The key (ID) of the Jira issue to retrieve, e.g., 'PROJ-123'."]
) -> ToolResult:
    """Retrieve all fields and details for a single Jira issue by its key."""
    if not issue_key:
        raise ToolError("Argument validation error: 'issue_key' cannot be empty.")

    access_token = await get_atlassian_access_token()
    if isinstance(access_token, ToolError):
        raise access_token

    async with JiraClient(access_token) as client:
        issue = await client.get_jira_issue(issue_key)

    return ToolResult(
        content=f"Successfully retrieved details for issue '{issue_key}'.",
        structured_content=issue.as_flat_dict(),
    )


@dr_mcp_tool(tags={"jira", "create", "add", "issue"})
async def jira_create_issue(
    *,
    project_key: Annotated[str, "The key of the project where the issue should be created."],
    summary: Annotated[str, "A brief summary or title for the new issue."],
    issue_type: Annotated[str, "The type of issue to create (e.g., 'Task', 'Bug', 'Story')."],
    description: Annotated[str | None, "Detailed description of the issue."] = None,
) -> ToolResult:
    """Create a new Jira issue with mandatory project, summary, and type information."""
    if not all([project_key, summary, issue_type]):
        raise ToolError(
            "Argument validation error: project_key, summary, and issue_type are required fields."
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
            raise ToolError(
                f"Unexpected issue type `{issue_type}`. Possible values are {possible_issue_types}."
            )

    async with JiraClient(access_token) as client:
        issue_key = await client.create_jira_issue(
            project_key=project_key,
            summary=summary,
            issue_type_id=issue_type_id,
            description=description,
        )

    return ToolResult(
        content=f"Successfully created issue '{issue_key}'.",
        structured_content={"newIssueKey": issue_key, "projectKey": project_key},
    )


@dr_mcp_tool(tags={"jira", "update", "edit", "issue"})
async def jira_update_issue(
    *,
    issue_key: Annotated[str, "The key (ID) of the Jira issue to retrieve, e.g., 'PROJ-123'."],
    fields_to_update: Annotated[
        dict[str, Any],
        "A dictionary of field names and their new values (e.g., {'summary': 'New content'}).",
    ],
) -> ToolResult:
    """
    Modify descriptive fields or custom fields on an existing Jira issue using its key.
    If you want to update issue status you should use `jira_transition_issue` tool instead.

    Some fields needs very specific schema to allow update.
    You should follow jira rest api guidance.
    Good example is description field:
        "description": {
            "type": "text",
            "version": 1,
            "text": [
                {
                    "type": "paragraph",
                    "content": [
                        {
                            "type": "text",
                            "text": "[HERE YOU PUT REAL DESCRIPTION]"
                        }
                    ]
                }
            ]
        }
    """
    if not issue_key:
        raise ToolError("Argument validation error: 'issue_key' cannot be empty.")
    if not fields_to_update or not isinstance(fields_to_update, dict):
        raise ToolError(
            "Argument validation error: 'fields_to_update' must be a non-empty dictionary."
        )

    access_token = await get_atlassian_access_token()
    if isinstance(access_token, ToolError):
        raise access_token

    async with JiraClient(access_token) as client:
        updated_fields = await client.update_jira_issue(
            issue_key=issue_key, fields=fields_to_update
        )

    updated_fields_str = ",".join(updated_fields)
    return ToolResult(
        content=f"Successfully updated issue '{issue_key}'. Fields modified: {updated_fields_str}.",
        structured_content={"updatedIssueKey": issue_key, "fields": updated_fields},
    )


@dr_mcp_tool(tags={"jira", "update", "transition", "issue"})
async def jira_transition_issue(
    *,
    issue_key: Annotated[str, "The key (ID) of the Jira issue to transition, e.g. 'PROJ-123'."],
    transition_name: Annotated[
        str, "The exact name of the target status/transition (e.g., 'In Progress')."
    ],
) -> ToolResult:
    """
    Move a Jira issue through its defined workflow to a new status.
    This leverages Jira's workflow engine directly.
    """
    if not all([issue_key, transition_name]):
        raise ToolError("Argument validation error: issue_key and transition name/ID are required.")

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
                f"Possible values are {available_transitions_str}."
            )

    async with JiraClient(access_token) as client:
        await client.transition_jira_issue(issue_key=issue_key, transition_id=transition_id)

    return ToolResult(
        content=f"Successfully transitioned issue '{issue_key}' to status '{transition_name}'.",
        structured_content={
            "transitionedIssueKey": issue_key,
            "newStatusName": transition_name,
            "newStatusId": transition_id,
        },
    )
