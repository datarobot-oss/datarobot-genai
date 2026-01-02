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

from fastmcp.exceptions import ToolError
from fastmcp.tools.tool import ToolResult

from datarobot_genai.drmcp.core.mcp_instance import dr_mcp_tool
from datarobot_genai.drmcp.tools.clients.atlassian import get_atlassian_access_token
from datarobot_genai.drmcp.tools.clients.jira import JiraClient

logger = logging.getLogger(__name__)


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

    try:
        async with JiraClient(access_token) as client:
            issue = await client.get_jira_issue(issue_key)
    except Exception as e:
        logger.error(f"Unexpected error while getting Jira issue: {e}")
        raise ToolError(
            f"An unexpected error occurred while getting Jira issue '{issue_key}': {str(e)}"
        )

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

    try:
        async with JiraClient(access_token) as client:
            issue_key = await client.create_jira_issue(
                project_key=project_key,
                summary=summary,
                issue_type_id=issue_type_id,
                description=description,
            )
    except Exception as e:
        logger.error(f"Unexpected error while creating Jira issue: {e}")
        raise ToolError(f"An unexpected error occurred while creating Jira issue: {str(e)}")

    return ToolResult(
        content=f"Successfully created issue '{issue_key}'.",
        structured_content={"newIssueKey": issue_key, "projectKey": project_key},
    )
