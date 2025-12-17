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
        logger.error(f"Unexpected error getting Jira issue: {e}")
        raise ToolError(
            f"An unexpected error occurred while getting Jira issue '{issue_key}': {str(e)}"
        )

    return ToolResult(
        content=f"Successfully retrieved details for issue '{issue_key}'.",
        structured_content=issue.as_flat_dict(),
    )
