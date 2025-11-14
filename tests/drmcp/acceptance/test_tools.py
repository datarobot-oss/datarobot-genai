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

"""
Test tools for acceptance/E2E tests.

These tools are loaded by the MCP server and are available for testing purposes.
"""

from datarobot_genai.drmcp.core.auth import get_auth_context
from datarobot_genai.drmcp.core.mcp_instance import dr_mcp_tool


@dr_mcp_tool()
async def get_auth_context_user_info() -> str:
    """
    Tool that retrieves and returns user info like ID, name, and email from the auth context.

    This tool extracts user information (ID, name, email) from the current
    authorization context and returns it as a formatted string.

    Returns
    -------
    String with user information from the auth context in the format:
    "User ID: <id>, User Name: <name>, User Email: <email>"
    """
    auth_ctx = await get_auth_context()
    return (
        f"User ID: {auth_ctx.user.id}, "
        f"User Name: {auth_ctx.user.name}, "
        f"User Email: {auth_ctx.user.email}"
    )
