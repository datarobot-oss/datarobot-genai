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

"""Test tool for elicitation testing.

This module registers a test tool that can be used to test elicitation support.
It should be imported in tests that need it.
"""

from fastmcp import Context
from fastmcp.server.context import AcceptedElicitation
from fastmcp.server.context import DeclinedElicitation
from mcp.types import ClientCapabilities
from mcp.types import ElicitationCapability

from datarobot_genai.drmcp.core.mcp_instance import mcp


@mcp.tool(
    name="get_user_greeting",
    description=(
        "Get a personalized greeting for a user. "
        "Requires a username - if not provided, will request it via elicitation."
    ),
    tags={"test", "elicitation"},
)
async def get_user_greeting(ctx: Context, username: str | None = None) -> dict:
    """
    Get a personalized greeting for a user.

    This tool demonstrates FastMCP's built-in elicitation by requiring a username parameter.
    If username is not provided, it uses ctx.elicit() to request it from the user.

    Args:
        ctx: FastMCP context (automatically injected)
        username: The username to greet. If None, elicitation will be triggered.

    Returns
    -------
        Dictionary with greeting message or error if elicitation was declined/cancelled
    """
    if not username:
        # Check if client supports elicitation before using it
        # According to MCP spec, elicitation is a client capability
        # We use check_client_capability to verify support
        try:
            has_elicitation = ctx.session.check_client_capability(
                ClientCapabilities(elicitation=ElicitationCapability())
            )
        except (AttributeError, TypeError):
            # If check_client_capability doesn't exist or fails, assume no support
            has_elicitation = False

        if not has_elicitation:
            # According to MCP spec, when elicitation is not supported, return a no-op response
            # rather than throwing an error
            return {
                "status": "skipped",
                "message": (
                    "Elicitation not supported by client. "
                    "Username parameter is required when client does not support elicitation."
                ),
                "elicitation_supported": False,
            }

        # Use FastMCP's built-in elicitation
        result = await ctx.elicit(
            message="Username is required to generate a personalized greeting",
            response_type=str,
        )

        if isinstance(result, AcceptedElicitation):
            username = result.data
        elif isinstance(result, DeclinedElicitation):
            return {
                "status": "error",
                "error": "Username declined by user",
                "message": "Cannot generate greeting without username",
            }
        else:  # CancelledElicitation
            return {
                "status": "error",
                "error": "Operation cancelled",
                "message": "Greeting request was cancelled",
            }

    return {
        "status": "success",
        "message": f"Hello, {username}! Welcome to the DataRobot MCP server.",
        "username": username,
    }
