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
        # Try to use elicitation - if client supports it, this will work
        # If not, we'll get an error that we can handle
        try:
            result = await ctx.elicit(
                message="Username is required to generate a personalized greeting",
                response_type=str,
            )
        except Exception as e:
            # If elicitation fails, check if it's because client doesn't support it
            error_msg = str(e).lower()
            # Check for common elicitation not supported errors
            if any(
                keyword in error_msg
                for keyword in ["elicitation", "not supported", "capability", "unsupported"]
            ):
                # According to MCP spec, when elicitation is not supported, return a no-op response
                return {
                    "status": "skipped",
                    "message": (
                        "Elicitation not supported by client. "
                        "Username parameter is required when client does not support elicitation."
                    ),
                    "elicitation_supported": False,
                }
            # Re-raise if it's a different error
            raise

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
