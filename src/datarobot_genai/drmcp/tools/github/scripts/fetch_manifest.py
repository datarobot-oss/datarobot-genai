#!/usr/bin/env python
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

"""Script to fetch and update the GitHub tools manifest.

This is a dev-time script that fetches all available tools from the GitHub MCP
server and updates the github_tools.json manifest file.

Usage:
    GITHUB_TOKEN=xxx python -m datarobot_genai.drmcp.tools.github.scripts.fetch_manifest

Environment variables:
    GITHUB_TOKEN: Required. A valid GitHub token with appropriate permissions.
"""

import asyncio
import logging
import os
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_token_from_env() -> str | None:
    """Get GitHub token from environment variable."""
    return os.getenv("GITHUB_TOKEN")


async def main() -> int:
    """Fetch tools from GitHub MCP server and update the manifest."""
    # Import here to avoid circular imports and allow running as script
    from datarobot_genai.drmcp.tools.clients.github import GitHubMCPClient  # noqa: PLC0415
    from datarobot_genai.drmcp.tools.clients.github import GitHubMCPError  # noqa: PLC0415
    from datarobot_genai.drmcp.tools.github.register import update_manifest_cache  # noqa: PLC0415

    # Get token from environment
    token = get_token_from_env()
    if not token:
        logger.error("No GitHub token found. Set GITHUB_TOKEN environment variable.")
        return 1

    logger.info("Fetching tools from GitHub MCP server...")

    try:
        async with GitHubMCPClient(token, toolsets=["all"]) as client:
            tools = await client.list_tools()
    except GitHubMCPError as e:
        logger.error(f"Failed to fetch tools: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1

    logger.info(f"Fetched {len(tools)} tools from GitHub MCP server")

    # Update manifest using the register module's function
    update_manifest_cache(tools)

    # Print summary
    logger.info("Manifest updated successfully")
    logger.info(f"Tools: {', '.join(t['name'] for t in tools[:5])}...")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
