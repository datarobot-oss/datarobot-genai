#!/usr/bin/env python3

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
Integration test MCP server.

This server works standalone (base tools only) or detects and loads
user modules if they exist in the project structure.
"""

from pathlib import Path
from typing import Any

from datarobot_genai.drmcp import create_mcp_server

# Import elicitation test tool to register it with the MCP server
try:
    from datarobot_genai.drmcp.test_utils import elicitation_test_tool  # noqa: F401
except ImportError:
    # Test utils not available (e.g., running in production)
    pass

# Import user components (will be used conditionally)
try:
    from app.core.server_lifecycle import ServerLifecycle  # type: ignore  # noqa: F401
    from app.core.user_config import get_user_config  # type: ignore  # noqa: F401
    from app.core.user_credentials import get_user_credentials  # type: ignore  # noqa: F401

except ImportError:
    # These imports will fail when running from library without user modules
    pass


def detect_user_modules() -> Any:
    """
    Detect if user modules exist in the project.

    Returns
    -------
        Tuple of (config_factory, credentials_factory, lifecycle, module_paths) or None
    """
    # Try to find app directory
    # When run from library: won't find it
    # When run from project: will find it
    current_dir = Path.cwd()

    # Look for app in current directory or parent directories
    for search_dir in [current_dir, current_dir.parent, current_dir.parent.parent]:
        app_dir = search_dir / "app"
        app_core_dir = app_dir / "core"
        if app_core_dir.exists():
            # Found user directory - load user modules
            try:
                module_paths = [
                    (str(app_dir / "tools"), "app.tools"),
                    (str(app_dir / "prompts"), "app.prompts"),
                    (str(app_dir / "resources"), "app.resources"),
                ]

                return (
                    get_user_config,
                    get_user_credentials,
                    ServerLifecycle(),
                    module_paths,
                )
            except ImportError:
                # User modules don't exist or can't be imported
                pass

    return None


def main() -> None:
    """Run the integration test MCP server."""
    # Try to detect and load user modules
    user_components = detect_user_modules()

    if user_components:
        # User modules found - create server with user extensions
        config_factory, credentials_factory, lifecycle, module_paths = user_components
        server = create_mcp_server(
            config_factory=config_factory,
            credentials_factory=credentials_factory,
            lifecycle=lifecycle,
            additional_module_paths=module_paths,
            transport="stdio",
        )
    else:
        # No user modules - create server with base tools only
        server = create_mcp_server(transport="stdio")

    server.run()


if __name__ == "__main__":
    main()
