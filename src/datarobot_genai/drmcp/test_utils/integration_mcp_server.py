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

When running under stdio there are no HTTP headers, so get_sdk_client() and
get_datarobot_access_token() would raise. We patch both to fall back to
credentials (from env) so integration tests can use DATAROBOT_API_TOKEN
without injecting headers.
"""

import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import datarobot as dr
from datarobot.context import Context as DRContext

from datarobot_genai.drmcp import create_mcp_server
from datarobot_genai.drmcp.core import clients
from datarobot_genai.drmcp.core.clients import get_sdk_client as _original_get_sdk_client
from datarobot_genai.drmcp.core.credentials import get_credentials
from datarobot_genai.drmcp.test_utils.stubs.dr_client_stubs import test_create_dr_client
from datarobot_genai.drtools.clients import datarobot as tools_datarobot_client

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


async def _get_datarobot_access_token_stdio_fallback() -> str:
    """Return DataRobot token from credentials for stdio (no headers)."""
    creds = get_credentials()
    token = creds.datarobot.application_api_token
    if not token:
        from fastmcp.exceptions import ToolError

        raise ToolError("DataRobot API token not available (stdio and no DATAROBOT_API_TOKEN).")
    return token


def _patch_get_sdk_client_for_stdio() -> None:
    """Patch get_sdk_client and get_datarobot_access_token for stdio (no headers)."""
    from datarobot_genai.drmcp.core import clients
    from datarobot_genai.drtools.clients import datarobot as tools_datarobot_client

    def get_sdk_client_with_credentials_fallback(headers_auth_only: bool = False) -> Any:
        try:
            return _original_get_sdk_client(headers_auth_only=headers_auth_only)
        except ValueError:
            creds = get_credentials()
            token = creds.datarobot.application_api_token
            if not token:
                raise
            dr.Client(token=token, endpoint=creds.datarobot.endpoint)
            DRContext.use_case = None
            return dr

    clients.get_sdk_client = get_sdk_client_with_credentials_fallback
    tools_datarobot_client.get_datarobot_access_token = _get_datarobot_access_token_stdio_fallback


def _apply_dr_client_stubs() -> None:
    """
    Replace the real DataRobot client with stubs
    that areself-contained (patches token + client for stdio).
    """
    stub_dr = test_create_dr_client()
    # get_api_client() does dr.client.get_client(); stub must have that for prompt registration.
    # dr.utils.pagination.unpaginate expects client.get(...).json()
    # to return {"data": [...], "next": url or None}.
    # Return empty page so registration finishes immediately instead of hanging.
    mock_rest = MagicMock()
    mock_rest.get.return_value.json.return_value = {"data": [], "next": None}
    stub_dr.client = MagicMock()
    stub_dr.client.get_client = lambda: mock_rest

    clients.get_sdk_client = lambda *args, **kwargs: stub_dr
    tools_datarobot_client.DataRobotClient.get_client = lambda self: stub_dr  # type: ignore[method-assign]
    # Tools call get_datarobot_access_token() before DataRobotClient; patch for stdio (no headers).
    tools_datarobot_client.get_datarobot_access_token = _get_datarobot_access_token_stdio_fallback


def main() -> None:
    """Run the integration test MCP server."""
    if os.environ.get("MCP_USE_CLIENT_STUBS", "true") == "true":
        _apply_dr_client_stubs()
    elif os.environ.get("MCP_SERVER_NAME") == "integration":
        _patch_get_sdk_client_for_stdio()

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
            load_native_mcp_tools=True,
        )
    else:
        # No user modules - create server with base tools only
        server = create_mcp_server(transport="stdio", load_native_mcp_tools=True)

    server.run()


if __name__ == "__main__":
    main()
