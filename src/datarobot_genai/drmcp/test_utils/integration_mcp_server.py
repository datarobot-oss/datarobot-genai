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

When running under stdio there are no HTTP headers, so request_user_dr_sdk()
and get_datarobot_access_token() would raise. We patch token resolution to fall
back to credentials (from env) so integration tests can use DATAROBOT_API_TOKEN
without injecting headers.
"""

import os
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import Mock

import datarobot as dr
import datarobot_predict.deployment as _dr_predict_deployment

from datarobot_genai.drmcp import create_mcp_server
from datarobot_genai.drmcp.core.dynamic_prompts import register as prompt_register
from datarobot_genai.drmcp.core.feature_flags import FeatureFlag
from datarobot_genai.drmcp.core.lineage.manager import LineageManager
from datarobot_genai.drmcp.test_utils.stubs.dr_client_stubs import test_create_dr_client
from datarobot_genai.drmcp.test_utils.stubs.prediction_result_stub import (
    test_create_prediction_result,
)
from datarobot_genai.drmcp.test_utils.stubs.prompt_stubs import get_stub_prompt_template_versions
from datarobot_genai.drmcp.test_utils.stubs.prompt_stubs import get_stub_prompt_templates
from datarobot_genai.drtools.core.clients import datarobot as tools_datarobot_client
from datarobot_genai.drtools.core.credentials import get_credentials
from datarobot_genai.drtools.core.exceptions import ToolError

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


def _stub_prompt_template_versions(
    prompt_template_ids: list[str],
    headers_auth_only: bool = False,
) -> dict[str, list[Any]]:
    """Stub that matches dr_lib.get_datarobot_prompt_template_versions signature."""
    return get_stub_prompt_template_versions(prompt_template_ids)


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


def _get_datarobot_access_token_stdio_fallback(*, headers_auth_only: bool = True) -> str:
    """Return DataRobot token from credentials for stdio (no headers)."""
    del headers_auth_only  # stdio has no headers; always use application credentials.
    creds = get_credentials()
    token = creds.datarobot.datarobot_api_token
    if not token:
        raise ToolError("DataRobot API token not available (stdio and no DATAROBOT_API_TOKEN).")
    return token


@contextmanager
def _stub_thread_safe_request_user_client(
    self: Any, *, headers_auth_only: bool = True
) -> Generator[MagicMock, None, None]:
    del headers_auth_only
    """No-op stub for ThreadSafeDataRobotClient.request_user_client.

    Tools use dr.X.Y() directly inside the with block; the real SDK classes are
    monkey-patched in _apply_dr_client_stubs so no actual HTTP calls are made.
    """
    yield MagicMock()


@contextmanager
def _stub_request_user_dr_sdk(*, headers_auth_only: bool = True) -> Generator[Any, None, None]:
    del headers_auth_only
    """No-op stub for request_user_dr_sdk; SDK classes are stubbed globally."""
    yield dr


def _patch_datarobot_token_for_stdio() -> None:
    """Patch get_datarobot_access_token for stdio (no headers)."""
    tools_datarobot_client.get_datarobot_access_token = _get_datarobot_access_token_stdio_fallback


def _apply_predict_stubs() -> None:
    """Patch datarobot_predict.deployment.predict so predict_realtime works with StubDeployment."""
    _dr_predict_deployment.predict = test_create_prediction_result


def _apply_prompt_stubs() -> None:
    """Patch register module so prompt registration uses stub templates/versions in this process."""
    prompt_register.get_datarobot_prompt_templates = get_stub_prompt_templates  # type: ignore[assignment]
    prompt_register.get_datarobot_prompt_template_versions = _stub_prompt_template_versions


def apply_lineage_manager_stubs() -> None:
    LineageManager.__init__ = Mock(return_value=None)  # type: ignore[assignment]
    LineageManager.sync_mcp_tools = AsyncMock()  # type: ignore[assignment]
    LineageManager.sync_mcp_prompts = AsyncMock()  # type: ignore[assignment]
    FeatureFlag.is_mcp_tools_gallery_support_enabled_for_static_mcp_container_user = AsyncMock()  # type: ignore[assignment]


def _apply_dr_sdk_stubs(stub_dr: Any, mock_rest: MagicMock) -> None:
    """Monkey-patch datarobot SDK entry points used by ThreadSafeDataRobotClient tools."""
    dr.Project.get = stub_dr.Project.get  # type: ignore[method-assign]
    dr.Model.get = stub_dr.Model.get  # type: ignore[method-assign]
    dr.Deployment.get = stub_dr.Deployment.get  # type: ignore[method-assign]
    dr.Dataset.get = stub_dr.Dataset.get  # type: ignore[method-assign]
    dr.Dataset.list = stub_dr.Dataset.list  # type: ignore[method-assign]
    dr.Dataset.iterate = stub_dr.Dataset.iterate  # type: ignore[method-assign]
    dr.DataStore.list = stub_dr.DataStore.list  # type: ignore[method-assign]
    dr.UseCase.get = stub_dr.UseCase.get  # type: ignore[method-assign]
    dr.BatchPredictionJob = stub_dr.BatchPredictionJob  # type: ignore[misc]
    dr.client.get_client = lambda: mock_rest  # type: ignore[method-assign]


def _apply_dr_client_stubs() -> None:
    """Replace the real DataRobot client with stubs (patches token + client for stdio)."""
    stub_dr = test_create_dr_client()
    # request_user_dr_client() uses dr.client.get_client(); stub needs that for prompts.
    # dr.utils.pagination.unpaginate expects client.get(...).json()
    # to return {"data": [...], "next": url or None}.
    # Return empty page so registration finishes immediately instead of hanging.
    mock_rest = MagicMock()
    mock_rest.get.return_value.json.return_value = {"data": [], "next": None}
    # Wire REST stubs from test_create_dr_client onto mock_rest
    if stub_dr.stub_rest_get:
        mock_rest.get = stub_dr.stub_rest_get
    if stub_dr.stub_rest_post:
        mock_rest.post = stub_dr.stub_rest_post
    stub_dr.client = MagicMock()
    stub_dr.client.get_client = lambda: mock_rest

    _apply_dr_sdk_stubs(stub_dr, mock_rest)

    tools_datarobot_client.request_user_dr_sdk = _stub_request_user_dr_sdk
    thread_safe_client = tools_datarobot_client.ThreadSafeDataRobotClient
    thread_safe_client.request_user_client = _stub_thread_safe_request_user_client  # type: ignore[method-assign]
    tools_datarobot_client.get_datarobot_access_token = _get_datarobot_access_token_stdio_fallback
    _apply_predict_stubs()
    _apply_prompt_stubs()


def main() -> None:
    """Run the integration test MCP server."""
    if os.environ.get("MCP_USE_CLIENT_STUBS", "true") == "true":
        _apply_dr_client_stubs()
        apply_lineage_manager_stubs()
    elif os.environ.get("MCP_SERVER_NAME") == "integration":
        _patch_datarobot_token_for_stdio()

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
