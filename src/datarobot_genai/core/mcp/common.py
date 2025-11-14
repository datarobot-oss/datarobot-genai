# Copyright 2025 DataRobot, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from typing import Any

from datarobot_genai.core.utils.auth import AuthContextHeaderHandler


class MCPConfig:
    """Configuration for MCP server connection."""

    def __init__(self, api_base: str | None = None, api_key: str | None = None) -> None:
        """Initialize MCP configuration from environment variables and runtime parameters."""
        self.external_mcp_url = os.environ.get("EXTERNAL_MCP_URL")
        self.external_mcp_headers = os.environ.get("EXTERNAL_MCP_HEADERS")
        self.external_mcp_transport = os.environ.get("EXTERNAL_MCP_TRANSPORT", "streamable-http")
        self.mcp_deployment_id = os.environ.get("MCP_DEPLOYMENT_ID")
        self.api_base = api_base or os.environ.get(
            "DATAROBOT_ENDPOINT", "https://app.datarobot.com"
        )
        self.api_key = api_key or os.environ.get("DATAROBOT_API_TOKEN")
        self.auth_context_handler = AuthContextHeaderHandler()
        self.server_config = self._get_server_config()

    def _authorization_bearer_header(self) -> dict[str, str]:
        """Return Authorization header with Bearer token or empty dict."""
        if not self.api_key:
            return {}
        auth = self.api_key if self.api_key.startswith("Bearer ") else f"Bearer {self.api_key}"
        return {"Authorization": auth}

    def _authorization_context_header(self) -> dict[str, str]:
        """Return X-DataRobot-Authorization-Context header or empty dict."""
        try:
            return self.auth_context_handler.get_header()
        except (LookupError, RuntimeError):
            # Authorization context not available (e.g., in tests)
            return {}

    def _get_server_config(self) -> dict[str, Any] | None:
        """
        Get MCP server configuration.

        Returns
        -------
            Server configuration dict with url, transport, and optional headers,
            or None if not configured.
        """
        if self.external_mcp_url:
            # External MCP URL - no authentication needed
            if self.external_mcp_headers:
                headers = json.loads(self.external_mcp_headers)
            else:
                headers = {}

            config = {
                "url": self.external_mcp_url.rstrip("/"),
                "transport": self.external_mcp_transport,
                "headers": headers,
            }
            return config
        elif self.mcp_deployment_id and self.api_key:
            # DataRobot deployment ID - requires authentication
            base_url = self.api_base.rstrip("/")
            url = f"{base_url}/deployments/{self.mcp_deployment_id}/directAccess/mcp"

            headers = {
                **self._authorization_bearer_header(),
                **self._authorization_context_header(),
            }

            return {
                "url": url,
                "transport": "streamable-http",
                "headers": headers,
            }

        return None
