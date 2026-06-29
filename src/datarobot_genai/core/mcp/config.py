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
import logging
import re
import socket
from typing import Any
from typing import Literal
from urllib.parse import urlparse

from datarobot.core.config import DataRobotAppFrameworkBaseSettings
from pydantic import field_validator

from datarobot_genai.core.utils.auth import AuthContextHeaderHandler

logger = logging.getLogger(__name__)


class MCPConfig(DataRobotAppFrameworkBaseSettings):
    """Configuration for MCP server connection.

    Derived values are exposed as properties rather than stored, avoiding
    Pydantic field validation/serialization concerns for internal helpers.
    """

    external_mcp_url: str | None = None
    external_mcp_headers: str | None = None
    external_mcp_transport: Literal["sse", "streamable-http"] = "streamable-http"
    mcp_deployment_id: str | None = None
    datarobot_endpoint: str | None = None
    datarobot_api_token: str | None = None
    authorization_context: dict[str, Any] | None = None
    forwarded_headers: dict[str, str] | None = None
    mcp_server_port: int | None = None

    _auth_context_handler: AuthContextHeaderHandler | None = None
    _server_config: dict[str, Any] | None = None

    @field_validator("external_mcp_headers", mode="before")
    @classmethod
    def validate_external_mcp_headers(cls, value: str | None) -> str | None:
        if value is None:
            return None

        candidate = value.strip()

        try:
            json.loads(candidate)
        except json.JSONDecodeError:
            msg = "external_mcp_headers must be valid JSON"
            logger.warning(msg)
            return None

        return candidate

    @field_validator("mcp_deployment_id", mode="before")
    @classmethod
    def validate_mcp_deployment_id(cls, value: str | None) -> str | None:
        if value is None:
            return None

        candidate = value.strip()

        if not re.fullmatch(r"[0-9a-fA-F]{24}", candidate):
            msg = "mcp_deployment_id must be a valid 24-character hex ID"
            logger.warning(msg)
            return None

        return candidate

    def _authorization_bearer_header(self) -> dict[str, str]:
        """Return Authorization header with Bearer token or empty dict."""
        if not self.datarobot_api_token:
            return {}
        auth = (
            self.datarobot_api_token
            if self.datarobot_api_token.startswith("Bearer ")
            else f"Bearer {self.datarobot_api_token}"
        )
        return {"Authorization": auth}

    @property
    def auth_context_handler(self) -> AuthContextHeaderHandler:
        if self._auth_context_handler is None:
            self._auth_context_handler = AuthContextHeaderHandler()
        return self._auth_context_handler

    @property
    def server_config(self) -> dict[str, Any] | None:
        if self._server_config is None:
            self._server_config = self._build_server_config()
        return self._server_config

    @property
    def is_local_server(self) -> bool:
        """True when the MCP server is a local process we start (resolved via
        mcp_server_port), as opposed to a DataRobot deployment or external URL.
        """
        return bool(
            self.mcp_server_port and not self.mcp_deployment_id and not self.external_mcp_url
        )

    def server_reachable(self, timeout: float = 1.0) -> bool:
        """TCP-probe the resolved MCP server's host:port.

        A reliable up/down signal for a local server (process listening or not).
        For remote hosts a load balancer may answer even when the MCP endpoint is
        down, so callers should gate this with `is_local_server`.
        """
        config = self.server_config
        if not config:
            return False
        parsed = urlparse(config["url"])
        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            return False

    def _authorization_context_header(self) -> dict[str, str]:
        """Return X-DataRobot-Authorization-Context header or empty dict."""
        try:
            return self.auth_context_handler.get_header(self.authorization_context)
        except (LookupError, RuntimeError):
            # Authorization context not available (e.g., in tests)
            return {}

    def _build_authenticated_headers(self) -> dict[str, str]:
        """Build headers for authenticated requests.

        Returns
        -------
            Dictionary containing forwarded headers (if available) and authentication headers.
        """
        headers: dict[str, str] = {}
        if self.forwarded_headers:
            headers.update(self.forwarded_headers)
        headers.update(self._authorization_bearer_header())
        headers.update(self._authorization_context_header())
        return headers

    def _build_server_config(self) -> dict[str, Any] | None:
        """
        Get MCP server configuration.

        Returns
        -------
            Server configuration dict with url, transport, and optional headers,
            or None if not configured.
        """
        if self.mcp_deployment_id:
            # DataRobot deployment ID - requires authentication
            if self.datarobot_endpoint is None:
                raise ValueError(
                    "When using a DataRobot hosted MCP deployment, datarobot_endpoint must be set."
                )
            if self.datarobot_api_token is None:
                raise ValueError(
                    "When using a DataRobot hosted MCP deployment, datarobot_api_token must be set."
                )

            base_url = self.datarobot_endpoint.rstrip("/")
            if not base_url.endswith("/api/v2"):
                base_url = f"{base_url}/api/v2"

            url = f"{base_url}/deployments/{self.mcp_deployment_id}/directAccess/mcp"
            headers = self._build_authenticated_headers()

            logger.info(f"Using DataRobot hosted MCP deployment: {url}")

            return {
                "url": url,
                "transport": "streamable-http",
                "headers": headers,
            }

        if self.external_mcp_url:
            # External MCP URL - no authentication needed
            headers = {}

            # Merge external headers if provided
            if self.external_mcp_headers:
                external_headers = json.loads(self.external_mcp_headers)
                headers.update(external_headers)

            logger.info(f"Using external MCP URL: {self.external_mcp_url}")

            return {
                "url": self.external_mcp_url.rstrip("/"),
                "transport": self.external_mcp_transport,
                "headers": headers,
            }

        # No MCP configuration found, setup localhost if running locally
        if self.mcp_server_port:
            url = f"http://localhost:{self.mcp_server_port}"
            headers = self._build_authenticated_headers()
            logger.info(f"Using localhost MCP server: {url}")
            return {
                "url": f"{url}/mcp",
                "transport": "streamable-http",
                "headers": headers,
            }

        return None
