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
import re
from typing import Any
from typing import Literal
from urllib.parse import urlparse

from datarobot.core.config import DataRobotAppFrameworkBaseSettings
from pydantic import field_validator

from datarobot_genai.core.utils.auth import AuthContextHeaderHandler


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

    _auth_context_handler: AuthContextHeaderHandler | None = None
    _server_config: dict[str, Any] | None = None

    @field_validator("external_mcp_headers", mode="before")
    @classmethod
    def validate_external_mcp_headers(cls, value: str | None) -> str | None:
        if value is None:
            return None

        if not isinstance(value, str):
            msg = "external_mcp_headers must be a JSON string"
            raise TypeError(msg)

        candidate = value.strip()

        try:
            json.loads(candidate)
        except json.JSONDecodeError as exc:
            msg = "external_mcp_headers must be valid JSON"
            raise ValueError(msg) from exc

        return candidate

    @field_validator("mcp_deployment_id", mode="before")
    @classmethod
    def validate_mcp_deployment_id(cls, value: str | None) -> str | None:
        if value is None:
            return None

        if not isinstance(value, str):
            msg = "mcp_deployment_id must be a string"
            raise TypeError(msg)

        candidate = value.strip()

        if not re.fullmatch(r"[0-9a-fA-F]{24}", candidate):
            msg = "mcp_deployment_id must be a valid 24-character hex ID"
            raise ValueError(msg)

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

    def _authorization_context_header(self) -> dict[str, str]:
        """Return X-DataRobot-Authorization-Context header or empty dict."""
        try:
            return self.auth_context_handler.get_header(self.authorization_context)
        except (LookupError, RuntimeError):
            # Authorization context not available (e.g., in tests)
            return {}

    def _build_server_config(self) -> dict[str, Any] | None:
        """
        Get MCP server configuration.

        Returns
        -------
            Server configuration dict with url, transport, and optional headers,
            or None if not configured.
        """
        if self.external_mcp_url:
            # External MCP URL - no authentication needed
            headers: dict[str, str] = {}

            # Forward headers for localhost connections
            if self.forwarded_headers:
                try:
                    parsed_url = urlparse(self.external_mcp_url)
                    hostname = parsed_url.hostname or ""
                    # Check if hostname is localhost or 127.0.0.1
                    if hostname in ("localhost", "127.0.0.1", "::1"):
                        headers.update(self.forwarded_headers)
                except Exception:
                    # If URL parsing fails, fall back to simple string check
                    if "localhost" in self.external_mcp_url or "127.0.0.1" in self.external_mcp_url:
                        headers.update(self.forwarded_headers)

            # Merge external headers if provided
            if self.external_mcp_headers:
                external_headers = json.loads(self.external_mcp_headers)
                headers.update(external_headers)

            return {
                "url": self.external_mcp_url.rstrip("/"),
                "transport": self.external_mcp_transport,
                "headers": headers,
            }

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

            # Start with forwarded headers if available
            headers: dict[str, str] = {}
            if self.forwarded_headers:
                headers.update(self.forwarded_headers)

            # Add authentication headers
            headers.update(self._authorization_bearer_header())
            headers.update(self._authorization_context_header())

            return {
                "url": url,
                "transport": "streamable-http",
                "headers": headers,
            }

        return None
