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
from typing import Any
from typing import Literal
from typing import cast

import httpx
from datarobot.core.config import DataRobotAppFrameworkBaseSettings
from pydantic import field_validator

from datarobot_genai.core.utils.auth import AuthContextHeaderHandler
from datarobot_genai.dragent.deployment_urls import build_deployment_mcp_url
from datarobot_genai.dragent.deployment_urls import build_local_mcp_url
from datarobot_genai.dragent.deployment_urls import normalize_api_v2_endpoint
from datarobot_genai.dragent.deployment_urls import workload_mcp_url_from_endpoint

logger = logging.getLogger(__name__)

#: Timeout for the workload endpoint lookup.
WORKLOAD_LOOKUP_TIMEOUT_SECONDS = 10.0
_WORKLOAD_ENDPOINT_CACHE: dict[tuple[str, str], str] = {}

#: The one workload status whose reported endpoint is settled.
_WORKLOAD_RUNNING_STATUS = "running"


def clear_workload_endpoint_cache() -> None:
    """Forget every cached workload endpoint (used by tests)."""
    _WORKLOAD_ENDPOINT_CACHE.clear()


def lookup_workload_endpoint(
    workload_id: str,
    *,
    endpoint: str,
    token: str,
    timeout: float = WORKLOAD_LOOKUP_TIMEOUT_SECONDS,
) -> str | None:
    """Return the endpoint the platform serves ``workload_id`` from, or *None*.

    A workload's URL cannot be composed from its ID and the caller's endpoint,
    because the shape depends on a server-side Workload API setting the caller
    cannot see.

    Parameters
    ----------
    workload_id:
        The DataRobot workload ID.
    endpoint:
        DataRobot API endpoint
    token:
        DataRobot API token used for the lookup.
    timeout:
        Seconds to wait for the Workload API.

    Returns
    -------
    str | None
        The workload's ``endpoint`` field, or *None* when the workload cannot be
        read or reports no endpoint yet.
    """
    cache_key = (endpoint, workload_id)
    if cached := _WORKLOAD_ENDPOINT_CACHE.get(cache_key):
        return cached

    url = f"{normalize_api_v2_endpoint(endpoint)}/workloads/{workload_id}/"
    try:
        response = httpx.get(
            url,
            headers={"Authorization": f"Bearer {token.removeprefix('Bearer ').strip()}"},
            timeout=timeout,
        )
        response.raise_for_status()
        # ValueError covers a non-JSON body (json.JSONDecodeError subclasses it).
        payload: dict[str, Any] = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning(
            "Could not read the endpoint of workload %s from %s: %s. The agent has no MCP "
            "server as a result — the workload's route cannot be composed without this "
            "answer. Check that the agent's API token may read the workload, or set "
            "EXTERNAL_MCP_URL to address it directly.",
            workload_id,
            url,
            exc,
        )
        return None

    resolved = payload.get("endpoint")
    if not isinstance(resolved, str) or not resolved.strip():
        logger.warning(
            "Workload %s reported no endpoint (status %r); it may not be running yet.",
            workload_id,
            payload.get("status"),
        )
        return None

    resolved = resolved.strip()
    status = payload.get("status")
    if status == _WORKLOAD_RUNNING_STATUS:
        _WORKLOAD_ENDPOINT_CACHE[cache_key] = resolved
    else:
        logger.info(
            "Workload %s is %r, so its endpoint is not cached; it will be resolved again.",
            workload_id,
            status,
        )
    logger.info("Workload %s is served from %s", workload_id, resolved)
    return resolved


class MCPConfig(DataRobotAppFrameworkBaseSettings):
    """Configuration for MCP server connection.

    Derived values are exposed as properties rather than stored, avoiding
    Pydantic field validation/serialization concerns for internal helpers.
    """

    external_mcp_url: str | None = None
    external_mcp_headers: str | None = None
    external_mcp_transport: Literal["sse", "streamable-http"] = "streamable-http"
    mcp_deployment_id: str | None = None
    mcp_workload_id: str | None = None
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

    @field_validator("mcp_workload_id", mode="before")
    @classmethod
    def validate_mcp_workload_id(cls, value: str | None) -> str | None:
        if value is None:
            return None

        candidate = value.strip()

        if not re.fullmatch(r"[0-9a-fA-F]{24}", candidate):
            msg = "mcp_workload_id must be a valid 24-character hex ID"
            logger.warning(msg)
            return None

        return candidate

    @field_validator("mcp_server_port", mode="after")
    @classmethod
    def validate_mcp_server_port(cls, value: int | None) -> int | None:
        # Pydantic already coerces/rejects the type; only the range is ours.
        if value is not None and not 1 <= value <= 65535:
            logger.warning("mcp_server_port must be between 1 and 65535; ignoring")
            return None
        return value

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

    def _config_kind(self) -> Literal["workload", "deployment", "external", "local"] | None:
        """Single source of truth for which MCP server config is active.


        Precedence: workload > deployment > external > local.
        """
        if self.mcp_workload_id:
            return "workload"
        if self.mcp_deployment_id:
            return "deployment"
        if self.external_mcp_url:
            return "external"
        if self.mcp_server_port:
            return "local"
        return None

    @property
    def is_local_server(self) -> bool:
        """True when the MCP server is a local process we start (resolved via
        mcp_server_port), as opposed to a DataRobot deployment or external URL.
        """
        return self._config_kind() == "local"

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

    def _resolve_workload_mcp_url(self, workload_id: str) -> str | None:
        """Return the MCP URL of a workload, asking the platform where it is served.

        The lookup is the only source.  A workload's route depends on whether the
        cluster fronts workloads with different API Gateway — a server-side setting,
        with a per-enclave Host that is not derivable from ``datarobot_endpoint`` —
        so there is no template worth falling back to: it would be right on some
        clusters and quietly wrong on others.  Without an answer the agent has no
        MCP server (see the warning the lookup logs).  Since failed lookups are not
        cached and this runs per request, a workload that is not serving yet
        resolves on a later request rather than needing a restart.
        """
        assert self.datarobot_endpoint is not None  # caller checked
        assert self.datarobot_api_token is not None  # caller checked

        workload_endpoint = lookup_workload_endpoint(
            workload_id,
            endpoint=self.datarobot_endpoint,
            token=self.datarobot_api_token,
        )
        if workload_endpoint is None:
            return None
        return workload_mcp_url_from_endpoint(workload_endpoint)

    def _build_server_config(self) -> dict[str, Any] | None:
        """
        Get MCP server configuration.

        Returns
        -------
            Server configuration dict with url, transport, and optional headers,
            or None if not configured.
        """
        kind = self._config_kind()

        if kind == "workload":
            # DataRobot workload MCP - requires authentication
            if self.datarobot_endpoint is None:
                raise ValueError(
                    "When using a DataRobot workload MCP, datarobot_endpoint must be set."
                )
            if self.datarobot_api_token is None:
                raise ValueError(
                    "When using a DataRobot workload MCP, datarobot_api_token must be set."
                )

            # cast: kind == "workload" guarantees the workload ID is set
            url = self._resolve_workload_mcp_url(cast(str, self.mcp_workload_id))
            if url is None:
                # Nowhere to connect: better no MCP server than a guessed URL.
                return None
            headers = self._build_authenticated_headers()

            logger.info(f"Using DataRobot workload MCP: {url}")

            return {
                "url": url,
                "transport": "streamable-http",
                "headers": headers,
            }

        if kind == "deployment":
            # DataRobot deployment ID - requires authentication
            if self.datarobot_endpoint is None:
                raise ValueError(
                    "When using a DataRobot hosted MCP deployment, datarobot_endpoint must be set."
                )
            if self.datarobot_api_token is None:
                raise ValueError(
                    "When using a DataRobot hosted MCP deployment, datarobot_api_token must be set."
                )

            url = build_deployment_mcp_url(
                self.datarobot_endpoint,
                # cast: kind == "deployment" guarantees the deployment ID is set
                cast(str, self.mcp_deployment_id),
            )
            headers = self._build_authenticated_headers()

            logger.info(f"Using DataRobot hosted MCP deployment: {url}")

            return {
                "url": url,
                "transport": "streamable-http",
                "headers": headers,
            }

        if kind == "external":
            # External MCP URL - no authentication needed
            headers = {}

            # Merge external headers if provided
            if self.external_mcp_headers:
                external_headers = json.loads(self.external_mcp_headers)
                headers.update(external_headers)

            logger.info(f"Using external MCP URL: {self.external_mcp_url}")

            return {
                # cast: kind == "external" guarantees external_mcp_url is set
                "url": cast(str, self.external_mcp_url).rstrip("/"),
                "transport": self.external_mcp_transport,
                "headers": headers,
            }

        if kind == "local":
            # cast: kind == "local" guarantees the port is set
            url = build_local_mcp_url(cast(int, self.mcp_server_port))
            headers = self._build_authenticated_headers()
            logger.info(f"Using localhost MCP server: {url}")
            return {
                "url": url,
                "transport": "streamable-http",
                "headers": headers,
            }

        return None
