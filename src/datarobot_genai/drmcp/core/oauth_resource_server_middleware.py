# Copyright 2026 DataRobot, Inc.
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

"""ASGI middleware for MCP OAuth resource-server handshake on streamable HTTP."""

from http import HTTPStatus
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.responses import Response

from datarobot_genai.drmcpbase.oauth_protected_resource_metadata.manager import (
    MCPOAuthProtectedResourceMetadataManager,
)

from .config import get_config
from .constants import MCP_OAUTH_REALM
from .constants import MCP_PATH_ENDPOINT
from .routes_utils import build_oauth_protected_resource_metadata_url
from .routes_utils import prefix_mount_path


def _normalize_path(path: str) -> str:
    return path.rstrip("/") or "/"


def _extract_bearer_token(authorization_header: str) -> str | None:
    scheme, _, credentials = authorization_header.partition(" ")
    if scheme.lower() != "bearer":
        return None
    token = credentials.strip()
    return token or None


def _is_mcp_request_path(path: str, mcp_path: str) -> bool:
    normalized_path = _normalize_path(path)
    normalized_mcp_path = _normalize_path(mcp_path)
    return normalized_path == normalized_mcp_path or normalized_path.startswith(
        normalized_mcp_path + "/"
    )


def create_oauth_resource_server_middleware() -> type[BaseHTTPMiddleware] | None:
    """Return the OAuth resource-server middleware class when OAuth metadata is configured."""
    config = get_config()
    if not config.mcp_enable_unauthenticated_well_known_route:
        return None

    manager = MCPOAuthProtectedResourceMetadataManager(
        mcp_oauth_metadata=config.mcp_oauth_metadata,
    )
    if manager.get_protected_resource_metadata() is None:
        return None
    return MCPOAuthResourceServerMiddleware


class MCPOAuthResourceServerMiddleware(BaseHTTPMiddleware):
    """Challenge unauthenticated MCP HTTP requests with a 401 WWW-Authenticate response."""

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        mcp_path = prefix_mount_path(MCP_PATH_ENDPOINT)
        if not _is_mcp_request_path(request.url.path, mcp_path):
            return await call_next(request)

        bearer_token = _extract_bearer_token(request.headers.get("authorization", ""))
        if bearer_token is None:
            return self._unauthorized_response(request)

        return await call_next(request)

    @staticmethod
    def _unauthorized_response(request: Request) -> JSONResponse:
        metadata_url = build_oauth_protected_resource_metadata_url(request)
        www_authenticate = f'Bearer realm="{MCP_OAUTH_REALM}", resource_metadata="{metadata_url}"'
        return JSONResponse(
            status_code=HTTPStatus.UNAUTHORIZED,
            content={
                "error": "unauthorized",
                "error_description": "Bearer token required",
            },
            headers={"WWW-Authenticate": www_authenticate},
        )
