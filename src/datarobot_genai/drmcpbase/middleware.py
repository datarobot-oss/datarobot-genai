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

"""FastMCP middleware utilities shared across DataRobot MCP servers.

These middleware classes are runtime-specific (FastMCP).
Callers inject header and auth-context hooks so
each server can wire its own auth resolution (for example ``drtools.core.auth``).
"""

import logging
from collections.abc import Callable
from typing import Any

from fastmcp.server.dependencies import get_http_headers
from fastmcp.server.middleware import Middleware

logger = logging.getLogger(__name__)

HeaderInjector = Callable[[dict[str, str]], None]
AuthContextExtractor = Callable[[dict[str, str]], Any | None]
AuthContextSetter = Callable[[Any | None], None]


def read_http_headers() -> dict[str, str]:
    """Return current FastMCP HTTP headers, including authorization headers."""
    # fastmcp 3.x strips authorization headers by default; include them for auth
    return get_http_headers(include_all=True)


class RequestHeadersMiddleware(Middleware):
    """Read FastMCP request headers and forward them to an injector callback."""

    def __init__(self, inject_headers: HeaderInjector) -> None:
        self._inject_headers = inject_headers

    async def on_call_tool(self, context: Any, call_next: Any) -> Any:
        headers = _safe_read_http_headers()
        self._inject_headers(headers)
        return await call_next(context)


class OAuthMiddleWare(Middleware):
    """Inject HTTP headers and authorization context for FastMCP tool calls.

    Parameters
    ----------
    inject_headers:
        Called with request headers (for example ``drtools.core.auth.set_request_headers``).
    extract_auth_context:
        Parses authorization context from headers; return ``None`` when unavailable.
    set_auth_context:
        Stores parsed authorization context for the current request.
    auth_context_state_key:
        Key used when attaching auth context to FastMCP context state.
    attach_to_fastmcp_state:
        When True, also persist auth context on ``context.fastmcp_context`` state.
    """

    def __init__(
        self,
        *,
        inject_headers: HeaderInjector,
        extract_auth_context: AuthContextExtractor,
        set_auth_context: AuthContextSetter,
        auth_context_state_key: str = "authorization_context",
        attach_to_fastmcp_state: bool = True,
    ) -> None:
        self._inject_headers = inject_headers
        self._extract_auth_context = extract_auth_context
        self._set_auth_context = set_auth_context
        self._auth_context_state_key = auth_context_state_key
        self._attach_to_fastmcp_state = attach_to_fastmcp_state

    async def on_call_tool(self, context: Any, call_next: Any) -> Any:
        headers = _safe_read_http_headers()
        self._inject_headers(headers)

        auth_context = self._safe_extract_auth_context(headers)
        if not auth_context:
            logger.debug("No valid authorization context extracted from request headers.")

        self._set_auth_context(auth_context)

        if self._attach_to_fastmcp_state and context.fastmcp_context is not None:
            await context.fastmcp_context.set_state(self._auth_context_state_key, auth_context)
            logger.debug("Authorization context attached to FastMCP state.")

        return await call_next(context)

    def _safe_extract_auth_context(self, headers: dict[str, str]) -> Any | None:
        try:
            return self._extract_auth_context(headers)
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning("Failed to extract auth context from headers: %s", exc, exc_info=True)
            return None
        except Exception as exc:
            logger.error("Unexpected error extracting auth context: %s", exc, exc_info=True)
            return None


def _safe_read_http_headers() -> dict[str, str]:
    try:
        return read_http_headers()
    except Exception as exc:
        logger.warning("Failed to read HTTP headers: %s", exc, exc_info=True)
        return {}


def register_oauth_middleware(mcp: Any, middleware: Middleware) -> None:
    """Register a middleware instance with a FastMCP server."""
    mcp.add_middleware(middleware)
    logger.info("OAuth middleware registered successfully")
