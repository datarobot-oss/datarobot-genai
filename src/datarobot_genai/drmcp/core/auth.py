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

from __future__ import annotations

import logging
from typing import Any

from datarobot.auth.session import AuthCtx
from datarobot.models.genai.agent.auth import ToolAuth
from fastmcp.server.dependencies import get_context
from fastmcp.server.dependencies import get_http_headers
from fastmcp.server.middleware import CallNext
from fastmcp.server.middleware import Middleware
from fastmcp.server.middleware import MiddlewareContext
from fastmcp.tools.tool import ToolResult

from datarobot_genai.core.utils.auth import AsyncOAuthTokenProvider
from datarobot_genai.core.utils.auth import AuthContextHeaderHandler

logger = logging.getLogger(__name__)


AUTH_CTX_KEY = "authorization_context"


class OAuthMiddleWare(Middleware):
    """Middleware that parses `x-datarobot-authorization-context` for tool calls.

    The header is expected to be a JWT-encoded token representing an authentication
    context compatible with :class:`datarobot.auth.session.AuthCtx`.

    Attributes
    ----------
    auth_handler : AuthContextHeaderHandler
        Handler for encoding/decoding JWT tokens containing auth context.
    """

    def __init__(self, auth_handler: AuthContextHeaderHandler | None = None) -> None:
        self.auth_handler = auth_handler or AuthContextHeaderHandler()

    async def on_call_tool(
        self, context: MiddlewareContext, call_next: CallNext[Any, ToolResult]
    ) -> ToolResult:
        """Parse header and attach an AuthCtx to the context before running the tool.

        Parameters
        ----------
        context : MiddlewareContext
            The middleware context that will be passed to the tool.
        call_next : CallNext[Any, ToolResult]
            The next handler in the middleware chain.

        Returns
        -------
        ToolResult
            The result from the tool execution.
        """
        auth_context = self._extract_auth_context()
        if not auth_context:
            logger.debug("No valid authorization context extracted from request headers.")

        if context.fastmcp_context is not None:
            context.fastmcp_context.set_state(AUTH_CTX_KEY, auth_context)
            logger.debug("Authorization context attached to state.")

        return await call_next(context)

    def _extract_auth_context(self) -> AuthCtx | None:
        """Extract and validate authentication context from request headers.

        Returns
        -------
        Optional[AuthCtx]
            The validated authentication context, or None if extraction fails.
        """
        try:
            headers = get_http_headers()
            return self.auth_handler.get_context(headers)
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning("Failed to extract auth context from headers: %s", exc, exc_info=True)
            return None
        except Exception as exc:
            logger.error("Unexpected error extracting auth context: %s", exc, exc_info=True)
            return None


async def must_get_auth_context() -> AuthCtx:
    """Retrieve the AuthCtx from the current request context or raise error.

    Raises
    ------
    RuntimeError
        If no authorization context is found in the request.

    Returns
    -------
    AuthCtx
        The authorization context associated with the current request.
    """
    context = get_context()

    auth_ctx = context.get_state(AUTH_CTX_KEY)
    if not auth_ctx:
        raise RuntimeError("Could not retrieve authorization context from FastMCP context state.")

    return auth_ctx


async def get_access_token(provider_type: str | None = None) -> str:
    """Retrieve access token from the DataRobot OAuth Provider Service.

    OAuth access tokens can be retrieved only for providers where the user completed
    the OAuth flow and granted consent.

    Note:
        *   Currently, only On-Behalf-Of (OBO) tokens are supported, which allow tools to
            act on behalf of the authenticated user, after the user has granted his consent.

    Parameters
    ----------
    provider_type : str, optional
        The name of the OAuth provider. It should match the name of the provider configured
        during provider setup. If no value is provided and only one OAuth provider exists, that
        provider will be used. If multiple providers exist and none is specified, an error will be
        raised.

    Returns
    -------
    The oauth access token.
    """
    auth_ctx = await must_get_auth_context()
    logger.debug("Retrieved authorization context")

    oauth_token_provider = AsyncOAuthTokenProvider(auth_ctx)
    oauth_access_token = await oauth_token_provider.get_token(
        auth_type=ToolAuth.OBO,
        provider_type=provider_type,
    )
    return oauth_access_token


def initialize_oauth_middleware(mcp: Any) -> None:
    """Initialize and register OAuth middleware with the MCP server.

    Parameters
    ----------
    mcp : FastMCP
        The FastMCP server instance to register the middleware with.
    secret_key : Optional[str]
        Secret key for JWT validation. If None, uses the value from config.
    """
    mcp.add_middleware(OAuthMiddleWare())
    logger.info("OAuth middleware registered successfully")
