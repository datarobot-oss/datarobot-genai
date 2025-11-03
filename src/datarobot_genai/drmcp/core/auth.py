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
from datarobot.models.genai.agent.auth import OAuthAccessTokenProvider
from datarobot.models.genai.agent.auth import ToolAuth
from fastmcp.server.dependencies import get_context
from fastmcp.server.dependencies import get_http_headers
from fastmcp.server.middleware import CallNext
from fastmcp.server.middleware import Middleware
from fastmcp.server.middleware import MiddlewareContext
from fastmcp.tools.tool import ToolResult

from datarobot_genai.core.utils.auth import AuthContextHeaderHandler
from datarobot_genai.drmcp import get_config

logger = logging.getLogger(__name__)


class OAuthMiddleWare(Middleware):
    """Middleware that parses `x-datarobot-authorization-context` for tool calls.

    The header is expected to be a JWT-encoded token representing an authentication
    context compatible with :class:`datarobot.auth.session.AuthCtx`.

    Attributes
    ----------
    auth_handler : AuthContextHeaderHandler
        Handler for encoding/decoding JWT tokens containing auth context.
    """

    def __init__(self, secret_key: str | None = None) -> None:
        """Initialize the middleware with authentication handler.

        Parameters
        ----------
        secret_key : Optional[str]
            Secret key for JWT validation. If None, uses the value from config.
        """
        secret_key = secret_key or get_config().session_secret_key
        self.auth_handler = AuthContextHeaderHandler(secret_key)

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

        if context.fastmcp_context is not None:
            context.fastmcp_context.auth_context = auth_context

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


async def get_auth_context() -> AuthCtx:
    """Retrieve the AuthCtx from the current request context, if available.

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
    auth_ctx = getattr(context, "auth_context", None)
    if not auth_ctx:
        raise RuntimeError("No authorization context found.")

    return auth_ctx


async def get_access_token(provider: str | None = None) -> str:
    """Retrieve access token from the DataRobot OAuth Provider Service.

    OAuth access tokens can be retrieved only for providers where the user completed
    the OAuth flow and granted consent.

    Note:
        *   Currently, only On-Behalf-Of (OBO) tokens are supported, which allow tools to
            act on behalf of the authenticated user, after the user has granted his consent.

    Parameters
    ----------
    provider : str, optional
        The name of the OAuth provider. It should match the name of the provider configured
        during provider setup. If no value is provided and only one OAuth provider exists, that
        provider will be used. If multiple providers exist and none is specified, an error will be
        raised.

    Returns
    -------
    The oauth access token.
    """
    token_provider = OAuthAccessTokenProvider(await get_auth_context())
    access_token = token_provider.get_token(ToolAuth.OBO, provider)
    return access_token
