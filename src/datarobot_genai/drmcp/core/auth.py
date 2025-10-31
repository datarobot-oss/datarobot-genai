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

from datarobot_genai.core.utils.b64 import decode_b64_to_json

logger = logging.getLogger(__name__)


class OAuthMiddleWare(Middleware):
    """Middleware that parses `x-datarobot-authorization-context` for tool calls.

    The header is expected to be a base64-encoded JSON object representing an
    authentication context compatible with :class:`datarobot.auth.session.AuthCtx`.
    """

    auth_ctx_header: str = "x-datarobot-authorization-context"

    async def on_call_tool(
        self, context: MiddlewareContext, call_next: CallNext[Any, ToolResult]
    ) -> ToolResult:
        """Parse header and attach an AuthCtx to the context before running the tool."""
        headers = get_http_headers()

        auth_ctx: AuthCtx | None = None
        raw = headers.get(self.auth_ctx_header)

        if auth_ctx_dict := decode_b64_to_json(raw):
            try:
                auth_ctx = AuthCtx(**auth_ctx_dict)
            except Exception as exc:
                logger.exception("Error when parsing %s header: %s", self.auth_ctx_header, exc)

        setattr(context, "auth_context", auth_ctx)

        return await call_next(context)


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
    auth_ctx = getattr(get_context(), "auth_context", None)
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
