# Copyright 2026 DataRobot, Inc. and its affiliates.
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

import logging
from collections.abc import AsyncIterator
from collections.abc import Awaitable
from collections.abc import Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar

from fastapi import Request
from nat.data_models.api_server import Request as NATRequest
from nat.data_models.authentication import AuthenticatedContext
from nat.data_models.authentication import AuthFlowType
from nat.data_models.authentication import AuthProviderBaseConfig
from nat.data_models.interactive import HumanResponse
from nat.data_models.interactive import InteractionPrompt
from nat.runtime.session import Session
from nat.runtime.session import SessionManager
from nat.runtime.user_metadata import RequestAttributes
from pydantic import BaseModel
from starlette.requests import HTTPConnection

from datarobot_genai.core.utils.auth import AuthContextHeaderHandler

from .request import DRAgentRunAgentInput
from .response import DRAgentEventResponse

logger = logging.getLogger(__name__)


_auth_handler = AuthContextHeaderHandler()


def _resolve_dr_user_id(headers: dict[str, str]) -> str | None:
    """Resolve ``user_id`` from DataRobot-specific request headers.

    Tries ``X-DataRobot-Authorization-Context`` (signed app-context JWT
    carrying the full identity) first, then ``X-DataRobot-User-Id``
    (populated by predictions-gateway for any authenticated deployment
    request, including direct API-token calls).

    Returns ``None`` if neither header yields a value; callers should fall
    through to NAT's standard extractor (Bearer/JWT/cookie/API-key) in
    that case.
    """
    auth_ctx = _auth_handler.get_context(headers)
    if auth_ctx is not None:
        logger.debug(
            "Resolved user_id from X-DataRobot-Authorization-Context: %s", auth_ctx.user.id
        )
        return auth_ctx.user.id

    user_id = headers.get("x-datarobot-user-id") or headers.get("X-DataRobot-User-Id")
    if user_id:
        logger.debug("Resolved user_id from X-DataRobot-User-Id header: %s", user_id)
        return user_id

    return None


# ContextVar used by _PerUserCompatibleAgentExecutor to forward the incoming A2A HTTP
# request headers into the NAT context so auth providers (e.g. OAuth2CrossApplicationAccess)
# can read them via Context.get().metadata.headers.  Module-level so the same var is
# shared across all DRAgentAGUISessionManager instances (ContextVars are per-async-task).
_a2a_headers: ContextVar[dict[str, str] | None] = ContextVar("_a2a_headers", default=None)


def _build_metadata_from_headers(headers: dict[str, str]) -> RequestAttributes:
    """Build a :class:`RequestAttributes` carrying the given headers.

    Isolates access to NAT internals (``RequestAttributes._request``) in a
    single place so upstream changes only require one update.
    """
    attrs = RequestAttributes()
    attrs._request = NATRequest(headers=headers)
    return attrs


class DRAgentAGUISessionManager(SessionManager):
    @asynccontextmanager
    async def session(
        self,
        user_id: str | None = None,
        http_connection: HTTPConnection | None = None,
        user_message_id: str | None = None,
        conversation_id: str | None = None,
        user_input_callback: Callable[[InteractionPrompt], Awaitable[HumanResponse]] | None = None,
        user_authentication_callback: Callable[
            [AuthProviderBaseConfig, AuthFlowType], Awaitable[AuthenticatedContext | None]
        ]
        | None = None,
    ) -> AsyncIterator[Session]:
        """Resolve user_id from A2A preset or DataRobot headers before delegating to NAT."""
        if user_id is None:
            preset = self._context_state.user_id.get()
            if preset:
                user_id = preset

        if user_id is None and isinstance(http_connection, Request):
            user_id = _resolve_dr_user_id(dict(http_connection.headers))

        # Inject A2A request headers BEFORE super().session() so they are
        # available during per-user builder creation (agent card discovery
        # reads Context.get().metadata.headers). This is safe because NAT's
        # session() does NOT call set_metadata_from_http_request() when
        # http_connection is None (the A2A case).
        token_metadata = None
        preset_headers = _a2a_headers.get()
        if preset_headers:
            attrs = _build_metadata_from_headers(preset_headers)
            token_metadata = self._context_state._metadata.set(attrs)

        # Wrap the entire super().session() in try/finally so _metadata is
        # always reset — even if super().session().__aenter__() raises
        # (e.g. per-user builder creation failure).
        try:
            async with super().session(
                user_id=user_id,
                http_connection=http_connection,
                user_message_id=user_message_id,
                conversation_id=conversation_id,
                user_input_callback=user_input_callback,
                user_authentication_callback=user_authentication_callback,
            ) as sess:
                yield sess
        finally:
            if token_metadata is not None:
                self._context_state._metadata.reset(token_metadata)

    def get_workflow_input_schema(self) -> type[BaseModel]:
        """Get workflow input schema for OpenAPI documentation."""
        return DRAgentRunAgentInput

    def get_workflow_streaming_output_schema(self) -> type[BaseModel]:
        """Get workflow streaming output schema for OpenAPI documentation."""
        return DRAgentEventResponse
