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
from typing import Literal

from a2a.types import InvalidParamsError
from a2a.utils.errors import ServerError
from fastapi import Request
from nat.data_models.api_server import Request as NATRequest
from nat.data_models.authentication import AuthenticatedContext
from nat.data_models.authentication import AuthFlowType
from nat.data_models.authentication import AuthProviderBaseConfig
from nat.data_models.interactive import HumanResponse
from nat.data_models.interactive import InteractionPrompt
from nat.data_models.user_info import UserInfo
from nat.runtime.session import Session
from nat.runtime.session import SessionManager
from nat.runtime.user_manager import UserManager
from nat.runtime.user_metadata import RequestAttributes
from pydantic import BaseModel
from starlette.requests import HTTPConnection
from starlette.websockets import WebSocket

from datarobot_genai.core.utils.auth import AuthContextHeaderHandler

from .request import DRAgentRunAgentInput
from .response import DRAgentEventResponse

logger = logging.getLogger(__name__)


_auth_handler = AuthContextHeaderHandler()

_AUTH_CONTEXT_HEADER = "x-datarobot-authorization-context"
_GATEWAY_USER_ID_HEADER = "x-datarobot-user-id"
_INVALID_AUTH_CONTEXT_MSG = (
    "X-DataRobot-Authorization-Context header is present but invalid or expired"
)

# Constant key used by DRAgentAGUISessionManager.session() to satisfy NAT's
# per-user workflow requirement when no caller identity is present. Purely a
# workflow-keying fallback — not an identity claim. Must not be used by any
# code that depends on the result being a real user.
DEFAULT_DR_AGENT_USER_ID = "default-user"


class DRAgentUserManager(UserManager):
    """Add DataRobot signed auth-context resolution to NAT's standard identity extractors.

    NAT 1.6 replaced the extensible context-based user_id resolution with
    ``UserManager.extract_user_from_connection()`` (#1775), which only supports
    standard auth (Bearer JWT, cookies, API key). DataRobot passes user identity
    via ``X-DataRobot-Authorization-Context`` (signed app-context JWT), which the
    vanilla extractor does not understand. This subclass handles that header
    first, then delegates to ``super().extract_user_from_connection()`` so
    standard auth still works.
    """

    @classmethod
    def extract_user_from_connection(cls, connection: Request | WebSocket) -> UserInfo | None:
        if isinstance(connection, Request):
            auth_ctx = _auth_handler.get_context(dict(connection.headers))
            if auth_ctx is not None:
                logger.debug(
                    "Resolved user_id from X-DataRobot-Authorization-Context: %s", auth_ctx.user.id
                )
                return UserInfo._from_session_cookie(auth_ctx.user.id)
        return super().extract_user_from_connection(connection)


# ContextVar used to forward incoming A2A HTTP request headers.  Set by
# :class:`~datarobot_genai.dragent.frontends.fastapi._PerUserCompatibleAgentExecutor`
# during message execution and by :class:`DRAgentA2AStarletteApplication` during
# public agent-card GET so auth and card selection can read gateway identity.
_a2a_headers: ContextVar[dict[str, str] | None] = ContextVar("_a2a_headers", default=None)


def normalise_headers(headers: dict[str, str] | None) -> dict[str, str] | None:
    if not headers:
        return None
    return {k.lower(): v for k, v in headers.items()}


def resolve_identity_from_headers(
    headers: dict[str, str] | None,
    *,
    on_invalid_auth_context: Literal["error", "none"] = "error",
) -> str | None:
    """Extract gateway-validated user identity from forwarded HTTP headers.

    Resolution order (first match wins):

    1. ``X-DataRobot-Authorization-Context`` -- signed JWT forwarded by
       components in the agent application template.  Decoded via
       :data:`_auth_handler` and hashed through
       ``UserInfo._from_session_cookie`` to produce the same UUID5 workflow
       key as the AG-UI path.  When this header is present but validation
       fails, behaviour depends on *on_invalid_auth_context*: ``"error"``
       raises :class:`~a2a.utils.errors.ServerError` with
       :class:`~a2a.types.InvalidParamsError`` (no fall-through to other
       headers or ``context_id``); ``"none"`` returns ``None``.
    2. ``X-DataRobot-User-Id`` -- raw DataRobot user ID injected by the API
       gateway, tied to the API-key owner.  Used only when the auth-context
       header is absent.  Same ``_from_session_cookie`` transform is applied
       for key-format consistency.
    3. ``None`` -- no gateway-provided identity (local dev).

    Returns ``None`` when *headers* are absent or contain no recognised
    identity header.
    """
    if not headers:
        return None

    if _AUTH_CONTEXT_HEADER in headers:
        try:
            auth_ctx = _auth_handler.get_context(headers)
        except Exception:
            logger.warning("Failed to decode auth-context header", exc_info=True)
            auth_ctx = None
        if auth_ctx is None:
            if on_invalid_auth_context == "error":
                raise ServerError(error=InvalidParamsError(message=_INVALID_AUTH_CONTEXT_MSG))
            return None
        return UserInfo._from_session_cookie(auth_ctx.user.id).get_user_id()

    raw_user_id = headers.get(_GATEWAY_USER_ID_HEADER)
    if raw_user_id:
        return UserInfo._from_session_cookie(raw_user_id).get_user_id()

    return None


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
        """Bridge A2A preset, resolve DR headers, default for per-user, and inject A2A headers.

        NAT 1.6+ assigns ``self._context_state.user_id`` from the explicit ``user_id``
        argument only. The A2A adapter calls ``session()`` with no arguments; our
        executor sets ``context_id`` on the context var first, but the parent
        ``session()`` would overwrite it with ``None``. Copy any preset non-empty
        value into ``user_id`` before delegating so per-user workflows work without
        a Bearer JWT (local dev / message-only A2A).

        Additionally, A2A HTTP request headers stored in the module-level
        ``_a2a_headers`` ContextVar by :class:`_PerUserCompatibleAgentExecutor` are
        injected into ``ContextState._metadata``.
        """
        if user_id is None:
            preset = self._context_state.user_id.get()
            if preset:
                user_id = preset

        # Resolve identity via our UserManager (knows DR signed auth-context in
        # addition to NAT's standard extractors). Done explicitly rather than
        # via a module rebind so the default-user fallback below stays scoped
        # to this session() — DRAgentUserManager remains a pure identity
        # resolver that other callers can safely use.
        if user_id is None and isinstance(http_connection, (Request, WebSocket)):
            user_info = DRAgentUserManager.extract_user_from_connection(http_connection)
            if user_info is not None:
                user_id = user_info.get_user_id()

        # Per-user workflows need *a* key in the per-user-builder dict.
        # Fall back to a constant so they do not crash when no identity is
        # available (e.g. locust against a deployed agent). This is a
        # workflow-keying concern, not an identity claim — all such requests
        # share one builder/state.
        if user_id is None and self.is_workflow_per_user:
            logger.debug(
                "No identity resolved for per-user workflow — using %s as key",
                DEFAULT_DR_AGENT_USER_ID,
            )
            user_id = DEFAULT_DR_AGENT_USER_ID

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
