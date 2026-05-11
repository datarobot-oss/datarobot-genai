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

from fastapi import Request
from nat.data_models.authentication import AuthenticatedContext
from nat.data_models.authentication import AuthFlowType
from nat.data_models.authentication import AuthProviderBaseConfig
from nat.data_models.interactive import HumanResponse
from nat.data_models.interactive import InteractionPrompt
from nat.data_models.user_info import UserInfo
from nat.runtime.session import Session
from nat.runtime.session import SessionManager
from nat.runtime.user_manager import UserManager
from pydantic import BaseModel
from starlette.requests import HTTPConnection

from datarobot_genai.core.utils.auth import AuthContextHeaderHandler

from .request import DRAgentRunAgentInput
from .response import DRAgentEventResponse

logger = logging.getLogger(__name__)


def _patch_user_manager() -> None:
    """Extend ``UserManager.extract_user_from_connection`` with DataRobot auth.

    NAT 1.6 replaced the extensible context-based user_id resolution with
    ``UserManager.extract_user_from_connection()`` (#1775) which only supports
    standard auth (Bearer JWT, cookies, API key).  DataRobot passes user identity
    via ``X-DataRobot-Authorization-Context``, which UserManager doesn't know about.

    This monkey-patches ``extract_user_from_connection`` to try the DataRobot header
    first, falling back to the original implementation for standard auth.
    """
    if getattr(UserManager.extract_user_from_connection, "_dr_patched", False):
        return

    auth_handler = AuthContextHeaderHandler()
    original_extract = UserManager.extract_user_from_connection

    def _extract_with_dr_auth(cls: type, connection: HTTPConnection) -> UserInfo | None:
        if isinstance(connection, Request):
            auth_ctx = auth_handler.get_context(dict(connection.headers))
            if auth_ctx is not None:
                user_info = UserInfo._from_session_cookie(auth_ctx.user.id)
                logger.debug(
                    "Resolved user_id from DataRobot auth context: %s", user_info.get_user_id()
                )
                return user_info
        return original_extract.__func__(cls, connection)

    _extract_with_dr_auth._dr_patched = True  # type: ignore[attr-defined]
    UserManager.extract_user_from_connection = classmethod(_extract_with_dr_auth)  # type: ignore[assignment]


_patch_user_manager()


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
        """Bridge A2A and other callers that preset ``ContextState.user_id``.

        NAT 1.6+ assigns ``self._context_state.user_id`` from the explicit ``user_id``
        argument only. The A2A adapter calls ``session()`` with no arguments; our
        executor sets ``context_id`` on the context var first, but the parent
        ``session()`` would overwrite it with ``None``. Copy any preset non-empty
        value into ``user_id`` before delegating so per-user workflows work without
        a Bearer JWT (local dev / message-only A2A).
        """
        if user_id is None:
            preset = self._context_state.user_id.get()
            if preset:
                user_id = preset
        async with super().session(
            user_id=user_id,
            http_connection=http_connection,
            user_message_id=user_message_id,
            conversation_id=conversation_id,
            user_input_callback=user_input_callback,
            user_authentication_callback=user_authentication_callback,
        ) as sess:
            yield sess

    def get_workflow_input_schema(self) -> type[BaseModel]:
        """Get workflow input schema for OpenAPI documentation."""
        return DRAgentRunAgentInput

    def get_workflow_streaming_output_schema(self) -> type[BaseModel]:
        """Get workflow streaming output schema for OpenAPI documentation."""
        return DRAgentEventResponse
