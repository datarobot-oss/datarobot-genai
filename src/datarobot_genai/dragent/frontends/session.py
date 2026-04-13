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

import contextvars
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Request
from nat.runtime.session import SessionManager
from pydantic import BaseModel
from starlette.requests import HTTPConnection

from datarobot_genai.core.utils.auth import AuthContextHeaderHandler

from .request import DRAgentRunAgentInput
from .response import DRAgentEventResponse

logger = logging.getLogger(__name__)

_auth_context_handler = AuthContextHeaderHandler()


def _extract_user_id_from_dr_auth_context(request: Request) -> str | None:
    """Extract user ID from X-DataRobot-Authorization-Context header."""
    auth_ctx = _auth_context_handler.get_context(dict(request.headers))
    if auth_ctx is not None:
        return auth_ctx.user.id
    return None


class DRAgentAGUISessionManager(SessionManager):
    def get_workflow_input_schema(self) -> type[BaseModel]:
        """Get workflow input schema for OpenAPI documentation."""
        return DRAgentRunAgentInput

    def get_workflow_streaming_output_schema(self) -> type[BaseModel]:
        """Get workflow streaming output schema for OpenAPI documentation."""
        return DRAgentEventResponse

    async def set_metadata_from_http_request(
        self, request: Request
    ) -> tuple[contextvars.Token, contextvars.Token]:
        """Extend base metadata extraction."""
        return await super().set_metadata_from_http_request(request)

    @asynccontextmanager
    async def session(  # type: ignore[override]
        self,
        *,
        user_id: str | None = None,
        http_connection: HTTPConnection | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator:  # type: ignore[type-arg]
        """Override session to extract user_id from DataRobot auth context header.

        The DataRobot UI does not set the ``nat-session`` cookie that NAT normally uses
        to identify users for per-user workflows.  Instead, DataRobot passes user identity
        via the ``X-DataRobot-Authorization-Context`` JWT header.  We decode that header
        and pass the user ID to NAT's session so it resolves identity correctly.
        """
        if user_id is None and isinstance(http_connection, Request):
            user_id = _extract_user_id_from_dr_auth_context(http_connection)
            if user_id is not None:
                logger.debug("Set user_id from DataRobot auth context: %s", user_id)

        async with super().session(user_id=user_id, http_connection=http_connection, **kwargs) as s:
            yield s
