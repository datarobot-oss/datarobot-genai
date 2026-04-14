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

    @asynccontextmanager
    async def session(  # type: ignore[override]
        self,
        *,
        user_id: str | None = None,
        http_connection: HTTPConnection | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator:  # type: ignore[type-arg]
        """Override session to extract user_id from DataRobot auth context.

        Resolves user_id from two sources (in order):
        1. HTTP requests: ``X-DataRobot-Authorization-Context`` JWT header
        2. A2A requests: context var set by the A2A executor before calling session()

        This is needed because NAT 1.6 no longer allows set_metadata_from_http_request
        to set user_id (it gets overwritten), and the parent session() doesn't know
        about the DataRobot auth context header.
        """
        if user_id is None and isinstance(http_connection, Request):
            user_id = _extract_user_id_from_dr_auth_context(http_connection)
            if user_id is not None:
                logger.debug("Set user_id from DataRobot auth context: %s", user_id)

        # For A2A: the executor sets user_id via context var before calling session()
        if user_id is None:
            user_id = self._context_state.user_id.get()

        async with super().session(user_id=user_id, http_connection=http_connection, **kwargs) as s:
            yield s
