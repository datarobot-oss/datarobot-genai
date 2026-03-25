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

from fastapi import Request
from nat.runtime.session import SessionManager
from pydantic import BaseModel

from datarobot_genai.core.utils.auth import AuthContextHeaderHandler

from .request import DRAgentRunAgentInput
from .response import DRAgentEventResponse

logger = logging.getLogger(__name__)

_auth_context_handler = AuthContextHeaderHandler()


class DRAgentAGUISessionManager(SessionManager):
    def get_workflow_input_schema(self) -> type[BaseModel]:
        """Get workflow input schema for OpenAPI documentation."""
        return DRAgentRunAgentInput

    def get_workflow_streaming_output_schema(self) -> type[BaseModel]:
        """Get workflow streaming output schema for OpenAPI documentation."""
        return DRAgentEventResponse

    async def set_metadata_from_http_request(
        self, request: Request
    ) -> tuple[contextvars.Token | None, contextvars.Token | None]:
        """Extend base metadata extraction to also set user_id from DataRobot auth context.

        The DataRobot UI does not set the ``nat-session`` cookie that NAT normally uses
        to identify users for per-user workflows.  Instead, DataRobot passes user identity
        via the ``X-DataRobot-Authorization-Context`` JWT header.  We decode that header
        and use the contained user ID so that per-user workflows receive a stable identity.
        """
        result = await super().set_metadata_from_http_request(request)

        # Only attempt extraction if user_id wasn't already set (e.g. via nat-session cookie).
        if self._context_state.user_id.get():
            return result

        auth_ctx = _auth_context_handler.get_context(dict(request.headers))
        if auth_ctx is not None:
            user_id = auth_ctx.user.id
            self._context_state.user_id.set(user_id)
            logger.debug("Set user_id from DataRobot auth context: %s", user_id)

        return result
