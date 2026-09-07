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
from datetime import timedelta
from typing import TYPE_CHECKING
from typing import Any

from nat.plugins.mcp.client.client_base import AuthAdapter
from nat.plugins.mcp.client.client_base import MCPStreamableHTTPClient


if TYPE_CHECKING:
    import httpx
    from nat.authentication.interfaces import AuthProviderBase

logger = logging.getLogger(__name__)


class DataRobotAuthAdapter(AuthAdapter):
    async def _get_auth_headers(
        self, request: httpx.Request | None = None, response: httpx.Response | None = None
    ) -> dict[str, str]:
        """Get authentication headers from the NAT auth provider."""
        try:
            # Use the user_id passed to this AuthAdapter instance
            auth_result = await self.auth_provider.authenticate(
                user_id=self.user_id, response=response
            )
            as_kwargs = auth_result.as_requests_kwargs()
            return as_kwargs["headers"]
        except Exception as e:
            logger.warning("Failed to get auth token: %s", e)
            return {}


class DataRobotMCPStreamableHTTPClient(MCPStreamableHTTPClient):
    def __init__(
        self,
        url: str,
        auth_provider: AuthProviderBase | None = None,
        user_id: str | None = None,
        tool_call_timeout: timedelta = timedelta(seconds=60),
        auth_flow_timeout: timedelta = timedelta(seconds=300),
        reconnect_enabled: bool = True,
        reconnect_max_attempts: int = 2,
        reconnect_initial_backoff: float = 0.5,
        reconnect_max_backoff: float = 50.0,
    ):
        super().__init__(
            url=url,
            auth_provider=auth_provider,
            user_id=user_id,
            tool_call_timeout=tool_call_timeout,
            auth_flow_timeout=auth_flow_timeout,
            reconnect_enabled=reconnect_enabled,
            reconnect_max_attempts=reconnect_max_attempts,
            reconnect_initial_backoff=reconnect_initial_backoff,
            reconnect_max_backoff=reconnect_max_backoff,
        )
        effective_user_id = user_id or (
            auth_provider.config.default_user_id if auth_provider else None
        )
        self._httpx_auth = (
            DataRobotAuthAdapter(auth_provider, effective_user_id) if auth_provider else None
        )


def make_input_schema_enum_safe(tool_fn: Any) -> Any:
    """Workaround for BUZZOK-30556. Configure the registered input schema
    with ``use_enum_values=True`` so NAT's ``_convert_input_pydantic``
    extracts plain strings via ``getattr`` instead of ``Enum`` instances.
    Without this, NAT's downstream ``session_tool.input_schema.model_validate
    (kwargs)`` rejects ``Enum`` instances whose class identity differs from
    the cached enum class (which happens when LangChain builds its own
    ``args_schema`` from the same JSON schema).
    """
    schema = tool_fn.input_schema
    if schema is None or schema is type(None):  # noqa: E721
        return tool_fn
    schema.model_config["use_enum_values"] = True
    schema.model_rebuild(force=True)
    return tool_fn
