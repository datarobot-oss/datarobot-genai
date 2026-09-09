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

"""The deprecated single-server MCP configuration.

``MCPConfig`` used to be a settings class: constructing it read the environment,
resolved exactly one server out of four mutually-exclusive variables by precedence, and
handed back a connection dict with credentials already baked in. That is why an agent
could reach only one MCP server, why the way you *addressed* a server decided which
credentials it received, and why a second configured address was discarded in silence.

It is now a plain shape with no sources and no behavior. The environment read belongs
to the application's own config; addresses are declared as ``MCP_SERVERS`` entries and
resolved through :mod:`datarobot_genai.core.mcp.target`:

    config = resolve_config()
    target = build_target(
        config.resolve_mcp_server("analytics"),
        datarobot_endpoint=config.resolve_datarobot_endpoint(),
        datarobot_api_token=config.resolve_datarobot_api_token(),
    )
"""

from __future__ import annotations

import warnings
from typing import Any
from typing import Literal

from datarobot.core.config import DEFAULT_MCP_SERVER_NAME
from datarobot.core.config import MCPServerRef
from pydantic import BaseModel

_REMOVED_MEMBER_MESSAGE = (
    "MCPConfig.{name} was removed. It read the environment and resolved one server, "
    "which is what limited an agent to a single MCP server. Resolve a named server "
    "instead: build_target(resolve_config().resolve_mcp_server(name), ...) from "
    "datarobot_genai.core.mcp."
)


class MCPConfig(BaseModel):
    """Deprecated. A single MCP server's address, as a plain value.

    Kept so that code holding one of these can be migrated in one step rather than
    rewritten blind, via :meth:`to_ref`. Constructing it no longer reads the
    environment, and it no longer produces headers or a connection dict.
    """

    external_mcp_url: str | None = None
    external_mcp_headers: str | None = None
    external_mcp_transport: Literal["sse", "streamable-http"] = "streamable-http"
    mcp_deployment_id: str | None = None
    mcp_workload_id: str | None = None
    datarobot_endpoint: str | None = None
    datarobot_api_token: str | None = None
    mcp_server_port: int | None = None

    def __init__(self, **data: Any) -> None:
        warnings.warn(
            "MCPConfig is deprecated and no longer reads the environment or builds "
            "headers. Declare servers in MCP_SERVERS and resolve them with "
            "resolve_config().resolve_mcp_server(name) plus build_target().",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(**data)

    def to_ref(self, name: str = DEFAULT_MCP_SERVER_NAME) -> MCPServerRef:
        """Convert this to the declared shape that replaces it.

        Raises ``ValueError`` when it holds no address or more than one -- the two cases
        the old precedence contest resolved silently.
        """
        import json  # noqa: PLC0415

        addresses = {
            "workload_id": self.mcp_workload_id,
            "deployment_id": self.mcp_deployment_id,
            "url": self.external_mcp_url,
            "local_port": self.mcp_server_port,
        }
        set_addresses = {k: v for k, v in addresses.items() if v not in (None, "")}
        extra: dict[str, Any] = {}
        if "url" in set_addresses:
            if self.external_mcp_headers:
                extra["headers"] = json.loads(self.external_mcp_headers)
            extra["transport"] = self.external_mcp_transport
        return MCPServerRef(name=name, **set_addresses, **extra)

    def __getattr__(self, name: str) -> Any:
        # The removed members were properties, so an ordinary AttributeError would read
        # as a typo rather than as the change that removed them.
        if name in {
            "server_config",
            "is_local_server",
            "auth_context_handler",
            "forwarded_headers",
            "authorization_context",
        }:
            raise AttributeError(_REMOVED_MEMBER_MESSAGE.format(name=name))
        raise AttributeError(name)
