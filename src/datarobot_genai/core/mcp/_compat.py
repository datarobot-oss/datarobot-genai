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

"""Interim home for the declared MCP server shape, until the SDK ships it.

!!! REMOVE WHEN THE `datarobot` RELEASE CARRYING `MCPServerRef` IS PINNED !!!

``MCPServerRef`` and the two resolvers belong in ``datarobot.core.config``, beside
``LLMConfig`` and ``resolve_llm_config``: the application's ``config.py`` annotates
``list[MCPServerRef]``, so both the app and genai have to be able to name the type, and
the settings base class is what the resolvers hang off.

That release is not out yet, and this work should not be blocked on one. So genai hosts
both, and prefers the SDK's the moment it appears. Removing this module means deleting
the fallbacks below and importing straight from ``datarobot.core.config`` -- the names
and behaviour are identical by construction.

The resolvers are module functions **taking the config explicitly**, which the LLM seam
deliberately forbids for its own resolver. The reason that rule exists is that a
zero-argument ``resolve_llm_config()`` hides *which* config it read and becomes a place
to grow routing logic. These take the config as an argument, do nothing but delegate,
and exist only until the method is on the base class.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator

try:  # pragma: no cover - exercised by whichever datarobot release is installed
    from datarobot.core.config import DEFAULT_MCP_SERVER_NAME
    from datarobot.core.config import MCP_ADDRESS_FIELDS
    from datarobot.core.config import MCPServerKind
    from datarobot.core.config import MCPServerRef

    SDK_HAS_MCP_SERVER_REF = True
except ImportError:  # pragma: no cover - the pre-release path
    SDK_HAS_MCP_SERVER_REF = False

    DEFAULT_MCP_SERVER_NAME = "default"
    MCP_ADDRESS_FIELDS = ("deployment_id", "workload_id", "local_port", "url")
    _LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})

    # `(str, Enum)` rather than `StrEnum`, deliberately: this definition is replaced by
    # the SDK's the moment it ships, and the SDK supports Pythons that have no StrEnum.
    # The two must behave identically, including how `str()` renders a member.
    class MCPServerKind(str, Enum):  # type: ignore[no-redef]  # noqa: UP042
        """Which hosting option an :class:`MCPServerRef` addresses."""

        WORKLOAD = "workload"
        DEPLOYMENT = "deployment"
        LOCAL = "local"
        EXTERNAL = "external"

    class MCPServerRef(BaseModel):  # type: ignore[no-redef]
        """One MCP server as an application declares it: a name, and exactly one address.

        Declared intent, not a resolved fact: pure, no I/O, validated at config load, so
        a malformed ID or a second address is an error where it was written rather than
        a ``None`` that later reads as "no server configured".
        """

        model_config = ConfigDict(frozen=True)

        # Hyphens allowed: the name is the prefix on every tool the server exposes, and
        # `-` is valid in a tool name.
        name: str = Field(default=DEFAULT_MCP_SERVER_NAME, pattern=r"^[a-z][a-z0-9_-]*$")
        deployment_id: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{24}$")
        workload_id: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{24}$")
        local_port: int | None = Field(default=None, ge=1, le=65535)
        local_host: str = "localhost"
        url: str | None = None
        headers: dict[str, str] = Field(default_factory=dict)
        transport: Literal["sse", "streamable-http"] = "streamable-http"
        identity: Literal["dr_service", "dr_user", "none"] | None = None

        @model_validator(mode="after")
        def _exactly_one_address(self) -> MCPServerRef:
            addressed = [f for f in MCP_ADDRESS_FIELDS if getattr(self, f) is not None]
            if len(addressed) != 1:
                raise ValueError(
                    f"MCP server {self.name!r} must set exactly one of "
                    f"{list(MCP_ADDRESS_FIELDS)}; got {addressed or 'none'}."
                )
            return self

        @model_validator(mode="after")
        def _no_loopback_under_url(self) -> MCPServerRef:
            # `url` means "third-party", which sends no DataRobot credentials. A local
            # server usually does not enforce auth, so this mistake surfaces on deploy,
            # not on the laptop where it was made.
            if self.url and urlsplit(self.url).hostname in _LOOPBACK_HOSTS:
                raise ValueError(
                    f"MCP server {self.name!r} addresses a loopback host through `url`, "
                    f"which sends no DataRobot credentials. Use `local_port` instead."
                )
            return self

        @property
        def kind(self) -> MCPServerKind:
            if self.workload_id is not None:
                return MCPServerKind.WORKLOAD
            if self.deployment_id is not None:
                return MCPServerKind.DEPLOYMENT
            if self.local_port is not None:
                return MCPServerKind.LOCAL
            return MCPServerKind.EXTERNAL


# The remote addresses, which are mutually exclusive. MCP_SERVER_PORT is deliberately
# absent: it names the port an MCP *server* process binds, not an address a client chose,
# and application templates set it unconditionally for their bundled server. It stays a
# lowest-precedence fallback below.
_LEGACY_ADDRESS_SETTINGS = {
    "workload_id": ("mcp_workload_id", "MCP_WORKLOAD_ID"),
    "deployment_id": ("mcp_deployment_id", "MCP_DEPLOYMENT_ID"),
    "url": ("external_mcp_url", "EXTERNAL_MCP_URL"),
}


def _legacy_value(config: Any, field: str) -> Any:
    """Read one singular pre-``mcp_servers`` setting off a config.

    A declared field goes through the config's normal sources. A field the config does
    not declare at all is read straight from the environment, so an existing ``.env`` or
    runtime parameter keeps working against a config never updated to declare it.
    """
    if field in type(config).model_fields:
        return getattr(config, field, None)
    from datarobot.core.config import getenv  # noqa: PLC0415

    return getenv(field.upper())


def _synthesise_default_server(config: Any) -> MCPServerRef | None:
    """Build the ``default`` server from the singular pre-``mcp_servers`` settings."""
    addresses = {
        key: _legacy_value(config, field) for key, (field, _) in _LEGACY_ADDRESS_SETTINGS.items()
    }
    set_addresses = {k: v for k, v in addresses.items() if v not in (None, "")}
    if len(set_addresses) > 1:
        offenders = sorted(_LEGACY_ADDRESS_SETTINGS[k][1] for k in set_addresses)
        raise ValueError(
            f"{', '.join(offenders)} are mutually exclusive but all are set. Set one, or "
            f"declare each server as an entry in MCP_SERVERS."
        )
    if not set_addresses:
        # MCP_SERVER_PORT last, and only when nothing remote is set. See the note above.
        local_port = _legacy_value(config, "mcp_server_port")
        if local_port in (None, ""):
            return None
        set_addresses = {"local_port": local_port}

    extra: dict[str, Any] = {}
    if "url" in set_addresses:
        raw_headers = _legacy_value(config, "external_mcp_headers")
        if raw_headers:
            # Deliberately not caught: EXTERNAL_MCP_HEADERS used to log a warning and
            # resolve to no headers at all, which reads as "this server needs none".
            extra["headers"] = json.loads(raw_headers)
        transport = _legacy_value(config, "external_mcp_transport")
        if transport:
            extra["transport"] = transport

    return MCPServerRef(name=DEFAULT_MCP_SERVER_NAME, **set_addresses, **extra)


def resolve_mcp_servers(config: Any) -> list[MCPServerRef]:
    """Return every MCP server a config declares, preferring the SDK's own resolver."""
    if hasattr(config, "resolve_mcp_servers"):
        return list(config.resolve_mcp_servers())

    declared: list[MCPServerRef] = list(getattr(config, "mcp_servers", None) or [])
    names = [ref.name for ref in declared]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(
            f"MCP server names must be unique; {duplicates} appear more than once in mcp_servers."
        )
    # The singular settings are a fallback for a configuration that has not adopted
    # `mcp_servers` at all -- not entries merged into one that has. Declaring a fleet
    # supersedes them wholesale, so the list you wrote is the fleet you get. Merging
    # per-name would silently add a `default` local server from MCP_SERVER_PORT, which
    # the application templates set unconditionally for their bundled server.
    if not declared:
        legacy = _synthesise_default_server(config)
        if legacy is not None:
            declared.append(legacy)
    return declared


def resolve_mcp_server(config: Any, name: str = DEFAULT_MCP_SERVER_NAME) -> MCPServerRef:
    """Return one declared MCP server by name, preferring the SDK's own resolver.

    Raises ``LookupError`` if the name is not configured. Deliberately fatal: a typo
    would otherwise be indistinguishable from a working server, and this is also the
    only thing that catches a config provider registered too late, since the fallback
    config contains none of the app's server names.
    """
    if hasattr(config, "resolve_mcp_server"):
        return config.resolve_mcp_server(name)

    servers = resolve_mcp_servers(config)
    for ref in servers:
        if ref.name == name:
            return ref
    raise LookupError(
        f"No MCP server named {name!r}. Configured: "
        f"{[ref.name for ref in servers] or 'none'}. If it should come from MCP_SERVERS, "
        f"check that the application registered its config provider before its "
        f"configuration was first read."
    )
