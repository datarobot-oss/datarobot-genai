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

"""Resolving a declared MCP server into one you can connect to.

``MCPServerRef`` is declared intent -- "analytics is workload ``6a72...``" -- and lives
in the SDK. :class:`MCPTarget` is the resolved fact, "it is at ``https://.../mcp``".
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from datarobot.core.config import MCPServerKind
from datarobot.core.config import MCPServerRef
from pydantic import BaseModel
from pydantic import ConfigDict

from datarobot_genai.core.utils.auth import AuthContextHeaderHandler
from datarobot_genai.dragent.deployment_urls import build_deployment_mcp_url
from datarobot_genai.dragent.deployment_urls import build_local_mcp_url
from datarobot_genai.dragent.deployment_urls import build_workload_mcp_url

logger = logging.getLogger(__name__)

#: ``MCPTargetKind`` is ``MCPServerKind``. The kind is a property of the declared ref,
#: so it is owned by the SDK; this alias is the name the resolution side reads better by.
MCPTargetKind = MCPServerKind


@lru_cache(maxsize=1)
def auth_context_handler() -> AuthContextHeaderHandler:
    """Return the process-wide authorization-context header handler.

    Stateless apart from the ``SESSION_SECRET_KEY`` it reads once; call
    ``auth_context_handler.cache_clear()`` after changing that key.
    """
    return AuthContextHeaderHandler()


class MCPTarget(BaseModel):
    """One MCP server, resolved: where it is, and what identity it accepts.

    Built by :func:`build_target` and nowhere else; the thing you configure is the
    :class:`MCPServerRef`. Holds the ref rather than copying out of it, so the source ID
    stays available for telemetry. Frozen and carrying no request state, so it is safe
    to resolve once and share across requests.

    Attributes
    ----------
    ref : MCPServerRef
        The declaration this was resolved from.
    url : str
        The resolved address.
    api_token : str or None
        The token to present; ``None`` when the server gets no DataRobot identity.
    """

    model_config = ConfigDict(frozen=True)

    ref: MCPServerRef
    url: str
    api_token: str | None = None

    @property
    def kind(self) -> MCPServerKind:
        return self.ref.kind

    @property
    def name(self) -> str:
        return self.ref.name


def _require_endpoint(ref: MCPServerRef, endpoint: str | None) -> str:
    """Return the endpoint, which composes the URL, so a DataRobot kind always needs it."""
    if not endpoint:
        raise ValueError(
            f"MCP server {ref.name!r} is a DataRobot {ref.kind.value}, so DATAROBOT_ENDPOINT "
            f"must be set."
        )
    return endpoint


def _resolve_service_token(ref: MCPServerRef, token: str | None) -> str | None:
    """Return the service token, required only of servers that present it.

    Separate from :func:`_require_endpoint` because composing a URL never needs the
    token: a cross-application-access server must build where no ``DATAROBOT_API_TOKEN``
    exists.
    """
    if not ref.sends_datarobot_credentials:
        return None
    if not token:
        raise ValueError(
            f"MCP server {ref.name!r} is a DataRobot {ref.kind.value} reached with "
            f"`auth_provider: {ref.resolved_auth_provider}`, so DATAROBOT_API_TOKEN must be "
            f"set. Servers using cross-application access do not need it; those reached "
            f"anonymously can set {ref.name}_mcp_auth_provider=none."
        )
    return token


def build_target(
    ref: MCPServerRef,
    *,
    datarobot_endpoint: str | None = None,
    datarobot_api_token: str | None = None,
) -> MCPTarget:
    """Resolve a declared server into a reachable target, or raise.

    Never returns ``None``: an unresolvable server fails here, naming itself, rather
    than degrading into "no MCP configured". Pure -- every kind composes its URL, so
    there is no network call. See :func:`build_workload_mcp_url` for the workload
    shapes.

    Raises
    ------
    ValueError
        A DataRobot-hosted kind with no endpoint or no API token.
    """
    kind = ref.kind

    if kind is MCPServerKind.EXTERNAL:
        # Whether a `url` server gets an identity is the ref's decision, not this
        # branch's; the host guard keeps the deliberate case from becoming a leak.
        assert ref.url is not None  # the exactly-one-address validator guarantees it
        ref.assert_credentials_allowed(datarobot_endpoint)
        return MCPTarget(
            ref=ref,
            url=ref.url.rstrip("/"),
            api_token=_resolve_service_token(ref, datarobot_api_token),
        )

    if kind is MCPServerKind.WORKLOAD:
        assert ref.workload_id is not None
        return MCPTarget(
            ref=ref,
            url=build_workload_mcp_url(_require_endpoint(ref, datarobot_endpoint), ref.workload_id),
            api_token=_resolve_service_token(ref, datarobot_api_token),
        )

    if kind is MCPServerKind.DEPLOYMENT:
        assert ref.deployment_id is not None
        return MCPTarget(
            ref=ref,
            url=build_deployment_mcp_url(
                _require_endpoint(ref, datarobot_endpoint), ref.deployment_id
            ),
            api_token=_resolve_service_token(ref, datarobot_api_token),
        )

    # LOCAL: DataRobot-hosted for credential purposes, but the token is optional --
    # a local server usually does not enforce auth.
    assert ref.local_port is not None
    return MCPTarget(
        ref=ref,
        url=build_local_mcp_url(ref.local_port, host=ref.local_host),
        api_token=datarobot_api_token or None,
    )


async def build_targets(
    refs: list[MCPServerRef],
    *,
    datarobot_endpoint: str | None = None,
    datarobot_api_token: str | None = None,
) -> list[MCPTarget]:
    """Resolve a whole fleet.

    Stays ``async`` for its callers' sake, but no longer needs to be: resolution became
    pure when the workload lookup was replaced by composition, so there is nothing left
    to overlap. It previously fanned the lookups out across threads because four
    workloads resolved in sequence was up to forty seconds of startup.

    Raises whatever :func:`build_target` raises, for the first server that fails.
    """
    return [
        build_target(
            ref,
            datarobot_endpoint=datarobot_endpoint,
            datarobot_api_token=datarobot_api_token,
        )
        for ref in refs
    ]


def build_datarobot_mcp_headers(
    *,
    endpoint: str | None = None,
    api_token: str | None = None,
    forwarded: dict[str, str] | None = None,
    auth_context: dict[str, Any] | None = None,
    extra: dict[str, str] | None = None,
    base: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build the DataRobot credentials to present to an MCP server.

    The same for every DataRobot-hosted server, which is what lets one auth provider
    instance be shared by name. The order is load-bearing:

    1. ``base`` -- static headers for this connection
    2. forwarded headers from the inbound request
    3. ``Authorization: Bearer <service token>``
    4. ``x-datarobot-api-key`` -- unless already forwarded
    5. ``X-DataRobot-Authorization-Context``
    6. ``extra`` -- an explicit override, so it wins

    Step 4 goes to every server: the workload gateway needs it and the others ignore it,
    so nothing varies per server. ``endpoint`` is unused; the token authenticates.
    """
    headers: dict[str, str] = dict(base or {})
    if forwarded:
        headers.update(forwarded)

    if api_token:
        headers["Authorization"] = (
            api_token if api_token.startswith("Bearer ") else f"Bearer {api_token}"
        )
        # A forwarded key is the caller's own scoped token and
        # outranks the service one.
        forwarded_names = {name.lower() for name in (forwarded or {})}
        if "x-datarobot-api-key" not in forwarded_names:
            headers["x-datarobot-api-key"] = api_token.removeprefix("Bearer ").strip()

    try:
        headers.update(auth_context_handler().get_header(auth_context))
    except (LookupError, RuntimeError):
        # No authorization context in scope (e.g. in tests).
        pass

    if extra:
        headers.update(extra)
    return headers


def build_headers(
    target: MCPTarget,
    *,
    forwarded: dict[str, str] | None = None,
    auth_context: dict[str, Any] | None = None,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """Headers for one resolved target, for callers that build clients themselves.

    A thin wrapper over :func:`build_datarobot_mcp_headers`, which the NAT auth provider
    calls instead, so the two paths agree by construction.
    """
    # `auth_provider: none` -- static headers only.
    if not target.ref.sends_datarobot_credentials:
        return {**target.ref.headers, **(extra or {})}

    return build_datarobot_mcp_headers(
        api_token=target.api_token,
        forwarded=forwarded,
        auth_context=auth_context,
        extra=extra,
        base=target.ref.headers,
    )


def build_server_config(
    target: MCPTarget,
    *,
    forwarded: dict[str, str] | None = None,
    auth_context: dict[str, Any] | None = None,
    extra: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Render a target as the ``{url, transport, headers}`` dict the MCP clients take.

    A fresh dict every call, so two requests never share one header set.
    """
    return {
        "url": target.url,
        "transport": target.ref.transport,
        "headers": build_headers(
            target, forwarded=forwarded, auth_context=auth_context, extra=extra
        ),
    }


def resolve_mcp_targets(names: list[str] | None = None) -> list[MCPTarget]:
    """Resolve the application's configured MCP servers into targets.

    The convenience entry point for agents that build their own tool list rather than
    declaring MCP clients in ``workflow.yaml``. Reads the application's config through
    the registered provider, so the fleet comes from ``MCP_SERVERS`` (or the singular
    settings it supersedes).

    Parameters
    ----------
    names:
        Resolve only these servers, in this order. ``None`` resolves every configured
        server. An unknown name raises ``LookupError``.
    """
    from datarobot_genai.core.config import resolve_config  # noqa: PLC0415

    config = resolve_config()
    refs = (
        [config.resolve_mcp_server(name) for name in names]
        if names is not None
        else config.resolve_mcp_servers()
    )
    return [
        build_target(
            ref,
            datarobot_endpoint=config.resolve_datarobot_endpoint(),
            datarobot_api_token=config.resolve_datarobot_api_token(),
        )
        for ref in refs
    ]


async def aresolve_mcp_targets(names: list[str] | None = None) -> list[MCPTarget]:
    """Async :func:`resolve_mcp_targets`, resolving workload lookups concurrently."""
    from datarobot_genai.core.config import resolve_config  # noqa: PLC0415

    config = resolve_config()
    refs = (
        [config.resolve_mcp_server(name) for name in names]
        if names is not None
        else config.resolve_mcp_servers()
    )
    return await build_targets(
        refs,
        datarobot_endpoint=config.resolve_datarobot_endpoint(),
        datarobot_api_token=config.resolve_datarobot_api_token(),
    )
