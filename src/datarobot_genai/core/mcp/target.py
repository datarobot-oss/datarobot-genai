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

"""Resolving a declared MCP server into one you can actually connect to.

Two types, and the split between them is the point:

``MCPServerRef`` (from ``datarobot.core.config``)
    Declared intent -- "the analytics server is workload ``6a72...``". Pure, validated
    at config load, no I/O.
``MCPTarget``
    Resolved fact -- "it is reachable at ``https://.../mcp``". Built by
    :func:`build_target`, which may perform a network call and may fail.

The kind is consulted twice, and that is the whole reason the target exists: once in
:func:`build_target` to decide *how to find the server*, and again in
:func:`build_headers` to decide *what to send it*. Both answers are per-server, so
neither can be resolved once from the environment for a whole fleet.
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

    Stateless apart from the signing key it reads once from ``SESSION_SECRET_KEY``, and
    the context it encodes arrives as an argument. Holding one of these on a config
    object is what made that object unsafe to copy, so it lives here instead. Call
    ``auth_context_handler.cache_clear()`` to rebuild it after changing that key.
    """
    return AuthContextHeaderHandler()


class MCPTarget(BaseModel):
    """One MCP server, resolved: where it is, and what identity it accepts.

    Constructed by :func:`build_target` at workflow build and nowhere else. It appears
    in no ``.env``, no ``config.py`` and no ``workflow.yaml`` -- the thing you configure
    is the :class:`MCPServerRef`.

    Holds the ref rather than copying out of it, so the source ID stays available for
    telemetry ("which deployment") instead of being consumed into a URL. ``url`` is the
    only genuinely new fact this type adds.

    Frozen, and it carries no request state, which is what makes "resolve once, share
    across every request and user" safe rather than risky. (Frozen but not hashable --
    the static-headers dict rules that out -- so it is shared by reference, never used
    as a cache key.)

    Attributes
    ----------
    ref : MCPServerRef
        The declaration this was resolved from: name, transport, static headers, and
        the source ID.
    url : str
        The resolved address.
    api_token : str or None
        The DataRobot service token to present, and **always** ``None`` for a
        third-party server. Making this a field rather than an argument to
        :func:`build_headers` is a security property: "external implies no token" is
        then a construction-time invariant enforced in one place, instead of a rule
        every call site has to remember.
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
    """Return the service token, requiring it only of servers that actually present it.

    Deliberately separate from :func:`_require_endpoint`: composing a URL needs the
    endpoint and never the token. Demanding both together is what made a
    cross-application-access server fail to build on a deployment with no
    ``DATAROBOT_API_TOKEN`` -- which is the normal state for XAA, where the identity is
    an exchanged per-user token rather than a service one.
    """
    if ref.api_token:
        return ref.api_token
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

    Never returns ``None``. An unresolvable server used to degrade into "no MCP server
    configured", which is a legitimate state and therefore indistinguishable from
    success; here it fails at build, naming the server.

    Pure: every kind composes its URL, so this performs no network call and cannot
    block. A workload used to cost an HTTP GET against the Workload API, on the
    reasoning that its route could not be derived from its ID; it can, from the same
    ``DR_WORKLOAD_EXTERNAL_URL_HOST`` signal the MCP server itself uses to publish that
    route. See :func:`build_workload_mcp_url` for the two shapes and the two cases that
    still need an explicit ``url``.

    Raises
    ------
    ValueError
        A DataRobot-hosted kind with no endpoint or no API token.
    """
    kind = ref.kind

    if kind is MCPServerKind.EXTERNAL:
        # `url` defaults to sending no DataRobot identity, but that is now the ref's
        # DEFAULT rather than this branch's decision -- a DataRobot-hosted server that
        # happens to be addressed by URL can declare otherwise. The host guard is what
        # keeps the deliberate case from becoming an accidental leak.
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
        api_token=ref.api_token or datarobot_api_token or None,
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

    The same for **every** DataRobot-hosted server, which is what lets one auth provider
    instance be shared by name -- the ordinary NAT model. Nothing here is per-server:

    1. ``base`` -- static headers configured for this connection
    2. forwarded headers from the inbound request
    3. ``Authorization: Bearer <service token>``
    4. ``x-datarobot-api-key`` -- unless already forwarded
    5. ``X-DataRobot-Authorization-Context``
    6. ``extra`` -- an explicit override, so it wins

    The order is load-bearing and the numbering is the contract.

    Step 4 used to be workload-only, on the theory that the Workload API gateway needs
    it and nothing else does. It is now unconditional: a deployment or a local process
    ignores an unknown header, so sending it always costs nothing and removes the only
    fact that varied per server -- which is what made a shared provider unsafe. A
    third-party server never reaches this function, because it names
    ``auth_provider: none``.

    ``endpoint`` is accepted for symmetry with the callers and is not read; the token is
    what authenticates.
    """
    headers: dict[str, str] = dict(base or {})
    if forwarded:
        headers.update(forwarded)

    if api_token:
        headers["Authorization"] = (
            api_token if api_token.startswith("Bearer ") else f"Bearer {api_token}"
        )
        # DO NOT SIMPLIFY: a forwarded key is the caller's own scoped token and outranks
        # the service one. `Authorization` itself is never forwarded (the context
        # extractor passes only x-datarobot-* and x-untrusted-*), so this is the only
        # step where a forwarded credential can actually be overwritten.
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

    Used by the framework adapters (``register.py`` / ``agent.py``), where there is no
    NAT auth provider to delegate to. On the NAT path the provider does this instead,
    via :func:`build_datarobot_mcp_headers`; the two agree by construction, because this
    is a thin wrapper over it.
    """
    # A server that sends no DataRobot identity gets its static headers and nothing
    # else. That is what `auth_provider: none` means, and it is the default for `url`.
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

    A fresh dict every call, with no memoisation, so two requests can never share one
    header set.
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
