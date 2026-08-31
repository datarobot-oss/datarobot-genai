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

import asyncio
import logging
import time
from functools import lru_cache
from typing import Any

import httpx
from pydantic import BaseModel
from pydantic import ConfigDict

from datarobot_genai.core.mcp._compat import MCPServerKind
from datarobot_genai.core.mcp._compat import MCPServerRef
from datarobot_genai.core.mcp._compat import resolve_mcp_server
from datarobot_genai.core.mcp._compat import resolve_mcp_servers
from datarobot_genai.core.utils.auth import AuthContextHeaderHandler
from datarobot_genai.dragent.deployment_urls import build_deployment_mcp_url
from datarobot_genai.dragent.deployment_urls import build_local_mcp_url
from datarobot_genai.dragent.deployment_urls import normalize_api_v2_endpoint
from datarobot_genai.dragent.deployment_urls import workload_mcp_url_from_endpoint

logger = logging.getLogger(__name__)

#: ``MCPTargetKind`` is ``MCPServerKind``. The kind is a property of the declared ref,
#: so it is owned by the SDK; this alias is the name the resolution side reads better by.
MCPTargetKind = MCPServerKind

#: Timeout for the workload endpoint lookup.
WORKLOAD_LOOKUP_TIMEOUT_SECONDS = 10.0

#: How long a resolved workload endpoint stays usable before it is looked up again.
#:
#: This cache used to have no expiry and was cleared only by tests, so a workload that
#: moved needed a process restart -- an emergent property of a module dict rather than
#: anyone's decision. A bounded lifetime makes the answer "a moved workload is picked up
#: within this window", which is a policy that can be argued with.
WORKLOAD_ENDPOINT_CACHE_TTL_SECONDS = 300.0

# (endpoint, workload_id) -> (resolved endpoint, monotonic expiry)
_WORKLOAD_ENDPOINT_CACHE: dict[tuple[str, str], tuple[str, float]] = {}

#: The one workload status whose reported endpoint is settled.
_WORKLOAD_RUNNING_STATUS = "running"


def clear_workload_endpoint_cache() -> None:
    """Forget every cached workload endpoint (used by tests)."""
    _WORKLOAD_ENDPOINT_CACHE.clear()


@lru_cache(maxsize=1)
def auth_context_handler() -> AuthContextHeaderHandler:
    """Return the process-wide authorization-context header handler.

    Stateless apart from the signing key it reads once from ``SESSION_SECRET_KEY``, and
    the context it encodes arrives as an argument. Holding one of these on a config
    object is what made that object unsafe to copy, so it lives here instead. Call
    ``auth_context_handler.cache_clear()`` to rebuild it after changing that key.
    """
    return AuthContextHeaderHandler()


def lookup_workload_endpoint(
    workload_id: str,
    *,
    endpoint: str,
    token: str,
    timeout: float = WORKLOAD_LOOKUP_TIMEOUT_SECONDS,
) -> str | None:
    """Return the endpoint the platform serves ``workload_id`` from, or *None*.

    A workload's URL cannot be composed from its ID and the caller's endpoint, because
    the shape depends on a server-side Workload API setting the caller cannot see.

    Parameters
    ----------
    workload_id:
        The DataRobot workload ID.
    endpoint:
        DataRobot API endpoint.
    token:
        DataRobot API token used for the lookup.
    timeout:
        Seconds to wait for the Workload API.

    Returns
    -------
    str | None
        The workload's ``endpoint`` field, or *None* when the workload cannot be read
        or reports no endpoint yet.
    """
    cache_key = (endpoint, workload_id)
    cached = _WORKLOAD_ENDPOINT_CACHE.get(cache_key)
    if cached is not None:
        value, expires_at = cached
        if time.monotonic() < expires_at:
            return value
        del _WORKLOAD_ENDPOINT_CACHE[cache_key]

    url = f"{normalize_api_v2_endpoint(endpoint)}/workloads/{workload_id}/"
    try:
        response = httpx.get(
            url,
            headers={"Authorization": f"Bearer {token.removeprefix('Bearer ').strip()}"},
            timeout=timeout,
        )
        response.raise_for_status()
        # ValueError covers a non-JSON body (json.JSONDecodeError subclasses it).
        payload: dict[str, Any] = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning(
            "Could not read the endpoint of workload %s from %s: %s. Check that the "
            "agent's API token may read the workload.",
            workload_id,
            url,
            exc,
        )
        return None

    resolved = payload.get("endpoint")
    if not isinstance(resolved, str) or not resolved.strip():
        logger.warning(
            "Workload %s reported no endpoint (status %r); it may not be running yet.",
            workload_id,
            payload.get("status"),
        )
        return None

    resolved = resolved.strip()
    status = payload.get("status")
    if status == _WORKLOAD_RUNNING_STATUS:
        _WORKLOAD_ENDPOINT_CACHE[cache_key] = (
            resolved,
            time.monotonic() + WORKLOAD_ENDPOINT_CACHE_TTL_SECONDS,
        )
    else:
        logger.info(
            "Workload %s is %r, so its endpoint is not cached; it will be resolved again.",
            workload_id,
            status,
        )
    logger.info("Workload %s is served from %s", workload_id, resolved)
    return resolved


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


def _require_datarobot_credentials(
    ref: MCPServerRef, endpoint: str | None, token: str | None
) -> tuple[str, str]:
    if not endpoint:
        raise ValueError(
            f"MCP server {ref.name!r} is a DataRobot {ref.kind.value}, so DATAROBOT_ENDPOINT "
            f"must be set."
        )
    if not token:
        raise ValueError(
            f"MCP server {ref.name!r} is a DataRobot {ref.kind.value}, so DATAROBOT_API_TOKEN "
            f"must be set."
        )
    return endpoint, token


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

    Raises
    ------
    ValueError
        A DataRobot-hosted kind with no endpoint or no API token.
    LookupError
        A workload whose endpoint cannot be read. There is no URL template to fall back
        on -- a guess would be right on some clusters and quietly wrong on others.
    """
    kind = ref.kind

    if kind is MCPServerKind.EXTERNAL:
        # No DataRobot identity, by construction: `url` means third-party.
        assert ref.url is not None  # the exactly-one-address validator guarantees it
        return MCPTarget(ref=ref, url=ref.url.rstrip("/"), api_token=None)

    if kind is MCPServerKind.WORKLOAD:
        endpoint, token = _require_datarobot_credentials(
            ref, datarobot_endpoint, datarobot_api_token
        )
        assert ref.workload_id is not None
        workload_endpoint = lookup_workload_endpoint(
            ref.workload_id, endpoint=endpoint, token=token
        )
        if workload_endpoint is None:
            raise LookupError(
                f"MCP server {ref.name!r} is workload {ref.workload_id}, whose endpoint could "
                f"not be read from {endpoint}. A workload's route cannot be composed from its "
                f"ID, so there is nothing to fall back to."
            )
        return MCPTarget(
            ref=ref, url=workload_mcp_url_from_endpoint(workload_endpoint), api_token=token
        )

    if kind is MCPServerKind.DEPLOYMENT:
        endpoint, token = _require_datarobot_credentials(
            ref, datarobot_endpoint, datarobot_api_token
        )
        assert ref.deployment_id is not None
        return MCPTarget(
            ref=ref,
            url=build_deployment_mcp_url(endpoint, ref.deployment_id),
            api_token=token,
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
    """Resolve a whole fleet, running the workload lookups concurrently.

    Every workload costs one HTTP call at :data:`WORKLOAD_LOOKUP_TIMEOUT_SECONDS`, so
    four workloads resolved in sequence is up to forty seconds of startup. Deployment,
    local and external refs need no call at all and complete immediately.

    Raises whatever :func:`build_target` raises, for the first server that fails.
    """
    if not refs:
        return []
    return list(
        await asyncio.gather(
            *(
                asyncio.to_thread(
                    build_target,
                    ref,
                    datarobot_endpoint=datarobot_endpoint,
                    datarobot_api_token=datarobot_api_token,
                )
                for ref in refs
            )
        )
    )


def build_headers(
    target: MCPTarget,
    *,
    forwarded: dict[str, str] | None = None,
    auth_context: dict[str, Any] | None = None,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build the headers to send one MCP server, in an order that is load-bearing.

    1. forwarded headers from the inbound request
    2. ``Authorization: Bearer <service token>``
    3. ``x-datarobot-api-key`` -- workloads only, and only if not already forwarded
    4. ``X-DataRobot-Authorization-Context``
    5. ``extra`` -- an explicit override, so it wins

    Parameters
    ----------
    target:
        The server being called. Its ``kind`` decides steps 2-4.
    forwarded:
        Headers forwarded from the inbound request (``x-datarobot-*`` and
        ``x-untrusted-*`` only).
    auth_context:
        The authorization context to encode into step 4's header.
    extra:
        Headers configured explicitly for this connection. Merged last.
    """
    # `external` skips steps 1-4 entirely. That IS "no DataRobot credentials", and it is
    # why a loopback `url` is rejected at config load.
    if target.kind is MCPServerKind.EXTERNAL:
        return {**target.ref.headers, **(extra or {})}

    headers: dict[str, str] = {}
    if forwarded:
        headers.update(forwarded)

    if target.api_token:
        token = target.api_token
        headers["Authorization"] = token if token.startswith("Bearer ") else f"Bearer {token}"
        # DO NOT SIMPLIFY: a forwarded key is the caller's own scoped token and outranks
        # the service one. `Authorization` itself is never forwarded (the context
        # extractor passes only x-datarobot-* and x-untrusted-*), so this is the only
        # step where a forwarded credential can actually be overwritten.
        forwarded_names = {name.lower() for name in (forwarded or {})}
        if target.kind is MCPServerKind.WORKLOAD and "x-datarobot-api-key" not in forwarded_names:
            headers["x-datarobot-api-key"] = token.removeprefix("Bearer ").strip()

    try:
        headers.update(auth_context_handler().get_header(auth_context))
    except (LookupError, RuntimeError):
        # No authorization context in scope (e.g. in tests).
        pass

    if extra:
        headers.update(extra)
    return headers


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
        [resolve_mcp_server(config, name) for name in names]
        if names is not None
        else resolve_mcp_servers(config)
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
        [resolve_mcp_server(config, name) for name in names]
        if names is not None
        else resolve_mcp_servers(config)
    )
    return await build_targets(
        refs,
        datarobot_endpoint=config.resolve_datarobot_endpoint(),
        datarobot_api_token=config.resolve_datarobot_api_token(),
    )
