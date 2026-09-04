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

"""Shared URL construction utilities for DataRobot deployment endpoints.

These helpers are the single source of truth for the URL patterns used to reach
a DataRobot-hosted A2A agent and to look up its agent card from the central
registry.  Both the server side (advertising its own URL in the agent card) and
the client side (deriving the RPC base URL from a ``deployment_id``) use these
functions so that the patterns stay in sync automatically.

When a workload is served from an Envoy-fronted enclave,
:func:`resolve_external_workload_base` is the agent's own gateway route and
:func:`resolve_external_workload_api_endpoint` is the enclave's ``/api/v2``
base (memory service, and so on) — distinct from the control-hub
``DATAROBOT_ENDPOINT``.
"""

import os

from datarobot_genai.dragent.constants import A2A_MOUNT_PATH

DEPLOYMENT_DIRECT_ACCESS_PATH = "directAccess"
DEPLOYMENT_A2A_PATH = f"{DEPLOYMENT_DIRECT_ACCESS_PATH}/{A2A_MOUNT_PATH}"
WORKLOAD_A2A_PATH = A2A_MOUNT_PATH
DEPLOYMENT_MCP_PATH = "directAccess/mcp"
MCP_PATH = "mcp"

_DEFAULT_DATAROBOT_ENDPOINT = "https://app.datarobot.com/api/v2"
_API_V2_SUFFIX = "/api/v2"

#: Enclave host an Envoy API gateway serves this workload from. Injected only there.
WORKLOAD_EXTERNAL_HOST_ENV = "DR_WORKLOAD_EXTERNAL_URL_HOST"
#: Path prefix the Envoy API gateway routes to this workload. Injected only there.
WORKLOAD_EXTERNAL_PREFIX_ENV = "DR_WORKLOAD_EXTERNAL_URL_PREFIX"


def join_mount_path(base: str, mount_path: str) -> str:
    """Append ``mount_path`` to ``base``, returning a single trailing slash either way.

    A generic composer, not an A2A-specific policy: ``DRAgentA2AConfig`` never actually
    produces an empty ``mount_path`` (mounting A2A at the application root is rejected
    there), but this function still handles it, collapsing to ``base``'s own trailing
    slash rather than leaving a ``//`` behind. Slashes on both sides are stripped before
    joining so callers need not agree on which side owns the separator.

    Parameters
    ----------
    base:
        Path the mount hangs off, e.g. ``"deployments/xyz/directAccess"``.
    mount_path:
        Path suffix A2A is mounted under, possibly empty.

    Returns
    -------
    str
        ``"{base}/{mount_path}/"``, or ``"{base}/"`` when ``mount_path`` is empty.
    """
    root = base.rstrip("/")
    suffix = mount_path.strip("/")
    return f"{root}/{suffix}/" if suffix else f"{root}/"


def normalize_api_v2_endpoint(endpoint: str) -> str:
    """Return ``endpoint`` with exactly one trailing ``/api/v2`` and no trailing slash.

    Callers hand us ``DATAROBOT_ENDPOINT`` in both spellings — with and without
    the ``/api/v2`` suffix — and the deployment / workload routes live below it.
    Normalising in one place keeps ``.../api/v2/api/v2/...`` and
    ``.../deployments/...`` (missing ``/api/v2``) out of the URL builders.

    Parameters
    ----------
    endpoint:
        DataRobot endpoint
    -------
    str
        The endpoint with a single ``/api/v2`` suffix.
    """
    base = endpoint.rstrip("/")
    if base.endswith(_API_V2_SUFFIX):
        return base
    return f"{base}{_API_V2_SUFFIX}"


def resolve_datarobot_endpoint(require: bool = False) -> str | None:
    """Return the effective DataRobot API endpoint from the environment.

    Checks environment variables in priority order:

    1. ``DATAROBOT_PUBLIC_API_ENDPOINT`` — preferred for externally reachable URLs
       (on-prem deployments often set ``DATAROBOT_ENDPOINT`` to an internal k8s
       address while ``DATAROBOT_PUBLIC_API_ENDPOINT`` holds the public URL).
    2. ``DATAROBOT_ENDPOINT`` — standard SDK variable.
    3. Built-in default (``https://app.datarobot.com/api/v2``) when ``require``
       is *False*.

    Parameters
    ----------
    require:
        When *True*, raises :class:`ValueError` if neither env var is set.
        When *False* (default), falls back to the built-in default endpoint.

    Returns
    -------
    str | None
        The resolved endpoint string, or *None* only when ``require`` is
        *False* **and** both env vars are unset (returns the default instead,
        so in practice this always returns a non-None value when
        ``require=False``).

    Raises
    ------
    ValueError
        If ``require=True`` and neither ``DATAROBOT_PUBLIC_API_ENDPOINT`` nor
        ``DATAROBOT_ENDPOINT`` is set.
    """
    endpoint = os.getenv("DATAROBOT_PUBLIC_API_ENDPOINT") or os.getenv("DATAROBOT_ENDPOINT")
    if endpoint:
        return endpoint
    if require:
        raise ValueError("DATAROBOT_PUBLIC_API_ENDPOINT or DATAROBOT_ENDPOINT must be set.")
    return _DEFAULT_DATAROBOT_ENDPOINT


def _external_workload_host() -> str | None:
    """Return the Envoy host with a scheme, or *None* when the gateway vars are incomplete.

    Both ``DR_WORKLOAD_EXTERNAL_URL_HOST`` and ``DR_WORKLOAD_EXTERNAL_URL_PREFIX`` are
    required — the same presence signal as :func:`resolve_external_workload_base`.  The
    prefix is this workload's own route and is not returned here.
    """
    host = os.getenv(WORKLOAD_EXTERNAL_HOST_ENV, "").strip()
    prefix = os.getenv(WORKLOAD_EXTERNAL_PREFIX_ENV, "").strip()
    if not (host and prefix):
        return None
    if "://" not in host:
        host = f"https://{host}"
    return host.rstrip("/")


def resolve_external_workload_base() -> str | None:
    """Return the API gateway's base URL for this workload, or *None* when not behind one.

    An Envoy API gateway serves a workload from a per-enclave host and path prefix that
    ``DATAROBOT_ENDPOINT`` cannot derive, so it injects both as env vars.  Their presence is
    the signal that the URLs composed elsewhere in this module are unreachable; both are
    required.  The host is accepted with or without a scheme (``https://`` assumed).

    Returns
    -------
    str | None
        ``https://{host}/{prefix}``, no trailing slash, or *None* when either var is unset.
    """
    host = _external_workload_host()
    if host is None:
        return None
    prefix = os.getenv(WORKLOAD_EXTERNAL_PREFIX_ENV, "").strip()
    return f"{host}/{prefix.strip('/')}"


def resolve_external_workload_api_endpoint() -> str | None:
    """Return the enclave API gateway's ``/api/v2`` endpoint, or *None* when not behind one.

    Same presence signal as :func:`resolve_external_workload_base` (both host and prefix).
    The prefix is this workload's own route and is not part of the platform API base —
    only the host is used, so the DataRobot client talks to services on the enclave
    (memory, and so on) rather than the control hub.

    Returns
    -------
    str | None
        ``https://{host}/api/v2``, no trailing slash, or *None* when either var is unset.
    """
    host = _external_workload_host()
    if host is None:
        return None
    return normalize_api_v2_endpoint(host)


def build_deployment_a2a_url(
    endpoint: str, deployment_id: str, mount_path: str = A2A_MOUNT_PATH
) -> str:
    """Construct the A2A direct-access URL for a DataRobot deployment.

    ``directAccess`` forwards the full prefixed path to the container, so the suffix
    the agent actually mounted A2A under has to appear here too or the advertised URL
    would not resolve.

    Parameters
    ----------
    endpoint:
        DataRobot API endpoint base URL, e.g. ``https://app.datarobot.com/api/v2``.
        A trailing slash is stripped before composing the URL.
    deployment_id:
        The DataRobot deployment ID.
    mount_path:
        Path suffix A2A is mounted under inside the container, ``"a2a"`` by default.
        ``DRAgentA2AConfig`` never passes ``""`` (it rejects mounting A2A at the
        application root), but this generic composer still accepts it.

    Returns
    -------
    str
        A URL of the form ``{endpoint}/deployments/{deployment_id}/directAccess/a2a/``,
        with ``a2a`` replaced by ``mount_path`` and omitted entirely when it is empty.
    """
    base = endpoint.rstrip("/")
    return join_mount_path(
        f"{base}/deployments/{deployment_id}/{DEPLOYMENT_DIRECT_ACCESS_PATH}", mount_path
    )


def build_deployment_agent_card_url(endpoint: str, deployment_id: str) -> str:
    """Construct the agent card registry URL for a DataRobot deployment.

    Parameters
    ----------
    endpoint:
        DataRobot API endpoint base URL, e.g. ``https://app.datarobot.com/api/v2``.
        A trailing slash is stripped before composing the URL.
    deployment_id:
        The DataRobot deployment ID.

    Returns
    -------
    str
        A URL of the form ``{endpoint}/deployments/{deployment_id}/agentCard/``.
    """
    base = endpoint.rstrip("/")
    return f"{base}/deployments/{deployment_id}/agentCard/"


def build_workload_a2a_url(
    endpoint: str, workload_id: str, mount_path: str = A2A_MOUNT_PATH
) -> str:
    """Construct the A2A URL for a DataRobot workload.

    The workload route forwards the full prefixed path to the container, so the suffix
    the agent actually mounted A2A under has to appear here too or the advertised URL
    would not resolve.

    Parameters
    ----------
    endpoint:
        DataRobot API endpoint base URL, e.g. ``https://app.datarobot.com/api/v2``.
        A trailing slash is stripped before composing the URL.
    workload_id:
        The DataRobot workload ID.
    mount_path:
        Path suffix A2A is mounted under inside the container, ``"a2a"`` by default.
        ``DRAgentA2AConfig`` never passes ``""`` (it rejects mounting A2A at the
        application root), but this generic composer still accepts it.

    Returns
    -------
    str
        A URL of the form ``{endpoint}/endpoints/workloads/{workload_id}/a2a/``, with
        ``a2a`` replaced by ``mount_path`` and omitted entirely when it is empty.
    """
    base = endpoint.removesuffix("/")
    return join_mount_path(f"{base}/endpoints/workloads/{workload_id}", mount_path)


def build_deployment_mcp_url(endpoint: str, deployment_id: str) -> str:
    """Construct the MCP  URL.

    Parameters
    ----------
    endpoint:
        DataRobot endpoint, with or without the ``/api/v2`` suffix.
    deployment_id:
        The DataRobot deployment ID.

    Returns
    -------
    str
        A URL of the form
        ``{endpoint}/api/v2/deployments/{deployment_id}/directAccess/mcp``.
    """
    base = normalize_api_v2_endpoint(endpoint)
    return f"{base}/deployments/{deployment_id}/{DEPLOYMENT_MCP_PATH}"


def workload_mcp_url_from_endpoint(workload_endpoint: str, path: str = MCP_PATH) -> str:
    """Append the MCP path to the endpoint the platform reported for a workload.

    This is the only way to address a workload's MCP server: its route cannot be
    composed from a workload ID, due to different endpoints for the API Gateway.
    We look up the endpoint from the workload API
    (:func:`datarobot_genai.core.mcp.config.lookup_workload_endpoint`) and append
    the path to what it reports.

    Parameters
    ----------
    workload_endpoint:
        The workload's ``endpoint`` field as returned by the Workload API.
    path:
        Path the MCP server is served from, relative to the endpoint.

    Returns
    -------
    str
        ``{workload_endpoint}/{path}``, with duplicate slashes avoided.
    """
    return f"{workload_endpoint.rstrip('/')}/{path.lstrip('/')}"


def build_local_mcp_url(port: int, host: str = "localhost") -> str:
    """Construct the MCP URL for an MCP server running locally.

    Parameters
    ----------
    port:
        Port the local MCP server listens on.
    host:
        Host name, ``localhost`` by default.

    Returns
    -------
    str
        A URL of the form ``http://localhost:{port}/mcp``.
    """
    return f"http://{host}:{port}/{MCP_PATH}"


def build_workload_agent_card_url(endpoint: str, workload_id: str) -> str:
    """Construct the agent card URL for a DataRobot workload.

    Parameters
    ----------
    endpoint:
        DataRobot API endpoint base URL, e.g. ``https://app.datarobot.com/api/v2``.
        A trailing slash is stripped before composing the URL.
    workload_id:
        The DataRobot workload ID.

    Returns
    -------
    str
        A URL of the form ``{endpoint}/workloads/{workload_id}/agentCard/``.
    """
    base = endpoint.removesuffix("/")
    return f"{base}/workloads/{workload_id}/agentCard/"


def build_agent_cards_registry_url(endpoint: str) -> str:
    """Construct the URL for the central agent card registry.

    The central registry lists all agent cards within the user's organisation
    (tenant context) and requires only API-token authentication, not the
    per-agent AuthZ that the agent's own card endpoint demands.

    Parameters
    ----------
    endpoint:
        DataRobot API endpoint base URL, e.g. ``https://app.datarobot.com/api/v2``.
        A trailing slash is stripped before composing the URL.

    Returns
    -------
    str
        A URL of the form ``{endpoint}/agentCards/``.
    """
    base = endpoint.rstrip("/")
    return f"{base}/agentCards/"
