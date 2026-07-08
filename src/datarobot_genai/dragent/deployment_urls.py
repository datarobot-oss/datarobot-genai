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
"""

import os

# Default in-process A2A mount path (relative, without leading slash).
# Override via ``general.front_end.a2a.mount_path`` in workflow.yaml.
DEFAULT_A2A_MOUNT_PATH = "a2a"

# DataRobot gateway prefix for direct deployment access.
# The full gateway segment is ``_DIRECT_ACCESS_GATEWAY_PREFIX + normalized mount_path``,
# e.g. ``"directAccess/a2a"`` for the default mount path.
_DIRECT_ACCESS_GATEWAY_PREFIX = "directAccess"

# Backward-compatible constant — equals the gateway segment when mount_path is the default.
A2A_DIRECT_ACCESS_PATH = f"{_DIRECT_ACCESS_GATEWAY_PREFIX}/{DEFAULT_A2A_MOUNT_PATH}"


def normalize_a2a_mount_path(mount_path: str) -> str:
    """Return a canonical Starlette mount path: leading slash, no trailing slash.

    ``"/"`` is returned as-is (root mount).  Empty or whitespace-only input
    also yields ``"/"`` — Starlette requires at least ``"/"`` for a catch-all.

    Examples::

        normalize_a2a_mount_path("a2a")       == "/a2a"
        normalize_a2a_mount_path("/api/a2a/") == "/api/a2a"
        normalize_a2a_mount_path("/")         == "/"
        normalize_a2a_mount_path("")          == "/"
    """
    path = mount_path.strip()
    if not path:
        return "/"
    if not path.startswith("/"):
        path = "/" + path
    if path != "/":
        path = path.rstrip("/")
    return path


def build_a2a_local_url(host: str, port: int, mount_path: str) -> str:
    """Return the local agent-card base URL with a trailing slash.

    The ``mount_path`` argument should already be normalized (as produced by
    :func:`normalize_a2a_mount_path`), but the function tolerates unnormalized
    input by normalizing it internally.

    Examples::

        build_a2a_local_url("localhost", 8000, "/a2a")  == "http://localhost:8000/a2a/"
        build_a2a_local_url("localhost", 8000, "/")     == "http://localhost:8000/"
    """
    normalized = normalize_a2a_mount_path(mount_path)
    if normalized == "/":
        return f"http://{host}:{port}/"
    return f"http://{host}:{port}{normalized}/"


_DEFAULT_DATAROBOT_ENDPOINT = "https://app.datarobot.com/api/v2"


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


def build_deployment_a2a_url(
    endpoint: str,
    deployment_id: str,
    mount_path: str = DEFAULT_A2A_MOUNT_PATH,
) -> str:
    """Construct the A2A direct-access URL for a DataRobot deployment.

    The DataRobot gateway exposes deployment A2A endpoints under the
    ``directAccess{mount_path}`` sub-path.  When ``mount_path`` is the default
    ``"a2a"``, this produces the canonical ``directAccess/a2a/`` URL; custom
    mount paths (e.g. ``"/api/my-agent"``) are reflected in the gateway path so
    the advertised agent-card URL matches the actual endpoint.

    Parameters
    ----------
    endpoint:
        DataRobot API endpoint base URL, e.g. ``https://app.datarobot.com/api/v2``.
        A trailing slash is stripped before composing the URL.
    deployment_id:
        The DataRobot deployment ID.
    mount_path:
        The A2A mount path, as configured via ``general.front_end.a2a.mount_path``.
        Defaults to ``DEFAULT_A2A_MOUNT_PATH`` (``"a2a"``).

    Returns
    -------
    str
        A URL of the form
        ``{endpoint}/deployments/{deployment_id}/directAccess{mount_path}/``.
    """
    base = endpoint.rstrip("/")
    normalized = normalize_a2a_mount_path(mount_path)
    if normalized == "/":
        # Root mount: gateway segment is just "directAccess" (no extra slash before "/")
        gateway_segment = _DIRECT_ACCESS_GATEWAY_PREFIX
    else:
        gateway_segment = f"{_DIRECT_ACCESS_GATEWAY_PREFIX}{normalized}"
    return f"{base}/deployments/{deployment_id}/{gateway_segment}/"


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
