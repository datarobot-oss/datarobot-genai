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

DEPLOYMENT_A2A_PATH = "directAccess/a2a"
WORKLOAD_A2A_PATH = "a2a"
DEPLOYMENT_MCP_PATH = "directAccess/mcp"
MCP_PATH = "mcp"

_DEFAULT_DATAROBOT_ENDPOINT = "https://app.datarobot.com/api/v2"
_API_V2_SUFFIX = "/api/v2"
AGENT_MEMORY_DATAROBOT_ENDPOINT_ENV = "AGENT_MEMORY_DATAROBOT_ENDPOINT"
CUSTOM_MODEL_WEB_SERVER_URL_ENV = "CUSTOM_MODEL_WEB_SERVER_URL"
PUBLIC_API_ENDPOINT_ENV = "DATAROBOT_PUBLIC_API_ENDPOINT"
DATAROBOT_ENDPOINT_ENV = "DATAROBOT_ENDPOINT"


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


def resolve_memory_api_endpoint(
    *,
    explicit_endpoint: str | None = None,
    require: bool = True,
) -> str | None:
    """Return the DataRobot API base URL for Memory Service routes.

    Deployed custom models commonly receive ``DATAROBOT_ENDPOINT`` pointing at
    in-cluster ``datarobot-nginx``, which does not route ``/memory/`` to the
    Memory Service. Prefer publicly reachable endpoints injected at deploy time.

    Checks sources in priority order:

    1. ``explicit_endpoint`` — workflow config or caller override.
    2. ``AGENT_MEMORY_DATAROBOT_ENDPOINT`` — deploy-time public API URL.
    3. ``DATAROBOT_PUBLIC_API_ENDPOINT`` — externally reachable platform URL.
    4. ``CUSTOM_MODEL_WEB_SERVER_URL`` — custom model web server base URL.
    5. ``DATAROBOT_ENDPOINT`` — in-cluster fallback.

    Parameters
    ----------
    explicit_endpoint:
        Optional caller-provided endpoint override.
    require:
        When *True*, raises :class:`RuntimeError` if no endpoint can be resolved.
        When *False*, returns *None* instead.

    Returns
    -------
    str | None
        Normalized API v2 base URL, or *None* when ``require=False`` and unset.

    Raises
    ------
    RuntimeError
        If ``require=True`` and no endpoint source is configured.
    """
    if explicit_endpoint:
        return normalize_api_v2_endpoint(explicit_endpoint)

    for env_var in (
        AGENT_MEMORY_DATAROBOT_ENDPOINT_ENV,
        PUBLIC_API_ENDPOINT_ENV,
        CUSTOM_MODEL_WEB_SERVER_URL_ENV,
        DATAROBOT_ENDPOINT_ENV,
    ):
        value = os.getenv(env_var, "").strip()
        if value:
            return normalize_api_v2_endpoint(value)

    if require:
        raise RuntimeError(
            "DataRobot endpoint is not set. Configure memory.datarobot_endpoint, "
            f"{AGENT_MEMORY_DATAROBOT_ENDPOINT_ENV}, {PUBLIC_API_ENDPOINT_ENV}, "
            f"{CUSTOM_MODEL_WEB_SERVER_URL_ENV}, or {DATAROBOT_ENDPOINT_ENV} when "
            "using agent memory."
        )
    return None


def build_deployment_a2a_url(endpoint: str, deployment_id: str) -> str:
    """Construct the A2A direct-access URL for a DataRobot deployment.

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
        A URL of the form ``{endpoint}/deployments/{deployment_id}/directAccess/a2a/``.
    """
    base = endpoint.rstrip("/")
    return f"{base}/deployments/{deployment_id}/{DEPLOYMENT_A2A_PATH}/"


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


def build_workload_a2a_url(endpoint: str, workload_id: str) -> str:
    """Construct the A2A URL for a DataRobot workload.

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
        A URL of the form ``{endpoint}/endpoints/workloads/{workload_id}/a2a/``.
    """
    base = endpoint.removesuffix("/")
    return f"{base}/endpoints/workloads/{workload_id}/{WORKLOAD_A2A_PATH}/"


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
