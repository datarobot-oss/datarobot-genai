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

A2A_DIRECT_ACCESS_PATH = "directAccess/a2a"

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
    return f"{base}/deployments/{deployment_id}/{A2A_DIRECT_ACCESS_PATH}/"


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
