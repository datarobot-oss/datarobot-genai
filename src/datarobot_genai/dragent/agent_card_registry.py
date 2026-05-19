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

"""Client for the central DataRobot agent card registry.

The central registry provides a tenant-scoped list of agent cards that requires
only standard DataRobot API-token authentication (``DATAROBOT_API_TOKEN``).
This avoids the chicken-and-egg problem where an individual agent's card
endpoint is behind per-agent AuthN/AuthZ.
"""

import logging
from typing import Any

import httpx
from a2a.types import AgentCard
from datarobot.core.config import DataRobotAppFrameworkBaseSettings

from datarobot_genai.dragent.deployment_urls import build_agent_cards_registry_url

logger = logging.getLogger(__name__)


class DataRobotRegistrySettings(DataRobotAppFrameworkBaseSettings):
    """DataRobot connection settings for the central agent card registry.

    Loads ``DATAROBOT_API_TOKEN`` and ``DATAROBOT_ENDPOINT`` from env vars
    (including Runtime Parameters), ``.env``, file secrets, or Pulumi config
    using the standard :class:`DataRobotAppFrameworkBaseSettings` priority
    chain.
    """

    datarobot_api_token: str | None = None
    datarobot_endpoint: str | None = None


class AgentCardRegistryError(RuntimeError):
    """Raised when the central agent card registry lookup fails."""


async def fetch_agent_card_from_registry(
    *,
    deployment_id: str | None = None,
    external_id: str | None = None,
    api_token: str | None = None,
    endpoint: str | None = None,
    timeout: float = 30.0,
) -> AgentCard:
    """Fetch an :class:`~a2a.types.AgentCard` from the central DataRobot registry.

    Exactly one of ``deployment_id`` or ``external_id`` must be provided.

    Parameters
    ----------
    deployment_id:
        The DataRobot deployment ID to look up.
    external_id:
        The external agent identifier to look up.
    api_token:
        DataRobot API token.  Falls back to ``DATAROBOT_API_TOKEN`` resolved
        via :class:`DataRobotRegistrySettings` when *None*.
    endpoint:
        DataRobot API endpoint.  Falls back to ``DATAROBOT_ENDPOINT`` resolved
        via :class:`DataRobotRegistrySettings` when *None*.
    timeout:
        HTTP request timeout in seconds.

    Returns
    -------
    AgentCard
        The deserialized agent card from the registry.

    Raises
    ------
    AgentCardRegistryError
        If the lookup fails (missing credentials, HTTP error, or no matching
        card in the registry).
    """
    if not deployment_id and not external_id:
        raise AgentCardRegistryError(
            "Either 'deployment_id' or 'external_id' must be provided for registry lookup."
        )
    if deployment_id and external_id:
        raise AgentCardRegistryError(
            "Specify exactly one of 'deployment_id' or 'external_id', not both."
        )

    # Resolve credentials via DataRobotAppFrameworkBaseSettings
    settings = DataRobotRegistrySettings()
    resolved_token = api_token or settings.datarobot_api_token
    if not resolved_token:
        raise AgentCardRegistryError(
            "DataRobot API token is required for agent card registry lookup. "
            "Set the DATAROBOT_API_TOKEN environment variable or provide it explicitly."
        )

    resolved_endpoint = endpoint or settings.datarobot_endpoint
    if not resolved_endpoint:
        raise AgentCardRegistryError(
            "DataRobot API endpoint is required for agent card registry lookup. "
            "Set the DATAROBOT_ENDPOINT environment variable or provide it explicitly."
        )

    registry_url = build_agent_cards_registry_url(resolved_endpoint)

    # Build query parameters
    params: dict[str, str] = {}
    if deployment_id:
        params["deploymentIds"] = deployment_id
    else:
        params["externalIds"] = external_id  # type: ignore[assignment]

    headers = {"Authorization": f"Bearer {resolved_token}"}

    lookup_key = f"deployment_id={deployment_id}" if deployment_id else f"external_id={external_id}"
    logger.info("Fetching agent card from central registry (%s): %s", lookup_key, registry_url)

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
            response = await client.get(registry_url, params=params, headers=headers)
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise AgentCardRegistryError(
            f"Agent card registry request failed with HTTP {exc.response.status_code} "
            f"for {lookup_key}. Verify your API token and that the agent is registered."
        ) from exc
    except httpx.HTTPError as exc:
        raise AgentCardRegistryError(
            f"Agent card registry request failed for {lookup_key}: {exc}"
        ) from exc

    body: dict[str, Any] = response.json()
    data: list[dict[str, Any]] = body.get("data", [])

    if not data:
        raise AgentCardRegistryError(
            f"No agent card found in the central registry for {lookup_key}. "
            "Verify that the deployment exists and is registered in your organisation."
        )

    agent_card_json = data[0].get("agentCard")
    if not agent_card_json:
        raise AgentCardRegistryError(
            f"Registry entry for {lookup_key} exists but contains no 'agentCard' payload."
        )

    try:
        card = AgentCard.model_validate(agent_card_json)
    except Exception as exc:
        raise AgentCardRegistryError(
            f"Failed to parse agent card from registry for {lookup_key}: {exc}"
        ) from exc

    logger.info(
        "Successfully fetched agent card from registry (%s): name=%s, url=%s",
        lookup_key,
        card.name,
        card.url,
    )
    return card




