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

The :class:`AgentCardRegistry` supports **batch fetching** so that many
function groups sharing the same workflow can resolve all their cards in a
minimum number of HTTP round-trips instead of N+1 individual requests.

.. note::
    The registry API uses AND semantics when both ``deploymentIds`` and
    ``externalIds`` parameters are provided in a single request, which would
    return empty results.  Therefore :meth:`AgentCardRegistry.prefetch` issues
    **separate** HTTP calls for deployment IDs and external IDs.
"""

from __future__ import annotations

import asyncio
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


def _resolve_settings(
    api_token: str | None = None,
    endpoint: str | None = None,
) -> tuple[str, str]:
    """Return validated ``(api_token, endpoint)`` from explicit values or settings."""
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
    return resolved_token, resolved_endpoint


def _parse_registry_response(body: dict[str, Any]) -> dict[str, AgentCard]:
    """Parse a paginated registry response into ``{id: AgentCard}``.

    Each record is indexed by *both* ``deploymentId`` and ``externalId``
    (when present) so callers can look up by either key type.
    """
    cards: dict[str, AgentCard] = {}
    for entry in body.get("data", []):
        raw_card = entry.get("agentCard")
        if not raw_card:
            logger.warning(
                "Registry entry %s has no 'agentCard' payload — skipping.",
                entry.get("id", "?"),
            )
            continue
        try:
            card = AgentCard.model_validate(raw_card)
        except Exception:
            logger.warning(
                "Failed to parse agent card for registry entry %s — skipping.",
                entry.get("id", "?"),
                exc_info=True,
            )
            continue

        if dep_id := entry.get("deploymentId"):
            cards[dep_id] = card
        if ext_id := entry.get("externalId"):
            cards[ext_id] = card
    return cards


class AgentCardRegistry:
    """Async-safe, batch-capable cache for agent cards from the central registry.

    Designed to solve the **N+1 problem**: when a workflow defines many
    function groups that all need registry lookup, callers can
    :meth:`prefetch` all IDs in a minimum number of HTTP calls, then
    :meth:`get` each card from the local cache.

    .. important::
        The registry API uses AND semantics when both ``deploymentIds`` and
        ``externalIds`` are sent in a single request.  This class always
        issues **separate** requests for each ID type.

    Usage::

        registry = AgentCardRegistry()

        # Batch-prefetch — issues at most 2 HTTP calls (one per ID type)
        await registry.prefetch(
            deployment_ids=["dep-1", "dep-2"],
            external_ids=["ext-3"],
        )

        # Individual lookups hit the local cache (no HTTP)
        card = await registry.get(deployment_id="dep-1")

    The :meth:`get` method also works standalone — if the card is not cached
    it fetches it individually.
    """

    def __init__(
        self,
        api_token: str | None = None,
        endpoint: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._api_token = api_token
        self._endpoint = endpoint
        self._timeout = timeout
        self._cache: dict[str, AgentCard] = {}
        self._lock = asyncio.Lock()

    async def _fetch(self, params: dict[str, str]) -> dict[str, AgentCard]:
        """Execute a single registry HTTP GET and return parsed cards."""
        token, endpoint = _resolve_settings(self._api_token, self._endpoint)
        registry_url = build_agent_cards_registry_url(endpoint)
        headers = {"Authorization": f"Bearer {token}"}

        logger.info(
            "Fetching agent cards from registry: %s (params=%s)", registry_url, params,
        )

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(self._timeout)) as client:
                response = await client.get(registry_url, params=params, headers=headers)
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise AgentCardRegistryError(
                f"Agent card registry request failed with HTTP "
                f"{exc.response.status_code}. Verify your API token and that "
                f"the agents are registered."
            ) from exc
        except httpx.HTTPError as exc:
            raise AgentCardRegistryError(
                f"Agent card registry request failed: {exc}"
            ) from exc

        cards = _parse_registry_response(response.json())
        logger.info("Fetched %d agent card(s) from registry.", len(cards))
        return cards

    async def prefetch(
        self,
        *,
        deployment_ids: list[str] | None = None,
        external_ids: list[str] | None = None,
    ) -> None:
        """Batch-fetch and cache agent cards in a minimum number of HTTP calls.

        Issues **separate** requests for ``deployment_ids`` and
        ``external_ids`` because the API uses AND semantics when both
        parameters are present in a single request.

        Already-cached IDs are skipped.  This method is concurrency-safe.

        Parameters
        ----------
        deployment_ids:
            Deployment IDs to prefetch.
        external_ids:
            External IDs to prefetch.
        """
        async with self._lock:
            missing_dep = [d for d in (deployment_ids or []) if d not in self._cache]
            missing_ext = [e for e in (external_ids or []) if e not in self._cache]

            if not missing_dep and not missing_ext:
                logger.debug("All requested agent cards already cached — skipping prefetch.")
                return

            if missing_dep:
                cards = await self._fetch({"deploymentIds": ",".join(missing_dep)})
                self._cache.update(cards)

            if missing_ext:
                cards = await self._fetch({"externalIds": ",".join(missing_ext)})
                self._cache.update(cards)

    async def get(
        self,
        *,
        deployment_id: str | None = None,
        external_id: str | None = None,
    ) -> AgentCard:
        """Return a single agent card, using the cache or fetching on demand.

        Exactly one of ``deployment_id`` or ``external_id`` must be provided.

        Raises
        ------
        AgentCardRegistryError
            If the card cannot be found or the request fails.
        """
        if bool(deployment_id) == bool(external_id):
            raise AgentCardRegistryError(
                "Specify exactly one of 'deployment_id' or 'external_id'."
            )

        lookup_key: str = deployment_id or external_id  # type: ignore[assignment]

        # Fast path — already cached
        if lookup_key in self._cache:
            return self._cache[lookup_key]

        # Slow path — fetch individually (double-check under lock)
        async with self._lock:
            if lookup_key in self._cache:
                return self._cache[lookup_key]

            params: dict[str, str] = (
                {"deploymentIds": deployment_id}
                if deployment_id
                else {"externalIds": external_id}  # type: ignore[dict-item]
            )
            cards = await self._fetch(params)
            self._cache.update(cards)

        if lookup_key not in self._cache:
            id_label = (
                f"deployment_id='{deployment_id}'"
                if deployment_id
                else f"external_id='{external_id}'"
            )
            raise AgentCardRegistryError(
                f"No agent card found in the central registry for {id_label}. "
                "Verify that the deployment exists and is registered in your organisation."
            )

        return self._cache[lookup_key]


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_default_registry: AgentCardRegistry | None = None
_registry_lock = asyncio.Lock()


async def get_default_registry() -> AgentCardRegistry:
    """Return the module-level :class:`AgentCardRegistry` singleton.

    Created lazily on first access.  Credentials are resolved from
    :class:`DataRobotRegistrySettings` at instantiation time.
    """
    global _default_registry
    if _default_registry is None:
        async with _registry_lock:
            if _default_registry is None:
                _default_registry = AgentCardRegistry()
    return _default_registry


def reset_default_registry() -> None:
    """Reset the module-level singleton (for testing)."""
    global _default_registry
    _default_registry = None
