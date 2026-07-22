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

"""Startup prefetch for central agent card registry lookups.

Collects ``registry`` blocks from ``authenticated_a2a_client`` function groups
in the loaded NAT config and batch-fetches all agent cards before the server
accepts traffic (when enabled via ``AGENT_CARD_REGISTRY_PREFETCH_ON_STARTUP``).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from datarobot_genai.dragent.agent_card_registry import AgentCardRegistryConfig
from datarobot_genai.dragent.agent_card_registry import get_default_registry
from datarobot_genai.dragent.plugins.auth_a2a_client import AuthenticatedA2AClientConfig

if TYPE_CHECKING:
    from nat.data_models.config import Config

logger = logging.getLogger(__name__)


class _WarmState:
    """Mutable container for registry prefetch warm state."""

    warm: bool = False


def is_registry_warm() -> bool:
    """Return whether the latest startup prefetch completed successfully.

    When prefetch is disabled or there are no registry-backed function groups,
    this returns ``True`` (nothing to warm).
    """
    return _WarmState.warm


def reset_registry_warm_state() -> None:
    """Reset warm state (for tests)."""
    _WarmState.warm = False


def is_prefetch_on_startup_enabled() -> bool:
    """Return whether startup registry prefetch is enabled."""
    return AgentCardRegistryConfig().agent_card_registry_prefetch_on_startup


def collect_registry_lookup_ids(
    config: Config,
) -> tuple[list[str], list[str]]:
    """Return deduplicated ``(deployment_ids, external_ids)`` from *config*.

    Only ``authenticated_a2a_client`` function groups with a ``registry`` block
    contribute IDs. Order is preserved (first-seen).
    """
    deployment_ids: list[str] = []
    external_ids: list[str] = []
    seen_dep: set[str] = set()
    seen_ext: set[str] = set()

    function_groups = getattr(config, "function_groups", None) or {}
    for fg_config in function_groups.values():
        if not isinstance(fg_config, AuthenticatedA2AClientConfig):
            continue
        if fg_config.registry is None:
            continue
        if dep_id := fg_config.registry.deployment_id:
            if dep_id not in seen_dep:
                seen_dep.add(dep_id)
                deployment_ids.append(dep_id)
        if ext_id := fg_config.registry.external_id:
            if ext_id not in seen_ext:
                seen_ext.add(ext_id)
                external_ids.append(ext_id)

    return deployment_ids, external_ids


async def warmup_registry_from_config(config: Config) -> None:
    """Batch-prefetch agent cards for all registry-backed A2A clients in *config*.

    No-op when prefetch is disabled or when no registry lookups are configured.
    On failure, logs an error and leaves :func:`is_registry_warm` as ``False``.
    """
    if not is_prefetch_on_startup_enabled():
        logger.debug("Agent card registry prefetch on startup is disabled.")
        _WarmState.warm = True
        return

    deployment_ids, external_ids = collect_registry_lookup_ids(config)
    if not deployment_ids and not external_ids:
        logger.debug("No registry-backed A2A function groups; skipping agent card prefetch.")
        _WarmState.warm = True
        return

    _WarmState.warm = False
    logger.info(
        "Prefetching agent cards from central registry (deployment_ids=%s, external_ids=%s)",
        deployment_ids,
        external_ids,
    )

    try:
        registry = await get_default_registry()
        await registry.prefetch(
            deployment_ids=deployment_ids or None,
            external_ids=external_ids or None,
        )
    except Exception:
        logger.exception(
            "Agent card registry prefetch failed; registry-backed A2A tools may "
            "degrade until the central registry is reachable."
        )
        return

    _WarmState.warm = True
    logger.info(
        "Agent card registry prefetch complete (%d deployment ID(s), %d external ID(s)).",
        len(deployment_ids),
        len(external_ids),
    )
