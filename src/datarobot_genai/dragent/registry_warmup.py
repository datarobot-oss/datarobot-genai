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
accepts traffic.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from typing import NamedTuple

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

    When there are no registry-backed function groups, this returns ``True``
    (nothing to warm).
    """
    return _WarmState.warm


def reset_registry_warm_state() -> None:
    """Reset warm state (for tests)."""
    _WarmState.warm = False


class RegistryLookupIds(NamedTuple):
    """Registry lookup IDs collected from a workflow config, by ID kind."""

    deployment_ids: list[str]
    external_ids: list[str]
    workload_ids: list[str]

    def is_empty(self) -> bool:
        """Return whether no registry lookups were configured at all."""
        return not (self.deployment_ids or self.external_ids or self.workload_ids)


def collect_registry_lookup_ids(config: Config) -> RegistryLookupIds:
    """Return deduplicated registry lookup IDs per kind from *config*.

    Only ``authenticated_a2a_client`` function groups with a ``registry`` block
    contribute IDs. Order is preserved (first-seen).
    """
    collected = RegistryLookupIds(deployment_ids=[], external_ids=[], workload_ids=[])
    seen: dict[str, set[str]] = {"deployment": set(), "external": set(), "workload": set()}

    function_groups = getattr(config, "function_groups", None) or {}
    for fg_config in function_groups.values():
        if not isinstance(fg_config, AuthenticatedA2AClientConfig):
            continue
        if fg_config.registry is None:
            continue
        for kind, lookup_id, ids in (
            ("deployment", fg_config.registry.deployment_id, collected.deployment_ids),
            ("external", fg_config.registry.external_id, collected.external_ids),
            ("workload", fg_config.registry.workload_id, collected.workload_ids),
        ):
            if lookup_id and lookup_id not in seen[kind]:
                seen[kind].add(lookup_id)
                ids.append(lookup_id)

    return collected


async def warmup_registry_from_config(config: Config) -> None:
    """Batch-prefetch agent cards for all registry-backed A2A clients in *config*.

    No-op when no registry lookups are configured. On failure, logs an error and
    leaves :func:`is_registry_warm` as ``False``.
    """
    collected = collect_registry_lookup_ids(config)
    if collected.is_empty():
        logger.debug("No registry-backed A2A function groups; skipping agent card prefetch.")
        _WarmState.warm = True
        return

    _WarmState.warm = False
    logger.info(
        "Prefetching agent cards from central registry "
        "(deployment_ids=%s, external_ids=%s, workload_ids=%s)",
        collected.deployment_ids,
        collected.external_ids,
        collected.workload_ids,
    )

    try:
        registry = await get_default_registry()
        await registry.prefetch(
            deployment_ids=collected.deployment_ids or None,
            external_ids=collected.external_ids or None,
            workload_ids=collected.workload_ids or None,
        )
    except Exception:
        logger.exception(
            "Agent card registry prefetch failed; registry-backed A2A tools may "
            "degrade until the central registry is reachable."
        )
        return

    _WarmState.warm = True
    logger.info(
        "Agent card registry prefetch complete "
        "(%d deployment ID(s), %d external ID(s), %d workload ID(s)).",
        len(collected.deployment_ids),
        len(collected.external_ids),
        len(collected.workload_ids),
    )
