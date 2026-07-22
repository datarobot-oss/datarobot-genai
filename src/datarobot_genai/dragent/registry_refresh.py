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

"""Background refresh loop for the central agent card registry cache."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from datarobot_genai.dragent.agent_card_registry import AgentCardRegistry
from datarobot_genai.dragent.agent_card_registry import AgentCardRegistryConfig
from datarobot_genai.dragent.agent_card_registry import get_default_registry

if TYPE_CHECKING:
    from nat.data_models.config import Config

logger = logging.getLogger(__name__)


def is_registry_refresh_enabled() -> bool:
    """Return whether the background registry refresh loop is enabled."""
    return AgentCardRegistryConfig().agent_card_registry_refresh_interval_seconds > 0


def get_registry_refresh_interval_seconds() -> int:
    """Return the configured background refresh interval in seconds."""
    return AgentCardRegistryConfig().agent_card_registry_refresh_interval_seconds


async def registry_refresh_loop(
    registry: AgentCardRegistry,
    interval_seconds: int,
) -> None:
    """Periodically refresh soft-expired registered agent cards."""
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            await registry.refresh_all_registered()
        except Exception:
            logger.exception("Background agent card registry refresh failed")


@asynccontextmanager
async def registry_refresh_lifespan(_config: Config) -> AsyncIterator[None]:
    """Start the background refresh task for the registry singleton.

    No-op when ``AGENT_CARD_REGISTRY_REFRESH_INTERVAL_SECONDS`` is ``0`` or when
    no registry-backed IDs were registered at config-parse time.
    """
    interval = get_registry_refresh_interval_seconds()
    if interval <= 0:
        logger.debug("Agent card registry background refresh is disabled.")
        yield
        return

    registry = await get_default_registry()
    if not registry.has_registered_lookups():
        logger.debug("No registered agent card IDs; skipping background refresh task.")
        yield
        return

    logger.info(
        "Starting agent card registry background refresh (interval=%ds)",
        interval,
    )
    task = asyncio.create_task(registry_refresh_loop(registry, interval))
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        logger.debug("Agent card registry background refresh task stopped.")
