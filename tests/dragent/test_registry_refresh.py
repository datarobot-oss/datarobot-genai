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

import asyncio
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from a2a.types import AgentCard

from datarobot_genai.dragent.agent_card_registry import AgentCardRegistry
from datarobot_genai.dragent.agent_card_registry import AgentCardRegistryConfig
from datarobot_genai.dragent.agent_card_registry import AgentCardRegistryError
from datarobot_genai.dragent.agent_card_registry_backends import MemoryAgentCardCacheBackend
from datarobot_genai.dragent.registry_refresh import get_registry_refresh_interval_seconds
from datarobot_genai.dragent.registry_refresh import is_registry_refresh_enabled
from datarobot_genai.dragent.registry_refresh import registry_refresh_lifespan
from datarobot_genai.dragent.registry_refresh import registry_refresh_loop

_MODULE = "datarobot_genai.dragent.registry_refresh"

_SAMPLE_AGENT_CARD = AgentCard.model_validate(
    {
        "name": "Refresh Agent",
        "description": "Test agent",
        "url": "https://agent.example.com/a2a/",
        "version": "1.0.0",
        "skills": [],
        "defaultInputModes": ["text"],
        "defaultOutputModes": ["text"],
        "capabilities": {"streaming": False},
    }
)


def _parsed(cards: dict):
    from datarobot_genai.dragent.agent_card_registry import ParsedRegistryCards

    return ParsedRegistryCards(cards=cards, key_types={key: "dep" for key in cards})


class TestAgentCardRegistryRefresh:
    @pytest.fixture
    def mock_fetch(self):
        with patch.object(AgentCardRegistry, "_fetch", new_callable=AsyncMock) as m:
            yield m

    async def test_refresh_skips_fresh_entries(self, mock_fetch):
        mock_fetch.return_value = _parsed({"dep-1": _SAMPLE_AGENT_CARD})
        registry = AgentCardRegistry(
            api_token="tok",
            endpoint="https://ep",
            cache_ttl=3600,
            cache_backend=MemoryAgentCardCacheBackend(),
        )
        registry.register(deployment_id="dep-1")
        await registry.get(deployment_id="dep-1")

        mock_fetch.reset_mock()
        await registry.refresh_all_registered()
        mock_fetch.assert_not_awaited()

    async def test_refresh_refetches_soft_expired_entries(self, mock_fetch):
        refreshed = _SAMPLE_AGENT_CARD.model_copy(update={"version": "1.0.1"})
        mock_fetch.return_value = _parsed({"dep-1": _SAMPLE_AGENT_CARD})
        registry = AgentCardRegistry(
            api_token="tok",
            endpoint="https://ep",
            cache_ttl=60,
            cache_backend=MemoryAgentCardCacheBackend(),
        )
        registry.register(deployment_id="dep-1")
        await registry.get(deployment_id="dep-1")
        registry._age_cache_entry_for_test("dep-1", 120)

        mock_fetch.reset_mock()
        mock_fetch.return_value = _parsed({"dep-1": refreshed})
        await registry.refresh_all_registered()

        mock_fetch.assert_awaited_once_with({"deploymentIds": "dep-1"})

    async def test_refresh_logs_on_failure_without_raising(self, mock_fetch):
        mock_fetch.side_effect = [
            _parsed({"dep-1": _SAMPLE_AGENT_CARD}),
            AgentCardRegistryError("registry down"),
        ]
        registry = AgentCardRegistry(
            api_token="tok",
            endpoint="https://ep",
            cache_ttl=60,
            cache_backend=MemoryAgentCardCacheBackend(),
        )
        registry.register(deployment_id="dep-1")
        await registry.get(deployment_id="dep-1")
        registry._age_cache_entry_for_test("dep-1", 120)

        await registry.refresh_all_registered()

    async def test_refresh_no_op_without_registered_ids(self, mock_fetch):
        registry = AgentCardRegistry(
            api_token="tok",
            endpoint="https://ep",
            cache_ttl=3600,
            cache_backend=MemoryAgentCardCacheBackend(),
        )
        await registry.refresh_all_registered()
        mock_fetch.assert_not_awaited()


class TestRegistryRefreshConfig:
    def test_refresh_enabled_by_default(self):
        config = AgentCardRegistryConfig()
        assert config.agent_card_registry_refresh_interval_seconds == 30 * 60
        assert is_registry_refresh_enabled() is True

    def test_refresh_disabled_when_zero(self):
        with patch.dict("os.environ", {"AGENT_CARD_REGISTRY_REFRESH_INTERVAL_SECONDS": "0"}):
            config = AgentCardRegistryConfig()
            assert config.agent_card_registry_refresh_interval_seconds == 0
            assert is_registry_refresh_enabled() is False

    def test_refresh_interval_from_env(self):
        with patch.dict("os.environ", {"AGENT_CARD_REGISTRY_REFRESH_INTERVAL_SECONDS": "900"}):
            assert get_registry_refresh_interval_seconds() == 900


class TestRegistryRefreshLoop:
    async def test_loop_calls_refresh_after_interval(self):
        registry = AsyncMock()
        with patch(f"{_MODULE}.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            mock_sleep.side_effect = [None, asyncio.CancelledError()]

            with pytest.raises(asyncio.CancelledError):
                await registry_refresh_loop(registry, interval_seconds=60)

        registry.refresh_all_registered.assert_awaited_once()

    async def test_loop_continues_after_refresh_error(self):
        registry = AsyncMock()
        registry.refresh_all_registered.side_effect = [RuntimeError("boom"), None]
        with patch(f"{_MODULE}.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            mock_sleep.side_effect = [None, None, asyncio.CancelledError()]

            with pytest.raises(asyncio.CancelledError):
                await registry_refresh_loop(registry, interval_seconds=60)

        assert registry.refresh_all_registered.await_count == 2


class TestRegistryRefreshLifespan:
    async def test_lifespan_starts_and_stops_task(self):
        mock_registry = MagicMock()
        mock_registry.has_registered_lookups.return_value = True
        config = MagicMock()

        with (
            patch(f"{_MODULE}.get_registry_refresh_interval_seconds", return_value=60),
            patch(
                f"{_MODULE}.get_default_registry",
                AsyncMock(return_value=mock_registry),
            ),
            patch(f"{_MODULE}.asyncio.create_task") as mock_create_task,
        ):
            mock_task = AsyncMock()
            mock_task.cancel = MagicMock()
            mock_create_task.return_value = mock_task

            async with registry_refresh_lifespan(config):
                mock_create_task.assert_called_once()

            mock_task.cancel.assert_called_once()

    async def test_lifespan_no_op_when_disabled(self):
        config = MagicMock()
        with (
            patch(f"{_MODULE}.get_registry_refresh_interval_seconds", return_value=0),
            patch(f"{_MODULE}.asyncio.create_task") as mock_create_task,
        ):
            async with registry_refresh_lifespan(config):
                pass

            mock_create_task.assert_not_called()

    async def test_lifespan_no_op_without_registered_ids(self):
        mock_registry = MagicMock()
        mock_registry.has_registered_lookups.return_value = False
        config = MagicMock()

        with (
            patch(f"{_MODULE}.get_registry_refresh_interval_seconds", return_value=60),
            patch(
                f"{_MODULE}.get_default_registry",
                AsyncMock(return_value=mock_registry),
            ),
            patch(f"{_MODULE}.asyncio.create_task") as mock_create_task,
        ):
            async with registry_refresh_lifespan(config):
                pass

            mock_create_task.assert_not_called()
