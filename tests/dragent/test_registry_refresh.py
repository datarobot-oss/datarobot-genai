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

from datarobot_genai.dragent.agent_card_registry import AgentCardRegistry
from datarobot_genai.dragent.agent_card_registry import AgentCardRegistryError
from datarobot_genai.dragent.agent_card_registry import ParsedRegistryCards
from datarobot_genai.dragent.registry_refresh import registry_refresh_lifespan
from datarobot_genai.dragent.registry_refresh import registry_refresh_loop

_MODULE = "datarobot_genai.dragent.registry_refresh"

_SAMPLE_AGENT_CARD = {
    "name": "Test Agent",
    "description": "A test agent",
    "url": "https://agent.example.com/a2a/",
    "version": "1.0.0",
    "skills": [],
    "defaultInputModes": ["text"],
    "defaultOutputModes": ["text"],
    "capabilities": {"streaming": False},
}


def _card(**overrides):
    from a2a.types import AgentCard

    return AgentCard.model_validate({**_SAMPLE_AGENT_CARD, **overrides})


def _parsed(cards: dict) -> ParsedRegistryCards:
    return ParsedRegistryCards(cards=cards, key_types={key: "dep" for key in cards})


class TestAgentCardRegistryRefresh:
    @pytest.fixture
    def mock_fetch(self):
        with patch.object(AgentCardRegistry, "_fetch", new_callable=AsyncMock) as m:
            yield m

    async def test_refresh_skips_fresh_entries(self, mock_fetch):
        mock_fetch.return_value = _parsed({"dep-1": _card()})
        registry = AgentCardRegistry(api_token="tok", endpoint="https://ep", cache_ttl=3600)
        registry.register(deployment_id="dep-1")
        await registry.get(deployment_id="dep-1")

        mock_fetch.reset_mock()
        await registry.refresh_all_registered()
        mock_fetch.assert_not_awaited()

    async def test_refresh_refetches_soft_expired_entries(self, mock_fetch):
        mock_fetch.return_value = _parsed({"dep-1": _card()})
        registry = AgentCardRegistry(api_token="tok", endpoint="https://ep", cache_ttl=60)
        registry.register(deployment_id="dep-1")
        await registry.get(deployment_id="dep-1")
        registry._age_cache_entry_for_test("dep-1", 120)

        mock_fetch.reset_mock()
        mock_fetch.return_value = _parsed({"dep-1": _card(name="Refreshed Agent")})
        await registry.refresh_all_registered()

        mock_fetch.assert_awaited_once_with({"deploymentIds": "dep-1"})

    async def test_refresh_logs_on_failure_without_raising(self, mock_fetch):
        mock_fetch.side_effect = [
            _parsed({"dep-1": _card()}),
            AgentCardRegistryError("registry down"),
        ]
        registry = AgentCardRegistry(api_token="tok", endpoint="https://ep", cache_ttl=60)
        registry.register(deployment_id="dep-1")
        await registry.get(deployment_id="dep-1")
        registry._age_cache_entry_for_test("dep-1", 120)

        await registry.refresh_all_registered()

    async def test_refresh_no_op_without_registered_ids(self, mock_fetch):
        registry = AgentCardRegistry(api_token="tok", endpoint="https://ep", cache_ttl=3600)
        await registry.refresh_all_registered()
        mock_fetch.assert_not_awaited()


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

        class _FakeTask:
            def __init__(self) -> None:
                self.cancel = MagicMock()

            def __await__(self):
                async def _noop() -> None:
                    return None

                return _noop().__await__()

        fake_task = _FakeTask()

        def _create_task(coro):
            coro.close()
            return fake_task

        with (
            patch(
                f"{_MODULE}.get_default_registry",
                AsyncMock(return_value=mock_registry),
            ),
            patch(f"{_MODULE}.asyncio.create_task", side_effect=_create_task) as mock_create_task,
        ):
            async with registry_refresh_lifespan(config):
                mock_create_task.assert_called_once()

            fake_task.cancel.assert_called_once()

    async def test_lifespan_no_op_without_registered_ids(self):
        mock_registry = MagicMock()
        mock_registry.has_registered_lookups.return_value = False
        config = MagicMock()

        with (
            patch(
                f"{_MODULE}.get_default_registry",
                AsyncMock(return_value=mock_registry),
            ),
            patch(f"{_MODULE}.asyncio.create_task") as mock_create_task,
        ):
            async with registry_refresh_lifespan(config):
                pass

            mock_create_task.assert_not_called()
