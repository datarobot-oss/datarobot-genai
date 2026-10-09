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
import logging
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from nat.data_models.config import Config

from datarobot_genai.dragent.agent_card_registry import AgentCardRegistry
from datarobot_genai.dragent.agent_card_registry import AgentCardRegistryError
from datarobot_genai.dragent.agent_card_registry import ParsedRegistryCards
from datarobot_genai.dragent.agent_card_registry import reset_default_registry
from datarobot_genai.dragent.plugins.auth_a2a_client import AgentCardRegistryLookup
from datarobot_genai.dragent.plugins.auth_a2a_client import AuthenticatedA2AClientConfig
from datarobot_genai.dragent.registry_refresh import background_refresh_interval
from datarobot_genai.dragent.registry_refresh import refresh_sleep_seconds
from datarobot_genai.dragent.registry_refresh import registry_refresh_lifespan
from datarobot_genai.dragent.registry_refresh import registry_refresh_loop
from tests.dragent.test_agent_card_registry import _memory_registry

_MODULE = "datarobot_genai.dragent.registry_refresh"
_REGISTRY_SETTINGS_PATCH = "datarobot_genai.dragent.agent_card_registry._resolve_settings"


@pytest.fixture(autouse=True)
def _registry_credentials():
    reset_default_registry()
    with patch(_REGISTRY_SETTINGS_PATCH, return_value=("tok", "https://ep")):
        yield
    reset_default_registry()


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
    return ParsedRegistryCards(
        cards=cards,
        key_types={key: "deployment" for key in cards},
        registry_ids={},
    )


class TestAgentCardRegistryRefresh:
    @pytest.fixture
    def mock_fetch(self):
        with patch.object(AgentCardRegistry, "_fetch", new_callable=AsyncMock) as m:
            yield m

    async def test_refresh_skips_fresh_entries(self, mock_fetch):
        mock_fetch.return_value = _parsed({"dep-1": _card()})
        registry = _memory_registry(cache_ttl=3600)
        registry.register(deployment_id="dep-1")
        await registry.get(deployment_id="dep-1")

        mock_fetch.reset_mock()
        await registry.refresh_all_registered()
        mock_fetch.assert_not_awaited()

    async def test_refresh_refetches_soft_expired_entries(self, mock_fetch):
        mock_fetch.return_value = _parsed({"dep-1": _card()})
        registry = _memory_registry(cache_ttl=60)
        registry.register(deployment_id="dep-1")
        await registry.get(deployment_id="dep-1")
        registry._age_cache_entry_for_test("dep-1", 120)

        mock_fetch.reset_mock()
        mock_fetch.return_value = _parsed({"dep-1": _card(name="Refreshed Agent")})
        await registry.refresh_all_registered()

        mock_fetch.assert_awaited_once_with({"deploymentIds": "dep-1"})

    async def test_refresh_refetches_past_soft_ttl_within_hard_ttl(self, mock_fetch):
        mock_fetch.return_value = _parsed({"dep-1": _card()})
        registry = _memory_registry(
            cache_ttl=3600,
            soft_cache_ttl=60,
        )
        registry.register(deployment_id="dep-1")
        await registry.get(deployment_id="dep-1")
        registry._age_cache_entry_for_test("dep-1", 90)

        mock_fetch.reset_mock()
        mock_fetch.return_value = _parsed({"dep-1": _card(name="Refreshed Agent")})
        await registry.refresh_all_registered()

        mock_fetch.assert_awaited_once_with({"deploymentIds": "dep-1"})

    async def test_refresh_skips_within_soft_ttl(self, mock_fetch):
        mock_fetch.return_value = _parsed({"dep-1": _card()})
        registry = _memory_registry(
            cache_ttl=3600,
            soft_cache_ttl=60,
        )
        registry.register(deployment_id="dep-1")
        await registry.get(deployment_id="dep-1")
        registry._age_cache_entry_for_test("dep-1", 30)

        mock_fetch.reset_mock()
        await registry.refresh_all_registered()
        mock_fetch.assert_not_awaited()

    async def test_refresh_reraises_on_failure_without_logging(self, mock_fetch, caplog):
        """GIVEN a soft-expired card WHEN refresh fails THEN cache is kept, error raised, no log."""
        mock_fetch.side_effect = [
            _parsed({"dep-1": _card()}),
            AgentCardRegistryError("registry down"),
        ]
        registry = _memory_registry(cache_ttl=3600, soft_cache_ttl=60)
        registry.register(deployment_id="dep-1")
        await registry.get(deployment_id="dep-1")
        registry._age_cache_entry_for_test("dep-1", 90)

        with (
            caplog.at_level(logging.WARNING, logger="datarobot_genai.dragent.agent_card_registry"),
            pytest.raises(AgentCardRegistryError, match="registry down"),
        ):
            await registry.refresh_all_registered()

        assert await registry._backend.get_stale("dep-1", max_staleness_seconds=3600) is not None
        assert not any(
            "Background agent card registry refresh failed" in r.getMessage()
            for r in caplog.records
        )

    async def test_refresh_no_op_without_registered_ids(self, mock_fetch):
        registry = _memory_registry(cache_ttl=3600)
        await registry.refresh_all_registered()
        mock_fetch.assert_not_awaited()


class TestBackgroundRefreshInterval:
    def test_half_soft_ttl(self):
        assert background_refresh_interval(300) == 150

    def test_minimum_floor(self):
        assert background_refresh_interval(90) == 60

    def test_large_soft_ttl(self):
        assert background_refresh_interval(86400) == 43200


class TestRefreshSleepSeconds:
    def test_healthy_poll_applies_jitter_around_base(self):
        """GIVEN failures=0 WHEN computing sleep THEN delay is base × [0.5, 1.5]."""
        import random

        rng = random.Random(0)
        delays = [
            refresh_sleep_seconds(100, soft_cache_ttl=300, failures=0, rng=rng) for _ in range(50)
        ]
        assert all(50.0 <= d <= 150.0 for d in delays)
        assert min(delays) < 100.0 < max(delays)

    def test_failure_backoff_doubles_until_cap(self):
        """GIVEN consecutive failures WHEN computing sleep THEN delay grows then caps."""
        import random

        class _NoJitter:
            def uniform(self, a: float, b: float) -> float:
                return 1.0

        no_jitter = _NoJitter()
        assert refresh_sleep_seconds(60, soft_cache_ttl=120, failures=1, rng=no_jitter) == 120.0
        assert refresh_sleep_seconds(60, soft_cache_ttl=120, failures=2, rng=no_jitter) == 120.0
        # Cap is min(900, max(60, 120)) = 120 for short soft TTL above.
        # With a larger soft TTL the cap allows growth toward 15 minutes.
        assert refresh_sleep_seconds(60, soft_cache_ttl=3600, failures=1, rng=no_jitter) == 120.0
        assert refresh_sleep_seconds(60, soft_cache_ttl=3600, failures=2, rng=no_jitter) == 240.0
        assert refresh_sleep_seconds(60, soft_cache_ttl=3600, failures=8, rng=no_jitter) == 900.0
        # Sanity: jitter still applied when using a real RNG.
        jittered = refresh_sleep_seconds(60, soft_cache_ttl=3600, failures=1, rng=random.Random(1))
        assert 60.0 <= jittered <= 180.0


class TestRegistryRefreshLoop:
    async def test_loop_calls_refresh_after_interval(self):
        registry = AsyncMock()
        registry.soft_cache_ttl = 120
        with (
            patch(f"{_MODULE}.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
            patch(f"{_MODULE}.refresh_sleep_seconds", return_value=60.0) as mock_delay,
        ):
            mock_sleep.side_effect = [None, asyncio.CancelledError()]

            with pytest.raises(asyncio.CancelledError):
                await registry_refresh_loop(registry, interval_seconds=60)

        registry.refresh_all_registered.assert_awaited_once()
        mock_delay.assert_called_with(60, 120, 0)

    async def test_loop_continues_after_refresh_error(self):
        registry = AsyncMock()
        registry.soft_cache_ttl = 120
        registry.refresh_all_registered.side_effect = [RuntimeError("boom"), None]
        with (
            patch(f"{_MODULE}.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
            patch(
                f"{_MODULE}.refresh_sleep_seconds",
                side_effect=[60.0, 120.0, 60.0],
            ) as mock_delay,
        ):
            mock_sleep.side_effect = [None, None, asyncio.CancelledError()]

            with pytest.raises(asyncio.CancelledError):
                await registry_refresh_loop(registry, interval_seconds=60)

        assert registry.refresh_all_registered.await_count == 2
        assert mock_delay.call_args_list[0].args == (60, 120, 0)
        assert mock_delay.call_args_list[1].args == (60, 120, 1)
        assert mock_delay.call_args_list[2].args == (60, 120, 0)

    async def test_loop_logs_single_warning_on_refresh_failure(self, caplog):
        """GIVEN a refresh failure WHEN the loop catches it THEN one warning is logged."""
        registry = AsyncMock()
        registry.soft_cache_ttl = 120
        registry.refresh_all_registered.side_effect = AgentCardRegistryError("down")
        with (
            patch(f"{_MODULE}.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
            patch(f"{_MODULE}.refresh_sleep_seconds", return_value=60.0),
            caplog.at_level(logging.WARNING, logger=_MODULE),
        ):
            mock_sleep.side_effect = [None, asyncio.CancelledError()]

            with pytest.raises(asyncio.CancelledError):
                await registry_refresh_loop(registry, interval_seconds=60)

        failure_records = [
            r
            for r in caplog.records
            if "Background agent card registry refresh failed" in r.getMessage()
        ]
        assert len(failure_records) == 1
        assert failure_records[0].levelno == logging.WARNING
        assert failure_records[0].exc_info is not None
        assert "keeping cached entries" in failure_records[0].getMessage()
        assert "consecutive_failures=1" in failure_records[0].getMessage()

    async def test_loop_backs_off_on_registry_error_then_resets(self):
        """GIVEN registry errors THEN successes WHEN looping THEN failures reset after success."""
        registry = AsyncMock()
        registry.soft_cache_ttl = 300
        registry.refresh_all_registered.side_effect = [
            AgentCardRegistryError("down"),
            AgentCardRegistryError("still down"),
            None,
        ]
        with (
            patch(f"{_MODULE}.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
            patch(
                f"{_MODULE}.refresh_sleep_seconds",
                side_effect=[60.0, 120.0, 240.0, 60.0],
            ) as mock_delay,
        ):
            mock_sleep.side_effect = [None, None, None, asyncio.CancelledError()]

            with pytest.raises(asyncio.CancelledError):
                await registry_refresh_loop(registry, interval_seconds=60)

        assert [c.args[2] for c in mock_delay.call_args_list] == [0, 1, 2, 0]


class TestRegistryRefreshLifespan:
    async def test_lifespan_starts_and_stops_task(self):
        mock_registry = MagicMock()
        mock_registry.soft_cache_ttl = 1800
        mock_registry.has_registered_lookups.return_value = True
        config = Config(
            function_groups={
                "remote_agent": AuthenticatedA2AClientConfig(
                    registry=AgentCardRegistryLookup(deployment_id="dep-1"),
                    auth_provider="datarobot_auth",
                )
            }
        )

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

    async def test_lifespan_registers_ids_from_config_after_singleton_reset(self):
        """Background refresh must not depend on config-parse-time register() surviving L2 reset."""
        mock_registry = MagicMock()
        mock_registry.soft_cache_ttl = 1800
        mock_registry.has_registered_lookups.return_value = False
        config = Config(
            function_groups={
                "remote_agent": AuthenticatedA2AClientConfig(
                    registry=AgentCardRegistryLookup(workload_id="wl-1"),
                    auth_provider="datarobot_auth",
                )
            }
        )

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
                mock_registry.register.assert_called_once_with(workload_id="wl-1")
                mock_create_task.assert_called_once()

            fake_task.cancel.assert_called_once()

    async def test_lifespan_no_op_without_registry_backed_clients(self):
        config = Config(function_groups={})

        with (
            patch(f"{_MODULE}.get_default_registry", AsyncMock()) as mock_get_registry,
            patch(f"{_MODULE}.asyncio.create_task") as mock_create_task,
        ):
            async with registry_refresh_lifespan(config):
                pass

            mock_get_registry.assert_not_awaited()
            mock_create_task.assert_not_called()

    async def test_lifespan_uses_half_soft_cache_ttl_as_refresh_interval(self):
        mock_registry = MagicMock()
        mock_registry.soft_cache_ttl = 300
        config = Config(
            function_groups={
                "remote_agent": AuthenticatedA2AClientConfig(
                    registry=AgentCardRegistryLookup(deployment_id="dep-1"),
                    auth_provider="datarobot_auth",
                )
            }
        )

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
            patch(f"{_MODULE}.registry_refresh_loop") as mock_refresh_loop,
        ):
            async with registry_refresh_lifespan(config):
                mock_create_task.assert_called_once()
                mock_refresh_loop.assert_called_once_with(mock_registry, 150)

    async def test_lifespan_skips_refresh_when_caching_disabled(self):
        mock_registry = MagicMock()
        mock_registry.soft_cache_ttl = 0
        config = Config(
            function_groups={
                "remote_agent": AuthenticatedA2AClientConfig(
                    registry=AgentCardRegistryLookup(deployment_id="dep-1"),
                    auth_provider="datarobot_auth",
                )
            }
        )

        with (
            patch(
                f"{_MODULE}.get_default_registry",
                AsyncMock(return_value=mock_registry),
            ),
            patch(f"{_MODULE}.asyncio.create_task") as mock_create_task,
        ):
            async with registry_refresh_lifespan(config):
                pass

            mock_registry.register.assert_called_once_with(deployment_id="dep-1")
            mock_create_task.assert_not_called()
