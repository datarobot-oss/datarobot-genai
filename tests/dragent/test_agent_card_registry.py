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

from datetime import UTC
from datetime import datetime
from datetime import timedelta
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import httpx
import pytest
from a2a.types import AgentCard

from datarobot_genai.dragent.agent_card_registry import _MAX_PAGES
from datarobot_genai.dragent.agent_card_registry import AgentCardRegistry
from datarobot_genai.dragent.agent_card_registry import AgentCardRegistryConfig
from datarobot_genai.dragent.agent_card_registry import AgentCardRegistryError
from datarobot_genai.dragent.agent_card_registry import DataRobotRegistrySettings
from datarobot_genai.dragent.agent_card_registry import ParsedRegistryCards
from datarobot_genai.dragent.agent_card_registry import _parse_registry_response
from datarobot_genai.dragent.agent_card_registry import _resolve_settings
from datarobot_genai.dragent.agent_card_registry import get_default_registry
from datarobot_genai.dragent.agent_card_registry import get_default_registry_sync
from datarobot_genai.dragent.agent_card_registry import reset_default_registry
from datarobot_genai.dragent.agent_card_registry_backends import AgentCardCacheRecord
from datarobot_genai.dragent.agent_card_registry_backends import MemoryAgentCardCacheBackend
from datarobot_genai.dragent.agent_card_registry_backends import RegistryIds

_MODULE = "datarobot_genai.dragent.agent_card_registry"

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

_SAMPLE_AGENT_CARD_2 = {
    "name": "Second Agent",
    "description": "Another agent",
    "url": "https://agent2.example.com/a2a/",
    "version": "2.0.0",
    "skills": [],
    "defaultInputModes": ["text"],
    "defaultOutputModes": ["text"],
    "capabilities": {"streaming": True},
}


def _registry_response(*entries):
    """Build a registry response envelope from entry dicts."""
    return {"data": list(entries), "count": len(entries), "totalCount": len(entries)}


def _entry(dep_id=None, ext_id=None, wl_id=None, card=_SAMPLE_AGENT_CARD):
    return {
        "id": f"doc-{dep_id or wl_id or ext_id}",
        "deploymentId": dep_id,
        "externalId": ext_id,
        "workloadId": wl_id,
        "agentCard": card,
    }


def _card(**overrides) -> AgentCard:
    return AgentCard.model_validate({**_SAMPLE_AGENT_CARD, **overrides})


def _parsed(
    cards: dict,
    *,
    key_types: dict | None = None,
    registry_ids: dict | None = None,
) -> ParsedRegistryCards:
    if key_types is None:
        key_types = {key: "deployment" for key in cards}
    if registry_ids is None:
        registry_ids = {}
    return ParsedRegistryCards(cards=cards, key_types=key_types, registry_ids=registry_ids)


def _memory_registry(**kwargs) -> AgentCardRegistry:
    kwargs.setdefault("cache_backend", MemoryAgentCardCacheBackend())
    return AgentCardRegistry(**kwargs)


# ---------------------------------------------------------------------------
# Tests: AgentCardRegistryConfig
# ---------------------------------------------------------------------------


class TestAgentCardRegistryConfig:
    def test_default_ttl(self):
        config = AgentCardRegistryConfig()
        assert config.agent_card_registry_cache_ttl == 24 * 3600

    def test_ttl_zero_allowed(self):
        config = AgentCardRegistryConfig(agent_card_registry_cache_ttl=0)
        assert config.agent_card_registry_cache_ttl == 0

    def test_ttl_from_env(self):
        with patch.dict("os.environ", {"AGENT_CARD_REGISTRY_CACHE_TTL": "120"}):
            config = AgentCardRegistryConfig()
            assert config.agent_card_registry_cache_ttl == 120

    def test_default_on_duplicate(self):
        config = AgentCardRegistryConfig()
        assert config.agent_card_registry_on_duplicate == "first"

    def test_on_duplicate_from_env(self):
        with patch.dict("os.environ", {"AGENT_CARD_REGISTRY_ON_DUPLICATE": "error"}):
            config = AgentCardRegistryConfig()
            assert config.agent_card_registry_on_duplicate == "error"

    def test_on_duplicate_last(self):
        config = AgentCardRegistryConfig(agent_card_registry_on_duplicate="last")
        assert config.agent_card_registry_on_duplicate == "last"

    def test_memory_space_id_from_env(self):
        with patch.dict(
            "os.environ",
            {"AGENT_CARD_REGISTRY_MEMORY_SPACE_ID": "space-123"},
        ):
            config = AgentCardRegistryConfig()
            assert config.agent_card_registry_memory_space_id == "space-123"


# ---------------------------------------------------------------------------
# Tests: AgentCardCacheRecord
# ---------------------------------------------------------------------------


class TestAgentCardCacheRecord:
    def test_not_expired_within_ttl(self):
        entry = AgentCardCacheRecord(card=_card())
        assert entry.is_fresh(3600)

    def test_not_fresh_with_zero_ttl(self):
        entry = AgentCardCacheRecord(card=_card())
        assert not entry.is_fresh(0)

    def test_not_fresh_after_ttl(self):
        entry = AgentCardCacheRecord(card=_card())
        entry.fetched_at = datetime.now(UTC) - timedelta(seconds=100)
        assert not entry.is_fresh(50)

    def test_within_staleness(self):
        entry = AgentCardCacheRecord(card=_card())
        entry.fetched_at = datetime.now(UTC) - timedelta(seconds=100)
        assert entry.is_within_staleness(3600)
        assert not entry.is_within_staleness(50)

    def test_zero_max_staleness_never_servable(self):
        entry = AgentCardCacheRecord(card=_card())
        assert not entry.is_within_staleness(0)


# ---------------------------------------------------------------------------
# Tests: _resolve_settings
# ---------------------------------------------------------------------------


class TestResolveSettings:
    def test_explicit_values_used(self):
        token, endpoint = _resolve_settings(api_token="tok", endpoint="https://ep")
        assert token == "tok"
        assert endpoint == "https://ep"

    def test_token_comes_from_the_global_app_config(self):
        # The token is resolved off the global config (so a registered app config
        # supplies it), while the endpoint stays a local env-only setting.
        mock_settings = MagicMock(spec=DataRobotRegistrySettings)
        mock_settings.datarobot_endpoint = "https://ep"
        with (
            patch(f"{_MODULE}.DataRobotRegistrySettings", return_value=mock_settings),
            patch(f"{_MODULE}.resolve_config") as mock_resolve,
        ):
            mock_resolve.return_value.resolve_datarobot_api_token.return_value = "app-tok"
            token, endpoint = _resolve_settings()
        assert token == "app-tok"
        assert endpoint == "https://ep"

    def test_raises_when_no_token(self):
        mock_settings = MagicMock(spec=DataRobotRegistrySettings)
        mock_settings.datarobot_endpoint = None
        with (
            patch(f"{_MODULE}.DataRobotRegistrySettings", return_value=mock_settings),
            patch(f"{_MODULE}.resolve_config") as mock_resolve,
        ):
            mock_resolve.return_value.resolve_datarobot_api_token.return_value = None
            with pytest.raises(AgentCardRegistryError, match="API token is required"):
                _resolve_settings()

    def test_raises_when_no_endpoint(self):
        mock_settings = MagicMock(spec=DataRobotRegistrySettings)
        mock_settings.datarobot_endpoint = None
        with (
            patch(f"{_MODULE}.DataRobotRegistrySettings", return_value=mock_settings),
            patch(f"{_MODULE}.resolve_config") as mock_resolve,
        ):
            mock_resolve.return_value.resolve_datarobot_api_token.return_value = "tok"
            with pytest.raises(AgentCardRegistryError, match="API endpoint is required"):
                _resolve_settings()


# ---------------------------------------------------------------------------
# Tests: _parse_registry_response
# ---------------------------------------------------------------------------


class TestParseRegistryResponse:
    def test_indexes_by_both_ids(self):
        body = _registry_response(_entry(dep_id="dep-1", ext_id="ext-1"))
        parsed = _parse_registry_response(body)
        assert "dep-1" in parsed.cards
        assert "ext-1" in parsed.cards
        assert parsed.cards["dep-1"] is parsed.cards["ext-1"]
        assert parsed.key_types["dep-1"] == "deployment"
        assert parsed.key_types["ext-1"] == "external"
        assert parsed.registry_ids["dep-1"] == RegistryIds(
            deployment_id="dep-1", external_id="ext-1"
        )
        assert parsed.registry_ids["ext-1"] == RegistryIds(
            deployment_id="dep-1", external_id="ext-1"
        )

    def test_skips_entries_without_agent_card(self):
        body = _registry_response({"id": "x", "deploymentId": "d", "agentCard": None})
        assert _parse_registry_response(body).cards == {}

    def test_skips_entries_with_invalid_card(self):
        body = _registry_response({"id": "x", "deploymentId": "d", "agentCard": {"bad": True}})
        parsed = _parse_registry_response(body)
        assert parsed.cards == {}


class TestParseRegistryResponseDuplicates:
    """Tests for the on_duplicate strategy with duplicate external IDs."""

    def _body_with_duplicate_ext(self):
        return _registry_response(
            _entry(dep_id="dep-1", ext_id="shared-ext", card=_SAMPLE_AGENT_CARD),
            _entry(dep_id="dep-2", ext_id="shared-ext", card=_SAMPLE_AGENT_CARD_2),
        )

    def test_first_keeps_first_card(self):
        parsed = _parse_registry_response(self._body_with_duplicate_ext(), on_duplicate="first")
        assert parsed.cards["shared-ext"].name == "Test Agent"
        # Both deployment IDs are still indexed independently
        assert "dep-1" in parsed.cards
        assert "dep-2" in parsed.cards

    def test_last_keeps_last_card(self):
        parsed = _parse_registry_response(self._body_with_duplicate_ext(), on_duplicate="last")
        assert parsed.cards["shared-ext"].name == "Second Agent"

    def test_error_raises_on_duplicate(self):
        with pytest.raises(AgentCardRegistryError, match="Multiple agent cards found"):
            _parse_registry_response(self._body_with_duplicate_ext(), on_duplicate="error")

    def test_no_duplicate_no_error(self):
        """When external IDs are unique, 'error' strategy passes."""
        body = _registry_response(
            _entry(dep_id="dep-1", ext_id="ext-1"),
            _entry(dep_id="dep-2", ext_id="ext-2"),
        )
        parsed = _parse_registry_response(body, on_duplicate="error")
        assert "ext-1" in parsed.cards
        assert "ext-2" in parsed.cards

    def test_default_strategy_is_first(self):
        parsed = _parse_registry_response(self._body_with_duplicate_ext())
        assert parsed.cards["shared-ext"].name == "Test Agent"


_SAMPLE_AGENT_CARD_3 = {
    "name": "Third Agent",
    "description": "Third agent",
    "url": "https://agent3.example.com/a2a/",
    "version": "3.0.0",
    "skills": [],
    "defaultInputModes": ["text"],
    "defaultOutputModes": ["text"],
    "capabilities": {"streaming": False},
}


class TestParseRegistryResponseMultiIdBatch:
    """Simulate a batch query for externalIds=A,B where A has 3 cards and B has 1.

    The API returns 4 rows total.  The on_duplicate strategy must apply
    per-external-ID, and IDs without duplicates must be unaffected.
    """

    def _batch_response(self):
        """Three entries for ext_id 'agent-A', one for ext_id 'agent-B'."""
        return _registry_response(
            _entry(dep_id="dep-a1", ext_id="agent-A", card=_SAMPLE_AGENT_CARD),
            _entry(dep_id="dep-a2", ext_id="agent-A", card=_SAMPLE_AGENT_CARD_2),
            _entry(dep_id="dep-a3", ext_id="agent-A", card=_SAMPLE_AGENT_CARD_3),
            _entry(dep_id="dep-b1", ext_id="agent-B", card=_SAMPLE_AGENT_CARD_2),
        )

    def test_first_picks_first_of_three(self):
        parsed = _parse_registry_response(self._batch_response(), on_duplicate="first")
        # agent-A → first card (Test Agent)
        assert parsed.cards["agent-A"].name == "Test Agent"
        # agent-B has no duplicates → stored as-is
        assert parsed.cards["agent-B"].name == "Second Agent"
        # All deployment IDs are independently indexed
        assert len({parsed.cards[k].name for k in ["dep-a1", "dep-a2", "dep-a3", "dep-b1"]}) == 3

    def test_last_picks_last_of_three(self):
        parsed = _parse_registry_response(self._batch_response(), on_duplicate="last")
        # agent-A → last card (Third Agent)
        assert parsed.cards["agent-A"].name == "Third Agent"
        # agent-B unaffected
        assert parsed.cards["agent-B"].name == "Second Agent"

    def test_error_raises_for_duplicate_id_only(self):
        """Error is raised for agent-A (3 cards) — agent-B (1 card) never reached."""
        with pytest.raises(AgentCardRegistryError, match="agent-A"):
            _parse_registry_response(self._batch_response(), on_duplicate="error")


# ---------------------------------------------------------------------------
# Tests: AgentCardRegistry — core get/cache
# ---------------------------------------------------------------------------


class TestAgentCardRegistry:
    @pytest.fixture
    def mock_fetch(self):
        """Patch _fetch to avoid HTTP calls."""
        with patch.object(AgentCardRegistry, "_fetch", new_callable=AsyncMock) as m:
            yield m

    async def test_get_single_deployment_id(self, mock_fetch):
        expected = _card()
        mock_fetch.return_value = _parsed({"dep-1": expected})
        registry = _memory_registry(api_token="tok", endpoint="https://ep", cache_ttl=3600)
        card = await registry.get(deployment_id="dep-1")
        assert card is expected
        mock_fetch.assert_awaited_once_with({"deploymentIds": "dep-1"})

    async def test_get_single_external_id(self, mock_fetch):
        expected = _card()
        mock_fetch.return_value = _parsed(
            {"ext-1": expected},
            key_types={"ext-1": "external"},
        )
        registry = _memory_registry(api_token="tok", endpoint="https://ep", cache_ttl=3600)
        card = await registry.get(external_id="ext-1")
        assert card is expected
        mock_fetch.assert_awaited_once_with({"externalIds": "ext-1"})

    async def test_get_raises_when_both_ids(self):
        registry = _memory_registry(api_token="tok", endpoint="https://ep", cache_ttl=3600)
        with pytest.raises(AgentCardRegistryError, match="exactly one"):
            await registry.get(deployment_id="d", external_id="e")

    async def test_get_raises_when_neither_id(self):
        registry = _memory_registry(api_token="tok", endpoint="https://ep", cache_ttl=3600)
        with pytest.raises(AgentCardRegistryError, match="exactly one"):
            await registry.get()

    async def test_get_raises_when_not_found(self, mock_fetch):
        mock_fetch.return_value = _parsed({})
        registry = _memory_registry(api_token="tok", endpoint="https://ep", cache_ttl=3600)
        with pytest.raises(AgentCardRegistryError, match="No agent card found"):
            await registry.get(deployment_id="missing")

    async def test_get_uses_cache(self, mock_fetch):
        card = _card()
        mock_fetch.return_value = _parsed({"dep-1": card})
        registry = _memory_registry(api_token="tok", endpoint="https://ep", cache_ttl=3600)

        card1 = await registry.get(deployment_id="dep-1")
        card2 = await registry.get(deployment_id="dep-1")

        assert card1 is card2
        mock_fetch.assert_awaited_once()

    async def test_cache_ttl_zero_always_refetches(self, mock_fetch):
        """cache_ttl=0 means every get() triggers a fresh fetch."""
        card_mock = _card()
        mock_fetch.return_value = _parsed({"dep-1": card_mock})
        registry = _memory_registry(api_token="tok", endpoint="https://ep", cache_ttl=0)

        await registry.get(deployment_id="dep-1")
        await registry.get(deployment_id="dep-1")

        assert mock_fetch.await_count == 2

    async def test_expired_cache_triggers_refetch(self, mock_fetch):
        card = _card()
        mock_fetch.return_value = _parsed({"dep-1": card})
        registry = _memory_registry(api_token="tok", endpoint="https://ep", cache_ttl=60)

        await registry.get(deployment_id="dep-1")
        registry._age_cache_entry_for_test("dep-1", 120)

        refreshed = _card(name="Refreshed Agent")
        mock_fetch.return_value = _parsed({"dep-1": refreshed})
        await registry.get(deployment_id="dep-1")
        assert mock_fetch.await_count == 2


# ---------------------------------------------------------------------------
# Tests: AgentCardRegistry — stale-if-error
# ---------------------------------------------------------------------------


class TestAgentCardRegistryStaleIfError:
    @pytest.fixture
    def mock_fetch(self):
        with patch.object(AgentCardRegistry, "_fetch", new_callable=AsyncMock) as m:
            yield m

    @pytest.fixture
    def cache_backend(self):
        return MemoryAgentCardCacheBackend()

    async def test_serves_stale_when_refresh_fails(self, mock_fetch, cache_backend):
        stale_card = _card()
        mock_fetch.side_effect = [
            _parsed({"dep-1": stale_card}),
            AgentCardRegistryError("registry down"),
        ]
        registry = _memory_registry(
            api_token="tok",
            endpoint="https://ep",
            cache_ttl=60,
            cache_backend=cache_backend,
        )

        card1 = await registry.get(deployment_id="dep-1")
        fixed_now = datetime.now(UTC)
        cache_backend._entries["dep-1"].fetched_at = fixed_now - timedelta(seconds=60)
        with patch("datarobot_genai.dragent.agent_card_registry_backends.datetime") as mock_dt:
            mock_dt.now.return_value = fixed_now
            mock_dt.UTC = UTC
            card2 = await registry.get(deployment_id="dep-1")

        assert card1 is stale_card
        assert card2 is stale_card
        assert mock_fetch.await_count == 2

    async def test_raises_when_beyond_cache_ttl(self, mock_fetch, cache_backend):
        stale_card = _card()
        mock_fetch.side_effect = [
            _parsed({"dep-1": stale_card}),
            AgentCardRegistryError("registry down"),
        ]
        registry = _memory_registry(
            api_token="tok",
            endpoint="https://ep",
            cache_ttl=60,
            cache_backend=cache_backend,
        )

        await registry.get(deployment_id="dep-1")
        registry._age_cache_entry_for_test("dep-1", 120)

        with pytest.raises(AgentCardRegistryError, match="registry down"):
            await registry.get(deployment_id="dep-1")

    async def test_raises_when_no_cached_card(self, mock_fetch):
        mock_fetch.side_effect = AgentCardRegistryError("registry down")
        registry = _memory_registry(api_token="tok", endpoint="https://ep", cache_ttl=60)

        with pytest.raises(AgentCardRegistryError, match="registry down"):
            await registry.get(deployment_id="dep-1")

    async def test_stale_if_error_on_flush_pending_failure(self, mock_fetch, cache_backend):
        stale_card = _card()
        mock_fetch.side_effect = [
            _parsed({"dep-1": stale_card}),
            AgentCardRegistryError("registry down"),
        ]
        registry = _memory_registry(
            api_token="tok",
            endpoint="https://ep",
            cache_ttl=60,
            cache_backend=cache_backend,
        )
        await registry.get(deployment_id="dep-1")
        fixed_now = datetime.now(UTC)
        cache_backend._entries["dep-1"].fetched_at = fixed_now - timedelta(seconds=60)

        registry.register(deployment_id="dep-2")
        with patch("datarobot_genai.dragent.agent_card_registry_backends.datetime") as mock_dt:
            mock_dt.now.return_value = fixed_now
            mock_dt.UTC = UTC
            card = await registry.get(deployment_id="dep-1")

        assert card is stale_card

    async def test_raises_when_deregistered_after_soft_ttl(self, mock_fetch, cache_backend):
        stale_card = _card()
        mock_fetch.side_effect = [
            _parsed({"dep-1": stale_card}),
            _parsed({}),
        ]
        registry = _memory_registry(
            api_token="tok",
            endpoint="https://ep",
            cache_ttl=60,
            cache_backend=cache_backend,
        )
        await registry.get(deployment_id="dep-1")
        registry._age_cache_entry_for_test("dep-1", 120)

        with pytest.raises(AgentCardRegistryError, match="No agent card found"):
            await registry.get(deployment_id="dep-1")

        assert await cache_backend.get_stale("dep-1", max_staleness_seconds=60) is None

    async def test_deregistered_not_resurrected_by_stale_if_error(self, mock_fetch, cache_backend):
        stale_card = _card()
        mock_fetch.side_effect = [
            _parsed({"dep-1": stale_card}),
            _parsed({}),
            AgentCardRegistryError("registry down"),
        ]
        registry = _memory_registry(
            api_token="tok",
            endpoint="https://ep",
            cache_ttl=60,
            cache_backend=cache_backend,
        )
        await registry.get(deployment_id="dep-1")
        registry._age_cache_entry_for_test("dep-1", 60)

        with pytest.raises(AgentCardRegistryError, match="No agent card found"):
            await registry.get(deployment_id="dep-1")

        with pytest.raises(AgentCardRegistryError, match="registry down"):
            await registry.get(deployment_id="dep-1")

    async def test_deregistered_sibling_id_evicted_from_cache(self, mock_fetch, cache_backend):
        """Evicting on a deployment miss must drop the external-ID alias too."""
        stale_card = _card()
        id_pair = RegistryIds(deployment_id="dep-1", external_id="ext-1")
        mock_fetch.side_effect = [
            _parsed(
                {"dep-1": stale_card, "ext-1": stale_card},
                key_types={"dep-1": "deployment", "ext-1": "external"},
                registry_ids={"dep-1": id_pair, "ext-1": id_pair},
            ),
            _parsed({}),
        ]
        registry = _memory_registry(
            api_token="tok",
            endpoint="https://ep",
            cache_ttl=60,
            cache_backend=cache_backend,
        )
        await registry.get(deployment_id="dep-1")
        registry._age_cache_entry_for_test("dep-1", 60)

        with pytest.raises(AgentCardRegistryError, match="No agent card found"):
            await registry.get(deployment_id="dep-1")

        assert await cache_backend.get_stale("dep-1", max_staleness_seconds=3600) is None
        assert await cache_backend.get_stale("ext-1", max_staleness_seconds=3600) is None


# ---------------------------------------------------------------------------
# Tests: AgentCardRegistry — register + batch flush
# ---------------------------------------------------------------------------


class TestAgentCardRegistryRegisterFlush:
    @pytest.fixture
    def mock_fetch(self):
        with patch.object(AgentCardRegistry, "_fetch", new_callable=AsyncMock) as m:
            yield m

    async def test_register_then_get_flushes_batch(self, mock_fetch):
        """Registered IDs are batch-fetched on first get()."""
        mock_fetch.side_effect = [
            _parsed({"dep-1": _card(), "dep-2": _card(name="Second Agent")}),
            _parsed({"ext-1": _card(name="Third Agent")}, key_types={"ext-1": "external"}),
        ]
        registry = _memory_registry(api_token="tok", endpoint="https://ep", cache_ttl=3600)
        registry.register(deployment_id="dep-1")
        registry.register(deployment_id="dep-2")
        registry.register(external_id="ext-1")

        # First get triggers batch flush
        await registry.get(deployment_id="dep-1")

        assert mock_fetch.await_count == 2
        calls = mock_fetch.call_args_list
        assert calls[0].args[0] == {"deploymentIds": "dep-1,dep-2"} or calls[0].args[0] == {
            "deploymentIds": "dep-2,dep-1"
        }
        assert calls[1].args[0] == {"externalIds": "ext-1"}

        # Subsequent gets are cache hits
        mock_fetch.reset_mock()
        await registry.get(deployment_id="dep-2")
        await registry.get(external_id="ext-1")
        mock_fetch.assert_not_awaited()

    async def test_register_deduplicates(self, mock_fetch):
        mock_fetch.return_value = _parsed({"dep-1": _card()})
        registry = _memory_registry(api_token="tok", endpoint="https://ep", cache_ttl=3600)
        registry.register(deployment_id="dep-1")
        registry.register(deployment_id="dep-1")
        registry.register(deployment_id="dep-1")

        await registry.get(deployment_id="dep-1")
        mock_fetch.assert_awaited_once_with({"deploymentIds": "dep-1"})

    async def test_pending_cleared_after_flush(self, mock_fetch):
        mock_fetch.return_value = _parsed({"dep-1": _card()})
        registry = _memory_registry(api_token="tok", endpoint="https://ep", cache_ttl=3600)
        registry.register(deployment_id="dep-1")

        await registry.get(deployment_id="dep-1")
        assert len(registry._pending_deployment_ids) == 0
        assert len(registry._pending_external_ids) == 0
        assert registry.has_registered_lookups() is True


# ---------------------------------------------------------------------------
# Tests: AgentCardRegistry — prefetch
# ---------------------------------------------------------------------------


class TestAgentCardRegistryPrefetch:
    @pytest.fixture
    def mock_fetch(self):
        with patch.object(AgentCardRegistry, "_fetch", new_callable=AsyncMock) as m:
            yield m

    async def test_prefetch_deployment_ids(self, mock_fetch):
        mock_fetch.return_value = _parsed(
            {
                "dep-1": _card(),
                "dep-2": _card(name="Second Agent"),
            }
        )
        registry = _memory_registry(api_token="tok", endpoint="https://ep", cache_ttl=3600)
        await registry.prefetch(deployment_ids=["dep-1", "dep-2"])

        mock_fetch.assert_awaited_once_with({"deploymentIds": "dep-1,dep-2"})
        mock_fetch.reset_mock()
        await registry.get(deployment_id="dep-1")
        await registry.get(deployment_id="dep-2")
        mock_fetch.assert_not_awaited()

    async def test_prefetch_mixed_issues_separate_calls(self, mock_fetch):
        mock_fetch.side_effect = [
            _parsed({"dep-1": _card()}),
            _parsed({"ext-1": _card(name="Second Agent")}, key_types={"ext-1": "external"}),
        ]
        registry = _memory_registry(api_token="tok", endpoint="https://ep", cache_ttl=3600)
        await registry.prefetch(deployment_ids=["dep-1"], external_ids=["ext-1"])

        assert mock_fetch.await_count == 2
        calls = mock_fetch.call_args_list
        assert calls[0].args[0] == {"deploymentIds": "dep-1"}
        assert calls[1].args[0] == {"externalIds": "ext-1"}

    async def test_prefetch_skips_already_cached(self, mock_fetch):
        mock_fetch.return_value = _parsed({"dep-1": _card()})
        registry = _memory_registry(api_token="tok", endpoint="https://ep", cache_ttl=3600)

        await registry.prefetch(deployment_ids=["dep-1"])
        mock_fetch.reset_mock()

        mock_fetch.return_value = _parsed({"dep-2": _card(name="Second Agent")})
        await registry.prefetch(deployment_ids=["dep-1", "dep-2"])
        mock_fetch.assert_awaited_once_with({"deploymentIds": "dep-2"})


# ---------------------------------------------------------------------------
# Tests: AgentCardRegistry._fetch (HTTP integration)
# ---------------------------------------------------------------------------


class TestAgentCardRegistryFetch:
    @pytest.fixture
    def mock_httpx_client(self):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = _registry_response(
            _entry(dep_id="dep-1"),
            _entry(dep_id="dep-2", card=_SAMPLE_AGENT_CARD_2),
        )
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        return mock_client

    async def test_fetch_passes_params_and_auth(self, mock_httpx_client):
        with patch(f"{_MODULE}.httpx.AsyncClient", return_value=mock_httpx_client):
            registry = _memory_registry(
                api_token="my-tok", endpoint="https://app.dr.com/api/v2", cache_ttl=3600
            )
            parsed = await registry._fetch({"deploymentIds": "dep-1,dep-2"})

        assert "dep-1" in parsed.cards
        assert "dep-2" in parsed.cards
        call_kwargs = mock_httpx_client.get.call_args.kwargs
        assert call_kwargs["params"]["deploymentIds"] == "dep-1,dep-2"
        assert call_kwargs["params"]["limit"] == "100"
        assert call_kwargs["headers"]["Authorization"] == "Bearer my-tok"

    async def test_fetch_http_error(self):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 403
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Forbidden", request=MagicMock(), response=mock_response
        )

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch(f"{_MODULE}.httpx.AsyncClient", return_value=mock_client):
            registry = _memory_registry(api_token="tok", endpoint="https://ep", cache_ttl=3600)
            with pytest.raises(AgentCardRegistryError, match="HTTP 403"):
                await registry._fetch({"deploymentIds": "dep-1"})

    async def test_fetch_paginates_through_all_pages(self):
        """_fetch follows the 'next' link until all pages are consumed."""
        page1_response = MagicMock(spec=httpx.Response)
        page1_response.status_code = 200
        page1_response.json.return_value = {
            "data": [_entry(dep_id="dep-1")],
            "count": 1,
            "totalCount": 3,
            "next": "https://app.dr.com/api/v2/agentCards/?offset=1&limit=1",
            "previous": None,
        }
        page1_response.raise_for_status = MagicMock()

        page2_response = MagicMock(spec=httpx.Response)
        page2_response.status_code = 200
        page2_response.json.return_value = {
            "data": [_entry(dep_id="dep-2", card=_SAMPLE_AGENT_CARD_2)],
            "count": 1,
            "totalCount": 3,
            "next": "https://app.dr.com/api/v2/agentCards/?offset=2&limit=1",
            "previous": "https://app.dr.com/api/v2/agentCards/?offset=0&limit=1",
        }
        page2_response.raise_for_status = MagicMock()

        page3_response = MagicMock(spec=httpx.Response)
        page3_response.status_code = 200
        page3_response.json.return_value = {
            "data": [_entry(dep_id="dep-3", card=_SAMPLE_AGENT_CARD_3)],
            "count": 1,
            "totalCount": 3,
            "next": None,
            "previous": "https://app.dr.com/api/v2/agentCards/?offset=1&limit=1",
        }
        page3_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=[page1_response, page2_response, page3_response])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch(f"{_MODULE}.httpx.AsyncClient", return_value=mock_client):
            registry = _memory_registry(
                api_token="my-tok", endpoint="https://app.dr.com/api/v2", cache_ttl=3600
            )
            parsed = await registry._fetch({"deploymentIds": "dep-1,dep-2,dep-3"})

        assert "dep-1" in parsed.cards
        assert "dep-2" in parsed.cards
        assert "dep-3" in parsed.cards
        assert len(parsed.cards) == 3
        assert mock_client.get.await_count == 3

    async def test_fetch_no_pagination_when_next_absent(self):
        """When 'next' is absent or null, only one request is made."""
        single_response = MagicMock(spec=httpx.Response)
        single_response.status_code = 200
        single_response.json.return_value = {
            "data": [_entry(dep_id="dep-1")],
            "count": 1,
            "totalCount": 1,
        }
        single_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=single_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch(f"{_MODULE}.httpx.AsyncClient", return_value=mock_client):
            registry = _memory_registry(api_token="tok", endpoint="https://ep", cache_ttl=3600)
            parsed = await registry._fetch({"deploymentIds": "dep-1"})

        assert "dep-1" in parsed.cards
        assert mock_client.get.await_count == 1

    async def test_fetch_pagination_error_on_second_page_raises(self):
        """HTTP error on a pagination request propagates correctly."""
        page1_response = MagicMock(spec=httpx.Response)
        page1_response.status_code = 200
        page1_response.json.return_value = {
            "data": [_entry(dep_id="dep-1")],
            "count": 1,
            "totalCount": 2,
            "next": "https://app.dr.com/api/v2/agentCards/?offset=1&limit=1",
        }
        page1_response.raise_for_status = MagicMock()

        page2_response = MagicMock(spec=httpx.Response)
        page2_response.status_code = 500
        page2_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error", request=MagicMock(), response=page2_response
        )

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=[page1_response, page2_response])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch(f"{_MODULE}.httpx.AsyncClient", return_value=mock_client):
            registry = _memory_registry(api_token="tok", endpoint="https://ep", cache_ttl=3600)
            with pytest.raises(AgentCardRegistryError, match="HTTP 500"):
                await registry._fetch({"deploymentIds": "dep-1,dep-2"})

    async def test_fetch_stops_at_safety_limit(self):
        """Pagination stops after _MAX_PAGES to prevent infinite loops."""

        def _make_page(page_num: int, has_next: bool):
            resp = MagicMock(spec=httpx.Response)
            resp.status_code = 200
            resp.json.return_value = {
                "data": [_entry(dep_id=f"dep-{page_num}")],
                "count": 1,
                "totalCount": 9999,
                "next": f"https://ep/agentCards/?offset={page_num}&limit=100" if has_next else None,
            }
            resp.raise_for_status = MagicMock()
            return resp

        # All pages claim there's a next page (simulating a buggy API / infinite loop)
        pages = [_make_page(i, has_next=True) for i in range(_MAX_PAGES + 5)]

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=pages)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch(f"{_MODULE}.httpx.AsyncClient", return_value=mock_client):
            registry = _memory_registry(api_token="tok", endpoint="https://ep", cache_ttl=3600)
            parsed = await registry._fetch({"deploymentIds": "dep-0"})

        # Should have fetched exactly _MAX_PAGES (stopped at the safety limit)
        assert mock_client.get.await_count == _MAX_PAGES
        assert len(parsed.cards) == _MAX_PAGES


# ---------------------------------------------------------------------------
# Tests: get_default_registry / get_default_registry_sync singleton
# ---------------------------------------------------------------------------


class TestGetDefaultRegistry:
    @pytest.fixture(autouse=True)
    def _reset(self):
        reset_default_registry()
        yield
        reset_default_registry()

    async def test_returns_singleton(self):
        r1 = await get_default_registry()
        r2 = await get_default_registry()
        assert r1 is r2

    async def test_reset_clears_singleton(self):
        r1 = await get_default_registry()
        reset_default_registry()
        r2 = await get_default_registry()
        assert r1 is not r2

    def test_sync_returns_singleton(self):
        r1 = get_default_registry_sync()
        r2 = get_default_registry_sync()
        assert r1 is r2

    async def test_sync_and_async_share_singleton(self):
        r1 = get_default_registry_sync()
        r2 = await get_default_registry()
        assert r1 is r2


# ---------------------------------------------------------------------------
# Tests: workload ID lookups
# ---------------------------------------------------------------------------


class TestParseRegistryResponseWorkload:
    def test_indexes_by_workload_id(self):
        """GIVEN a workload-keyed entry WHEN parsed THEN it is indexed as 'workload'."""
        body = _registry_response(_entry(wl_id="wl-1"))
        parsed = _parse_registry_response(body)
        assert "wl-1" in parsed.cards
        assert parsed.key_types["wl-1"] == "workload"
        assert parsed.registry_ids["wl-1"] == RegistryIds(workload_id="wl-1")

    def test_workload_card_with_external_id_reachable_by_either(self):
        """GIVEN a workload card that also publishes an external ID
        WHEN parsed THEN both IDs resolve to the same card.
        """
        body = _registry_response(_entry(wl_id="wl-1", ext_id="ext-1"))
        parsed = _parse_registry_response(body)

        assert parsed.cards["wl-1"] is parsed.cards["ext-1"]
        assert parsed.key_types["wl-1"] == "workload"
        assert parsed.key_types["ext-1"] == "external"
        expected_ids = RegistryIds(external_id="ext-1", workload_id="wl-1")
        assert parsed.registry_ids["wl-1"] == expected_ids
        assert parsed.registry_ids["ext-1"] == expected_ids

    def test_duplicate_workload_id_always_overwrites(self):
        """GIVEN two entries for one workload ID (unique by platform design)
        WHEN parsed THEN the later entry wins without duplicate handling.
        """
        body = _registry_response(
            _entry(wl_id="wl-1"),
            _entry(wl_id="wl-1", card=_SAMPLE_AGENT_CARD_2),
        )
        parsed = _parse_registry_response(body, on_duplicate="error")
        assert parsed.cards["wl-1"].name == "Second Agent"


class TestAgentCardRegistryWorkload:
    @pytest.fixture
    def mock_fetch(self):
        with patch.object(AgentCardRegistry, "_fetch", new_callable=AsyncMock) as m:
            yield m

    @pytest.fixture
    def cache_backend(self):
        return MemoryAgentCardCacheBackend()

    def _parsed_workload(self, cards: dict) -> ParsedRegistryCards:
        return _parsed(cards, key_types={key: "workload" for key in cards})

    async def test_get_single_workload_id(self, mock_fetch):
        """GIVEN a workload-keyed card WHEN got by workload ID THEN workloadIds is queried."""
        expected = _card()
        mock_fetch.return_value = self._parsed_workload({"wl-1": expected})
        registry = _memory_registry(api_token="tok", endpoint="https://ep", cache_ttl=3600)

        card = await registry.get(workload_id="wl-1")

        assert card is expected
        mock_fetch.assert_awaited_once_with({"workloadIds": "wl-1"})

    async def test_get_raises_when_workload_and_deployment_id(self):
        registry = _memory_registry(api_token="tok", endpoint="https://ep", cache_ttl=3600)
        with pytest.raises(AgentCardRegistryError, match="exactly one"):
            await registry.get(deployment_id="dep-1", workload_id="wl-1")

    async def test_get_raises_when_workload_and_external_id(self):
        registry = _memory_registry(api_token="tok", endpoint="https://ep", cache_ttl=3600)
        with pytest.raises(AgentCardRegistryError, match="exactly one"):
            await registry.get(external_id="ext-1", workload_id="wl-1")

    async def test_not_found_message_names_workload_id(self, mock_fetch):
        mock_fetch.return_value = _parsed({})
        registry = _memory_registry(api_token="tok", endpoint="https://ep", cache_ttl=3600)
        with pytest.raises(AgentCardRegistryError, match="workload_id='wl-missing'"):
            await registry.get(workload_id="wl-missing")

    async def test_get_uses_cache(self, mock_fetch):
        mock_fetch.return_value = self._parsed_workload({"wl-1": _card()})
        registry = _memory_registry(api_token="tok", endpoint="https://ep", cache_ttl=3600)

        first = await registry.get(workload_id="wl-1")
        second = await registry.get(workload_id="wl-1")

        assert first is second
        mock_fetch.assert_awaited_once()

    async def test_register_then_get_flushes_batch(self, mock_fetch):
        """GIVEN registered workload IDs WHEN one is got THEN both are fetched in one call."""
        mock_fetch.return_value = self._parsed_workload(
            {"wl-1": _card(), "wl-2": _card(name="Second Agent")}
        )
        registry = _memory_registry(api_token="tok", endpoint="https://ep", cache_ttl=3600)
        registry.register(workload_id="wl-1")
        registry.register(workload_id="wl-2")
        assert registry.has_registered_lookups() is True

        await registry.get(workload_id="wl-1")

        mock_fetch.assert_awaited_once()
        assert mock_fetch.call_args_list[0].args[0].keys() == {"workloadIds"}
        assert set(mock_fetch.call_args_list[0].args[0]["workloadIds"].split(",")) == {
            "wl-1",
            "wl-2",
        }
        mock_fetch.reset_mock()
        await registry.get(workload_id="wl-2")
        mock_fetch.assert_not_awaited()

    async def test_prefetch_workload_ids(self, mock_fetch):
        mock_fetch.return_value = self._parsed_workload(
            {"wl-1": _card(), "wl-2": _card(name="Second Agent")}
        )
        registry = _memory_registry(api_token="tok", endpoint="https://ep", cache_ttl=3600)
        await registry.prefetch(workload_ids=["wl-1", "wl-2"])

        mock_fetch.assert_awaited_once_with({"workloadIds": "wl-1,wl-2"})
        mock_fetch.reset_mock()
        await registry.get(workload_id="wl-1")
        await registry.get(workload_id="wl-2")
        mock_fetch.assert_not_awaited()

    async def test_refresh_all_registered_includes_workload_ids(self, mock_fetch):
        mock_fetch.return_value = self._parsed_workload({"wl-1": _card()})
        registry = _memory_registry(api_token="tok", endpoint="https://ep", cache_ttl=60)
        registry.register(workload_id="wl-1")
        await registry.get(workload_id="wl-1")
        registry._age_cache_entry_for_test("wl-1", 120)

        mock_fetch.reset_mock()
        await registry.refresh_all_registered()

        mock_fetch.assert_awaited_once_with({"workloadIds": "wl-1"})

    async def test_serves_stale_when_refresh_fails(self, mock_fetch, cache_backend):
        """GIVEN a soft-expired workload card WHEN the registry is down THEN it is served."""
        stale_card = _card()
        mock_fetch.side_effect = [
            self._parsed_workload({"wl-1": stale_card}),
            AgentCardRegistryError("registry down"),
        ]
        registry = _memory_registry(
            api_token="tok",
            endpoint="https://ep",
            cache_ttl=60,
            cache_backend=cache_backend,
        )
        first = await registry.get(workload_id="wl-1")
        fixed_now = datetime.now(UTC)
        cache_backend._entries["wl-1"].fetched_at = fixed_now - timedelta(seconds=60)
        with patch("datarobot_genai.dragent.agent_card_registry_backends.datetime") as mock_dt:
            mock_dt.now.return_value = fixed_now
            mock_dt.UTC = UTC
            second = await registry.get(workload_id="wl-1")

        assert first is stale_card
        assert second is stale_card
        assert mock_fetch.await_count == 2

    async def test_deregistered_workload_evicted_from_cache(self, mock_fetch, cache_backend):
        """GIVEN a stopped workload WHEN its card is gone THEN the alias is evicted too."""
        stale_card = _card()
        ids = RegistryIds(external_id="ext-1", workload_id="wl-1")
        mock_fetch.side_effect = [
            _parsed(
                {"wl-1": stale_card, "ext-1": stale_card},
                key_types={"wl-1": "workload", "ext-1": "external"},
                registry_ids={"wl-1": ids, "ext-1": ids},
            ),
            _parsed({}),
        ]
        registry = _memory_registry(
            api_token="tok",
            endpoint="https://ep",
            cache_ttl=60,
            cache_backend=cache_backend,
        )
        await registry.get(workload_id="wl-1")
        registry._age_cache_entry_for_test("wl-1", 60)

        with pytest.raises(AgentCardRegistryError, match="No agent card found"):
            await registry.get(workload_id="wl-1")

        assert await cache_backend.get_stale("wl-1", max_staleness_seconds=3600) is None
        assert await cache_backend.get_stale("ext-1", max_staleness_seconds=3600) is None


# ---------------------------------------------------------------------------
# Tests: ID kinds are never mixed in one request
# ---------------------------------------------------------------------------


class TestRegistryIdKindIsolation:
    """``deploymentIds`` + ``workloadIds`` in one request is an HTTP 400."""

    @pytest.fixture
    def mock_fetch(self):
        with patch.object(AgentCardRegistry, "_fetch", new_callable=AsyncMock) as m:
            m.return_value = _parsed({})
            yield m

    @staticmethod
    def _assert_one_id_kind_per_request(mock_fetch):
        id_params = {"deploymentIds", "externalIds", "workloadIds"}
        for call in mock_fetch.call_args_list:
            params = call.args[0]
            assert len(id_params & params.keys()) == 1, params
            assert not {"deploymentIds", "workloadIds"} <= params.keys(), params

    async def test_prefetch_of_all_kinds_issues_one_call_per_kind(self, mock_fetch):
        registry = _memory_registry(api_token="tok", endpoint="https://ep", cache_ttl=3600)
        await registry.prefetch(
            deployment_ids=["dep-1"],
            external_ids=["ext-1"],
            workload_ids=["wl-1"],
        )

        assert mock_fetch.await_count == 3
        self._assert_one_id_kind_per_request(mock_fetch)
        assert [call.args[0] for call in mock_fetch.call_args_list] == [
            {"deploymentIds": "dep-1"},
            {"externalIds": "ext-1"},
            {"workloadIds": "wl-1"},
        ]

    async def test_flush_pending_of_all_kinds_issues_one_call_per_kind(self, mock_fetch):
        registry = _memory_registry(api_token="tok", endpoint="https://ep", cache_ttl=3600)
        registry.register(deployment_id="dep-1")
        registry.register(external_id="ext-1")
        registry.register(workload_id="wl-1")

        with pytest.raises(AgentCardRegistryError, match="No agent card found"):
            await registry.get(workload_id="wl-1")

        # 3 flush calls (one per kind) + 1 individual retry for the missing key
        assert mock_fetch.await_count == 4
        self._assert_one_id_kind_per_request(mock_fetch)

    async def test_fetch_rejects_deployment_and_workload_ids_together(self):
        """The guard fires before any HTTP call is made."""
        registry = _memory_registry(api_token="tok", endpoint="https://ep", cache_ttl=3600)
        with patch(f"{_MODULE}.httpx.AsyncClient") as mock_client_cls:
            with pytest.raises(AgentCardRegistryError, match="same agent card registry request"):
                await registry._fetch({"deploymentIds": "dep-1", "workloadIds": "wl-1"})

        mock_client_cls.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: ID list chunking (API caps each ID parameter at 20 values)
# ---------------------------------------------------------------------------


class TestAgentCardRegistryIdChunking:
    @staticmethod
    def _sequenced_client(bodies):
        """Return an httpx client mock answering each GET with the next body."""
        responses = []
        for body in bodies:
            response = MagicMock(spec=httpx.Response)
            response.status_code = 200
            response.json.return_value = body
            response.raise_for_status = MagicMock()
            responses.append(response)

        client = AsyncMock()
        client.get = AsyncMock(side_effect=responses)
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        return client

    @staticmethod
    def _requested_ids(client, param):
        """Return the per-request ID lists sent under *param*."""
        return [call.kwargs["params"][param].split(",") for call in client.get.call_args_list]

    async def test_more_than_20_ids_split_into_several_requests(self):
        """GIVEN 21 deployment IDs WHEN fetched THEN two capped requests are issued."""
        ids = [f"dep-{i}" for i in range(21)]
        client = self._sequenced_client(
            [
                _registry_response(*[_entry(dep_id=i) for i in ids[:20]]),
                _registry_response(_entry(dep_id=ids[20])),
            ]
        )

        with patch(f"{_MODULE}.httpx.AsyncClient", return_value=client):
            registry = _memory_registry(api_token="tok", endpoint="https://ep", cache_ttl=3600)
            parsed = await registry._fetch({"deploymentIds": ",".join(ids)})

        requested = self._requested_ids(client, "deploymentIds")
        assert [len(chunk) for chunk in requested] == [20, 1]
        assert [i for chunk in requested for i in chunk] == ids
        assert set(parsed.cards) == set(ids)

    async def test_workload_ids_are_chunked_too(self):
        ids = [f"wl-{i}" for i in range(25)]
        client = self._sequenced_client(
            [
                _registry_response(*[_entry(wl_id=i) for i in ids[:20]]),
                _registry_response(*[_entry(wl_id=i) for i in ids[20:]]),
            ]
        )

        with patch(f"{_MODULE}.httpx.AsyncClient", return_value=client):
            registry = _memory_registry(api_token="tok", endpoint="https://ep", cache_ttl=3600)
            parsed = await registry._fetch({"workloadIds": ",".join(ids)})

        assert [len(chunk) for chunk in self._requested_ids(client, "workloadIds")] == [20, 5]
        assert set(parsed.cards) == set(ids)
        assert parsed.key_types["wl-24"] == "workload"

    async def test_20_ids_stay_in_one_request(self):
        ids = [f"dep-{i}" for i in range(20)]
        client = self._sequenced_client([_registry_response(*[_entry(dep_id=i) for i in ids])])

        with patch(f"{_MODULE}.httpx.AsyncClient", return_value=client):
            registry = _memory_registry(api_token="tok", endpoint="https://ep", cache_ttl=3600)
            await registry._fetch({"deploymentIds": ",".join(ids)})

        assert client.get.await_count == 1

    async def test_on_duplicate_applies_across_chunks(self):
        """GIVEN a duplicate external ID split across two chunks
        WHEN fetched THEN on_duplicate resolves it as one result set.
        """
        ids = [f"ext-{i}" for i in range(20)] + ["ext-dup"]
        client = self._sequenced_client(
            [
                _registry_response(_entry(ext_id="ext-dup")),
                _registry_response(_entry(ext_id="ext-dup", card=_SAMPLE_AGENT_CARD_2)),
            ]
        )

        with patch(f"{_MODULE}.httpx.AsyncClient", return_value=client):
            registry = _memory_registry(
                api_token="tok",
                endpoint="https://ep",
                cache_ttl=3600,
                on_duplicate="first",
            )
            parsed = await registry._fetch({"externalIds": ",".join(ids)})

        assert client.get.await_count == 2
        assert parsed.cards["ext-dup"].name == "Test Agent"

    async def test_prefetch_chunks_registered_ids(self):
        """GIVEN more than 20 registered workload IDs WHEN prefetched THEN requests are capped."""
        ids = [f"wl-{i}" for i in range(21)]
        client = self._sequenced_client(
            [
                _registry_response(*[_entry(wl_id=i) for i in ids[:20]]),
                _registry_response(_entry(wl_id=ids[20])),
            ]
        )

        with patch(f"{_MODULE}.httpx.AsyncClient", return_value=client):
            registry = _memory_registry(api_token="tok", endpoint="https://ep", cache_ttl=3600)
            await registry.prefetch(workload_ids=ids)

        assert [len(chunk) for chunk in self._requested_ids(client, "workloadIds")] == [20, 1]
