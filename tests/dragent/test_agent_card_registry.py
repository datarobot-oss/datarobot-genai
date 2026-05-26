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

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import httpx
import pytest

from datarobot_genai.dragent.agent_card_registry import AgentCardRegistry
from datarobot_genai.dragent.agent_card_registry import AgentCardRegistryConfig
from datarobot_genai.dragent.agent_card_registry import AgentCardRegistryError
from datarobot_genai.dragent.agent_card_registry import DataRobotRegistrySettings
from datarobot_genai.dragent.agent_card_registry import _CacheEntry
from datarobot_genai.dragent.agent_card_registry import _MAX_PAGES
from datarobot_genai.dragent.agent_card_registry import _parse_registry_response
from datarobot_genai.dragent.agent_card_registry import _resolve_settings
from datarobot_genai.dragent.agent_card_registry import get_default_registry
from datarobot_genai.dragent.agent_card_registry import get_default_registry_sync
from datarobot_genai.dragent.agent_card_registry import reset_default_registry

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


def _entry(dep_id=None, ext_id=None, card=_SAMPLE_AGENT_CARD):
    return {
        "id": f"doc-{dep_id or ext_id}",
        "deploymentId": dep_id,
        "externalId": ext_id,
        "agentCard": card,
    }


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


# ---------------------------------------------------------------------------
# Tests: _CacheEntry
# ---------------------------------------------------------------------------


class TestCacheEntry:
    def test_not_expired_within_ttl(self):
        entry = _CacheEntry(MagicMock())
        assert not entry.is_expired(3600)

    def test_expired_with_zero_ttl(self):
        entry = _CacheEntry(MagicMock())
        assert entry.is_expired(0)

    def test_expired_after_ttl(self):
        entry = _CacheEntry(MagicMock())
        entry.fetched_at -= 100  # Simulate 100s ago
        assert entry.is_expired(50)


# ---------------------------------------------------------------------------
# Tests: _resolve_settings
# ---------------------------------------------------------------------------


class TestResolveSettings:
    def test_explicit_values_used(self):
        token, endpoint = _resolve_settings(api_token="tok", endpoint="https://ep")
        assert token == "tok"
        assert endpoint == "https://ep"

    def test_raises_when_no_token(self):
        mock_settings = MagicMock(spec=DataRobotRegistrySettings)
        mock_settings.datarobot_api_token = None
        mock_settings.datarobot_endpoint = None
        with patch(f"{_MODULE}.DataRobotRegistrySettings", return_value=mock_settings):
            with pytest.raises(AgentCardRegistryError, match="API token is required"):
                _resolve_settings()

    def test_raises_when_no_endpoint(self):
        mock_settings = MagicMock(spec=DataRobotRegistrySettings)
        mock_settings.datarobot_api_token = "tok"
        mock_settings.datarobot_endpoint = None
        with patch(f"{_MODULE}.DataRobotRegistrySettings", return_value=mock_settings):
            with pytest.raises(AgentCardRegistryError, match="API endpoint is required"):
                _resolve_settings()


# ---------------------------------------------------------------------------
# Tests: _parse_registry_response
# ---------------------------------------------------------------------------


class TestParseRegistryResponse:
    def test_indexes_by_both_ids(self):
        body = _registry_response(_entry(dep_id="dep-1", ext_id="ext-1"))
        cards = _parse_registry_response(body)
        assert "dep-1" in cards
        assert "ext-1" in cards
        assert cards["dep-1"] is cards["ext-1"]

    def test_skips_entries_without_agent_card(self):
        body = _registry_response({"id": "x", "deploymentId": "d", "agentCard": None})
        assert _parse_registry_response(body) == {}

    def test_skips_entries_with_invalid_card(self):
        body = _registry_response({"id": "x", "deploymentId": "d", "agentCard": {"bad": True}})
        cards = _parse_registry_response(body)
        assert cards == {}


class TestParseRegistryResponseDuplicates:
    """Tests for the on_duplicate strategy with duplicate external IDs."""

    def _body_with_duplicate_ext(self):
        return _registry_response(
            _entry(dep_id="dep-1", ext_id="shared-ext", card=_SAMPLE_AGENT_CARD),
            _entry(dep_id="dep-2", ext_id="shared-ext", card=_SAMPLE_AGENT_CARD_2),
        )

    def test_first_keeps_first_card(self):
        cards = _parse_registry_response(self._body_with_duplicate_ext(), on_duplicate="first")
        assert cards["shared-ext"].name == "Test Agent"
        # Both deployment IDs are still indexed independently
        assert "dep-1" in cards
        assert "dep-2" in cards

    def test_last_keeps_last_card(self):
        cards = _parse_registry_response(self._body_with_duplicate_ext(), on_duplicate="last")
        assert cards["shared-ext"].name == "Second Agent"

    def test_error_raises_on_duplicate(self):
        with pytest.raises(AgentCardRegistryError, match="Multiple agent cards found"):
            _parse_registry_response(self._body_with_duplicate_ext(), on_duplicate="error")

    def test_no_duplicate_no_error(self):
        """When external IDs are unique, 'error' strategy passes."""
        body = _registry_response(
            _entry(dep_id="dep-1", ext_id="ext-1"),
            _entry(dep_id="dep-2", ext_id="ext-2"),
        )
        cards = _parse_registry_response(body, on_duplicate="error")
        assert "ext-1" in cards
        assert "ext-2" in cards

    def test_default_strategy_is_first(self):
        cards = _parse_registry_response(self._body_with_duplicate_ext())
        assert cards["shared-ext"].name == "Test Agent"


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
        cards = _parse_registry_response(self._batch_response(), on_duplicate="first")
        # agent-A → first card (Test Agent)
        assert cards["agent-A"].name == "Test Agent"
        # agent-B has no duplicates → stored as-is
        assert cards["agent-B"].name == "Second Agent"
        # All deployment IDs are independently indexed
        assert len({cards[k].name for k in ["dep-a1", "dep-a2", "dep-a3", "dep-b1"]}) == 3

    def test_last_picks_last_of_three(self):
        cards = _parse_registry_response(self._batch_response(), on_duplicate="last")
        # agent-A → last card (Third Agent)
        assert cards["agent-A"].name == "Third Agent"
        # agent-B unaffected
        assert cards["agent-B"].name == "Second Agent"

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
        mock_fetch.return_value = {"dep-1": MagicMock(name="card1")}
        registry = AgentCardRegistry(api_token="tok", endpoint="https://ep", cache_ttl=3600)
        card = await registry.get(deployment_id="dep-1")
        assert card is mock_fetch.return_value["dep-1"]
        mock_fetch.assert_awaited_once_with({"deploymentIds": "dep-1"})

    async def test_get_single_external_id(self, mock_fetch):
        mock_fetch.return_value = {"ext-1": MagicMock(name="card1")}
        registry = AgentCardRegistry(api_token="tok", endpoint="https://ep", cache_ttl=3600)
        card = await registry.get(external_id="ext-1")
        assert card is mock_fetch.return_value["ext-1"]
        mock_fetch.assert_awaited_once_with({"externalIds": "ext-1"})

    async def test_get_raises_when_both_ids(self):
        registry = AgentCardRegistry(api_token="tok", endpoint="https://ep", cache_ttl=3600)
        with pytest.raises(AgentCardRegistryError, match="exactly one"):
            await registry.get(deployment_id="d", external_id="e")

    async def test_get_raises_when_neither_id(self):
        registry = AgentCardRegistry(api_token="tok", endpoint="https://ep", cache_ttl=3600)
        with pytest.raises(AgentCardRegistryError, match="exactly one"):
            await registry.get()

    async def test_get_raises_when_not_found(self, mock_fetch):
        mock_fetch.return_value = {}
        registry = AgentCardRegistry(api_token="tok", endpoint="https://ep", cache_ttl=3600)
        with pytest.raises(AgentCardRegistryError, match="No agent card found"):
            await registry.get(deployment_id="missing")

    async def test_get_uses_cache(self, mock_fetch):
        mock_fetch.return_value = {"dep-1": MagicMock(name="card1")}
        registry = AgentCardRegistry(api_token="tok", endpoint="https://ep", cache_ttl=3600)

        card1 = await registry.get(deployment_id="dep-1")
        card2 = await registry.get(deployment_id="dep-1")

        assert card1 is card2
        mock_fetch.assert_awaited_once()

    async def test_cache_ttl_zero_always_refetches(self, mock_fetch):
        """cache_ttl=0 means every get() triggers a fresh fetch."""
        card_mock = MagicMock(name="card1")
        mock_fetch.return_value = {"dep-1": card_mock}
        registry = AgentCardRegistry(api_token="tok", endpoint="https://ep", cache_ttl=0)

        await registry.get(deployment_id="dep-1")
        await registry.get(deployment_id="dep-1")

        assert mock_fetch.await_count == 2

    async def test_expired_cache_triggers_refetch(self, mock_fetch):
        mock_fetch.return_value = {"dep-1": MagicMock(name="card1")}
        registry = AgentCardRegistry(api_token="tok", endpoint="https://ep", cache_ttl=60)

        await registry.get(deployment_id="dep-1")
        # Simulate expiry
        registry._cache["dep-1"].fetched_at -= 120

        mock_fetch.return_value = {"dep-1": MagicMock(name="card1-refreshed")}
        await registry.get(deployment_id="dep-1")
        assert mock_fetch.await_count == 2


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
            {"dep-1": MagicMock(name="c1"), "dep-2": MagicMock(name="c2")},
            {"ext-1": MagicMock(name="c3")},
        ]
        registry = AgentCardRegistry(api_token="tok", endpoint="https://ep", cache_ttl=3600)
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
        mock_fetch.return_value = {"dep-1": MagicMock(name="c1")}
        registry = AgentCardRegistry(api_token="tok", endpoint="https://ep", cache_ttl=3600)
        registry.register(deployment_id="dep-1")
        registry.register(deployment_id="dep-1")
        registry.register(deployment_id="dep-1")

        await registry.get(deployment_id="dep-1")
        mock_fetch.assert_awaited_once_with({"deploymentIds": "dep-1"})

    async def test_pending_cleared_after_flush(self, mock_fetch):
        mock_fetch.return_value = {"dep-1": MagicMock(name="c1")}
        registry = AgentCardRegistry(api_token="tok", endpoint="https://ep", cache_ttl=3600)
        registry.register(deployment_id="dep-1")

        await registry.get(deployment_id="dep-1")
        assert len(registry._pending_deployment_ids) == 0
        assert len(registry._pending_external_ids) == 0


# ---------------------------------------------------------------------------
# Tests: AgentCardRegistry — prefetch
# ---------------------------------------------------------------------------


class TestAgentCardRegistryPrefetch:
    @pytest.fixture
    def mock_fetch(self):
        with patch.object(AgentCardRegistry, "_fetch", new_callable=AsyncMock) as m:
            yield m

    async def test_prefetch_deployment_ids(self, mock_fetch):
        mock_fetch.return_value = {
            "dep-1": MagicMock(name="card1"),
            "dep-2": MagicMock(name="card2"),
        }
        registry = AgentCardRegistry(api_token="tok", endpoint="https://ep", cache_ttl=3600)
        await registry.prefetch(deployment_ids=["dep-1", "dep-2"])

        mock_fetch.assert_awaited_once_with({"deploymentIds": "dep-1,dep-2"})
        mock_fetch.reset_mock()
        await registry.get(deployment_id="dep-1")
        await registry.get(deployment_id="dep-2")
        mock_fetch.assert_not_awaited()

    async def test_prefetch_mixed_issues_separate_calls(self, mock_fetch):
        mock_fetch.side_effect = [
            {"dep-1": MagicMock(name="card1")},
            {"ext-1": MagicMock(name="card2")},
        ]
        registry = AgentCardRegistry(api_token="tok", endpoint="https://ep", cache_ttl=3600)
        await registry.prefetch(deployment_ids=["dep-1"], external_ids=["ext-1"])

        assert mock_fetch.await_count == 2
        calls = mock_fetch.call_args_list
        assert calls[0].args[0] == {"deploymentIds": "dep-1"}
        assert calls[1].args[0] == {"externalIds": "ext-1"}

    async def test_prefetch_skips_already_cached(self, mock_fetch):
        mock_fetch.return_value = {"dep-1": MagicMock(name="card1")}
        registry = AgentCardRegistry(api_token="tok", endpoint="https://ep", cache_ttl=3600)

        await registry.prefetch(deployment_ids=["dep-1"])
        mock_fetch.reset_mock()

        mock_fetch.return_value = {"dep-2": MagicMock(name="card2")}
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
            registry = AgentCardRegistry(
                api_token="my-tok", endpoint="https://app.dr.com/api/v2", cache_ttl=3600
            )
            cards = await registry._fetch({"deploymentIds": "dep-1,dep-2"})

        assert "dep-1" in cards
        assert "dep-2" in cards
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
            registry = AgentCardRegistry(api_token="tok", endpoint="https://ep", cache_ttl=3600)
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
        mock_client.get = AsyncMock(
            side_effect=[page1_response, page2_response, page3_response]
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch(f"{_MODULE}.httpx.AsyncClient", return_value=mock_client):
            registry = AgentCardRegistry(
                api_token="my-tok", endpoint="https://app.dr.com/api/v2", cache_ttl=3600
            )
            cards = await registry._fetch({"deploymentIds": "dep-1,dep-2,dep-3"})

        assert "dep-1" in cards
        assert "dep-2" in cards
        assert "dep-3" in cards
        assert len(cards) == 3
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
            registry = AgentCardRegistry(
                api_token="tok", endpoint="https://ep", cache_ttl=3600
            )
            cards = await registry._fetch({"deploymentIds": "dep-1"})

        assert "dep-1" in cards
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
            registry = AgentCardRegistry(api_token="tok", endpoint="https://ep", cache_ttl=3600)
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
            registry = AgentCardRegistry(api_token="tok", endpoint="https://ep", cache_ttl=3600)
            cards = await registry._fetch({"deploymentIds": "dep-0"})

        # Should have fetched exactly _MAX_PAGES (stopped at the safety limit)
        assert mock_client.get.await_count == _MAX_PAGES
        assert len(cards) == _MAX_PAGES


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
