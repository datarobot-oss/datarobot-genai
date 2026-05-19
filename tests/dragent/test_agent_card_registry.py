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
from datarobot_genai.dragent.agent_card_registry import AgentCardRegistryError
from datarobot_genai.dragent.agent_card_registry import DataRobotRegistrySettings
from datarobot_genai.dragent.agent_card_registry import _parse_registry_response
from datarobot_genai.dragent.agent_card_registry import _resolve_settings
from datarobot_genai.dragent.agent_card_registry import get_default_registry
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
        # Should not raise; just skips
        cards = _parse_registry_response(body)
        assert cards == {}


# ---------------------------------------------------------------------------
# Tests: AgentCardRegistry
# ---------------------------------------------------------------------------


class TestAgentCardRegistry:
    @pytest.fixture
    def mock_fetch(self):
        """Patch _fetch to avoid HTTP calls."""
        with patch.object(AgentCardRegistry, "_fetch", new_callable=AsyncMock) as m:
            yield m

    async def test_get_single_deployment_id(self, mock_fetch):
        mock_fetch.return_value = {"dep-1": MagicMock(name="card1")}
        registry = AgentCardRegistry(api_token="tok", endpoint="https://ep")
        card = await registry.get(deployment_id="dep-1")
        assert card is mock_fetch.return_value["dep-1"]
        mock_fetch.assert_awaited_once_with({"deploymentIds": "dep-1"})

    async def test_get_single_external_id(self, mock_fetch):
        mock_fetch.return_value = {"ext-1": MagicMock(name="card1")}
        registry = AgentCardRegistry(api_token="tok", endpoint="https://ep")
        card = await registry.get(external_id="ext-1")
        assert card is mock_fetch.return_value["ext-1"]
        mock_fetch.assert_awaited_once_with({"externalIds": "ext-1"})

    async def test_get_raises_when_both_ids(self):
        registry = AgentCardRegistry(api_token="tok", endpoint="https://ep")
        with pytest.raises(AgentCardRegistryError, match="exactly one"):
            await registry.get(deployment_id="d", external_id="e")

    async def test_get_raises_when_neither_id(self):
        registry = AgentCardRegistry(api_token="tok", endpoint="https://ep")
        with pytest.raises(AgentCardRegistryError, match="exactly one"):
            await registry.get()

    async def test_get_raises_when_not_found(self, mock_fetch):
        mock_fetch.return_value = {}  # Empty result
        registry = AgentCardRegistry(api_token="tok", endpoint="https://ep")
        with pytest.raises(AgentCardRegistryError, match="No agent card found"):
            await registry.get(deployment_id="missing")

    async def test_get_uses_cache(self, mock_fetch):
        mock_fetch.return_value = {"dep-1": MagicMock(name="card1")}
        registry = AgentCardRegistry(api_token="tok", endpoint="https://ep")

        card1 = await registry.get(deployment_id="dep-1")
        card2 = await registry.get(deployment_id="dep-1")

        assert card1 is card2
        # Only one HTTP call — second get() used cache
        mock_fetch.assert_awaited_once()

    async def test_prefetch_deployment_ids(self, mock_fetch):
        mock_fetch.return_value = {
            "dep-1": MagicMock(name="card1"),
            "dep-2": MagicMock(name="card2"),
        }
        registry = AgentCardRegistry(api_token="tok", endpoint="https://ep")
        await registry.prefetch(deployment_ids=["dep-1", "dep-2"])

        mock_fetch.assert_awaited_once_with({"deploymentIds": "dep-1,dep-2"})
        # Subsequent gets should not trigger fetch
        mock_fetch.reset_mock()
        await registry.get(deployment_id="dep-1")
        await registry.get(deployment_id="dep-2")
        mock_fetch.assert_not_awaited()

    async def test_prefetch_external_ids(self, mock_fetch):
        mock_fetch.return_value = {"ext-1": MagicMock(name="card1")}
        registry = AgentCardRegistry(api_token="tok", endpoint="https://ep")
        await registry.prefetch(external_ids=["ext-1"])

        mock_fetch.assert_awaited_once_with({"externalIds": "ext-1"})

    async def test_prefetch_mixed_issues_separate_calls(self, mock_fetch):
        """deployment_ids and external_ids must be separate HTTP calls."""
        mock_fetch.side_effect = [
            {"dep-1": MagicMock(name="card1")},
            {"ext-1": MagicMock(name="card2")},
        ]
        registry = AgentCardRegistry(api_token="tok", endpoint="https://ep")
        await registry.prefetch(deployment_ids=["dep-1"], external_ids=["ext-1"])

        assert mock_fetch.await_count == 2
        calls = mock_fetch.call_args_list
        assert calls[0].args[0] == {"deploymentIds": "dep-1"}
        assert calls[1].args[0] == {"externalIds": "ext-1"}

    async def test_prefetch_skips_already_cached(self, mock_fetch):
        mock_fetch.return_value = {"dep-1": MagicMock(name="card1")}
        registry = AgentCardRegistry(api_token="tok", endpoint="https://ep")

        await registry.prefetch(deployment_ids=["dep-1"])
        mock_fetch.reset_mock()

        # dep-1 already cached, only dep-2 should be fetched
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
            _entry(dep_id="dep-1"), _entry(dep_id="dep-2", card=_SAMPLE_AGENT_CARD_2),
        )
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        return mock_client

    async def test_fetch_passes_params_and_auth(self, mock_httpx_client):
        with patch(f"{_MODULE}.httpx.AsyncClient", return_value=mock_httpx_client):
            registry = AgentCardRegistry(api_token="my-tok", endpoint="https://app.dr.com/api/v2")
            cards = await registry._fetch({"deploymentIds": "dep-1,dep-2"})

        assert "dep-1" in cards
        assert "dep-2" in cards
        call_kwargs = mock_httpx_client.get.call_args.kwargs
        assert call_kwargs["params"] == {"deploymentIds": "dep-1,dep-2"}
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
            registry = AgentCardRegistry(api_token="tok", endpoint="https://ep")
            with pytest.raises(AgentCardRegistryError, match="HTTP 403"):
                await registry._fetch({"deploymentIds": "dep-1"})


# ---------------------------------------------------------------------------
# Tests: get_default_registry singleton
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
