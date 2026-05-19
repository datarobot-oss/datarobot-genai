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

import json
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import httpx
import pytest

from datarobot_genai.dragent.agent_card_registry import AgentCardRegistryError
from datarobot_genai.dragent.agent_card_registry import DataRobotRegistrySettings
from datarobot_genai.dragent.agent_card_registry import fetch_agent_card_from_registry

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

_REGISTRY_RESPONSE = {
    "data": [
        {
            "id": "abc123",
            "deploymentId": "dep-001",
            "externalId": "ext-001",
            "createdAt": "2026-01-01T00:00:00Z",
            "updatedAt": "2026-01-01T00:00:00Z",
            "agentCard": _SAMPLE_AGENT_CARD,
        }
    ],
    "count": 1,
    "totalCount": 1,
    "next": None,
    "previous": None,
}


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


class TestFetchAgentCardValidation:
    async def test_raises_when_neither_id_provided(self):
        with pytest.raises(AgentCardRegistryError, match="Either 'deployment_id' or 'external_id'"):
            await fetch_agent_card_from_registry()

    async def test_raises_when_both_ids_provided(self):
        with pytest.raises(AgentCardRegistryError, match="not both"):
            await fetch_agent_card_from_registry(
                deployment_id="dep-1", external_id="ext-1"
            )

    async def test_raises_when_no_api_token(self):
        mock_settings = MagicMock(spec=DataRobotRegistrySettings)
        mock_settings.datarobot_api_token = None
        mock_settings.datarobot_endpoint = None
        with patch(f"{_MODULE}.DataRobotRegistrySettings", return_value=mock_settings):
            with pytest.raises(AgentCardRegistryError, match="API token is required"):
                await fetch_agent_card_from_registry(deployment_id="dep-1")

    async def test_raises_when_no_endpoint(self):
        mock_settings = MagicMock(spec=DataRobotRegistrySettings)
        mock_settings.datarobot_api_token = "some-token"
        mock_settings.datarobot_endpoint = None
        with patch(f"{_MODULE}.DataRobotRegistrySettings", return_value=mock_settings):
            with pytest.raises(AgentCardRegistryError, match="API endpoint is required"):
                await fetch_agent_card_from_registry(deployment_id="dep-1")


# ---------------------------------------------------------------------------
# Successful lookups
# ---------------------------------------------------------------------------


class TestFetchAgentCardSuccess:
    @pytest.fixture
    def mock_httpx_client(self):
        """Provide a mock httpx.AsyncClient that returns a valid registry response."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = _REGISTRY_RESPONSE
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        return mock_client

    async def test_lookup_by_deployment_id(self, mock_httpx_client):
        with patch(f"{_MODULE}.httpx.AsyncClient", return_value=mock_httpx_client):
            card = await fetch_agent_card_from_registry(
                deployment_id="dep-001",
                api_token="test-token",
                endpoint="https://app.datarobot.com/api/v2",
            )

        assert card.name == "Test Agent"
        assert str(card.url) == "https://agent.example.com/a2a/"

        call_kwargs = mock_httpx_client.get.call_args
        assert call_kwargs.kwargs["params"]["deploymentIds"] == "dep-001"
        assert "Bearer test-token" in call_kwargs.kwargs["headers"]["Authorization"]

    async def test_lookup_by_external_id(self, mock_httpx_client):
        with patch(f"{_MODULE}.httpx.AsyncClient", return_value=mock_httpx_client):
            card = await fetch_agent_card_from_registry(
                external_id="ext-001",
                api_token="test-token",
                endpoint="https://app.datarobot.com/api/v2",
            )

        assert card.name == "Test Agent"
        call_kwargs = mock_httpx_client.get.call_args
        assert call_kwargs.kwargs["params"]["externalIds"] == "ext-001"

    async def test_uses_settings_when_not_explicit(self, mock_httpx_client):
        mock_settings = MagicMock(spec=DataRobotRegistrySettings)
        mock_settings.datarobot_api_token = "settings-token"
        mock_settings.datarobot_endpoint = "https://settings.datarobot.com/api/v2"
        with (
            patch(f"{_MODULE}.DataRobotRegistrySettings", return_value=mock_settings),
            patch(f"{_MODULE}.httpx.AsyncClient", return_value=mock_httpx_client),
        ):
            card = await fetch_agent_card_from_registry(deployment_id="dep-001")

        assert card.name == "Test Agent"
        call_kwargs = mock_httpx_client.get.call_args
        assert "Bearer settings-token" in call_kwargs.kwargs["headers"]["Authorization"]


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestFetchAgentCardErrors:
    async def test_empty_data_raises(self):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [], "count": 0, "totalCount": 0}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch(f"{_MODULE}.httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(AgentCardRegistryError, match="No agent card found"):
                await fetch_agent_card_from_registry(
                    deployment_id="missing",
                    api_token="tok",
                    endpoint="https://app.datarobot.com/api/v2",
                )

    async def test_http_error_raises(self):
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
            with pytest.raises(AgentCardRegistryError, match="HTTP 403"):
                await fetch_agent_card_from_registry(
                    deployment_id="dep-001",
                    api_token="tok",
                    endpoint="https://app.datarobot.com/api/v2",
                )

    async def test_missing_agent_card_payload_raises(self):
        response_data = {
            "data": [
                {
                    "id": "abc123",
                    "deploymentId": "dep-001",
                    "agentCard": None,
                }
            ],
            "count": 1,
        }
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = response_data
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch(f"{_MODULE}.httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(AgentCardRegistryError, match="contains no 'agentCard' payload"):
                await fetch_agent_card_from_registry(
                    deployment_id="dep-001",
                    api_token="tok",
                    endpoint="https://app.datarobot.com/api/v2",
                )





