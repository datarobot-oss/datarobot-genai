# Copyright 2025 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for async Confluence client using httpx."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import httpx
import pytest

from datarobot_genai.drmcp.tools.clients.confluence import ConfluenceClient
from datarobot_genai.drmcp.tools.clients.confluence import ConfluencePage


class TestConfluencePageParsing:
    """Test ConfluenceClient._parse_response method."""

    @pytest.fixture
    def client(self) -> ConfluenceClient:
        """Create client for testing parse method."""
        return ConfluenceClient("test_token")

    def test_parse_full_response(self, client: ConfluenceClient) -> None:
        """Test parsing complete API response."""
        data = {
            "id": "12345",
            "title": "Test Page",
            "space": {"id": "67890", "key": "TEST"},
            "body": {"storage": {"value": "<p>Content</p>"}},
        }
        result = client._parse_response(data)

        assert isinstance(result, ConfluencePage)
        assert result.page_id == "12345"
        assert result.title == "Test Page"
        assert result.space_id == "67890"
        assert result.space_key == "TEST"
        assert result.body == "<p>Content</p>"


class TestConfluenceClientGetPageById:
    """Test ConfluenceClient.get_page_by_id method."""

    @pytest.fixture
    def mock_cloud_id(self) -> str:
        return "test-cloud-123"

    @pytest.mark.asyncio
    async def test_get_page_by_id_success(self, mock_cloud_id: str) -> None:
        """Test successful page retrieval by ID."""
        mock_page_data = {
            "id": "12345",
            "title": "Test Page",
            "space": {"id": "67890", "key": "TEST"},
            "body": {"storage": {"value": "<p>Content</p>"}},
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_page_data
        mock_response.raise_for_status = MagicMock()

        with patch(
            "datarobot_genai.drmcp.tools.clients.confluence.get_atlassian_cloud_id",
            new_callable=AsyncMock,
            return_value=mock_cloud_id,
        ):
            client = ConfluenceClient("test_token")
            client._client = AsyncMock(spec=httpx.AsyncClient)
            client._client.get = AsyncMock(return_value=mock_response)

            result = await client.get_page_by_id("12345")

            assert result.page_id == "12345"
            assert result.title == "Test Page"
            client._client.get.assert_awaited_once()
            call_url = client._client.get.call_args[0][0]
            assert f"/ex/confluence/{mock_cloud_id}/wiki/rest/api/content/12345" in call_url

    @pytest.mark.asyncio
    async def test_get_page_by_id_not_found(self, mock_cloud_id: str) -> None:
        """Test 404 error handling."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch(
            "datarobot_genai.drmcp.tools.clients.confluence.get_atlassian_cloud_id",
            new_callable=AsyncMock,
            return_value=mock_cloud_id,
        ):
            client = ConfluenceClient("test_token")
            client._client = AsyncMock(spec=httpx.AsyncClient)
            client._client.get = AsyncMock(return_value=mock_response)

            with pytest.raises(ValueError, match="not found"):
                await client.get_page_by_id("nonexistent")


class TestConfluenceClientGetPageByTitle:
    """Test ConfluenceClient.get_page_by_title method."""

    @pytest.fixture
    def mock_cloud_id(self) -> str:
        return "test-cloud-123"

    @pytest.mark.asyncio
    async def test_get_page_by_title_success(self, mock_cloud_id: str) -> None:
        """Test successful page retrieval by title."""
        mock_page_data = {
            "id": "12345",
            "title": "Test Page",
            "space": {"id": "67890", "key": "TEST"},
            "body": {"storage": {"value": "<p>Content</p>"}},
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": [mock_page_data]}
        mock_response.raise_for_status = MagicMock()

        with patch(
            "datarobot_genai.drmcp.tools.clients.confluence.get_atlassian_cloud_id",
            new_callable=AsyncMock,
            return_value=mock_cloud_id,
        ):
            client = ConfluenceClient("test_token")
            client._client = AsyncMock(spec=httpx.AsyncClient)
            client._client.get = AsyncMock(return_value=mock_response)

            result = await client.get_page_by_title("Test Page", "TEST")

            assert result.page_id == "12345"
            assert result.title == "Test Page"
            # Verify params passed
            call_kwargs = client._client.get.call_args[1]
            assert call_kwargs["params"]["title"] == "Test Page"
            assert call_kwargs["params"]["spaceKey"] == "TEST"

    @pytest.mark.asyncio
    async def test_get_page_by_title_not_found(self, mock_cloud_id: str) -> None:
        """Test empty results handling."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status = MagicMock()

        with patch(
            "datarobot_genai.drmcp.tools.clients.confluence.get_atlassian_cloud_id",
            new_callable=AsyncMock,
            return_value=mock_cloud_id,
        ):
            client = ConfluenceClient("test_token")
            client._client = AsyncMock(spec=httpx.AsyncClient)
            client._client.get = AsyncMock(return_value=mock_response)

            with pytest.raises(ValueError, match="not found"):
                await client.get_page_by_title("Nonexistent", "TEST")
