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

from unittest.mock import AsyncMock
from unittest.mock import patch

import httpx
import pytest

from datarobot_genai.drmcp.tools.clients.confluence import ConfluenceClient
from datarobot_genai.drmcp.tools.clients.confluence import ConfluencePage


def make_response(status_code: int, json_data: dict, cloud_id: str) -> httpx.Response:
    """Create a mock httpx.Response with a request attached."""
    request = httpx.Request("GET", f"https://api.atlassian.com/ex/confluence/{cloud_id}")
    return httpx.Response(status_code, json=json_data, request=request)


class TestConfluenceClient:
    """Test ConfluenceClient class."""

    @pytest.fixture
    def mock_access_token(self) -> str:
        """Mock access token."""
        return "test_access_token_456"

    @pytest.fixture
    def mock_cloud_id(self) -> str:
        """Mock cloud ID."""
        return "test-cloud-id-123"

    @pytest.fixture
    def mock_page_response(self) -> dict:
        """Mock Confluence REST API page response."""
        return {
            "id": "12345",
            "title": "Test Page",
            "space": {"id": "67890", "key": "TEST"},
            "body": {"storage": {"value": "<p>Content</p>"}},
        }

    @pytest.mark.asyncio
    async def test_get_page_by_id_success(
        self, mock_access_token: str, mock_cloud_id: str, mock_page_response: dict
    ) -> None:
        """Test successfully getting a page by ID."""
        with patch(
            "datarobot_genai.drmcp.tools.clients.confluence.get_atlassian_cloud_id",
            new_callable=AsyncMock,
            return_value=mock_cloud_id,
        ):
            async with ConfluenceClient(mock_access_token) as client:

                async def mock_get(_url: str, **_kwargs: dict) -> httpx.Response:
                    return make_response(200, mock_page_response, mock_cloud_id)

                client._client.get = mock_get

                result = await client.get_page_by_id("12345")

                assert result.page_id == "12345"
                assert result.title == "Test Page"
                assert result.space_key == "TEST"
                assert result.body == "<p>Content</p>"

    @pytest.mark.asyncio
    async def test_get_page_by_id_not_found(
        self, mock_access_token: str, mock_cloud_id: str
    ) -> None:
        """Test getting a page that doesn't exist."""
        with patch(
            "datarobot_genai.drmcp.tools.clients.confluence.get_atlassian_cloud_id",
            new_callable=AsyncMock,
            return_value=mock_cloud_id,
        ):
            async with ConfluenceClient(mock_access_token) as client:

                async def mock_get(_url: str, **_kwargs: dict) -> httpx.Response:
                    return make_response(404, {"message": "Not found"}, mock_cloud_id)

                client._client.get = mock_get

                with pytest.raises(ValueError, match="not found"):
                    await client.get_page_by_id("nonexistent")

    @pytest.mark.asyncio
    async def test_get_page_by_title_success(
        self, mock_access_token: str, mock_cloud_id: str, mock_page_response: dict
    ) -> None:
        """Test successfully getting a page by title."""
        with patch(
            "datarobot_genai.drmcp.tools.clients.confluence.get_atlassian_cloud_id",
            new_callable=AsyncMock,
            return_value=mock_cloud_id,
        ):
            async with ConfluenceClient(mock_access_token) as client:

                async def mock_get(_url: str, **_kwargs: dict) -> httpx.Response:
                    return make_response(200, {"results": [mock_page_response]}, mock_cloud_id)

                client._client.get = mock_get

                result = await client.get_page_by_title("Test Page", "TEST")

                assert result.page_id == "12345"
                assert result.title == "Test Page"

    @pytest.mark.asyncio
    async def test_get_page_by_title_not_found(
        self, mock_access_token: str, mock_cloud_id: str
    ) -> None:
        """Test getting a page by title that doesn't exist."""
        with patch(
            "datarobot_genai.drmcp.tools.clients.confluence.get_atlassian_cloud_id",
            new_callable=AsyncMock,
            return_value=mock_cloud_id,
        ):
            async with ConfluenceClient(mock_access_token) as client:

                async def mock_get(_url: str, **_kwargs: dict) -> httpx.Response:
                    return make_response(200, {"results": []}, mock_cloud_id)

                client._client.get = mock_get

                with pytest.raises(ValueError, match="not found"):
                    await client.get_page_by_title("Nonexistent", "TEST")

    def test_as_flat_dict(self) -> None:
        """Test ConfluencePage.as_flat_dict method."""
        page = ConfluencePage(
            page_id="12345",
            title="Test Page",
            space_id="67890",
            space_key="TEST",
            body="<p>Content</p>",
        )

        result = page.as_flat_dict()

        assert result == {
            "page_id": "12345",
            "title": "Test Page",
            "space_id": "67890",
            "space_key": "TEST",
            "body": "<p>Content</p>",
        }
