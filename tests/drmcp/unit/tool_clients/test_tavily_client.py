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
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from fastmcp.exceptions import ToolError

from datarobot_genai.drmcp.tools.clients.tavily import TavilyClient
from datarobot_genai.drmcp.tools.clients.tavily import TavilyImage
from datarobot_genai.drmcp.tools.clients.tavily import TavilySearchResult
from datarobot_genai.drmcp.tools.clients.tavily import get_tavily_access_token


class TestGetTavilyAccessToken:
    """Tests for get_tavily_access_token function."""

    @pytest.mark.asyncio
    async def test_returns_api_key_from_headers(self) -> None:
        """Test getting API key from HTTP headers."""
        with patch(
            "datarobot_genai.drmcp.tools.clients.tavily.get_http_headers",
            return_value={"x-tavily-api-key": "test-api-key-123"},
        ):
            result = await get_tavily_access_token()
            assert result == "test-api-key-123"

    @pytest.mark.asyncio
    async def test_raises_error_when_missing(self) -> None:
        """Test that missing API key raises ToolError."""
        with patch(
            "datarobot_genai.drmcp.tools.clients.tavily.get_http_headers",
            return_value={},
        ):
            with pytest.raises(ToolError, match="Tavily API key not found"):
                await get_tavily_access_token()


class TestTavilyModels:
    """Tests for Tavily data models."""

    def test_search_result_from_sdk(self) -> None:
        """Test TavilySearchResult.from_tavily_sdk."""
        result = TavilySearchResult.from_tavily_sdk(
            {
                "title": "Test",
                "url": "https://example.com",
                "content": "Content",
                "score": 0.95,
            }
        )
        assert result.title == "Test"
        assert result.score == pytest.approx(0.95)

    def test_image_from_string(self) -> None:
        """Test TavilyImage.from_tavily_sdk with string URL."""
        result = TavilyImage.from_tavily_sdk("https://example.com/img.jpg")
        assert result.url == "https://example.com/img.jpg"
        assert result.description is None

    def test_image_from_dict(self) -> None:
        """Test TavilyImage.from_tavily_sdk with dict."""
        result = TavilyImage.from_tavily_sdk(
            {
                "url": "https://example.com/img.jpg",
                "description": "A test image",
            }
        )
        assert result.description == "A test image"


class TestTavilyClient:
    """Tests for TavilyClient class."""

    @pytest.mark.asyncio
    async def test_search_calls_sdk(self) -> None:
        """Test that search calls the underlying SDK."""
        mock_response = {"query": "test", "results": [], "response_time": 0.5}

        with patch("datarobot_genai.drmcp.tools.clients.tavily.AsyncTavilyClient") as mock_class:
            mock_client = MagicMock()
            mock_client.search = AsyncMock(return_value=mock_response)
            mock_class.return_value = mock_client

            async with TavilyClient("api-key") as client:
                result = await client.search("test query")

            assert result == mock_response

    @pytest.mark.asyncio
    async def test_search_validates_empty_query(self) -> None:
        """Test that empty query raises ValueError."""
        async with TavilyClient("api-key") as client:
            with pytest.raises(ValueError, match="query cannot be empty"):
                await client.search("")

    @pytest.mark.asyncio
    async def test_search_validates_max_results(self) -> None:
        """Test that invalid max_results raises ValueError."""
        async with TavilyClient("api-key") as client:
            with pytest.raises(ValueError, match="max_results"):
                await client.search("test", max_results=0)
