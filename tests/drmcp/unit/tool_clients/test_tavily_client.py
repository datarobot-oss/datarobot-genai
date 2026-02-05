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
from datarobot_genai.drmcp.tools.clients.tavily import TavilyExtractResults
from datarobot_genai.drmcp.tools.clients.tavily import TavilyMapResults
from datarobot_genai.drmcp.tools.clients.tavily import TavilySearchResults
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


class TestTavilyClient:
    """Tests for TavilyClient class."""

    @pytest.mark.asyncio
    async def test_search_calls_sdk(self) -> None:
        """Test that search calls the underlying SDK."""
        mock_response = {"results": [], "response_time": 0.5}

        with patch("datarobot_genai.drmcp.tools.clients.tavily.AsyncTavilyClient") as mock_class:
            mock_client = MagicMock()
            mock_client.search = AsyncMock(return_value=mock_response)
            mock_class.return_value = mock_client

            async with TavilyClient("api-key") as client:
                result = await client.search("test query")

            assert result == TavilySearchResults(**mock_response)

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


class TestTavilyClientExtract:
    """Tests for TavilyClient.extract method."""

    @pytest.mark.asyncio
    async def test_extract_calls_sdk(self) -> None:
        """Test that extract calls the underlying SDK."""
        mock_response = {
            "results": [{"url": "https://example.com", "raw_content": "Content"}],
            "response_time": 1.5,
        }

        with patch("datarobot_genai.drmcp.tools.clients.tavily.AsyncTavilyClient") as mock_class:
            mock_client = MagicMock()
            mock_client.extract = AsyncMock(return_value=mock_response)
            mock_class.return_value = mock_client

            async with TavilyClient("api-key") as client:
                result = await client.extract("https://example.com")

            assert result == TavilyExtractResults(**mock_response)
            mock_client.extract.assert_called_once()

    @pytest.mark.asyncio
    async def test_extract_with_single_url(self) -> None:
        """Test extract converts single URL to list."""
        mock_response = {"results": [], "response_time": 0.5}

        with patch("datarobot_genai.drmcp.tools.clients.tavily.AsyncTavilyClient") as mock_class:
            mock_client = MagicMock()
            mock_client.extract = AsyncMock(return_value=mock_response)
            mock_class.return_value = mock_client

            async with TavilyClient("api-key") as client:
                await client.extract("https://example.com")

            call_kwargs = mock_client.extract.call_args[1]
            assert call_kwargs["urls"] == ["https://example.com"]

    @pytest.mark.asyncio
    async def test_extract_with_query_includes_chunks_per_source(self) -> None:
        """Test extract includes chunks_per_source when query is provided."""
        mock_response = {"results": [], "response_time": 0.5}

        with patch("datarobot_genai.drmcp.tools.clients.tavily.AsyncTavilyClient") as mock_class:
            mock_client = MagicMock()
            mock_client.extract = AsyncMock(return_value=mock_response)
            mock_class.return_value = mock_client

            async with TavilyClient("api-key") as client:
                await client.extract(
                    ["https://example.com"], query="auth setup", chunks_per_source=5
                )

            call_kwargs = mock_client.extract.call_args[1]
            assert call_kwargs["query"] == "auth setup"
            assert call_kwargs["chunks_per_source"] == 5

    @pytest.mark.asyncio
    async def test_extract_without_query_excludes_chunks_per_source(self) -> None:
        """Test extract excludes chunks_per_source when query is not provided."""
        mock_response = {"results": [], "response_time": 0.5}

        with patch("datarobot_genai.drmcp.tools.clients.tavily.AsyncTavilyClient") as mock_class:
            mock_client = MagicMock()
            mock_client.extract = AsyncMock(return_value=mock_response)
            mock_class.return_value = mock_client

            async with TavilyClient("api-key") as client:
                await client.extract(["https://example.com"])

            call_kwargs = mock_client.extract.call_args[1]
            assert "query" not in call_kwargs
            assert "chunks_per_source" not in call_kwargs

    @pytest.mark.asyncio
    async def test_extract_validates_empty_urls(self) -> None:
        """Test that empty urls raises ValueError."""
        async with TavilyClient("api-key") as client:
            with pytest.raises(ValueError, match="urls cannot be empty"):
                await client.extract([])

    @pytest.mark.asyncio
    async def test_extract_validates_max_urls(self) -> None:
        """Test that too many urls raises ValueError."""
        urls = [f"https://example{i}.com" for i in range(25)]
        async with TavilyClient("api-key") as client:
            with pytest.raises(ValueError, match="Maximum number of URLs is 20"):
                await client.extract(urls)

    @pytest.mark.asyncio
    async def test_extract_validates_chunks_per_source_zero(self) -> None:
        """Test that zero chunks_per_source raises ValueError."""
        async with TavilyClient("api-key") as client:
            with pytest.raises(ValueError, match="chunks_per_source must be greater than 0"):
                await client.extract("https://example.com", chunks_per_source=0)

    @pytest.mark.asyncio
    async def test_extract_validates_chunks_per_source_max(self) -> None:
        """Test that too high chunks_per_source raises ValueError."""
        async with TavilyClient("api-key") as client:
            with pytest.raises(ValueError, match="chunks_per_source must be smaller than or equal"):
                await client.extract("https://example.com", chunks_per_source=10)

    @pytest.mark.asyncio
    async def test_extract_validates_timeout_too_low(self) -> None:
        """Test that timeout below 1.0 raises ValueError."""
        async with TavilyClient("api-key") as client:
            with pytest.raises(ValueError, match="timeout must be between"):
                await client.extract("https://example.com", timeout=0.5)

    @pytest.mark.asyncio
    async def test_extract_validates_timeout_too_high(self) -> None:
        """Test that timeout above 60.0 raises ValueError."""
        async with TavilyClient("api-key") as client:
            with pytest.raises(ValueError, match="timeout must be between"):
                await client.extract("https://example.com", timeout=120.0)


class TestTavilyClientMap:
    """Tests for TavilyClient.map_ method."""

    @pytest.mark.asyncio
    async def test_map_calls_sdk(self) -> None:
        """Test that map calls the underlying SDK."""
        mock_response = {
            "results": ["https://example.com/url1", "https://example.com/url2"],
            "usage": {"credits": 4},
        }

        with patch("datarobot_genai.drmcp.tools.clients.tavily.AsyncTavilyClient") as mock_class:
            mock_client = MagicMock()
            mock_client.map = AsyncMock(return_value=mock_response)
            mock_class.return_value = mock_client

            async with TavilyClient("api-key") as client:
                result = await client.map_(url="https://example.com", include_usage=True)

            assert result == TavilyMapResults.from_tavily_sdk(mock_response)
            mock_client.map.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "function_kwargs,error_message",
        [
            ({"url": ""}, "url.*cannot be empty"),
            ({"url": "  "}, "url.*cannot be empty"),
            ({"url": "https://example.com", "limit": -1}, "limit must be greater than 0"),
            ({"url": "https://example.com", "limit": 201}, "limit must be less than 200"),
        ],
    )
    @pytest.mark.asyncio
    async def test_map_input_validations(self, function_kwargs: dict, error_message: str) -> None:
        """Test that map calls the underlying SDK."""
        with pytest.raises(ValueError, match=error_message):
            async with TavilyClient("api-key") as client:
                await client.map_(**function_kwargs)
