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
from datarobot_genai.drmcp.tools.clients.tavily import TavilyExtractFailedResult
from datarobot_genai.drmcp.tools.clients.tavily import TavilyExtractResult
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


class TestTavilyExtractModels:
    """Tests for Tavily Extract data models."""

    def test_extract_result_from_sdk(self) -> None:
        """Test TavilyExtractResult.from_tavily_sdk."""
        result = TavilyExtractResult.from_tavily_sdk(
            {
                "url": "https://example.com",
                "raw_content": "# Example\n\nThis is content.",
                "favicon": "https://example.com/favicon.ico",
            }
        )
        assert result.url == "https://example.com"
        assert result.raw_content == "# Example\n\nThis is content."
        assert result.favicon == "https://example.com/favicon.ico"
        assert result.images is None

    def test_extract_result_from_sdk_with_images(self) -> None:
        """Test TavilyExtractResult.from_tavily_sdk with images."""
        result = TavilyExtractResult.from_tavily_sdk(
            {
                "url": "https://example.com",
                "raw_content": "Content",
                "images": [
                    {"url": "https://example.com/img1.jpg", "description": "Image 1"},
                    {"url": "https://example.com/img2.jpg"},
                ],
            }
        )
        assert result.images is not None
        assert len(result.images) == 2
        assert result.images[0].url == "https://example.com/img1.jpg"
        assert result.images[0].description == "Image 1"
        assert result.images[1].description is None

    def test_extract_result_as_flat_dict(self) -> None:
        """Test TavilyExtractResult.as_flat_dict."""
        result = TavilyExtractResult(
            url="https://example.com",
            raw_content="Content",
            favicon="https://example.com/favicon.ico",
        )
        flat = result.as_flat_dict()
        assert flat["url"] == "https://example.com"
        assert flat["raw_content"] == "Content"
        assert flat["favicon"] == "https://example.com/favicon.ico"
        assert "images" not in flat

    def test_extract_result_as_flat_dict_with_images(self) -> None:
        """Test TavilyExtractResult.as_flat_dict with images."""
        result = TavilyExtractResult(
            url="https://example.com",
            raw_content="Content",
            images=[TavilyImage(url="https://example.com/img.jpg", description="An image")],
        )
        flat = result.as_flat_dict()
        assert "images" in flat
        assert flat["images"] == [{"url": "https://example.com/img.jpg", "description": "An image"}]

    def test_failed_result_from_sdk(self) -> None:
        """Test TavilyExtractFailedResult.from_tavily_sdk."""
        result = TavilyExtractFailedResult.from_tavily_sdk(
            {
                "url": "https://example.com/404",
                "error": "Page not found",
            }
        )
        assert result.url == "https://example.com/404"
        assert result.error == "Page not found"

    def test_failed_result_from_sdk_default_error(self) -> None:
        """Test TavilyExtractFailedResult.from_tavily_sdk with missing error."""
        result = TavilyExtractFailedResult.from_tavily_sdk(
            {
                "url": "https://example.com/error",
            }
        )
        assert result.error == "Unknown error"

    def test_failed_result_as_flat_dict(self) -> None:
        """Test TavilyExtractFailedResult.as_flat_dict."""
        result = TavilyExtractFailedResult(
            url="https://example.com/404",
            error="Page not found",
        )
        flat = result.as_flat_dict()
        assert flat == {"url": "https://example.com/404", "error": "Page not found"}


class TestTavilyClientExtract:
    """Tests for TavilyClient.extract method."""

    @pytest.mark.asyncio
    async def test_extract_calls_sdk(self) -> None:
        """Test that extract calls the underlying SDK."""
        mock_response = {
            "results": [{"url": "https://example.com", "raw_content": "Content"}],
            "failed_results": [],
            "response_time": 1.5,
        }

        with patch("datarobot_genai.drmcp.tools.clients.tavily.AsyncTavilyClient") as mock_class:
            mock_client = MagicMock()
            mock_client.extract = AsyncMock(return_value=mock_response)
            mock_class.return_value = mock_client

            async with TavilyClient("api-key") as client:
                result = await client.extract("https://example.com")

            assert result == mock_response
            mock_client.extract.assert_called_once()

    @pytest.mark.asyncio
    async def test_extract_with_single_url(self) -> None:
        """Test extract converts single URL to list."""
        mock_response = {"results": [], "failed_results": [], "response_time": 0.5}

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
        mock_response = {"results": [], "failed_results": [], "response_time": 0.5}

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
        mock_response = {"results": [], "failed_results": [], "response_time": 0.5}

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
