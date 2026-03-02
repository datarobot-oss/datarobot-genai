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

from collections.abc import Iterator
from unittest.mock import patch

import pytest

from datarobot_genai.drtools.clients.tavily import TavilyCrawlResults
from datarobot_genai.drtools.clients.tavily import TavilyMapResults
from datarobot_genai.drtools.clients.tavily import TavilySearchResults
from datarobot_genai.drtools.tavily.tools import tavily_crawl
from datarobot_genai.drtools.tavily.tools import tavily_map
from datarobot_genai.drtools.tavily.tools import tavily_search


@pytest.fixture
def mock_tavily_auth() -> Iterator[None]:
    with patch(
        "datarobot_genai.drtools.tavily.tools.get_tavily_access_token",
        return_value="test-api-key",
    ):
        yield


class TestTavilySearch:
    """Tests for tavily_search tool."""

    @pytest.mark.asyncio
    async def test_basic_search(self, mock_tavily_auth: None) -> None:
        """Test basic search returns expected structure."""
        mock_response = TavilySearchResults.from_tavily_sdk(
            {
                "results": [
                    {
                        "title": "Result 1",
                        "url": "https://example.com",
                        "content": "Content",
                        "score": 0.9,
                    },
                ],
                "response_time": 0.5,
            }
        )

        with patch("datarobot_genai.drtools.clients.tavily.TavilyClient.search") as mock:
            mock.return_value = mock_response
            result = await tavily_search(query="test query")

        _, structured = result.to_mcp_result()
        assert structured["resultCount"] == 1

    @pytest.mark.asyncio
    async def test_search_with_answer_and_images(self, mock_tavily_auth: None) -> None:
        """Test search with answer and images."""
        mock_response = TavilySearchResults.from_tavily_sdk(
            {
                "results": [{"title": "R1", "url": "https://x.com", "content": "C", "score": 0.9}],
                "answer": "AI summary",
                "images": [{"url": "https://img.com/1.jpg", "description": "Desc"}],
                "response_time": 1.0,
            }
        )

        with patch("datarobot_genai.drtools.clients.tavily.TavilyClient.search") as mock:
            mock.return_value = mock_response
            result = await tavily_search(query="test", include_images=True, include_answer=True)

        _, structured = result.to_mcp_result()
        assert structured["answer"] == "AI summary"
        assert len(structured["images"]) == 1


class TestTavilyMap:
    """Tests for tavily_map tool."""

    @pytest.mark.asyncio
    async def test_map_default(self, mock_tavily_auth: None) -> None:
        """Test map returns expected things."""
        mock_response = TavilyMapResults(
            **{
                "results": ["https://example.com/url1", "https://example.com/url2"],
            }
        )

        with patch("datarobot_genai.drtools.clients.tavily.TavilyClient.map_") as mock:
            mock.return_value = mock_response
            result = await tavily_map(url="https://example.com", include_usage=True)

        _, structured = result.to_mcp_result()
        assert structured["count"] == 2
        assert len(structured["results"]) == 2
        assert structured["usageCredits"] is None

    @pytest.mark.asyncio
    async def test_map_with_include_usage(self, mock_tavily_auth: None) -> None:
        """Test map returns expected things."""
        mock_response = TavilyMapResults(
            **{
                "results": ["https://example.com/url1", "https://example.com/url2"],
                "usage": {"credits": 4},
            }
        )

        with patch("datarobot_genai.drtools.clients.tavily.TavilyClient.map_") as mock:
            mock.return_value = mock_response
            result = await tavily_map(url="https://example.com", include_usage=True)

        _, structured = result.to_mcp_result()
        assert structured["count"] == 2
        assert len(structured["results"]) == 2
        assert structured["usageCredits"] == 4


class TestTavilyCrawl:
    """Tests for tavily_crawl tool."""

    @pytest.mark.asyncio
    async def test_basic_crawl(self, mock_tavily_auth: None) -> None:
        """Test basic crawl returns expected structure."""
        mock_response = TavilyCrawlResults.from_tavily_sdk(
            {
                "base_url": "https://example.com",
                "results": [
                    {
                        "url": "https://example.com/page1",
                        "raw_content": "# Page 1\n\nContent here.",
                    },
                ],
                "response_time": 2.5,
            }
        )

        with patch("datarobot_genai.drtools.clients.tavily.TavilyClient.crawl") as mock:
            mock.return_value = mock_response
            result = await tavily_crawl(url="https://example.com")

        _, structured = result.to_mcp_result()
        assert structured["baseUrl"] == "https://example.com"
        assert structured["resultCount"] == 1
        assert structured["results"][0]["url"] == "https://example.com/page1"
