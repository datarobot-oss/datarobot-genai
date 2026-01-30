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

from datarobot_genai.drmcp.tools.tavily.tools import tavily_search


@pytest.fixture
def mock_tavily_auth() -> Iterator[None]:
    with patch(
        "datarobot_genai.drmcp.tools.tavily.tools.get_tavily_access_token",
        return_value="test-api-key",
    ):
        yield


class TestTavilySearch:
    """Tests for tavily_search tool."""

    @pytest.mark.asyncio
    async def test_basic_search(self, mock_tavily_auth: None) -> None:
        """Test basic search returns expected structure."""
        mock_response = {
            "query": "test query",
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

        with patch("datarobot_genai.drmcp.tools.clients.tavily.TavilyClient.search") as mock:
            mock.return_value = mock_response
            result = await tavily_search(query="test query")

        _, structured = result.to_mcp_result()
        assert structured["resultCount"] == 1

    @pytest.mark.asyncio
    async def test_search_with_answer_and_images(self, mock_tavily_auth: None) -> None:
        """Test search with answer and images."""
        mock_response = {
            "query": "test",
            "results": [{"title": "R1", "url": "https://x.com", "content": "C", "score": 0.9}],
            "answer": "AI summary",
            "images": [{"url": "https://img.com/1.jpg", "description": "Desc"}],
            "response_time": 1.0,
        }

        with patch("datarobot_genai.drmcp.tools.clients.tavily.TavilyClient.search") as mock:
            mock.return_value = mock_response
            result = await tavily_search(query="test", include_images=True, include_answer=True)

        _, structured = result.to_mcp_result()
        assert structured["answer"] == "AI summary"
        assert len(structured["images"]) == 1
