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

from datarobot_genai.drtools.dr_docs.tools import fetch_datarobot_doc_page
from datarobot_genai.drtools.dr_docs.tools import search_datarobot_agentic_docs


class TestSearchDatarobotAgenticDocs:
    """Tests for search_datarobot_agentic_docs MCP tool."""

    async def test_search_returns_results(self) -> None:
        """Test that search returns structured results on success."""
        mock_results = [
            {
                "url": "https://docs.datarobot.com/en/docs/agentic-ai/index.html",
                "title": "Agentic AI Overview",
                "description": "Introduction to DataRobot agentic AI",
            },
            {
                "url": "https://docs.datarobot.com/en/docs/agentic-ai/agentic-glossary.html",
                "title": "Agentic AI Glossary",
                "description": "Definitions of agentic AI terms",
            },
        ]

        with patch(
            "datarobot_genai.drtools.dr_docs.local_tools.search_docs",
            new_callable=AsyncMock,
        ) as mock_search:
            mock_search.return_value = mock_results

            result = await search_datarobot_agentic_docs(query="agentic", max_results=5)

            assert result.structured_content is not None
            content = result.structured_content
            assert content["status"] == "success"
            assert content["query"] == "agentic"
            assert content["total_results"] == 2
            assert content["result_0_title"] == "Agentic AI Overview"
            assert content["result_0_description"] == "Introduction to DataRobot agentic AI"
            assert (
                content["result_0_url"]
                == "https://docs.datarobot.com/en/docs/agentic-ai/index.html"
            )
            assert content["result_1_title"] == "Agentic AI Glossary"
            assert content["result_1_description"] == "Definitions of agentic AI terms"
            assert (
                content["result_1_url"]
                == "https://docs.datarobot.com/en/docs/agentic-ai/agentic-glossary.html"
            )

    async def test_search_no_results(self) -> None:
        """Test that search returns no_results status when no matches found."""
        with patch(
            "datarobot_genai.drtools.dr_docs.local_tools.search_docs",
            new_callable=AsyncMock,
        ) as mock_search:
            mock_search.return_value = []

            result = await search_datarobot_agentic_docs(query="nonexistent", max_results=5)

            assert result.structured_content is not None
            content = result.structured_content
            assert content["status"] == "no_results"
            assert content["query"] == "nonexistent"
            assert "No documentation pages found" in content["message"]


class TestFetchDatarobotDocPage:
    """Tests for fetch_datarobot_doc_page MCP tool."""

    async def test_fetch_returns_page_content(self) -> None:
        """Test that fetch returns page content on success."""
        mock_content = {
            "url": "https://docs.datarobot.com/en/docs/agentic-ai/agentic-glossary.html",
            "title": "Agentic AI Glossary",
            "content": "Definitions of agentic AI terms used in DataRobot...",
        }

        with patch(
            "datarobot_genai.drtools.dr_docs.local_tools.fetch_page_content",
            new_callable=AsyncMock,
        ) as mock_fetch:
            mock_fetch.return_value = mock_content

            result = await fetch_datarobot_doc_page(
                url="https://docs.datarobot.com/en/docs/agentic-ai/agentic-glossary.html"
            )

            assert result.structured_content is not None
            content = result.structured_content
            assert (
                content["url"]
                == "https://docs.datarobot.com/en/docs/agentic-ai/agentic-glossary.html"
            )
            assert content["title"] == "Agentic AI Glossary"
            assert "agentic AI terms" in content["content"]

    async def test_fetch_returns_error_for_invalid_url(self) -> None:
        """Test that fetch returns error content for invalid URLs."""
        mock_error = {
            "url": "https://example.com/not-docs/",
            "title": "Error",
            "content": "URL must be a DataRobot documentation page",
        }

        with patch(
            "datarobot_genai.drtools.dr_docs.local_tools.fetch_page_content",
            new_callable=AsyncMock,
        ) as mock_fetch:
            mock_fetch.return_value = mock_error

            result = await fetch_datarobot_doc_page(url="https://example.com/not-docs/")

            content = result.structured_content
            assert content["title"] == "Error"
            assert "must be a DataRobot documentation page" in content["content"]

    async def test_fetch_returns_error_on_failure(self) -> None:
        """Test that fetch returns error content when fetch fails."""
        _url = "https://docs.datarobot.com/en/docs/agentic-ai/index.html"
        mock_error = {
            "url": _url,
            "title": "Error",
            "content": f"Failed to fetch content from {_url}",
        }

        with patch(
            "datarobot_genai.drtools.dr_docs.local_tools.fetch_page_content",
            new_callable=AsyncMock,
        ) as mock_fetch:
            mock_fetch.return_value = mock_error

            result = await fetch_datarobot_doc_page(url=_url)

            content = result.structured_content
            assert content["title"] == "Error"
            assert "Failed to fetch content" in content["content"]
