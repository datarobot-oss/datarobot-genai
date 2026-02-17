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

from datarobot_genai.drmcp.tools.dr_docs.tools import fetch_datarobot_doc_page
from datarobot_genai.drmcp.tools.dr_docs.tools import search_datarobot_docs


class TestSearchDatarobotDocs:
    """Tests for search_datarobot_docs MCP tool."""

    async def test_search_returns_results(self) -> None:
        """Test that search returns structured results on success."""
        mock_results = [
            {
                "url": "https://docs.datarobot.com/en/docs/modeling/autopilot/",
                "title": "Autopilot",
                "description": "Automated machine learning",
            },
            {
                "url": "https://docs.datarobot.com/en/docs/mlops/deployment/",
                "title": "Deployment",
                "description": "Deploy models",
            },
        ]

        with patch(
            "datarobot_genai.drmcp.tools.dr_docs.tools.search_docs",
            new_callable=AsyncMock,
        ) as mock_search:
            mock_search.return_value = mock_results

            result = await search_datarobot_docs(query="autopilot", max_results=5)

            assert result.structured_content is not None
            content = result.structured_content
            assert content["status"] == "success"
            assert content["query"] == "autopilot"
            assert content["total_results"] == 2
            assert content["result_0_title"] == "Autopilot"
            assert (
                content["result_0_url"] == "https://docs.datarobot.com/en/docs/modeling/autopilot/"
            )
            assert content["result_1_title"] == "Deployment"

    async def test_search_no_results(self) -> None:
        """Test that search returns no_results status when no matches found."""
        with patch(
            "datarobot_genai.drmcp.tools.dr_docs.tools.search_docs",
            new_callable=AsyncMock,
        ) as mock_search:
            mock_search.return_value = []

            result = await search_datarobot_docs(query="nonexistent", max_results=5)

            assert result.structured_content is not None
            content = result.structured_content
            assert content["status"] == "no_results"
            assert content["query"] == "nonexistent"
            assert "No documentation pages found" in content["message"]

    async def test_search_includes_descriptions(self) -> None:
        """Test that search includes descriptions when present."""
        mock_results = [
            {
                "url": "https://docs.datarobot.com/en/docs/test/",
                "title": "Test Page",
                "description": "This is a test description",
            }
        ]

        with patch(
            "datarobot_genai.drmcp.tools.dr_docs.tools.search_docs",
            new_callable=AsyncMock,
        ) as mock_search:
            mock_search.return_value = mock_results

            result = await search_datarobot_docs(query="test", max_results=5)

            content = result.structured_content
            assert content["result_0_description"] == "This is a test description"

    async def test_search_omits_empty_descriptions(self) -> None:
        """Test that empty descriptions are omitted from results."""
        mock_results = [
            {
                "url": "https://docs.datarobot.com/en/docs/test/",
                "title": "Test Page",
                "description": "",
            }
        ]

        with patch(
            "datarobot_genai.drmcp.tools.dr_docs.tools.search_docs",
            new_callable=AsyncMock,
        ) as mock_search:
            mock_search.return_value = mock_results

            result = await search_datarobot_docs(query="test", max_results=5)

            content = result.structured_content
            # Empty description should not be included
            assert "result_0_description" not in content


class TestFetchDatarobotDocPage:
    """Tests for fetch_datarobot_doc_page MCP tool."""

    async def test_fetch_returns_page_content(self) -> None:
        """Test that fetch returns page content on success."""
        mock_content = {
            "url": "https://docs.datarobot.com/en/docs/modeling/autopilot/",
            "title": "Autopilot Overview",
            "content": "Autopilot is an automated machine learning feature that...",
        }

        with patch(
            "datarobot_genai.drmcp.tools.dr_docs.tools.fetch_page_content",
            new_callable=AsyncMock,
        ) as mock_fetch:
            mock_fetch.return_value = mock_content

            result = await fetch_datarobot_doc_page(
                url="https://docs.datarobot.com/en/docs/modeling/autopilot/"
            )

            assert result.structured_content is not None
            content = result.structured_content
            assert content["url"] == "https://docs.datarobot.com/en/docs/modeling/autopilot/"
            assert content["title"] == "Autopilot Overview"
            assert "automated machine learning" in content["content"]

    async def test_fetch_returns_error_for_invalid_url(self) -> None:
        """Test that fetch returns error content for invalid URLs."""
        mock_error = {
            "url": "https://example.com/not-docs/",
            "title": "Error",
            "content": "URL must be a DataRobot documentation page",
        }

        with patch(
            "datarobot_genai.drmcp.tools.dr_docs.tools.fetch_page_content",
            new_callable=AsyncMock,
        ) as mock_fetch:
            mock_fetch.return_value = mock_error

            result = await fetch_datarobot_doc_page(url="https://example.com/not-docs/")

            content = result.structured_content
            assert content["title"] == "Error"
            assert "must be a DataRobot documentation page" in content["content"]

    async def test_fetch_returns_error_on_failure(self) -> None:
        """Test that fetch returns error content when fetch fails."""
        mock_error = {
            "url": "https://docs.datarobot.com/en/docs/test/",
            "title": "Error",
            "content": "Failed to fetch content from https://docs.datarobot.com/en/docs/test/",
        }

        with patch(
            "datarobot_genai.drmcp.tools.dr_docs.tools.fetch_page_content",
            new_callable=AsyncMock,
        ) as mock_fetch:
            mock_fetch.return_value = mock_error

            result = await fetch_datarobot_doc_page(url="https://docs.datarobot.com/en/docs/test/")

            content = result.structured_content
            assert content["title"] == "Error"
            assert "Failed to fetch content" in content["content"]
