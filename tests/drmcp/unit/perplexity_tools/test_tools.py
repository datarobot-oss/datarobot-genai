# Copyright 2026 DataRobot, Inc.
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
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest
from fastmcp.exceptions import ToolError

from datarobot_genai.drmcp.tools.clients.perplexity import PerplexityError
from datarobot_genai.drmcp.tools.clients.perplexity import PerplexitySearchResult
from datarobot_genai.drmcp.tools.clients.perplexity import PerplexityThinkResult
from datarobot_genai.drmcp.tools.perplexity.tools import perplexity_search
from datarobot_genai.drmcp.tools.perplexity.tools import perplexity_think


@pytest.fixture
def get_perplexity_access_token_mock() -> Iterator[None]:
    """Mock Perplexity access token retrieval."""
    with patch(
        "datarobot_genai.drmcp.tools.perplexity.tools.get_perplexity_access_token",
        return_value="test_token",
    ):
        yield


@pytest.fixture
def mock_perplexity_search_results() -> list[PerplexitySearchResult]:
    """Mock Perplexity Search Results."""
    return [
        PerplexitySearchResult(
            snippet="This website gives info about foo",
            title="Foo Website",
            url="https://foo.com",
        ),
        PerplexitySearchResult(
            snippet="This website gives info about bar",
            title="Bar Website",
            url="https://bar.com",
        ),
    ]


@pytest.fixture
def mock_perplexity_think_results() -> PerplexityThinkResult:
    """Mock Perplexity Think Results."""
    return PerplexityThinkResult(
        **{
            "usage": {
                "completion_tokens": 1000,
                "cost": {
                    "input_tokens_cost": 0.1,
                    "output_tokens_cost": 0.2,
                    "total_cost": 250,
                },
                "prompt_tokens": 500,
                "total_tokens": 1500,
            },
            "answer": "Dummy answer",
            "citations": ["citation1", "citation2"],
        }
    )


@pytest.fixture
def mock_client_search_success(
    mock_perplexity_search_results: list[PerplexitySearchResult],
) -> Iterator[AsyncMock]:
    """Mock successful client search."""
    with patch(
        "datarobot_genai.drmcp.tools.perplexity.tools.PerplexityClient"
    ) as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.search = AsyncMock(return_value=mock_perplexity_search_results)
        mock_client_class.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_client_think_success(
    mock_perplexity_think_results: PerplexityThinkResult,
) -> Iterator[AsyncMock]:
    """Mock successful client think."""
    with patch(
        "datarobot_genai.drmcp.tools.perplexity.tools.PerplexityClient"
    ) as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.think = AsyncMock(return_value=mock_perplexity_think_results)
        mock_client_class.return_value = mock_client
        yield mock_client


class TestPerplexitySearch:
    """Test perplexity_search tool."""

    @pytest.mark.asyncio
    async def test_search_success_single_query(
        self,
        get_perplexity_access_token_mock: None,
        mock_client_search_success: AsyncMock,
        mock_perplexity_search_results: list[PerplexitySearchResult],
    ) -> None:
        """Test successful search with single query."""
        result = await perplexity_search(query="test query")

        assert result.content[0].text.startswith(
            "Successfully executed search for query 'test query'."
        )
        assert "2 result(s)" in result.content[0].text
        assert result.structured_content["count"] == 2
        assert len(result.structured_content["results"]) == 2
        assert result.structured_content["results"][0]["url"] == "https://foo.com"
        assert result.structured_content["results"][1]["url"] == "https://bar.com"
        assert result.structured_content["metadata"]["queriesExecuted"] == 1
        assert result.structured_content["metadata"]["filtersApplied"] == {
            "domains": None,
            "recency": None,
        }
        assert result.structured_content["metadata"]["extractionLimit"] == 2048

    @pytest.mark.asyncio
    async def test_search_success_list_of_queries(
        self,
        get_perplexity_access_token_mock: None,
        mock_client_search_success: AsyncMock,
        mock_perplexity_search_results: list[PerplexitySearchResult],
    ) -> None:
        """Test successful search with list of queries."""
        result = await perplexity_search(query=["test query 1", "test query 2"])

        assert result.content[0].text.startswith(
            "Successfully executed search for queries 'test query 1, test query 2'."
        )
        assert "2 result(s)" in result.content[0].text
        assert result.structured_content["count"] == 2
        assert len(result.structured_content["results"]) == 2
        assert result.structured_content["results"][0]["url"] == "https://foo.com"
        assert result.structured_content["results"][1]["url"] == "https://bar.com"
        assert result.structured_content["metadata"]["queriesExecuted"] == 2
        assert result.structured_content["metadata"]["filtersApplied"] == {
            "domains": None,
            "recency": None,
        }
        assert result.structured_content["metadata"]["extractionLimit"] == 2048

    @pytest.mark.asyncio
    async def test_search_success_with_filters(
        self,
        get_perplexity_access_token_mock: None,
        mock_client_search_success: AsyncMock,
        mock_perplexity_search_results: list[PerplexitySearchResult],
    ) -> None:
        """Test successful search with single query."""
        result = await perplexity_search(
            query="test query",
            search_domain_filter=["foo, bar"],
            recency="week",
            max_tokens_per_page=1000,
        )

        assert result.content[0].text.startswith(
            "Successfully executed search for query 'test query'."
        )
        assert "2 result(s)" in result.content[0].text
        assert result.structured_content["count"] == 2
        assert len(result.structured_content["results"]) == 2
        assert result.structured_content["results"][0]["url"] == "https://foo.com"
        assert result.structured_content["results"][1]["url"] == "https://bar.com"
        assert result.structured_content["metadata"]["queriesExecuted"] == 1
        assert result.structured_content["metadata"]["filtersApplied"] == {
            "domains": ["foo, bar"],
            "recency": "week",
        }
        assert result.structured_content["metadata"]["extractionLimit"] == 1000

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "function_kwargs,error_message",
        [
            ({"query": ""}, "query.*cannot be empty"),
            ({"query": []}, "query.*cannot be empty"),
            ({"query": [str(i) for i in range(10)]}, "query list cannot be bigger than 5"),
            ({"query": ["query1", ""]}, "query.*cannot contain empty str"),
            (
                {"query": "query1", "search_domain_filter": [str(i) for i in range(25)]},
                "maximum number of search domain filters is 20",
            ),
            ({"query": "query1", "max_results": 0}, "max_results must be greater than 0"),
            (
                {"query": "query1", "max_results": 21},
                "max_results must be smaller than or equal to 20",
            ),
            (
                {"query": "query1", "max_tokens_per_page": 0},
                "max_tokens_per_page must be greater than 0",
            ),
            (
                {"query": "query1", "max_tokens_per_page": 8193},
                "max_tokens_per_page must be smaller than or equal to 8192",
            ),
        ],
    )
    async def test_search_input_validation(
        self,
        get_perplexity_access_token_mock: None,
        function_kwargs: dict,
        error_message: str,
    ) -> None:
        """Test search -- input validation."""
        with pytest.raises(ToolError, match=error_message):
            await perplexity_search(**function_kwargs)

    @pytest.mark.asyncio
    async def test_search_oauth_error(self) -> None:
        """Test search when OAuth token retrieval fails."""
        with patch(
            "datarobot_genai.drmcp.tools.perplexity.tools.get_perplexity_access_token",
            return_value=ToolError("OAuth error"),
        ):
            with pytest.raises(ToolError) as exc_info:
                await perplexity_search(query="test")
            assert "oauth" in str(exc_info.value).lower() or "error" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_search_client_error(
        self,
        get_perplexity_access_token_mock: None,
    ) -> None:
        """Test search when client raises error."""
        with patch(
            "datarobot_genai.drmcp.tools.perplexity.tools.PerplexityClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.search = AsyncMock(side_effect=PerplexityError("Client error"))
            mock_client_class.return_value = mock_client

            with pytest.raises(ToolError) as exc_info:
                await perplexity_search(query="test")
            assert "client error" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_search_unexpected_error(
        self,
        get_perplexity_access_token_mock: None,
    ) -> None:
        """Test search when unexpected error occurs."""
        with patch(
            "datarobot_genai.drmcp.tools.perplexity.tools.PerplexityClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.search = AsyncMock(side_effect=Exception("Unexpected error"))
            mock_client_class.return_value = mock_client

            with pytest.raises(ToolError) as exc_info:
                await perplexity_search(query="test")
            assert "unexpected error" in str(exc_info.value).lower()


class TestPerplexityThink:
    """Test perplexity_think tool."""

    @pytest.mark.asyncio
    async def test_think_success(
        self,
        get_perplexity_access_token_mock: None,
        mock_client_think_success: AsyncMock,
        mock_perplexity_think_results: PerplexityThinkResult,
    ) -> None:
        """Test successful think."""
        result = await perplexity_think(prompt="test prompt")

        assert result.structured_content["model"] == "sonar"
        assert len(result.structured_content["citations"]) == 2
        assert result.structured_content["content"] == "Dummy answer"
        assert result.structured_content["usage"]["cost"]["totalCost"] == 250

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "function_kwargs,error_message",
        [
            ({"prompt": ""}, "prompt.*cannot be empty"),
            ({"prompt": "  "}, "prompt.*cannot be empty"),
        ],
    )
    async def test_think_input_validation(
        self,
        get_perplexity_access_token_mock: None,
        function_kwargs: dict,
        error_message: str,
    ) -> None:
        """Test think -- input validation."""
        with pytest.raises(ToolError, match=error_message):
            await perplexity_think(**function_kwargs)

    @pytest.mark.asyncio
    async def test_think_oauth_error(self) -> None:
        """Test think when OAuth token retrieval fails."""
        with patch(
            "datarobot_genai.drmcp.tools.perplexity.tools.get_perplexity_access_token",
            return_value=ToolError("OAuth error"),
        ):
            with pytest.raises(ToolError) as exc_info:
                await perplexity_think(prompt="test prompt")
            assert "oauth" in str(exc_info.value).lower() or "error" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_think_client_error(
        self,
        get_perplexity_access_token_mock: None,
    ) -> None:
        """Test think when client raises error."""
        with patch(
            "datarobot_genai.drmcp.tools.perplexity.tools.PerplexityClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.think = AsyncMock(side_effect=PerplexityError("Client error"))
            mock_client_class.return_value = mock_client

            with pytest.raises(ToolError) as exc_info:
                await perplexity_think(prompt="test prompt")
            assert "client error" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_think_unexpected_error(
        self,
        get_perplexity_access_token_mock: None,
    ) -> None:
        """Test think when unexpected error occurs."""
        with patch(
            "datarobot_genai.drmcp.tools.perplexity.tools.PerplexityClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.think = AsyncMock(side_effect=Exception("Unexpected error"))
            mock_client_class.return_value = mock_client

            with pytest.raises(ToolError) as exc_info:
                await perplexity_think(prompt="test prompt")
            assert "unexpected error" in str(exc_info.value).lower()
