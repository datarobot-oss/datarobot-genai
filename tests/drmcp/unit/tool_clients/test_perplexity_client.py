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
from perplexity.types import ChatMessageOutput
from perplexity.types import Choice
from perplexity.types import StreamChunk
from perplexity.types import search_create_response

from datarobot_genai.drtools.clients.perplexity import PerplexityClient
from datarobot_genai.drtools.clients.perplexity import PerplexityError


def make_response(
    status_code: int,
    json_data: dict | None = None,
    text: str | None = None,
    method: str = "GET",
) -> httpx.Response:
    """Create a mock httpx.Response with a request attached."""
    request = httpx.Request(method, "https://api.perplexity.ai")
    if json_data is not None:
        return httpx.Response(status_code, json=json_data, request=request)
    return httpx.Response(status_code, text=text or "", request=request)


class TestPerplexityClient:
    """Test PerplexityClient class."""

    @pytest.fixture
    def mock_access_token(self) -> str:
        """Mock access token."""
        return "test_access_token_123"

    @pytest.fixture
    def mock_perplexity_sdk_search_results(self) -> search_create_response.SearchCreateResponse:
        """Mock Perplexity SDK search results."""
        return search_create_response.SearchCreateResponse(
            id="dummy_id",
            results=[
                search_create_response.Result(
                    snippet="This website gives info about foo",
                    title="Foo Website",
                    url="https://foo.com",
                ),
                search_create_response.Result(
                    snippet="This website gives info about bar",
                    title="Bar Website",
                    url="https://bar.com",
                ),
            ],
        )

    @pytest.fixture
    def mock_perplexity_sdk_think_results(self) -> StreamChunk:
        """Mock Perplexity SDK think result."""
        return StreamChunk(
            id="dummy_id",
            choices=[
                Choice(
                    index=0,
                    delta=ChatMessageOutput(role="assistant", content="Answer"),
                    message=ChatMessageOutput(role="assistant", content="Answer"),
                )
            ],
            citations=["Citation 1", "Citation 2"],
            created=0,
            model="sonar",
        )

    @pytest.mark.asyncio
    async def test_search_success(
        self,
        mock_access_token: str,
        mock_perplexity_sdk_search_results: search_create_response.SearchCreateResponse,
    ) -> None:
        """Test search."""
        with patch("datarobot_genai.drtools.clients.perplexity.AsyncPerplexity") as mock:
            mock_sdk_client = mock.return_value
            mock_sdk_client.close = AsyncMock()
            mock_sdk_client.search.create = AsyncMock(
                return_value=mock_perplexity_sdk_search_results
            )

            async with PerplexityClient(mock_access_token) as client:
                results = await client.search(query="query")

            assert results[0].url == "https://foo.com"
            assert results[1].url == "https://bar.com"

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
    async def test_search_validation(
        self, mock_access_token: str, function_kwargs: dict, error_message: str
    ) -> None:
        """Test search -- input validation."""
        async with PerplexityClient(mock_access_token) as client:
            with pytest.raises(PerplexityError, match=error_message):
                await client.search(**function_kwargs)

    @pytest.mark.asyncio
    async def test_think_success(
        self,
        mock_access_token: str,
        mock_perplexity_sdk_think_results: StreamChunk,
    ) -> None:
        """Test search."""
        with patch("datarobot_genai.drtools.clients.perplexity.AsyncPerplexity") as mock:
            mock_sdk_client = mock.return_value
            mock_sdk_client.close = AsyncMock()
            mock_sdk_client.chat.completions.create = AsyncMock(
                return_value=mock_perplexity_sdk_think_results
            )

            async with PerplexityClient(mock_access_token) as client:
                result = await client.think(prompt="prompt", model="sonar")

            assert result.answer == "Answer"
            assert len(result.citations) == 2

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "function_kwargs,error_message",
        [
            ({"prompt": "", "model": "sonar"}, "prompt.*cannot be empty"),
            ({"prompt": " ", "model": "sonar"}, "prompt.*cannot be empty"),
        ],
    )
    async def test_think_validation(
        self, mock_access_token: str, function_kwargs: dict, error_message: str
    ) -> None:
        """Test think -- input validation."""
        async with PerplexityClient(mock_access_token) as client:
            with pytest.raises(PerplexityError, match=error_message):
                await client.think(**function_kwargs)
