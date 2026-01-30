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

import logging
from typing import Any
from typing import Literal

from fastmcp.exceptions import ToolError
from fastmcp.server.dependencies import get_http_headers
from perplexity import AsyncPerplexity
from perplexity.types import search_create_response
from pydantic import BaseModel
from pydantic import ConfigDict

logger = logging.getLogger(__name__)

MAX_QUERIES: int = 5
MAX_RESULTS: int = 20
MAX_TOKENS_PER_PAGE: int = 8192
MAX_SEARCH_DOMAIN_FILTER: int = 20

MAX_RESULTS_DEFAULT: int = 10
MAX_TOKENS_PER_PAGE_DEFAULT: int = 2048


async def get_perplexity_access_token() -> str | ToolError:
    """
    Get Perplexity API key from HTTP headers.

    At the moment of creating this fn. Perplexity does not support OAuth.
    It allows only API-KEY authorized flow.

    Returns
    -------
        Access token string on success, ToolError on failure

    Example:
        ```python
        token = await get_perplexity_access_token()
        if isinstance(token, ToolError):
            # Handle error
            return token
        # Use token
        ```
    """
    try:
        headers = get_http_headers()

        if api_key := headers.get("x-perplexity-api-key"):
            return api_key

        logger.warning("Perplexity API key not found in headers.")
        return ToolError(
            "Perplexity API key not found in headers. "
            "Please provide it via 'x-perplexity-api-key' header."
        )
    except Exception as e:
        logger.error(f"Unexpected error obtaining Perplexity API key: {e}.", exc_info=e)
        return ToolError("An unexpected error occured while obtaining Perplexity API key.")


class PerplexityError(Exception):
    """Exception for Perplexity API errors."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class PerplexitySearchResult(BaseModel):
    snippet: str
    title: str
    url: str
    date: str | None = None
    last_updated: str | None = None

    model_config = ConfigDict(populate_by_name=True)

    @classmethod
    def from_perplexity_sdk(cls, result: search_create_response.Result) -> "PerplexitySearchResult":
        """Create a PerplexitySearchResult from perplexity sdk response data."""
        return cls(**result.model_dump())

    def as_flat_dict(self) -> dict[str, Any]:
        """Return a flat dictionary representation of the search result."""
        return self.model_dump(by_alias=True)


class PerplexityClient:
    """Client for interacting with Perplexity API.
    Its simple wrapper around perplexity python sdk.
    """

    def __init__(self, access_token: str) -> None:
        self._client = AsyncPerplexity(api_key=access_token)

    async def search(
        self,
        query: str | list[str],
        search_domain_filter: list[str] | None = None,
        recency: Literal["hour", "day", "week", "month", "year"] | None = None,
        max_results: int = MAX_RESULTS_DEFAULT,
        max_tokens_per_page: int = MAX_TOKENS_PER_PAGE_DEFAULT,
    ) -> list[PerplexitySearchResult]:
        """
        Search using Perplexity.

        Args:
            query: Query to filter results.
            search_domain_filter: Up to 20 domains/URLs to allowlist or denylist.
            recency: Filter results by time period.
            max_results: Number of ranked results to return.
            max_tokens_per_page: Context extraction cap per page.

        Returns
        -------
            List of Perplexity search results.
        """
        if not query:
            raise PerplexityError("Error: query cannot be empty.")
        if query and isinstance(query, str) and not query.strip():
            raise PerplexityError("Error: query cannot be empty.")
        if query and isinstance(query, list) and len(query) > MAX_QUERIES:
            raise PerplexityError(f"Error: query list cannot be bigger than {MAX_QUERIES}.")
        if query and isinstance(query, list) and not all(q.strip() for q in query):
            raise PerplexityError("Error: query cannot contain empty str.")
        if search_domain_filter and len(search_domain_filter) > MAX_SEARCH_DOMAIN_FILTER:
            raise PerplexityError("Error: maximum number of search domain filters is 20.")
        if max_results <= 0:
            raise PerplexityError("Error: max_results must be greater than 0.")
        if max_results > MAX_RESULTS:
            raise PerplexityError("Error: max_results must be smaller than or equal to 20.")
        if max_tokens_per_page <= 0:
            raise PerplexityError("Error: max_tokens_per_page must be greater than 0.")
        if max_tokens_per_page > MAX_TOKENS_PER_PAGE:
            raise PerplexityError(
                "Error: max_tokens_per_page must be smaller than or equal to 8192."
            )

        max_results = min(max_results, MAX_RESULTS)
        max_tokens_per_page = min(max_tokens_per_page, MAX_TOKENS_PER_PAGE)

        search_result = await self._client.search.create(
            query=query,
            search_domain_filter=search_domain_filter,
            search_recency_filter=recency,
            max_results=max_results,
            max_tokens_per_page=max_tokens_per_page,
        )

        return [
            PerplexitySearchResult.from_perplexity_sdk(result) for result in search_result.results
        ]

    async def __aenter__(self) -> "PerplexityClient":
        """Async context manager entry."""
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any
    ) -> None:
        """Async context manager exit."""
        await self._client.close()
