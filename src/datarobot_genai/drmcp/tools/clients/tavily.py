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

"""Tavily API Client and utilities for API key authentication."""

import logging
from typing import Any
from typing import Literal

from fastmcp.exceptions import ToolError
from fastmcp.server.dependencies import get_http_headers
from pydantic import BaseModel
from pydantic import ConfigDict
from tavily import AsyncTavilyClient

logger = logging.getLogger(__name__)

MAX_RESULTS: int = 20
MAX_CHUNKS_PER_SOURCE: int = 3

MAX_RESULTS_DEFAULT: int = 5
CHUNKS_PER_SOURCE_DEFAULT: int = 1


async def get_tavily_access_token() -> str:
    """
    Get Tavily API key from HTTP headers.

    Returns
    -------
        API key string

    Raises
    ------
        ToolError: If API key is not found in headers
    """
    headers = get_http_headers()

    api_key = headers.get("x-tavily-api-key")
    if api_key:
        return api_key

    logger.warning("Tavily API key not found in headers")
    raise ToolError(
        "Tavily API key not found in headers. Please provide it via 'x-tavily-api-key' header."
    )


class TavilySearchResult(BaseModel):
    """A single search result from Tavily API."""

    title: str
    url: str
    content: str
    score: float

    model_config = ConfigDict(populate_by_name=True)

    @classmethod
    def from_tavily_sdk(cls, result: dict[str, Any]) -> "TavilySearchResult":
        """Create a TavilySearchResult from Tavily SDK response data."""
        return cls(
            title=result.get("title", ""),
            url=result.get("url", ""),
            content=result.get("content", ""),
            score=result.get("score", 0.0),
        )

    def as_flat_dict(self) -> dict[str, Any]:
        """Return a flat dictionary representation of the search result."""
        return self.model_dump(by_alias=True)


class TavilyImage(BaseModel):
    """An image result from Tavily API."""

    url: str
    description: str | None = None

    model_config = ConfigDict(populate_by_name=True)

    @classmethod
    def from_tavily_sdk(cls, image: dict[str, Any] | str) -> "TavilyImage":
        """Create a TavilyImage from Tavily SDK response data."""
        if isinstance(image, str):
            return cls(url=image)
        return cls(
            url=image.get("url", ""),
            description=image.get("description"),
        )


class TavilyClient:
    """Client for interacting with Tavily Search API.

    This is a wrapper around the official tavily-python SDK.
    """

    def __init__(self, api_key: str) -> None:
        self._client = AsyncTavilyClient(api_key=api_key)

    async def search(
        self,
        query: str,
        *,
        topic: Literal["general", "news", "finance"] = "general",
        search_depth: Literal["basic", "advanced"] = "basic",
        max_results: int = MAX_RESULTS_DEFAULT,
        time_range: Literal["day", "week", "month", "year"] | None = None,
        include_images: bool = False,
        include_image_descriptions: bool = False,
        chunks_per_source: int = CHUNKS_PER_SOURCE_DEFAULT,
        include_answer: bool = False,
    ) -> dict[str, Any]:
        """
        Perform a web search using Tavily API.

        Args:
            query: The search query to execute.
            topic: The category of search ("general", "news", or "finance").
            search_depth: The depth of search ("basic" or "advanced").
            max_results: Maximum number of results to return (1-20).
            time_range: Time range filter ("day", "week", "month", "year").
            include_images: Whether to include images in results.
            include_image_descriptions: Whether to include image descriptions.
            chunks_per_source: Maximum content snippets per URL (1-3).
            include_answer: Whether to include an AI-generated answer.

        Returns
        -------
            Dict with search results from Tavily API.

        Raises
        ------
            ValueError: If validation fails.
            TavilyInvalidAPIKeyError: If the API key is invalid.
            TavilyUsageLimitExceededError: If usage limit is exceeded.
            TavilyForbiddenError: If access is forbidden.
            TavilyBadRequestError: If the request is malformed.
        """
        # Validate inputs
        if not query:
            raise ValueError("query cannot be empty.")
        if isinstance(query, str) and not query.strip():
            raise ValueError("query cannot be empty.")
        if max_results <= 0:
            raise ValueError("max_results must be greater than 0.")
        if max_results > MAX_RESULTS:
            raise ValueError(f"max_results must be smaller than or equal to {MAX_RESULTS}.")
        if chunks_per_source <= 0:
            raise ValueError("chunks_per_source must be greater than 0.")
        if chunks_per_source > MAX_CHUNKS_PER_SOURCE:
            raise ValueError(
                f"chunks_per_source must be smaller than or equal to {MAX_CHUNKS_PER_SOURCE}."
            )

        # Clamp values to valid ranges
        max_results = min(max_results, MAX_RESULTS)
        chunks_per_source = min(chunks_per_source, MAX_CHUNKS_PER_SOURCE)

        # Build search parameters
        search_kwargs: dict[str, Any] = {
            "query": query,
            "topic": topic,
            "search_depth": search_depth,
            "max_results": max_results,
            "include_images": include_images,
            "include_image_descriptions": include_image_descriptions,
            "chunks_per_source": chunks_per_source,
            "include_answer": include_answer,
        }

        if time_range:
            search_kwargs["time_range"] = time_range

        return await self._client.search(**search_kwargs)

    async def __aenter__(self) -> "TavilyClient":
        """Async context manager entry."""
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any
    ) -> None:
        """Async context manager exit."""
        # AsyncTavilyClient doesn't have a close method, but we keep the context manager
        # pattern for consistency with other clients
        pass
