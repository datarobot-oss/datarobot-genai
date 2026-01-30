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

"""Tavily MCP tools for web search."""

import logging
from typing import Annotated
from typing import Literal

from fastmcp.tools.tool import ToolResult

from datarobot_genai.drmcp.core.mcp_instance import dr_mcp_tool
from datarobot_genai.drmcp.tools.clients.tavily import CHUNKS_PER_SOURCE_DEFAULT
from datarobot_genai.drmcp.tools.clients.tavily import MAX_CHUNKS_PER_SOURCE
from datarobot_genai.drmcp.tools.clients.tavily import MAX_RESULTS
from datarobot_genai.drmcp.tools.clients.tavily import MAX_RESULTS_DEFAULT
from datarobot_genai.drmcp.tools.clients.tavily import TavilyClient
from datarobot_genai.drmcp.tools.clients.tavily import TavilyImage
from datarobot_genai.drmcp.tools.clients.tavily import TavilySearchResult
from datarobot_genai.drmcp.tools.clients.tavily import get_tavily_access_token

logger = logging.getLogger(__name__)


@dr_mcp_tool(tags={"search", "tavily", "web", "websearch"})
async def tavily_search(
    *,
    query: Annotated[str, "The search query to execute."],
    topic: Annotated[
        Literal["general", "news", "finance"],
        "The category of search. Use 'general' for broad web search, "
        "'news' for recent news articles, or 'finance' for financial information.",
    ] = "general",
    search_depth: Annotated[
        Literal["basic", "advanced"],
        "The depth of search. 'basic' is faster and cheaper, "
        "'advanced' provides more comprehensive results.",
    ] = "basic",
    max_results: Annotated[
        int,
        f"Maximum number of search results to return (1-{MAX_RESULTS}).",
    ] = MAX_RESULTS_DEFAULT,
    time_range: Annotated[
        Literal["day", "week", "month", "year"] | None,
        "Filter results by time range. Use 'day' for last 24 hours, "
        "'week' for last 7 days, 'month' for last 30 days, or 'year' for last year.",
    ] = None,
    include_images: Annotated[
        bool,
        "Whether to include related images in the search results.",
    ] = False,
    include_image_descriptions: Annotated[
        bool,
        "Whether to include AI-generated descriptions for images. "
        "Only applicable when include_images is True.",
    ] = False,
    chunks_per_source: Annotated[
        int,
        f"Maximum number of content snippets to return per source URL (1-{MAX_CHUNKS_PER_SOURCE}).",
    ] = CHUNKS_PER_SOURCE_DEFAULT,
    include_answer: Annotated[
        bool,
        "Whether to include an AI-generated answer summarizing the search results.",
    ] = False,
) -> ToolResult:
    """
    Perform a real-time web search using Tavily API.

    Tavily is optimized for AI agents and provides clean, relevant search results
    suitable for LLM consumption. Use this tool to search the web for current
    information, news, financial data, or general knowledge.

    Usage:
        - Basic search: tavily_search(query="Python best practices 2026")
        - News search: tavily_search(query="AI regulations", topic="news", time_range="week")
        - Financial search: tavily_search(query="AAPL stock analysis", topic="finance")
        - Comprehensive search: tavily_search(
            query="climate change solutions",
            search_depth="advanced",
            max_results=10,
            include_answer=True
          )

    Note:
        - Advanced search depth consumes more API credits but provides better results
        - Time range filtering is useful for finding recent information
    """
    api_key = await get_tavily_access_token()

    async with TavilyClient(api_key) as client:
        response = await client.search(
            query=query,
            topic=topic,
            search_depth=search_depth,
            max_results=max_results,
            time_range=time_range,
            include_images=include_images,
            include_image_descriptions=include_image_descriptions,
            chunks_per_source=chunks_per_source,
            include_answer=include_answer,
        )

    results = [TavilySearchResult.from_tavily_sdk(r) for r in response.get("results", [])]

    images: list[TavilyImage] | None = None
    if include_images and response.get("images"):
        images = [TavilyImage.from_tavily_sdk(img) for img in response.get("images", [])]

    result_count = len(results)
    answer = response.get("answer")
    response_time = response.get("response_time", 0.0)

    answer_info = " with AI-generated answer" if answer else ""
    image_info = f" and {len(images)} images" if images else ""

    structured_content: dict = {
        "query": response.get("query", query),
        "results": [r.as_flat_dict() for r in results],
        "resultCount": result_count,
        "responseTime": response_time,
    }

    if answer:
        structured_content["answer"] = answer

    if images:
        structured_content["images"] = [
            {"url": img.url, "description": img.description} for img in images
        ]

    return ToolResult(
        content=(
            f"Successfully searched for '{query}'. "
            f"Found {result_count} results{answer_info}{image_info}."
        ),
        structured_content=structured_content,
    )
