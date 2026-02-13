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

"""Tavily MCP tools for web search and content extraction."""

import logging
from typing import Annotated
from typing import Literal

from fastmcp.tools.tool import ToolResult

from datarobot_genai.drmcp import dr_mcp_tool
from datarobot_genai.drmcp.tools.clients.tavily import CHUNKS_PER_SOURCE_DEFAULT
from datarobot_genai.drmcp.tools.clients.tavily import MAX_CHUNKS_PER_SOURCE
from datarobot_genai.drmcp.tools.clients.tavily import MAX_RESULTS
from datarobot_genai.drmcp.tools.clients.tavily import MAX_RESULTS_DEFAULT
from datarobot_genai.drmcp.tools.clients.tavily import TavilyClient
from datarobot_genai.drmcp.tools.clients.tavily import get_tavily_access_token

logger = logging.getLogger(__name__)


@dr_mcp_tool(tags={"tavily", "search", "web", "websearch"})
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

    return ToolResult(
        structured_content=response.as_flat_dict(
            include_images=include_images, include_answer=include_answer
        )
    )


@dr_mcp_tool(tags={"extract", "tavily", "web", "content"})
async def tavily_extract(
    *,
    urls: Annotated[
        str | list[str],
        "The URL or list of URLs to extract content from (max 20).",
    ],
    query: Annotated[
        str | None,
        "User intent for reranking extracted content chunks based on relevance.",
    ] = None,
    chunks_per_source: Annotated[
        int,
        "Max number of 500-char snippets per URL (1-5). Used when query is provided.",
    ] = 3,
    format: Annotated[
        Literal["markdown", "text"],
        "Format of extracted content. Markdown is usually more ergonomic for LLMs.",
    ] = "markdown",
    extract_depth: Annotated[
        Literal["basic", "advanced"],
        "Depth of extraction; 'advanced' handles complex tables and embedded content.",
    ] = "basic",
) -> ToolResult:
    """
    Extract content from web pages using Tavily Extract API.

    Use this tool to retrieve and parse content from one or more URLs. The extracted
    content is cleaned and formatted for LLM consumption.

    Usage:
        - Single URL: tavily_extract(urls="https://example.com/article")
        - Multiple URLs: tavily_extract(urls=["url1", "url2"])
        - With relevance query: tavily_extract(urls="url", query="auth setup", chunks_per_source=5)
        - Advanced: tavily_extract(urls="url", extract_depth="advanced")

    Limits (configurable in client):
        - Maximum 20 URLs per request
        - chunks_per_source: 1-5 (default 3)
        - timeout: 1.0-60.0 seconds (default 30.0)

    Note:
        - Advanced depth handles complex content but may be slower
    """
    api_key = await get_tavily_access_token()

    async with TavilyClient(api_key) as client:
        response = await client.extract(
            urls=urls,
            query=query,
            chunks_per_source=chunks_per_source,
            extract_depth=extract_depth,
            format=format,
        )

    return ToolResult(structured_content=response.as_flat_dict())


@dr_mcp_tool(tags={"map", "tavily", "discovery"})
async def tavily_map(
    *,
    url: Annotated[str, "The root URL to begin mapping."],
    instructions: Annotated[
        str | None, "Instructions to guide the mapper toward specific paths."
    ] = None,
    limit: Annotated[int, "Total links to process (default 50)."] = 50,
    include_usage: Annotated[
        bool, "Whether to include credit usage information in the response."
    ] = False,
) -> ToolResult:
    """
    Generate a structured map of a website to discover relevant sub-pages.
    API documentation: https://docs.tavily.com/documentation/api-reference/endpoint/map .
    """
    api_key = await get_tavily_access_token()

    async with TavilyClient(api_key) as client:
        response = await client.map_(
            url=url, instructions=instructions, limit=limit, include_usage=include_usage
        )

    return ToolResult(structured_content=response.as_flat_dict())


@dr_mcp_tool(tags={"crawl", "tavily", "web", "rag"})
async def tavily_crawl(
    *,
    url: Annotated[str, "The root URL to begin the traversal."],
    instructions: Annotated[
        str | None,
        "Natural language instructions for the crawler to filter relevant content.",
    ] = None,
    limit: Annotated[int, "Total number of links to process (1-500)."] = 20,
    max_depth: Annotated[int, "Maximum depth from base URL (1-5)."] = 1,
    exclude_paths: Annotated[
        list[str] | None,
        "Regex patterns to exclude URL paths (e.g., ['/blog/.*', '/archive/.*']).",
    ] = None,
    include_images: Annotated[bool, "Include images found during crawl."] = False,
) -> ToolResult:
    """
    Crawl a website using Tavily Crawl API for Crawl-to-RAG workflows.

    Use this tool to explore an entire site and retrieve relevant content based on
    natural language instructions. Ideal for building knowledge bases, extracting
    documentation, or gathering structured content from websites.

    Usage:
        - Basic crawl: tavily_crawl(url="https://docs.example.com")
        - With instructions: tavily_crawl(
            url="https://docs.example.com",
            instructions="Find all API documentation pages"
          )
        - Deep crawl: tavily_crawl(url="https://example.com", max_depth=3, limit=100)
        - Exclude paths: tavily_crawl(
            url="https://example.com",
            exclude_paths=["/blog/.*", "/archive/.*"]
          )

    Limits:
        - limit: 1-500 (default 20)
        - max_depth: 1-5 (default 1)

    Note:
        - Higher limits and depths consume more API credits
        - Use instructions to filter for relevant content and reduce noise
    """
    api_key = await get_tavily_access_token()

    async with TavilyClient(api_key) as client:
        response = await client.crawl(
            url=url,
            instructions=instructions,
            limit=limit,
            max_depth=max_depth,
            exclude_paths=exclude_paths,
            include_images=include_images,
        )

    return ToolResult(structured_content=response.as_flat_dict())
