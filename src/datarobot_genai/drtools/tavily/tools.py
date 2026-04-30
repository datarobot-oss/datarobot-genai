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
from typing import Any
from typing import Literal

from datarobot_genai.drtools.core import tool_metadata
from datarobot_genai.drtools.core.clients.tavily import CHUNKS_PER_SOURCE_DEFAULT
from datarobot_genai.drtools.core.clients.tavily import MAX_CHUNKS_PER_SOURCE
from datarobot_genai.drtools.core.clients.tavily import MAX_RESULTS
from datarobot_genai.drtools.core.clients.tavily import MAX_RESULTS_DEFAULT
from datarobot_genai.drtools.core.clients.tavily import TavilyClient
from datarobot_genai.drtools.core.clients.tavily import get_tavily_access_token
from datarobot_genai.drtools.core.exceptions import ToolError
from datarobot_genai.drtools.core.exceptions import ToolErrorKind

logger = logging.getLogger(__name__)

_TAVILY_SEARCH_API = "https://docs.tavily.com/documentation/api-reference/endpoint/search"
_TAVILY_EXTRACT_API = "https://docs.tavily.com/documentation/api-reference/endpoint/extract"
_TAVILY_MAP_API = "https://docs.tavily.com/documentation/api-reference/endpoint/map"
_TAVILY_CRAWL_API = "https://docs.tavily.com/documentation/api-reference/endpoint/crawl"


@tool_metadata(
    tags={"tavily", "search", "web", "websearch"},
    description=(
        "[Tavily—web search] Use when the user needs fresh facts from the public web by keyword "
        "(optional topic string: 'general', 'news', or 'finance'). Returns ranked snippets and "
        "optional short answer. "
        "Not for reading full pages when you already have URLs (tavily_extract), not site link "
        "discovery (tavily_map), not multi-page site harvest (tavily_crawl).\n\n"
        'Examples: tavily_search(query="Python best practices 2026"); '
        'topic="news", time_range="week"; topic="finance"; search_depth="advanced", '
        "include_answer=True. Advanced depth uses more credits.\n\n"
        f"Reference: {_TAVILY_SEARCH_API}"
    ),
)
async def tavily_search(
    *,
    query: Annotated[str, "The search query to execute."],
    topic: Annotated[
        Literal["general", "news", "finance"],
        "Pass exactly one of these strings (not a type or enum name): 'general', 'news', "
        "'finance'. Broad web vs news vs finance search; omit to default to 'general'.",
    ] = "general",
    search_depth: Annotated[
        Literal["basic", "advanced"],
        "Pass exactly one of these strings: 'basic', 'advanced'. Faster/cheaper vs deeper "
        "coverage; omit to default to 'basic'.",
    ] = "basic",
    max_results: Annotated[
        int,
        f"Maximum number of search results to return (1-{MAX_RESULTS}).",
    ] = MAX_RESULTS_DEFAULT,
    time_range: Annotated[
        Literal["day", "week", "month", "year"] | None,
        "Optional. Pass exactly one of these strings or omit: 'day', 'week', 'month', 'year' "
        "(last 24h / 7d / 30d / year).",
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
) -> dict[str, Any]:
    if not query or not query.strip():
        raise ToolError(
            "Argument validation error: 'query' cannot be empty.", kind=ToolErrorKind.VALIDATION
        )

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

    return response.as_flat_dict(include_images=include_images, include_answer=include_answer)


@tool_metadata(
    tags={"tavily", "extract", "web", "content"},
    description=(
        "[Tavily—read URLs] Use when you already have one or more page URLs and need cleaned "
        "body text or reranked chunks (optional query for relevance). Not broad keyword web "
        "search (tavily_search), not crawling an entire site (tavily_crawl).\n\n"
        'Examples: single url string; list of urls; tavily_extract(urls="...", query="...", '
        'chunks_per_source=5); extract_depth="advanced". Up to 20 URLs; chunks_per_source 1-5.\n\n'
        f"Reference: {_TAVILY_EXTRACT_API}"
    ),
)
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
        "Pass exactly one of these strings: 'markdown', 'text'. Omit to default to 'markdown'.",
    ] = "markdown",
    extract_depth: Annotated[
        Literal["basic", "advanced"],
        "Pass exactly one of these strings: 'basic', 'advanced'. Omit to default to 'basic'; "
        "'advanced' handles complex tables and embedded content.",
    ] = "basic",
) -> dict[str, Any]:
    if isinstance(urls, str):
        if not urls or not urls.strip():
            raise ToolError(
                "Argument validation error: 'urls' cannot be empty.", kind=ToolErrorKind.VALIDATION
            )
    elif isinstance(urls, list):
        if not urls:
            raise ToolError(
                "Argument validation error: 'urls' list cannot be empty.",
                kind=ToolErrorKind.VALIDATION,
            )
        for url in urls:
            if not url or not url.strip():
                raise ToolError(
                    "Argument validation error: URLs in list cannot be empty.",
                    kind=ToolErrorKind.VALIDATION,
                )

    if query is not None and (not query or not query.strip()):
        raise ToolError(
            "Argument validation error: 'query' cannot be empty.", kind=ToolErrorKind.VALIDATION
        )

    api_key = await get_tavily_access_token()

    async with TavilyClient(api_key) as client:
        response = await client.extract(
            urls=urls,
            query=query,
            chunks_per_source=chunks_per_source,
            extract_depth=extract_depth,
            format=format,
        )

    return response.as_flat_dict()


@tool_metadata(
    tags={"tavily", "map", "discovery"},
    description=(
        "[Tavily—site map] Use when you need a structured list of links under one root URL "
        "(find sections or docs paths before reading). Optional instructions steer which branches "
        "matter. Not keyword web search (tavily_search), not full page text (tavily_extract), "
        "not deep multi-hop crawl (tavily_crawl).\n\n"
        'Example: tavily_map(url="https://docs.example.com", instructions="API sections").\n\n'
        f"Reference: {_TAVILY_MAP_API}"
    ),
)
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
) -> dict[str, Any]:
    if not url or not url.strip():
        raise ToolError(
            "Argument validation error: 'url' cannot be empty.", kind=ToolErrorKind.VALIDATION
        )

    if instructions is not None and (not instructions or not instructions.strip()):
        raise ToolError(
            "Argument validation error: 'instructions' cannot be empty.",
            kind=ToolErrorKind.VALIDATION,
        )

    api_key = await get_tavily_access_token()

    async with TavilyClient(api_key) as client:
        response = await client.map_(
            url=url, instructions=instructions, limit=limit, include_usage=include_usage
        )

    return response.as_flat_dict()


@tool_metadata(
    tags={"tavily", "crawl", "web", "rag"},
    description=(
        "[Tavily—site crawl] Use when you need many related pages from one site guided by "
        "natural-language instructions (breadth/depth limits). Not single-URL read "
        "(tavily_extract), not link-only outline (tavily_map), not global keyword search "
        "(tavily_search).\n\n"
        "Examples: basic crawl on docs root; instructions to filter topics; max_depth and limit "
        "up to API max; exclude_paths regex list. Higher depth/limit uses more credits.\n\n"
        f"Reference: {_TAVILY_CRAWL_API}"
    ),
)
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
) -> dict[str, Any]:
    if not url or not url.strip():
        raise ToolError(
            "Argument validation error: 'url' cannot be empty.", kind=ToolErrorKind.VALIDATION
        )

    if instructions is not None and (not instructions or not instructions.strip()):
        raise ToolError(
            "Argument validation error: 'instructions' cannot be empty.",
            kind=ToolErrorKind.VALIDATION,
        )

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

    return response.as_flat_dict()
