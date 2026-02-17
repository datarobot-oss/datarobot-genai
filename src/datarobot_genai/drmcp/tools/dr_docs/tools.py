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

"""DataRobot Documentation search tools.

Provides MCP tools for searching and fetching content from the DataRobot
product documentation at https://docs.datarobot.com/en/docs/index.html.

No API keys or external services are required — the tool directly crawls
and indexes the public documentation site.
"""

import logging
from typing import Annotated

from fastmcp.tools.tool import ToolResult

from datarobot_genai.drmcp.core.mcp_instance import dr_mcp_tool
from datarobot_genai.drmcp.tools.clients.dr_docs import MAX_RESULTS
from datarobot_genai.drmcp.tools.clients.dr_docs import MAX_RESULTS_DEFAULT
from datarobot_genai.drmcp.tools.clients.dr_docs import fetch_page_content
from datarobot_genai.drmcp.tools.clients.dr_docs import search_docs

logger = logging.getLogger(__name__)


@dr_mcp_tool(tags={"datarobot", "docs", "documentation", "search"})
async def search_datarobot_docs(
    *,
    query: Annotated[
        str,
        "The search query describing what you want to find in the DataRobot documentation. "
        "Use natural language or keywords (e.g., 'how to deploy a model', 'time series', "
        "'custom inference model', 'bias and fairness').",
    ],
    max_results: Annotated[
        int,
        f"Maximum number of documentation pages to return (1-{MAX_RESULTS}).",
    ] = MAX_RESULTS_DEFAULT,
) -> ToolResult:
    """
    Search the DataRobot product documentation for relevant pages.

    This tool searches through the DataRobot documentation site at
    https://docs.datarobot.com/en/docs/ to find pages relevant to your query.
    It returns page titles and URLs that you can then fetch for full content.

    No API keys are required — the tool directly indexes the public
    documentation site and uses keyword-based relevance matching.

    Usage:
        - Find deployment docs: search_datarobot_docs(query="deploy a model")
        - Find time series info: search_datarobot_docs(query="time series modeling")
        - Find API docs: search_datarobot_docs(query="prediction API")
        - Broad search: search_datarobot_docs(query="custom model environment", max_results=10)

    Note:
        - The first call may take a few seconds to build the documentation index.
        - Subsequent calls use a cached index (refreshed hourly).
        - Use fetch_datarobot_doc_page to get the full content of a specific page.
    """
    results = await search_docs(query=query, max_results=max_results)

    if not results:
        return ToolResult(
            structured_content={
                "status": "no_results",
                "query": query,
                "message": "No documentation pages found matching your query. "
                "Try rephrasing with different keywords.",
            }
        )

    flat: dict[str, str | int] = {
        "status": "success",
        "query": query,
        "total_results": len(results),
    }
    for i, result in enumerate(results):
        flat[f"result_{i}_title"] = result["title"]
        flat[f"result_{i}_url"] = result["url"]
        if result.get("description"):
            flat[f"result_{i}_description"] = result["description"]

    return ToolResult(structured_content=flat)


@dr_mcp_tool(tags={"datarobot", "docs", "documentation", "fetch", "read"})
async def fetch_datarobot_doc_page(
    *,
    url: Annotated[
        str,
        "The full URL of the DataRobot documentation page to fetch. "
        "Must be a URL from docs.datarobot.com/en/docs/.",
    ],
) -> ToolResult:
    """
    Fetch and extract the text content of a specific DataRobot documentation page.

    Use this tool after search_datarobot_docs to retrieve the full content of a
    relevant documentation page. The content is extracted as clean text, suitable
    for reading and analysis.

    Usage:
        - Fetch a page: fetch_datarobot_doc_page(
            url="https://docs.datarobot.com/en/docs/mlops/deployment/index.html"
          )

    Note:
        - Only works with DataRobot documentation URLs (docs.datarobot.com).
        - Content is truncated to ~5000 characters for very long pages.
    """
    result = await fetch_page_content(url=url)

    return ToolResult(structured_content=result)
