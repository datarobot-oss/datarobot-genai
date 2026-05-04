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

"""Standalone DataRobot Agentic AI documentation search tools.

No API keys or external services are required — the tool directly indexes
the public documentation site using a TF-IDF search over page titles and
body text.

Provides standalone tool functions that can be wrapped as LangChain or other framework
(e.g. LlamaIndex) or MCP tools.

Example use with LangGraph::

    from datarobot_genai.drtools.dr_docs import search_datarobot_agentic_docs
    from langchain_core.tools import tool
    search_tool = tool(search_datarobot_agentic_docs)
    agent = create_agent(model, tools=[search_tool])

    # To call directly:
    result = await search_tool.ainvoke({
        "query": "MCP server setup",
        "max_results": 5
    })
"""

import logging
from typing import Annotated
from typing import Any

from datarobot_genai.drtools.core import tool_metadata
from datarobot_genai.drtools.core.clients.dr_docs import MAX_RESULTS
from datarobot_genai.drtools.core.clients.dr_docs import MAX_RESULTS_DEFAULT
from datarobot_genai.drtools.core.clients.dr_docs import fetch_page_content
from datarobot_genai.drtools.core.clients.dr_docs import search_docs
from datarobot_genai.drtools.core.exceptions import ToolError
from datarobot_genai.drtools.core.exceptions import ToolErrorKind

logger = logging.getLogger(__name__)


@tool_metadata(
    tags={"dr_docs", "datarobot", "docs", "documentation", "search"},
    description=(
        "[DR docs—search] Use when the user asks about DataRobot agentic AI product behavior, "
        "setup, or APIs covered on the public agentic-AI documentation site. Returns page titles "
        "and URLs (then fetch_datarobot_doc_page for full text). Not Confluence, not general web "
        "(tavily_search / perplexity_search)."
    ),
)
async def search_datarobot_agentic_docs(
    *,
    query: Annotated[
        str,
        "The search query describing what you want to find in the DataRobot agentic-AI docs. "
        "Use natural language or keywords (e.g., 'MCP server setup', 'agentic tools', "
        "'custom agent', 'authentication').",
    ],
    max_results: Annotated[
        int,
        f"Maximum number of documentation pages to return (allowable values: 1 to {MAX_RESULTS}).",
    ] = MAX_RESULTS_DEFAULT,
) -> dict[str, Any]:
    if not query or not query.strip():
        raise ToolError(
            "Argument validation error: 'query' cannot be empty.", kind=ToolErrorKind.VALIDATION
        )

    results = await search_docs(query=query, max_results=max_results)

    if not results:
        return {
            "status": "no_results",
            "query": query,
            "message": "No documentation pages found matching your query. "
            "Try rephrasing with different keywords.",
        }

    flat: dict[str, Any] = {
        "status": "success",
        "query": query,
        "total_results": len(results),
    }
    for i, result in enumerate(results):
        flat[f"result_{i}_title"] = result["title"]
        flat[f"result_{i}_url"] = result["url"]
        if result.get("description"):
            flat[f"result_{i}_description"] = result["description"]

    return flat


@tool_metadata(
    tags={"dr_docs", "datarobot", "docs", "documentation", "fetch", "read"},
    description=(
        "[DR docs—fetch page] Use when you already have a full docs.datarobot.com English docs URL "
        "(e.g. from search results) and need the page body as text. Not keyword search across the "
        "agentic docs index (search_datarobot_agentic_docs)."
    ),
)
async def fetch_datarobot_doc_page(
    *,
    url: Annotated[
        str,
        "The full URL of the DataRobot documentation page to fetch. "
        "Must be a URL from docs.datarobot.com/en/docs/.",
    ],
) -> dict[str, Any]:
    if not url or not url.strip():
        raise ToolError(
            "Argument validation error: 'url' cannot be empty.", kind=ToolErrorKind.VALIDATION
        )

    result = await fetch_page_content(url=url)
    return result
