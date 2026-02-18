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

"""Standalone (non-MCP) DataRobot Agentic AI documentation search tools.

Provides the same tool interface as the MCP-registered tools in tools.py, but
as plain async functions that return dict[str, Any] rather than ToolResult. No
MCP server or FastMCP dependency is required.

These functions are framework-agnostic and can be passed directly to any agent
that accepts Python callables, or wrapped with a framework-specific decorator
(e.g. LangChain's ``@tool``, LlamaIndex's ``FunctionTool``) at the call site.

Example — LangGraph::

    from langchain_core.tools import tool
    from datarobot_genai.drmcp.tools.dr_docs import search_datarobot_agentic_docs

    search_tool = tool(search_datarobot_agentic_docs)
    agent = create_react_agent(model, tools=[search_tool])
"""

import logging
from typing import Annotated
from typing import Any

from datarobot_genai.drmcp.tools.clients.dr_docs import MAX_RESULTS
from datarobot_genai.drmcp.tools.clients.dr_docs import MAX_RESULTS_DEFAULT
from datarobot_genai.drmcp.tools.clients.dr_docs import fetch_page_content
from datarobot_genai.drmcp.tools.clients.dr_docs import search_docs

logger = logging.getLogger(__name__)


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
        f"Maximum number of documentation pages to return (1 to {MAX_RESULTS}).",
    ] = MAX_RESULTS_DEFAULT,
) -> dict[str, Any]:
    """
    Search the DataRobot agentic-AI documentation for relevant pages.

    This tool searches through the DataRobot agentic-AI documentation at
    https://docs.datarobot.com/en/docs/agentic-ai/ to find pages relevant
    to your query. It returns page titles and URLs that you can then fetch
    for full content.

    No API keys are required — the tool indexes the public documentation
    site using TF-IDF over real page titles and body text.

    Usage:
        - Find MCP info: search_datarobot_agentic_docs(query="MCP server setup")
        - Find agent tools: search_datarobot_agentic_docs(query="agentic tools")
        - Broad search: search_datarobot_agentic_docs(query="authentication", max_results=10)

    Note:
        - The index covers only https://docs.datarobot.com/en/docs/agentic-ai/ (~28 pages).
        - Use fetch_datarobot_doc_page to get the full content of a specific page.
    """
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


async def fetch_datarobot_doc_page(
    *,
    url: Annotated[
        str,
        "The full URL of the DataRobot documentation page to fetch. "
        "Must be a URL from docs.datarobot.com/en/docs/.",
    ],
) -> dict[str, Any]:
    """
    Fetch and extract the text content of a specific DataRobot documentation page.

    Use this tool after search_datarobot_agentic_docs to retrieve the full content of a
    relevant documentation page. The content is extracted as clean text, suitable
    for reading and analysis.

    Usage:
        - Fetch a page: fetch_datarobot_doc_page(
            url="https://docs.datarobot.com/en/docs/mlops/deployment/index.html"
          )

    Note:
        - Only works with English DataRobot documentation URLs (e.g. docs.datarobot.com/en/docs/).
    """
    return await fetch_page_content(url=url)
