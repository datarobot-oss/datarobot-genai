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

"""DataRobot Agentic AI documentation search tools.

Provides MCP tools for searching and fetching content from the DataRobot
agentic-ai documentation at https://docs.datarobot.com/en/docs/agentic-ai/.

No API keys or external services are required — the tool directly indexes
the public documentation site using a TF-IDF search over page titles and
body text.

For use without an MCP server, import the tool functions directly from
:mod:`datarobot_genai.drmcp.tools.dr_docs.local_tools` (or via the
package ``__init__``).
"""

import logging
from typing import Annotated

from fastmcp.tools.tool import ToolResult

from datarobot_genai.drmcp.core.mcp_instance import dr_mcp_integration_tool
from datarobot_genai.drmcp.tools.clients.dr_docs import MAX_RESULTS
from datarobot_genai.drmcp.tools.clients.dr_docs import MAX_RESULTS_DEFAULT
from datarobot_genai.drmcp.tools.dr_docs.local_tools import (
    fetch_datarobot_doc_page as _fetch_datarobot_doc_page,
)
from datarobot_genai.drmcp.tools.dr_docs.local_tools import (
    search_datarobot_agentic_docs as _search_datarobot_agentic_docs,
)

logger = logging.getLogger(__name__)


@dr_mcp_integration_tool(tags={"datarobot", "docs", "documentation", "search"})
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
) -> ToolResult:
    """
    Search the DataRobot agentic-AI documentation for relevant pages.

    This tool searches through the DataRobot agentic-AI documentation at
    https://docs.datarobot.com/en/docs/agentic-ai/ to find pages relevant
    to your query. It returns page titles, URLs, and page contents.

    No API keys are required — the tool indexes the public documentation
    site using TF-IDF over real page titles and page text.

    Usage:
        - Find MCP info: search_datarobot_agentic_docs(query="MCP server setup")
        - Find agent tools: search_datarobot_agentic_docs(query="agentic tools")
        - Broad search: search_datarobot_agentic_docs(query="authentication", max_results=10)

    Note:
        - The index covers only https://docs.datarobot.com/en/docs/agentic-ai/ (~28 pages).
    """
    result = await _search_datarobot_agentic_docs(query=query, max_results=max_results)
    return ToolResult(structured_content=result)


@dr_mcp_integration_tool(tags={"datarobot", "docs", "documentation", "fetch", "read"})
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

    Use this tool to retrieve the full content of a relevant documentation page. The content is
    extracted as clean text, suitable for reading and analysis.

    Usage:
        - Fetch a page: fetch_datarobot_doc_page(
            url="https://docs.datarobot.com/en/docs/mlops/deployment/index.html"
          )

    Note:
        - Only works with English DataRobot documentation URLs (e.g. docs.datarobot.com/en/docs/).
    """
    result = await _fetch_datarobot_doc_page(url=url)
    return ToolResult(structured_content=result)
