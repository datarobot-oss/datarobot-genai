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

"""DataRobot Agentic AI documentation tools.

Exports tool functions that can be wrapped as LangChain or other framework (e.g. LlamaIndex) tools.

For use with LangGraph::

    from datarobot_genai.drmcp.tools.dr_docs import search_datarobot_agentic_docs
    from langchain_core.tools import tool
    search_tool = tool(search_datarobot_agentic_docs)
    agent = create_agent(model, tools=[search_tool])

    # To call directly:
    result = await search_tool.ainvoke({
        "query": "MCP server setup",
        "max_results": 5
    })
"""

from datarobot_genai.drmcp.tools.dr_docs.local_tools import fetch_datarobot_doc_page
from datarobot_genai.drmcp.tools.dr_docs.local_tools import search_datarobot_agentic_docs

__all__ = [
    "fetch_datarobot_doc_page",
    "search_datarobot_agentic_docs",
]
