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

Exports the standalone (non-MCP) tool functions for direct use in agent
demos and applications that do not run an MCP server.

Example::

    from datarobot_genai.drmcp.tools.dr_docs import (
        fetch_datarobot_doc_page,
        search_datarobot_agentic_docs,
    )

For use with LangGraph::

    from langchain_core.tools import tool as lc_tool
    from datarobot_genai.drmcp.tools.dr_docs import search_datarobot_agentic_docs

    search_tool = lc_tool(search_datarobot_agentic_docs)
    agent = create_react_agent(model, tools=[search_tool])
"""

from datarobot_genai.drmcp.tools.dr_docs.local_tools import fetch_datarobot_doc_page
from datarobot_genai.drmcp.tools.dr_docs.local_tools import search_datarobot_agentic_docs

__all__ = [
    "fetch_datarobot_doc_page",
    "search_datarobot_agentic_docs",
]
