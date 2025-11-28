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

from datarobot_genai.drmcp.core.dr_mcp_server import DataRobotMCPServer
from datarobot_genai.drmcp.core.mcp_instance import TaggedFastMCP


def test_mcp_server_capabilities():
    """Server should declare required MCP capabilities."""
    mcp = TaggedFastMCP()
    DataRobotMCPServer(mcp)

    opts = mcp._mcp_server.create_initialization_options()

    assert opts.capabilities.prompts.listChanged is True
    assert opts.capabilities.experimental == {"dynamic_prompts": {"enabled": True}}
