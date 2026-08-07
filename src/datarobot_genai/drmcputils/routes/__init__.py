# Copyright 2026 DataRobot, Inc.
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

"""Custom (non-MCP) HTTP routes shared by the DataRobot MCP servers."""

from datarobot_genai.drmcputils.routes.metadata import register_metadata_routes
from datarobot_genai.drmcputils.routes.tool_gallery import register_tool_gallery_routes
from datarobot_genai.drmcputils.routes.trailing_slash import DEFAULT_MCP_PATH
from datarobot_genai.drmcputils.routes.trailing_slash import SlashRule
from datarobot_genai.drmcputils.routes.trailing_slash import TrailingSlashNormalizer
from datarobot_genai.drmcputils.routes.trailing_slash import default_slash_rules
from datarobot_genai.drmcputils.routes.trailing_slash import mcp_slash_rule
from datarobot_genai.drmcputils.routes.trailing_slash import shared_route_slash_rules

__all__ = [
    "DEFAULT_MCP_PATH",
    "SlashRule",
    "TrailingSlashNormalizer",
    "default_slash_rules",
    "mcp_slash_rule",
    "register_metadata_routes",
    "register_tool_gallery_routes",
    "shared_route_slash_rules",
]
