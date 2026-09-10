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

"""Custom (non-MCP) HTTP routes shared by DataRobot MCP servers."""

from datarobot_genai.drmcpbase.routes.helpers import apply_list_filters
from datarobot_genai.drmcpbase.routes.helpers import parse_list_filters
from datarobot_genai.drmcpbase.routes.helpers import parse_pagination
from datarobot_genai.drmcpbase.routes.static import STATIC_BASE_PATH
from datarobot_genai.drmcpbase.routes.static import register_static_routes

__all__ = [
    "STATIC_BASE_PATH",
    "apply_list_filters",
    "parse_list_filters",
    "parse_pagination",
    "register_static_routes",
]
