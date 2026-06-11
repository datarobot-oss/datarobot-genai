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

"""Server-side panels: typed analytical artifacts persisted via the Files API.

`drtools`-only (fastmcp-free) so both DRMCP and global-mcp can consume it. The
CRUD tools in :mod:`datarobot_genai.drtools.panels.tools` are registered by each
MCP server's registry; the drmcp/global-mcp wiring lands with the panel
resources work (MODEL-23663).

Import directly from the submodules (e.g. :mod:`~datarobot_genai.drtools.panels.models`,
:mod:`~datarobot_genai.drtools.panels.store`); this package intentionally re-exports
nothing to avoid a public-API compatibility surface.
"""
