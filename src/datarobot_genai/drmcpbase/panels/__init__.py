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

"""Panels exposed as read-only MCP resources.

Resources live in ``drmcpbase`` (the shared resources/prompts layer) rather than
``drtools`` (tools) so both DRMCP and global-mcp can register them onto their own
FastMCP instance via :func:`register_panel_resources` — neither server's mcp
singleton is reachable from a shared layer, so registration is instance-passing.
"""

from .resources import register_panel_resources

__all__ = ["register_panel_resources"]
