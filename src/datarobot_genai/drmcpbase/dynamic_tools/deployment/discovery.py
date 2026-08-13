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
"""The single source of truth for how a deployment declares itself an MCP tool.

A DataRobot deployment becomes discoverable as an MCP tool by carrying the
tag ``tool=tool`` (see the ``datarobot-register-mcp-tool`` agent skill, which
sets it via ``Deployment.create_tag``). The public API filters multi-tag
queries with OR semantics, so every consumer must re-verify the exact
name/value pair on each result — this module is that shared predicate, used
by the startup batch registration, the async deployment lister, and the
request-time :class:`CustomModelToolProvider`.
"""

from collections.abc import Iterable
from collections.abc import Mapping
from typing import Any

TOOL_TAG_NAME = "tool"
TOOL_TAG_VALUE = "tool"

# Query params understood by GET /api/v2/deployments/ (OR-matched server-side).
TOOL_TAG_QUERY_PARAMS = {"tag_keys": TOOL_TAG_NAME, "tag_values": TOOL_TAG_VALUE}


def is_tool_tagged(tags: Iterable[Mapping[str, Any]] | None) -> bool:
    """Return True when a deployment's tags mark it as an MCP tool.

    ``tags`` is the ``tags`` list of a deployment API payload: mappings with
    ``name``/``value`` keys. Both must equal ``"tool"`` on a single tag.
    """
    return any(
        tag.get("name") == TOOL_TAG_NAME and tag.get("value") == TOOL_TAG_VALUE
        for tag in tags or ()
    )
