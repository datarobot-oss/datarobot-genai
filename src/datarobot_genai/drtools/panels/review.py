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

"""Panel review/introspection tools (ported from wren-mcp's bi_review).

``inspect_panel`` walks a panel's ``parents`` lineage without hydrating payloads;
``view_json_panel`` returns a Json panel's data, truncated structure-preservingly
for LLM consumption. Both are read-only and ``ENABLE_MCP_SANDBOX``-gated.
"""

from __future__ import annotations

import logging
from typing import Annotated
from typing import Any

from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.exceptions import ToolErrorKind
from datarobot_genai.drmcputils.panels.access import _get_store
from datarobot_genai.drmcputils.panels.access import _require_mcp_sandbox
from datarobot_genai.drmcputils.panels.models import Json
from datarobot_genai.drtools.core import tool_metadata
from datarobot_genai.drtools.panels.truncate import truncate_for_llm
from datarobot_genai.drtools.panels.truncate import truncate_source_code

logger = logging.getLogger(__name__)

_MAX_LINEAGE_DEPTH = 10


def _node_summary(panel: Any) -> dict[str, Any]:
    """Lineage-relevant metadata for one panel (no payload hydration)."""
    execution_context = panel.execution_context or {}
    if isinstance(execution_context.get("code"), str):
        execution_context = {
            **execution_context,
            "code": truncate_source_code(execution_context["code"]),
        }
    return {
        "id": panel.id,
        "type": panel.type.value,
        "title": panel.title,
        "parents": list(panel.parents),
        "execution_context": truncate_for_llm(execution_context),
        "updated_at": panel.updated_at,
    }


@tool_metadata(
    tags={"panels", "read", "lineage", "daria"},
    description=(
        "[Panels—inspect] Walk a panel's parent lineage recursively: execution context "
        "and parent graph for the panel and its ancestors, without reading payload data. "
        "Read-only. Use this to understand where a derived panel came from."
    ),
    display_name="Panels — Inspect",
    description_ui=(
        "Walks a panel's parent lineage recursively, returning execution context "
        "and the ancestor graph without reading payloads."
    ),
)
async def inspect_panel(
    panel_id: Annotated[str, "The panel ID whose lineage to inspect."],
) -> dict[str, Any]:
    _require_mcp_sandbox()
    if not panel_id:
        raise ToolError("panel_id must be provided", kind=ToolErrorKind.VALIDATION)

    store = _get_store()
    nodes: dict[str, dict[str, Any]] = {}

    async def _walk(pid: str, depth: int) -> None:
        if pid in nodes or depth > _MAX_LINEAGE_DEPTH:
            return
        try:
            panel = await store.get(pid)
        except Exception as exc:  # noqa: BLE001 - a missing ancestor shouldn't kill the walk
            # Includes ToolError: the Files backend maps missing blobs to a
            # NOT_FOUND ToolError, and a stale parent id must degrade gracefully.
            logger.warning("Could not load ancestor panel %s: %s", pid, exc)
            nodes[pid] = {"id": pid, "error": "panel unavailable"}
            return
        nodes[pid] = _node_summary(panel)
        for parent_id in panel.parents:
            await _walk(parent_id, depth + 1)

    # The root panel must exist; ancestors degrade gracefully.
    root = await store.get(panel_id)
    nodes[panel_id] = _node_summary(root)
    for parent_id in root.parents:
        await _walk(parent_id, 1)

    return {
        "panel_id": panel_id,
        "graph": nodes,
        "node_count": len(nodes),
        "max_depth": _MAX_LINEAGE_DEPTH,
    }


@tool_metadata(
    tags={"panels", "read", "json", "daria"},
    description=(
        "[Panels—view json] View a Json panel's structured data (large structures are "
        "truncated to preserve shape: first 5 array items, 200-char strings, 6 levels). "
        "Read-only; for tabular panels use preview_dataset_panel."
    ),
    display_name="Panels — View JSON panel",
    description_ui=(
        "Views a JSON panel's structured data, truncating large structures to preserve their shape."
    ),
)
async def view_json_panel(
    panel_id: Annotated[str, "Id of the Json panel to view."],
) -> dict[str, Any]:
    _require_mcp_sandbox()
    if not panel_id:
        raise ToolError("panel_id must be provided", kind=ToolErrorKind.VALIDATION)
    panel = await _get_store().get(panel_id)
    if not isinstance(panel, Json):
        raise ToolError(
            f"Panel {panel_id} is a {panel.type.value} panel; view_json_panel only supports "
            "Json panels (use preview_dataset_panel for tabular data).",
            kind=ToolErrorKind.VALIDATION,
        )
    return {
        "panel_id": panel_id,
        "title": panel.title,
        "description": panel.description,
        "data": truncate_for_llm(panel.data),
    }
