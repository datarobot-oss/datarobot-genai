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

The discovery/read surface for panels. URIs:

* ``panels://{source}``                     — list panels (metadata) in a source
* ``panels://{source}/{panel_id}``          — a single panel's metadata
* ``panels://{source}/{panel_id}/content``  — a panel's content (inline for
  Text/Json; a payload reference for Dataset/Chart)

All are read-only and gated on the per-user ``ENABLE_MCP_SANDBOX`` entitlement
(enforced inside each handler). The handlers live here in ``drmcpbase`` so both
DRMCP and global-mcp can register them onto their own FastMCP instance via
:func:`register_panel_resources`.
"""

from __future__ import annotations

import json
from typing import Any

from datarobot_genai.drmcputils.panels.access import _get_store
from datarobot_genai.drmcputils.panels.access import _require_mcp_sandbox
from datarobot_genai.drmcputils.panels.models import Json
from datarobot_genai.drmcputils.panels.models import Text
from datarobot_genai.drmcputils.panels.store import PanelStore


def _store() -> PanelStore:
    # Shared factory: conversation-scoped when the request carries the
    # x-datarobot-conversation-id header, unscoped otherwise.
    return _get_store()


async def panels_list_resource(source: str) -> str:
    _require_mcp_sandbox()
    panels = await _store().list(source=source)
    return json.dumps(
        {
            "source": source,
            "panels": [p.model_dump(mode="json") for p in panels],
            "count": len(panels),
        }
    )


async def panel_metadata_resource(source: str, panel_id: str) -> str:
    _require_mcp_sandbox()
    panel = await _store().get(panel_id)
    return panel.model_dump_json()


async def panel_content_resource(source: str, panel_id: str) -> str:
    _require_mcp_sandbox()
    panel = await _store().get(panel_id)
    if isinstance(panel, Text):
        return json.dumps({"type": "text", "text": panel.text})
    if isinstance(panel, Json):
        return json.dumps({"type": "json", "data": panel.data})
    return json.dumps(
        {
            "type": panel.type.value,
            "payload_files_id": panel.payload_files_id,
            "payload_name": panel.payload_name,
        }
    )


# (handler, resource-init kwargs) for each panel resource. Registered onto a
# server's FastMCP instance by register_panel_resources().
_PANEL_RESOURCES: list[tuple[Any, dict[str, Any]]] = [
    (
        panels_list_resource,
        {
            "uri": "panels://{source}",
            "name": "panels_list",
            "description": "List panels (metadata only) in a source ('main' or 'staging').",
            "mime_type": "application/json",
            "tags": {"panels", "read"},
        },
    ),
    (
        panel_metadata_resource,
        {
            "uri": "panels://{source}/{panel_id}",
            "name": "panel_metadata",
            "description": "A single panel's metadata by id.",
            "mime_type": "application/json",
            "tags": {"panels", "read"},
        },
    ),
    (
        panel_content_resource,
        {
            "uri": "panels://{source}/{panel_id}/content",
            "name": "panel_content",
            "description": (
                "A panel's content: inline text for Text panels, the dict for Json panels; "
                "for Dataset/Chart panels, a reference to the stored payload."
            ),
            "mime_type": "application/json",
            "tags": {"panels", "read"},
        },
    ),
]


def register_panel_resources(mcp: Any) -> None:
    """Register the panel resources onto *mcp* (a FastMCP instance).

    Instance-passing rather than a module-global decorator: ``drmcpbase`` is a
    shared layer with no mcp singleton, so each server (DRMCP, global-mcp) calls
    this with its own instance. The handlers are registered unwrapped so FastMCP
    sees the real coroutine functions and awaits them.
    """
    for handler, init_args in _PANEL_RESOURCES:
        mcp.resource(**init_args)(handler)
