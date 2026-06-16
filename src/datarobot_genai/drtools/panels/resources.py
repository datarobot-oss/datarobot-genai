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

Declares the discovery/read surface for panels via the ``@resource_metadata``
primitive (drtools-only, fastmcp-free); each MCP server registers these onto its
own FastMCP instance. URIs:

* ``panels://{source}``                 — list panels (metadata) in a source
* ``panels://{source}/{panel_id}``      — a single panel's metadata
* ``panels://{source}/{panel_id}/content`` — a panel's content (inline for
  Text/Json; a payload reference for Dataset/Chart)

All are read-only and gated on the per-user ``ENABLE_MCP_SANDBOX`` entitlement.
"""

from __future__ import annotations

import json

from datarobot_genai.drmcputils.files.store import DataRobotFilesBlobStore
from datarobot_genai.drmcputils.panels.access import _require_mcp_sandbox
from datarobot_genai.drmcputils.panels.models import Json
from datarobot_genai.drmcputils.panels.models import Text
from datarobot_genai.drmcputils.panels.store import PanelStore
from datarobot_genai.drtools.core import resource_metadata


def _store() -> PanelStore:
    return PanelStore(DataRobotFilesBlobStore())


@resource_metadata(
    uri="panels://{source}",
    name="panels_list",
    description="List panels (metadata only) in a source ('main' or 'staging').",
    mime_type="application/json",
    tags={"panels", "read"},
)
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


@resource_metadata(
    uri="panels://{source}/{panel_id}",
    name="panel_metadata",
    description="A single panel's metadata by id.",
    mime_type="application/json",
    tags={"panels", "read"},
)
async def panel_metadata_resource(source: str, panel_id: str) -> str:
    _require_mcp_sandbox()
    panel = await _store().get(panel_id)
    return panel.model_dump_json()


@resource_metadata(
    uri="panels://{source}/{panel_id}/content",
    name="panel_content",
    description=(
        "A panel's content: inline text for Text panels, the dict for Json panels; "
        "for Dataset/Chart panels, a reference to the stored payload."
    ),
    mime_type="application/json",
    tags={"panels", "read"},
)
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
