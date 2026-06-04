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

"""Panel CRUD tools (server-side store).

Each tool is gated at call time on the per-user **MCP_SANDBOX** entitlement
(fail-closed), reusing the existing sandbox entitlement rather than minting a
new flag. Panels are persisted via a :class:`PanelStore` backed by the
DataRobot Files API.
"""

from __future__ import annotations

import logging
from typing import Annotated
from typing import Any

from datarobot_genai.drtools.core import tool_metadata
from datarobot_genai.drtools.core.clients.datarobot import request_user_dr_client
from datarobot_genai.drtools.core.exceptions import ToolError
from datarobot_genai.drtools.core.exceptions import ToolErrorKind
from datarobot_genai.drtools.core.feature_flags import FeatureFlag
from datarobot_genai.drtools.files.store import DataRobotFilesBlobStore
from datarobot_genai.drtools.pagination import PAGINATION_MAX
from datarobot_genai.drtools.panels.models import Json
from datarobot_genai.drtools.panels.models import Text
from datarobot_genai.drtools.panels.store import DEFAULT_SOURCE
from datarobot_genai.drtools.panels.store import PanelStore

logger = logging.getLogger(__name__)

MCP_SANDBOX_FEATURE_FLAG = "MCP_SANDBOX"


def _require_mcp_sandbox() -> None:
    """Fail-closed unless the requesting user holds the MCP_SANDBOX entitlement."""
    try:
        with request_user_dr_client() as client:
            enabled = FeatureFlag.is_enabled(MCP_SANDBOX_FEATURE_FLAG, client=client)
    except ToolError:
        raise
    except Exception as exc:  # noqa: BLE001 - any FF lookup failure denies (fail-closed)
        raise ToolError(
            "Could not verify the MCP_SANDBOX entitlement required for panel tools.",
            kind=ToolErrorKind.AUTHENTICATION,
        ) from exc
    if not enabled:
        raise ToolError(
            "Panel tools require the MCP_SANDBOX entitlement.",
            kind=ToolErrorKind.AUTHENTICATION,
        )


def _get_store() -> PanelStore:
    return PanelStore(DataRobotFilesBlobStore())


@tool_metadata(
    tags={"panels", "read", "list", "daria"},
    description=(
        "[Panels—list] List panels (metadata only) in a source ('main' committed, "
        "'staging' session-scoped). Read-only. Next step: get_panel for a single panel's "
        "full metadata, or its content via the panels:// resource."
    ),
)
async def list_panels(
    *,
    source: Annotated[
        str, "Panel source: 'main' (committed) or 'staging' (session)."
    ] = DEFAULT_SOURCE,
    limit: Annotated[int, "Max panels to return (default 100)."] = PAGINATION_MAX,
) -> dict[str, Any]:
    _require_mcp_sandbox()
    panels = await _get_store().list(source=source, limit=limit)
    return {
        "panels": [p.model_dump(mode="json") for p in panels],
        "count": len(panels),
        "source": source,
    }


@tool_metadata(
    tags={"panels", "read", "daria"},
    description=(
        "[Panels—get] Fetch a single panel's metadata by id. Read-only. Bulky payloads "
        "(Dataset/Chart) are referenced by payload_files_id, not inlined here."
    ),
)
async def get_panel(
    panel_id: Annotated[str, "The panel id (returned by list_panels / create_*_panel)."],
) -> dict[str, Any]:
    _require_mcp_sandbox()
    if not panel_id:
        raise ToolError("panel_id must be provided", kind=ToolErrorKind.VALIDATION)
    panel = await _get_store().get(panel_id)
    return panel.model_dump(mode="json")


@tool_metadata(
    tags={"panels", "write", "create", "daria"},
    description=(
        "[Panels—create text] Create a Text panel (markdown narrative/report) in a source. "
        "Returns the created panel including its assigned id."
    ),
)
async def create_text_panel(
    *,
    title: Annotated[str, "Human-readable panel title."],
    text: Annotated[str, "Markdown body of the panel."],
    description: Annotated[str | None, "Optional short description."] = None,
    source: Annotated[
        str, "Target source ('main' or 'staging')."
    ] = DEFAULT_SOURCE,
    parents: Annotated[list[str] | None, "Parent panel ids for lineage."] = None,
) -> dict[str, Any]:
    _require_mcp_sandbox()
    if not title:
        raise ToolError("title must be provided", kind=ToolErrorKind.VALIDATION)
    panel = Text(title=title, text=text, description=description, parents=parents or [])
    created = await _get_store().create(panel, source=source)
    return created.model_dump(mode="json")


@tool_metadata(
    tags={"panels", "write", "create", "daria"},
    description=(
        "[Panels—create json] Create a Json panel (structured dict payload) in a source. "
        "Returns the created panel including its assigned id."
    ),
)
async def create_json_panel(
    *,
    title: Annotated[str, "Human-readable panel title."],
    data: Annotated[dict[str, Any], "JSON-serializable object stored on the panel."],
    description: Annotated[str | None, "Optional short description."] = None,
    source: Annotated[
        str, "Target source ('main' or 'staging')."
    ] = DEFAULT_SOURCE,
    parents: Annotated[list[str] | None, "Parent panel ids for lineage."] = None,
) -> dict[str, Any]:
    _require_mcp_sandbox()
    if not title:
        raise ToolError("title must be provided", kind=ToolErrorKind.VALIDATION)
    panel = Json(title=title, data=data, description=description, parents=parents or [])
    created = await _get_store().create(panel, source=source)
    return created.model_dump(mode="json")


@tool_metadata(
    tags={"panels", "write", "delete", "daria"},
    description=(
        "[Panels—delete] Delete a panel (manifest + payload) by id. Returns the deleted id."
    ),
)
async def delete_panel(
    panel_id: Annotated[str, "The id of the panel to delete."],
) -> dict[str, Any]:
    _require_mcp_sandbox()
    if not panel_id:
        raise ToolError("panel_id must be provided", kind=ToolErrorKind.VALIDATION)
    await _get_store().delete(panel_id)
    return {"deleted": True, "panel_id": panel_id}
