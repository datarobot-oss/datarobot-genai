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

Each tool is gated at call time on the per-user **ENABLE_MCP_SANDBOX** entitlement
(fail-closed), reusing the existing sandbox entitlement rather than minting a
new flag. Panels are persisted via a :class:`PanelStore` backed by the
DataRobot Files API.
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
from datarobot_genai.drmcputils.panels.models import Text
from datarobot_genai.drmcputils.panels.store import DEFAULT_SOURCE
from datarobot_genai.drtools.core import tool_metadata
from datarobot_genai.drtools.pagination import PAGINATION_MAX
from datarobot_genai.drtools.pagination import clamp_limit
from datarobot_genai.drtools.pagination import merge_pagination_metadata
from datarobot_genai.drtools.panels.schema_registry import SchemaRegistry
from datarobot_genai.drtools.panels.schema_registry import SchemaValidationError

logger = logging.getLogger(__name__)


@tool_metadata(
    tags={"panels", "read", "list", "daria"},
    description=(
        "[Panels—list] List panels (metadata only) in a source ('main' committed, "
        "'staging' session-scoped). Read-only. Next step: get_panel for a single panel's "
        "full metadata, or its content via the panels:// resource."
    ),
    display_name="Panels — List",
    description_ui=(
        "Returns a paginated list of panel metadata from a source, either the "
        "committed main store or the session-scoped staging store."
    ),
)
async def list_panels(
    *,
    source: Annotated[
        str, "Panel source: 'main' (committed) or 'staging' (session)."
    ] = DEFAULT_SOURCE,
    limit: Annotated[int, "Max panels to return (default/max 100)."] = PAGINATION_MAX,
    offset: Annotated[int, "Number of panels to skip, for paging."] = 0,
) -> dict[str, Any]:
    _require_mcp_sandbox()
    limit, note = clamp_limit(limit)
    offset = max(offset, 0)
    panels = await _get_store().list(source=source, limit=limit, offset=offset)
    results: dict[str, Any] = {
        "panels": [p.model_dump(mode="json") for p in panels],
        "count": len(panels),
        "source": source,
    }
    return merge_pagination_metadata(results, {}, note, offset=offset, limit=limit)


@tool_metadata(
    tags={"panels", "read", "daria"},
    description=(
        "[Panels—get] Fetch a single panel's metadata by ID. Read-only. Bulky payloads "
        "(Dataset/Chart) are referenced by payload_files_id, not inlined here."
    ),
    display_name="Panels — Get",
    description_ui=(
        "Fetches a single panel's metadata by ID, referencing bulky payloads "
        "rather than outputting them inline."
    ),
)
async def get_panel(
    panel_id: Annotated[str, "The panel ID (returned by list_panels / create_*_panel)."],
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
        "Returns the created panel including its assigned ID."
    ),
    display_name="Panels — Create text panel",
    description_ui="Creates a text panel containing a Markdown narrative or report in a source.",
)
async def create_text_panel(
    *,
    title: Annotated[str, "Human-readable panel title."],
    text: Annotated[str, "Markdown body of the panel."],
    description: Annotated[str | None, "Optional short description."] = None,
    source: Annotated[str, "Target source ('main' or 'staging')."] = DEFAULT_SOURCE,
    parents: Annotated[list[str] | None, "Parent panel IDs for lineage."] = None,
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
        "Returns the created panel including its assigned ID."
    ),
    display_name="Panels — Create JSON panel",
    description_ui="Creates a JSON panel storing a structured dictionary payload in a source.",
)
async def create_json_panel(
    *,
    title: Annotated[str, "Human-readable panel title."],
    data: Annotated[dict[str, Any], "JSON-serializable object stored on the panel."],
    description: Annotated[str | None, "Optional short description."] = None,
    source: Annotated[str, "Target source ('main' or 'staging')."] = DEFAULT_SOURCE,
    parents: Annotated[list[str] | None, "Parent panel IDs for lineage."] = None,
) -> dict[str, Any]:
    _require_mcp_sandbox()
    if not title:
        raise ToolError("title must be provided", kind=ToolErrorKind.VALIDATION)
    panel = Json(title=title, data=data, description=description, parents=parents or [])
    created = await _get_store().create(panel, source=source)
    return created.model_dump(mode="json")


@tool_metadata(
    tags={"panels", "schemas", "read", "list", "daria"},
    description=(
        "[Panels—schemas] List registered Pydantic schemas available for Json panel "
        "validation, optionally filtered by namespace. Read-only. Next step: "
        "describe_panel_schema for field details, validate_panel_data to check data."
    ),
    display_name="Panels — List schemas",
    description_ui=(
        "Lists registered schemas available for JSON panel validation, "
        "optionally filtered by namespace."
    ),
)
async def list_panel_schemas(
    *,
    namespace: Annotated[str | None, "Optional namespace filter (e.g. 'cuopt')."] = None,
) -> dict[str, Any]:
    _require_mcp_sandbox()
    schemas = SchemaRegistry.list_schemas(namespace=namespace)
    if not schemas:
        available_namespaces = {
            name.split(".")[0] for name in SchemaRegistry.list_schemas() if "." in name
        }
        return {
            "message": (
                f"No schemas found for namespace '{namespace}'"
                if namespace
                else "No schemas registered"
            ),
            "available_namespaces": sorted(available_namespaces),
            "hint": "Use list_panel_schemas() without a namespace to see all schemas.",
        }
    return {"schemas": schemas, "count": len(schemas)}


@tool_metadata(
    tags={"panels", "schemas", "read", "daria"},
    description=(
        "[Panels—schemas] Describe a registered schema in detail: fields with types and "
        "required status, the full JSON Schema, and an example value. Read-only."
    ),
    display_name="Panels — Describe schema",
    description_ui=(
        "Describes a registered schema in detail, including fields, types, "
        "required status, and an example value."
    ),
)
async def describe_panel_schema(
    schema_name: Annotated[str, "Full schema name (e.g. 'cuopt.VRPData')."],
) -> dict[str, Any]:
    _require_mcp_sandbox()
    if not schema_name:
        raise ToolError("schema_name must be provided", kind=ToolErrorKind.VALIDATION)
    try:
        return SchemaRegistry.describe(schema_name)
    except KeyError:
        return {
            "error": f"Schema '{schema_name}' not found",
            "available_schemas": sorted(SchemaRegistry.list_schemas().keys()),
            "hint": "Use list_panel_schemas() to see all available schemas.",
        }


@tool_metadata(
    tags={"panels", "schemas", "read", "daria"},
    description=(
        "[Panels—schemas] Validate data against a registered schema without creating a "
        "panel. Returns {valid: true, normalized_data} on success or {valid: false, "
        "errors} on failure. Read-only."
    ),
    display_name="Panels — Validate data",
    description_ui="Validates data against a registered schema without creating a panel.",
)
async def validate_panel_data(
    *,
    schema_name: Annotated[str, "Schema to validate against (e.g. 'cuopt.VRPData')."],
    data: Annotated[dict[str, Any], "The data to validate."],
) -> dict[str, Any]:
    _require_mcp_sandbox()
    if not schema_name:
        raise ToolError("schema_name must be provided", kind=ToolErrorKind.VALIDATION)
    try:
        normalized = SchemaRegistry.validate(schema_name, data)
    except KeyError:
        return {
            "valid": False,
            "error": f"Schema '{schema_name}' not found",
            "available_schemas": sorted(SchemaRegistry.list_schemas().keys()),
        }
    except SchemaValidationError as e:
        return {
            "valid": False,
            "errors": e.errors,
            "error_message": str(e),
            "hint": f"Use describe_panel_schema('{schema_name}') to see required fields.",
        }
    return {"valid": True, "normalized_data": normalized}


@tool_metadata(
    tags={"panels", "write", "delete", "daria"},
    description=(
        "[Panels—delete] Delete a panel (manifest + payload) by ID. Returns the deleted ID."
    ),
    display_name="Panels — Delete",
    description_ui="Deletes a panel and its payload by ID.",
)
async def delete_panel(
    panel_id: Annotated[str, "The ID of the panel to delete."],
) -> dict[str, Any]:
    _require_mcp_sandbox()
    if not panel_id:
        raise ToolError("panel_id must be provided", kind=ToolErrorKind.VALIDATION)
    await _get_store().delete(panel_id)
    return {"deleted": True, "panel_id": panel_id}
