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

"""Query the live Workload API OpenAPI specification.

The spec is fetched from ``{DataRobot endpoint}/openapi.yaml`` — the source of
truth for the deployed environment — and cached per process, so it never
drifts from the deployed API the way a bundled snapshot would. There is
intentionally no full-spec dump: responses are targeted and size-capped so the
multi-MB spec never floods agent context.
"""

import threading
from typing import Annotated
from typing import Any

import yaml
from datarobot.errors import ClientError

from datarobot_genai.drmcputils.client_exceptions import raise_tool_error_for_client_error
from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.exceptions import ToolErrorKind
from datarobot_genai.drtools.core import tool_metadata
from datarobot_genai.drtools.core.clients.datarobot_workload import WorkloadApiClient

# Cap on any single response body so the full spec (which can be several MB)
# never lands in the agent's context. Targeted lookups (one schema/path) stay
# well under this; anything larger is truncated with guidance to narrow the query.
MAX_SPEC_RESPONSE_CHARS = 12000

_SECTIONS = ("info", "paths", "schemas")

# Per-process cache holder: the parsed spec lives under the "spec" key.
_spec_cache: dict[str, dict[str, Any]] = {}
_spec_lock = threading.Lock()


def _load_spec() -> dict[str, Any]:
    """Fetch, parse, and cache the live OpenAPI spec (once per process)."""
    if "spec" not in _spec_cache:
        with _spec_lock:
            if "spec" not in _spec_cache:
                try:
                    text = WorkloadApiClient().get_openapi_spec_text()
                except ClientError as exc:
                    raise_tool_error_for_client_error(exc)
                try:
                    parsed = yaml.safe_load(text)
                except yaml.YAMLError as exc:
                    raise ToolError(
                        f"Failed to parse the OpenAPI spec as YAML: {exc}",
                        kind=ToolErrorKind.UPSTREAM,
                    ) from exc
                if not isinstance(parsed, dict):
                    raise ToolError(
                        "The OpenAPI spec endpoint returned unexpected content "
                        "(expected a YAML mapping).",
                        kind=ToolErrorKind.UPSTREAM,
                    )
                _spec_cache["spec"] = parsed
    return _spec_cache["spec"]


def _dump_capped(obj: Any) -> str:
    """YAML-dump an object, truncating to keep responses out of context bloat."""
    text = yaml.dump(obj, default_flow_style=False, sort_keys=False)
    if len(text) > MAX_SPEC_RESPONSE_CHARS:
        text = text[:MAX_SPEC_RESPONSE_CHARS] + (
            f"\n... [truncated at {MAX_SPEC_RESPONSE_CHARS} chars]. Narrow your "
            "query — request a specific schema_name, path, or search term instead "
            "of a whole section."
        )
    return text


def _schema_lookup(spec: dict[str, Any], schema_name: str) -> dict[str, Any]:
    schemas = spec.get("components", {}).get("schemas", {})
    if schema_name in schemas:
        return {"schema_name": schema_name, "definition": _dump_capped(schemas[schema_name])}
    for name, schema in schemas.items():
        if name.lower() == schema_name.lower():
            return {"schema_name": name, "definition": _dump_capped(schema)}
    similar = [n for n in schemas if schema_name.lower() in n.lower()]
    hint = f" Similar schemas: {similar[:10]}." if similar else ""
    raise ToolError(
        f"Schema {schema_name!r} not found in the OpenAPI spec.{hint} "
        "Use read_openapi_spec(section='schemas') to list all schemas.",
        kind=ToolErrorKind.NOT_FOUND,
    )


def _path_lookup(spec: dict[str, Any], path: str) -> dict[str, Any]:
    paths = spec.get("paths", {})
    if not path.startswith("/"):
        path = "/" + path
    if path in paths:
        return {"path": path, "definition": _dump_capped(paths[path])}
    matching = [p for p in paths if path in p]
    if matching:
        return {
            "path": path,
            "matches": [{"path": p, "definition": _dump_capped(paths[p])} for p in matching[:5]],
        }
    raise ToolError(
        f"Path {path!r} not found in the OpenAPI spec. "
        "Use read_openapi_spec(section='paths') to list all paths.",
        kind=ToolErrorKind.NOT_FOUND,
    )


def _search_spec(spec: dict[str, Any], search: str) -> dict[str, Any]:
    needle = search.lower()
    matched_schemas: list[str] = []
    matched_paths: list[str] = []
    matched_fields: list[str] = []

    schemas = spec.get("components", {}).get("schemas", {})
    for name, schema in schemas.items():
        if needle in name.lower():
            matched_schemas.append(name)
        for prop_name in schema.get("properties") or {}:
            if needle in prop_name.lower():
                matched_fields.append(f"{name}.{prop_name}")

    for path_name, path_def in spec.get("paths", {}).items():
        if needle in path_name.lower():
            matched_paths.append(path_name)
        for method, op in path_def.items():
            if isinstance(op, dict):
                op_id = op.get("operationId", "")
                summary = op.get("summary", "")
                if needle in op_id.lower() or needle in summary.lower():
                    matched_paths.append(f"{method.upper()} {path_name}: {summary}")

    return {
        "query": search,
        "schemas": matched_schemas[:15],
        "paths": matched_paths[:15],
        "fields": matched_fields[:20],
        "note": (
            "Use read_openapi_spec(schema_name=...) or read_openapi_spec(path=...) "
            "for full definitions."
        ),
    }


def _section_view(spec: dict[str, Any], section: str) -> dict[str, Any]:
    if section == "info":
        return {"info": _dump_capped(spec.get("info", {}))}
    if section == "schemas":
        schemas = spec.get("components", {}).get("schemas", {})
        return {
            "count": len(schemas),
            "schemas": sorted(schemas.keys()),
            "note": "Use read_openapi_spec(schema_name='SchemaName') for a definition.",
        }
    # section == "paths"
    paths = spec.get("paths", {})
    methods_by_path = {
        p: sorted(m.upper() for m in defn if m in ("get", "post", "put", "patch", "delete"))
        for p, defn in sorted(paths.items())
    }
    return {
        "count": len(paths),
        "paths": methods_by_path,
        "note": "Use read_openapi_spec(path='/path') for endpoint details.",
    }


def _overview(spec: dict[str, Any]) -> dict[str, Any]:
    info = spec.get("info", {})
    paths = spec.get("paths", {})
    schemas = spec.get("components", {}).get("schemas", {})
    return {
        "source": "live ({DataRobot endpoint}/openapi.yaml, cached per process)",
        "title": info.get("title"),
        "version": info.get("version"),
        "endpoint_count": len(paths),
        "schema_count": len(schemas),
        "key_paths": sorted(paths.keys())[:10],
        "usage": [
            "read_openapi_spec(section='paths') — list all API endpoints",
            "read_openapi_spec(section='schemas') — list all schema definitions",
            "read_openapi_spec(schema_name='CreateWorkloadRequest') — one schema",
            "read_openapi_spec(path='/workloads') — one endpoint",
            "read_openapi_spec(search='replica') — search by keyword",
        ],
    }


@tool_metadata(
    tags={"workload", "openapi", "spec", "datarobot", "get", "search"},
    description=(
        "[Workload—OpenAPI spec] Query the Workload API OpenAPI specification. Use "
        "this to understand endpoints, request/response formats, and valid field "
        "values before making API calls. The spec is fetched live from the deployed "
        "environment's /openapi.yaml (cached per process), so it never drifts from "
        "the running API. Queries are targeted and size-capped — there is no "
        "full-spec dump. Precedence when multiple arguments are set: schema_name, "
        "then path, then search, then section.\n\n"
        "Example (overview): read_openapi_spec()\n"
        "Example (schema):   read_openapi_spec(schema_name='CreateWorkloadRequest')\n"
        "Example (endpoint): read_openapi_spec(path='/workloads')\n"
        "Example (search):   read_openapi_spec(search='replica')"
    ),
    display_name="Workload — Read OpenAPI spec",
    description_ui=(
        "Query the live Workload API OpenAPI specification: overview, endpoint and "
        "schema lookups, or keyword search."
    ),
)
async def read_openapi_spec(
    *,
    section: Annotated[
        str | None,
        "Section to retrieve: 'info', 'paths', or 'schemas' (lists names, not full "
        "bodies). Omit for an overview.",
    ] = None,
    search: Annotated[
        str | None, "Keyword to search for in schema/field/path/operation names."
    ] = None,
    schema_name: Annotated[
        str | None, "Exact schema name to fetch (case-insensitive fallback)."
    ] = None,
    path: Annotated[str | None, "API path to inspect (e.g. '/workloads')."] = None,
) -> dict[str, Any]:
    if section is not None and section not in _SECTIONS:
        raise ToolError(
            f"Argument validation error: 'section' must be one of {_SECTIONS}. "
            "There is no full-spec dump — use targeted queries instead.",
            kind=ToolErrorKind.VALIDATION,
        )

    spec = _load_spec()

    if schema_name:
        return _schema_lookup(spec, schema_name)
    if path:
        return _path_lookup(spec, path)
    if search:
        return _search_spec(spec, search)
    if section:
        return _section_view(spec, section)
    return _overview(spec)
