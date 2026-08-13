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

"""Pre-defined MCP tool categories for header-based filtering.

Categories allow agents and the DR platform to request logical groups of tools
via the ``x-datarobot-mcp-tools`` header instead of enumerating individual
tool names.  Parents expand to all their leaf children; leaves expand to the
set of tool function names they contain.  Anything that does not match a known
category is kept as-is (treated as a plain tool name).

Hierarchy:
  dr_connectors
    dr_connector_confluence
    dr_connector_jira
    dr_connector_gdrive
    dr_connector_microsoft_sharepoint_onedrive
  dr_web_search
    dr_web_search_perplexity
    dr_web_search_tavily
  dr_documentation                       (leaf — no sub-categories)
  dr_use_cases                           (leaf — no sub-categories)
  dr_predictive
    dr_catalog
    dr_modeling
    dr_predictions
  dr_deployments                         (leaf — no sub-categories)
  dr_development
    dr_workload
    dr_file
  dr_visual
    dr_mcpapps
    dr_panels
  dr_db
    dr_vdb
  dr_user_tools                          (user-authored tools — outside the static taxonomy)
  dr_dynamic_tools                       (hosted tools — registered separately)
"""

from dataclasses import dataclass
from enum import StrEnum


class MCPToolCategory(StrEnum):
    # ── connector categories ─────────────────────────────────────────────────
    DR_CONNECTORS = "dr_connectors"
    DR_CONNECTOR_CONFLUENCE = "dr_connector_confluence"
    DR_CONNECTOR_JIRA = "dr_connector_jira"
    DR_CONNECTOR_GDRIVE = "dr_connector_gdrive"
    DR_CONNECTOR_MICROSOFT_SHAREPOINT_ONEDRIVE = "dr_connector_microsoft_sharepoint_onedrive"

    # ── web search categories ────────────────────────────────────────────────
    DR_WEB_SEARCH = "dr_web_search"
    DR_WEB_SEARCH_PERPLEXITY = "dr_web_search_perplexity"
    DR_WEB_SEARCH_TAVILY = "dr_web_search_tavily"

    # ── documentation (leaf) ────────────────────────────────────────────────
    DR_DOCUMENTATION = "dr_documentation"

    # ── use cases (leaf) ────────────────────────────────────────────────────
    DR_USE_CASES = "dr_use_cases"

    # ── predictive categories ────────────────────────────────────────────────
    DR_PREDICTIVE = "dr_predictive"
    DR_CATALOG = "dr_catalog"
    DR_MODELING = "dr_modeling"
    DR_PREDICTIONS = "dr_predictions"

    # ── deployments (leaf) ─────────────────────────────────────────────────
    DR_DEPLOYMENTS = "dr_deployments"

    # ── development categories ───────────────────────────────────────────────
    DR_DEVELOPMENT = "dr_development"
    DR_WORKLOAD = "dr_workload"
    DR_FILE = "dr_file"

    # ── visual categories ────────────────────────────────────────────────────
    DR_VISUAL = "dr_visual"
    DR_MCPAPPS = "dr_mcpapps"
    DR_PANELS = "dr_panels"

    # ── database categories ──────────────────────────────────────────────────
    DR_DB = "dr_db"
    DR_VDB = "dr_vdb"

    # ── special / marker-resolved ────────────────────────────────────────────
    # Tools the user authored in their own MCP server code (``dr_mcp_tool``'s
    # default USER_TOOL marker) — not part of the predefined static taxonomy.
    DR_USER_TOOLS = "dr_user_tools"
    DR_DYNAMIC_TOOLS = "dr_dynamic_tools"


# ── leaf category → tool names ───────────────────────────────────────────────

LEAF_CATEGORY_TOOLS: dict[str, frozenset[str]] = {
    MCPToolCategory.DR_CONNECTOR_CONFLUENCE: frozenset(
        {
            "confluence_get_page",
            "confluence_create_page",
            "confluence_add_comment",
            "confluence_search_space",
            "confluence_update_page",
        }
    ),
    MCPToolCategory.DR_CONNECTOR_JIRA: frozenset(
        {
            "jira_search_issues",
            "jira_get_issue",
            "jira_create_issue",
            "jira_update_issue",
            "jira_transition_issue",
        }
    ),
    MCPToolCategory.DR_CONNECTOR_GDRIVE: frozenset(
        {
            "gdrive_find_contents",
            "gdrive_read_and_export_content",
            "gdrive_create_file",
            "gdrive_update_metadata",
            "gdrive_manage_access",
        }
    ),
    MCPToolCategory.DR_CONNECTOR_MICROSOFT_SHAREPOINT_ONEDRIVE: frozenset(
        {
            "microsoft_graph_search_content",
            "microsoft_graph_share_item",
            "microsoft_graph_create_file",
            "microsoft_graph_update_metadata",
        }
    ),
    MCPToolCategory.DR_WEB_SEARCH_PERPLEXITY: frozenset(
        {
            "perplexity_search",
            "perplexity_sonar",
        }
    ),
    MCPToolCategory.DR_WEB_SEARCH_TAVILY: frozenset(
        {
            "tavily_search_web",
            "tavily_extract_text",
            "tavily_list_links",
            "tavily_crawl_site",
        }
    ),
    MCPToolCategory.DR_DOCUMENTATION: frozenset(
        {
            "search_datarobot_agentic_docs",
            "datarobot_docs_fetch_page",
        }
    ),
    MCPToolCategory.DR_USE_CASES: frozenset(
        {
            "datarobot_usecases_list",
            "usecases_list_assets",
        }
    ),
    MCPToolCategory.DR_CATALOG: frozenset(
        {
            "catalog_upload_dataset",
            "catalog_list_datasets",
            "catalog_get_preview",
            "catalog_list_datastores",
            "catalog_browse_datastore",
            "catalog_query_datastore",
            "catalog_analyze_dataset",
            "catalog_suggest_ml_problems",
            "catalog_get_eda_insights",
            "catalog_check_timeseries_eligibility",
        }
    ),
    MCPToolCategory.DR_MODELING: frozenset(
        {
            "modeling_list_projects",
            "modeling_get_project_dataset",
            "models_get_bestmodel",
            "modeling_score_dataset",
            "modeling_start_autopilot",
            "modeling_get_model_roc",
            "modeling_get_model_feature_impact",
            "modeling_get_model_lift_chart",
            "modeling_list_models",
            "modeling_get_modeldetails",
        }
    ),
    MCPToolCategory.DR_DEPLOYMENTS: frozenset(
        {
            "deployment_get_list",
            "deployment_get_model_info",
            "deployment_create_deployment",
            "deployment_get_prediction_history",
            "deployment_get_info",
            "deployment_generate_prediction_sample",
            "deployment_validate_prediction_data",
            "deployment_get_features",
        }
    ),
    MCPToolCategory.DR_PREDICTIONS: frozenset(
        {
            "predict_score_catalog_realtime",
            "predict_score_inline_realtime",
            "predict_batch_predictions_from_dataset",
            "predict_batch_predictions_from_partition",
            "predict_get_batch_job_status",
            "predict_get_batch_results",
        }
    ),
    MCPToolCategory.DR_WORKLOAD: frozenset(
        {
            "workload_list",
            "workload_get",
            "workload_create_payload_build",
            "workload_create",
            "workload_update",
            "workload_action_run",
            "workload_settings",
            "workload_artifact_replace",
            "workload_bundle_list",
            "workload_stats_get",
            "workload_logs_get",
            "workload_activity_get",
            "workload_proton_get",
            "artifact_get",
            "artifact_create",
            "artifact_update",
            "artifact_action_run",
            "artifact_repository_get",
            "artifact_repository_delete",
            "artifact_get_build",
            "artifact_build_run_action",
            "read_openapi_spec",
        }
    ),
    MCPToolCategory.DR_FILE: frozenset(
        {
            "file_import",
            "file_get_status",
            "file_list",
            "file_info",
            "file_read",
            "file_sign",
            "file_write",
            "file_upload",
            "file_manage",
        }
    ),
    MCPToolCategory.DR_MCPAPPS: frozenset(),  # placeholder — not yet implemented
    MCPToolCategory.DR_PANELS: frozenset(
        {
            "list_panels",
            "get_panel",
            "create_text_panel",
            "create_json_panel",
            "list_panel_schemas",
            "describe_panel_schema",
            "validate_panel_data",
            "delete_panel",
            "move_panel",
            "inspect_panel",
            "view_json_panel",
            "create_dataset_panel_from_connector",
            "preview_dataset_panel",
            "transform_panel",
            "filter_panel",
            "create_dataset_panel_from_catalog",
            "upload_dataset_panel_to_catalog",
            "query_datasets_to_panel",
            "get_prediction_history",
            "get_autopilot_status",
            "predict_with_deployment",
            "apply_what_if",
            "get_time_series_scoring_dataset_panel",
        }
    ),
    MCPToolCategory.DR_VDB: frozenset(
        {
            "vdb_create",
            "vdb_deploy",
            "vdb_get",
            "vdb_list",
            "vdb_query",
        }
    ),
    # Marker-resolved categories — tool names are resolved at request time
    # from each tool's ``meta.tool_category`` marker, not from this static
    # map.  Present so the names are recognised as categories; the empty set
    # is NOT what an allowlist expands them to (see MARKER_RESOLVED_CATEGORIES).
    MCPToolCategory.DR_USER_TOOLS: frozenset(),
    MCPToolCategory.DR_DYNAMIC_TOOLS: frozenset(),
}

# Categories whose membership is decided by a tool's ``meta.tool_category`` marker at
# request time rather than by any static list of names. They are the one kind of
# category ``resolve_to_tool_names`` cannot expand: it is a pure function over this
# taxonomy and never sees the server's catalog.
#
# So it passes them through as literal tokens and the *matcher* resolves them, where
# the tools are in hand (``drmcpbase.fastmcp_transforms.utils.is_tool_allowed``).
# Expanding them to the empty set instead — which is what the map above would do —
# made ``x-datarobot-mcp-tools: dr_user_tools`` a present-but-empty allowlist, and an
# empty allowlist is a hard deny: picking "Your own tools" hid every tool on the
# server. Neither of the two obvious readings of that header is "show me nothing".
MARKER_RESOLVED_CATEGORIES: frozenset[str] = frozenset(
    {
        MCPToolCategory.DR_USER_TOOLS.value,
        MCPToolCategory.DR_DYNAMIC_TOOLS.value,
    }
)

# ── parent category → leaf category names ────────────────────────────────────

PARENT_TO_CHILDREN: dict[str, frozenset[str]] = {
    MCPToolCategory.DR_CONNECTORS: frozenset(
        {
            MCPToolCategory.DR_CONNECTOR_CONFLUENCE,
            MCPToolCategory.DR_CONNECTOR_JIRA,
            MCPToolCategory.DR_CONNECTOR_GDRIVE,
            MCPToolCategory.DR_CONNECTOR_MICROSOFT_SHAREPOINT_ONEDRIVE,
        }
    ),
    MCPToolCategory.DR_WEB_SEARCH: frozenset(
        {
            MCPToolCategory.DR_WEB_SEARCH_PERPLEXITY,
            MCPToolCategory.DR_WEB_SEARCH_TAVILY,
        }
    ),
    MCPToolCategory.DR_PREDICTIVE: frozenset(
        {
            MCPToolCategory.DR_CATALOG,
            MCPToolCategory.DR_MODELING,
            MCPToolCategory.DR_PREDICTIONS,
        }
    ),
    MCPToolCategory.DR_DEVELOPMENT: frozenset(
        {
            MCPToolCategory.DR_WORKLOAD,
            MCPToolCategory.DR_FILE,
        }
    ),
    MCPToolCategory.DR_VISUAL: frozenset(
        {
            MCPToolCategory.DR_MCPAPPS,
            MCPToolCategory.DR_PANELS,
        }
    ),
    MCPToolCategory.DR_DB: frozenset(
        {
            MCPToolCategory.DR_VDB,
        }
    ),
}


# ── category → display label ─────────────────────────────────────────────────

# The human-readable name of every category, in UI display order. Single source of
# truth for the ``label`` on each node of ``GET /toolGallery/categories/`` (see
# ``drmcputils/category_tree.py``) and, through it, for the legal values of the
# gallery's ``category`` filter param — the keys here are the same ``dr_*`` strings
# emitted in each tool item's ``categories``.
#
# EVERY member of ``MCPToolCategory`` must appear, children included; a test pins
# that, so adding a category without a label fails CI rather than shipping a node
# labelled with its own raw ``dr_*`` string. This started life as a curated subset
# of five parents, which meant the filter panel could not reach 17 of the 120
# categorized tools — and on a user MCP it offered five categories the server has
# no tools in while omitting ``dr_user_tools``, the only bucket it does have.
# Which categories are *filterable* is now decided by the taxonomy's shape (top-level
# nodes), not by a hand-kept list that can silently fall behind it.
TOOL_CATEGORY_LABELS: dict[MCPToolCategory, str] = {
    # ── connectors ──────────────────────────────────────────────────────────
    MCPToolCategory.DR_CONNECTORS: "Data connectors",
    MCPToolCategory.DR_CONNECTOR_CONFLUENCE: "Confluence",
    MCPToolCategory.DR_CONNECTOR_JIRA: "Jira",
    MCPToolCategory.DR_CONNECTOR_GDRIVE: "Google Drive",
    MCPToolCategory.DR_CONNECTOR_MICROSOFT_SHAREPOINT_ONEDRIVE: "SharePoint & OneDrive",
    # ── web search ──────────────────────────────────────────────────────────
    MCPToolCategory.DR_WEB_SEARCH: "Web search",
    MCPToolCategory.DR_WEB_SEARCH_PERPLEXITY: "Perplexity",
    MCPToolCategory.DR_WEB_SEARCH_TAVILY: "Tavily",
    # ── documentation ───────────────────────────────────────────────────────
    MCPToolCategory.DR_DOCUMENTATION: "Documentation",
    # ── use cases ───────────────────────────────────────────────────────────
    MCPToolCategory.DR_USE_CASES: "Use cases",
    # ── predictive ──────────────────────────────────────────────────────────
    MCPToolCategory.DR_PREDICTIVE: "Predictive",
    MCPToolCategory.DR_CATALOG: "Data catalog",
    MCPToolCategory.DR_MODELING: "Modeling",
    MCPToolCategory.DR_PREDICTIONS: "Predictions",
    # ── deployments ─────────────────────────────────────────────────────────
    MCPToolCategory.DR_DEPLOYMENTS: "Deployments",
    # ── development ─────────────────────────────────────────────────────────
    MCPToolCategory.DR_DEVELOPMENT: "Software development & DevOps",
    MCPToolCategory.DR_WORKLOAD: "Workloads",
    MCPToolCategory.DR_FILE: "Files",
    # ── visual ──────────────────────────────────────────────────────────────
    MCPToolCategory.DR_VISUAL: "Data visualization",
    MCPToolCategory.DR_MCPAPPS: "Applications",
    MCPToolCategory.DR_PANELS: "Panels",
    # ── databases ───────────────────────────────────────────────────────────
    MCPToolCategory.DR_DB: "Databases",
    MCPToolCategory.DR_VDB: "Vector databases",
    # ── marker-resolved (user MCPs only) ────────────────────────────────────
    MCPToolCategory.DR_USER_TOOLS: "Your own tools",
    MCPToolCategory.DR_DYNAMIC_TOOLS: "Deployed tools",
}


def category_label(name: str) -> str:
    """Display label for a category, falling back to its raw ``dr_*`` name.

    The fallback exists so an unlabelled category degrades to something readable
    instead of a ``KeyError`` mid-response; the test that pins full coverage of
    ``MCPToolCategory`` is what keeps the fallback unreachable in practice.
    """
    try:
        return TOOL_CATEGORY_LABELS[MCPToolCategory(name)]
    except (KeyError, ValueError):
        return name


@dataclass(frozen=True, slots=True)
class ToolAllowlist:
    """A parsed ``x-datarobot-mcp-tools`` header, keeping *how* each name got here.

    Three buckets, because "the client named this tool" and "this name fell out of
    expanding a category" must not be treated alike:

    - ``explicit``: names the client wrote verbatim, plus unknown tokens (a typo stays
      a name that matches nothing). Admits any tool with that name, whatever it is.
    - ``derived``: names produced by expanding a *static* category. These describe
      DataRobot's own built-in tools, so they must only admit built-ins — a
      user-authored tool that happens to share a name is not the tool the category
      meant. Flattening these into ``explicit`` is what made ``dr_db`` admit a
      ``USER_TOOL`` called ``vdb_query`` while ``?category=dr_db`` returned nothing.
    - ``buckets``: marker-resolved categories (``dr_user_tools``/``dr_dynamic_tools``),
      which name no tools at all and are matched against a tool's own marker.

    Matching needs the tool object, so it lives beside the transform
    (``drmcpbase.fastmcp_transforms.utils.is_tool_allowed``); this stays a pure
    taxonomy type with no view of any catalog.
    """

    explicit: frozenset[str] = frozenset()
    derived: frozenset[str] = frozenset()
    buckets: frozenset[str] = frozenset()

    def may_admit_name(self, name: str) -> bool:
        """Report whether a tool with this name could be admitted at all.

        The cheap half of the decision, for callers holding only a name (``get_tool``
        rejects before resolving). ``True`` means "keep going", not "allowed": whether
        a derived name or a bucket admits *this* tool depends on its marker, which
        needs the tool in hand.
        """
        return name in self.explicit or name in self.derived or bool(self.buckets)

    def with_explicit(self, names: frozenset[str]) -> "ToolAllowlist":
        """Union in concrete tool names, as explicitly named (Tool Sets expansion).

        A Tool Set lists tool *functions* by name, which is the client naming them —
        so a user-authored tool in a bundle is admitted like any other.
        """
        return ToolAllowlist(self.explicit | names, self.derived, self.buckets)

    def __bool__(self) -> bool:
        return bool(self.explicit or self.derived or self.buckets)


def resolve_tool_allowlist(entries: frozenset[str]) -> ToolAllowlist:
    """Sort raw header tokens into explicit names, category-derived names and buckets.

    Resolution rules (per entry):
    1. Marker-resolved category  → a bucket, matched against the tool's own marker
    2. Parent category           → its leaf children's tool names, as *derived*
    3. Leaf category             → its tool names, as *derived*
    4. Anything else             → an *explicit* name (a plain tool name, or a typo
       that will simply match nothing — never an error)
    """
    explicit: set[str] = set()
    derived: set[str] = set()
    buckets: set[str] = set()
    for entry in entries:
        if entry in MARKER_RESOLVED_CATEGORIES:
            buckets.add(entry)
        elif entry in PARENT_TO_CHILDREN:
            for leaf in PARENT_TO_CHILDREN[entry]:
                derived.update(LEAF_CATEGORY_TOOLS.get(leaf, frozenset()))
        elif entry in LEAF_CATEGORY_TOOLS:
            derived.update(LEAF_CATEGORY_TOOLS[entry])
        else:
            explicit.add(entry)
    return ToolAllowlist(frozenset(explicit), frozenset(derived), frozenset(buckets))


def resolve_to_tool_names(entries: frozenset[str]) -> frozenset[str]:
    """Expand category names in *entries* to their constituent tool names.

    Resolution rules (applied per entry):
    1. Parent category  → expand to all leaf categories → expand each to tool names
    2. Leaf category    → expand to its tool names
    3. Anything else    → kept as-is (treated as a plain tool name)

    Unknown entries (typos, future categories) are silently kept as plain
    strings.  They will simply never match any registered tool name and the
    filter will ignore them — no error is raised.

    Marker-resolved categories expand to nothing here: their membership is a property
    of each tool's marker, not of this taxonomy. Use :func:`resolve_tool_allowlist`
    for the request path, which keeps them as buckets for the matcher to settle.

    Args:
        entries: Raw strings parsed from the ``x-datarobot-mcp-tools`` header.

    Returns
    -------
        Resolved set of tool function names (plain strings only).
    """
    allowlist = resolve_tool_allowlist(entries)
    return allowlist.explicit | allowlist.derived


def _parse_header_entries(raw: str | None) -> frozenset[str] | None:
    """Split a comma-separated header value into a frozenset of stripped tokens.

    Returns None when the header is absent or blank (means "no filter").
    """
    if raw is None:
        return None
    stripped = raw.strip()
    if not stripped:
        return None
    entries = frozenset(part.strip() for part in stripped.split(",") if part.strip())
    return entries if entries else None


def parse_toolset_names_header(raw: str | None) -> frozenset[str] | None:
    """Parse ``x-datarobot-mcp-toolsets`` to toolset bundle names (not tool names).

    Uses the same comma-separated token format as ``x-datarobot-mcp-tools``. Returns
    ``None`` when the header is absent or blank (no toolset filter).
    """
    return _parse_header_entries(raw)


def parse_tool_allowlist_header(raw: str | None) -> ToolAllowlist | None:
    """Parse the x-datarobot-mcp-tools header into a :class:`ToolAllowlist`.

    Static category names (e.g. ``dr_connectors``, ``dr_connector_jira``) expand to
    the tool names they contain; marker-resolved categories become buckets; plain
    tool names and unknown entries are kept as explicit names.

    Returns ``None`` when the header is absent or blank — *no filtering*. That is a
    different answer from an empty allowlist, which denies everything; see
    ``effective_tool_allowlist``.
    """
    entries = _parse_header_entries(raw)
    if entries is None:
        return None
    return resolve_tool_allowlist(entries)


# ── reverse index: tool name → its categories ────────────────────────────────

# Leaf category → its parent (if any).  Each leaf has at most one parent in this
# taxonomy; standalone leaves (e.g. dr_documentation) have none.
_LEAF_TO_PARENT: dict[str, str] = {
    leaf: parent for parent, leaves in PARENT_TO_CHILDREN.items() for leaf in leaves
}


def _build_tool_to_categories() -> dict[str, frozenset[str]]:
    """Map each tool name to its leaf category plus that leaf's parent (if any).

    This is the single-source-of-truth inverse of ``LEAF_CATEGORY_TOOLS`` — the
    tools-gallery and ARD catalog derive a tool's categories from here rather
    than duplicating them on each ``@tool_metadata`` decorator.
    """
    mapping: dict[str, set[str]] = {}
    for leaf, tools in LEAF_CATEGORY_TOOLS.items():
        parent = _LEAF_TO_PARENT.get(leaf)
        labels = {leaf, parent} if parent else {leaf}
        for tool_name in tools:
            mapping.setdefault(tool_name, set()).update(labels)
    return {name: frozenset(labels) for name, labels in mapping.items()}


TOOL_TO_CATEGORIES: dict[str, frozenset[str]] = _build_tool_to_categories()


def categories_for_tool(tool_name: str) -> list[str]:
    """Return the sorted category labels (leaf + parent) for *tool_name*.

    Empty list for hosted/dynamic tools and any name not in the static taxonomy.
    """
    return sorted(TOOL_TO_CATEGORIES.get(tool_name, frozenset()))
