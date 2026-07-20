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
  dr_proxied_user_mcp                    (global-mcp only — proxied user MCPs)
  dr_dynamic_tools                       (hosted tools — registered separately)
"""

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

    # ── special / hosted ─────────────────────────────────────────────────────
    DR_PROXIED_USER_MCP = "dr_proxied_user_mcp"
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
    # Hosted categories — tool names are dynamic and resolved at request time,
    # not from this static map.  Kept here so category names are recognised and
    # not passed through as plain (unknown) tool names.
    MCPToolCategory.DR_PROXIED_USER_MCP: frozenset(),
    MCPToolCategory.DR_DYNAMIC_TOOLS: frozenset(),
}

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


def resolve_to_tool_names(entries: frozenset[str]) -> frozenset[str]:
    """Expand category names in *entries* to their constituent tool names.

    Resolution rules (applied per entry):
    1. Parent category  → expand to all leaf categories → expand each to tool names
    2. Leaf category    → expand to its tool names
    3. Anything else    → kept as-is (treated as a plain tool name)

    Unknown entries (typos, future categories) are silently kept as plain
    strings.  They will simply never match any registered tool name and the
    filter will ignore them — no error is raised.

    Args:
        entries: Raw strings parsed from the ``x-datarobot-mcp-tools`` header.

    Returns
    -------
        Resolved set of tool function names (plain strings only).
    """
    resolved: set[str] = set()
    for entry in entries:
        if entry in PARENT_TO_CHILDREN:
            # Parent → expand each leaf child to tool names
            for leaf in PARENT_TO_CHILDREN[entry]:
                resolved.update(LEAF_CATEGORY_TOOLS.get(leaf, frozenset()))
        elif entry in LEAF_CATEGORY_TOOLS:
            # Leaf → expand to tool names
            resolved.update(LEAF_CATEGORY_TOOLS[entry])
        else:
            # Plain tool name or unknown category — pass through
            resolved.add(entry)
    return frozenset(resolved)


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


def parse_tool_allowlist_header(raw: str | None) -> frozenset[str] | None:
    """Parse the x-datarobot-mcp-tools header and resolve any category names.

    Category names (e.g. ``dr_connectors``, ``dr_connector_jira``) are expanded
    to the set of tool function names they contain.  Plain tool names and unknown
    entries are kept as-is.  Returns None when the header is absent or blank,
    meaning no tool filtering should be applied.
    """
    entries = _parse_header_entries(raw)
    if entries is None:
        return None
    return resolve_to_tool_names(entries)


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
