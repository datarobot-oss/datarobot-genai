"""DataRobot use case tools.

Ported from wren-mcp bi_use_case.py.
"""
from __future__ import annotations

import logging

from datarobot_genai.mcp_tools._registry import register_tool

logger = logging.getLogger(__name__)


async def list_use_cases(
    search: str | None = None,
    limit: int = 100,
) -> list[str]:
    """List DataRobot use cases with optional search filter.

    Returns a list of strings in the format 'name (id)'.
    """
    import datarobot as dr

    client = dr.Client()
    params: dict = {"limit": limit}
    if search:
        params["search"] = search
    response = client.get("useCases/", params=params)
    items = response.json().get("data", [])
    return [f"{item.get('name', 'Untitled')} ({item['id']})" for item in items]


async def list_use_case_assets(use_case_id: str) -> str:
    """List datasets, deployments, and experiments belonging to a use case.

    Returns a formatted string summary of all assets in the use case.
    """
    import datarobot as dr

    use_case = dr.UseCase.get(use_case_id)
    lines = [f"Assets for use case '{use_case.name}' ({use_case_id}):"]

    try:
        datasets = list(use_case.list_datasets())
        lines.append(f"\nDatasets ({len(datasets)}):")
        for d in datasets:
            lines.append(f"  [{d.id}] {d.name}")
    except Exception as exc:
        lines.append(f"\nDatasets: error — {exc}")

    try:
        deployments = list(use_case.list_deployments())
        lines.append(f"\nDeployments ({len(deployments)}):")
        for d in deployments:
            lines.append(f"  [{d.id}] {d.label}")
    except Exception as exc:
        lines.append(f"\nDeployments: error — {exc}")

    try:
        projects = list(use_case.list_projects())
        lines.append(f"\nExperiments ({len(projects)}):")
        for p in projects:
            lines.append(f"  [{p.id}] {p.project_name}")
    except Exception as exc:
        lines.append(f"\nExperiments: error — {exc}")

    return "\n".join(lines)


register_tool(
    "list_use_cases",
    list_use_cases,
    "List DataRobot use cases with optional search filter.",
    "wren_tools",
)
register_tool(
    "list_use_case_assets",
    list_use_case_assets,
    "List datasets, deployments, and experiments belonging to a use case.",
    "wren_tools",
)
