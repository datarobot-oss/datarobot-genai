"""NVIDIA cuOpt optimization tool.

Ported from wren-mcp cuopt_solve.py. Requires a cuOpt deployment to be
configured via the CUOPT_DEPLOYMENT_ID environment variable.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from datarobot_genai.mcp_tools._registry import register_tool

logger = logging.getLogger(__name__)


async def cuopt_solve(
    problem_definition: dict[str, Any],
    preview: bool = False,
) -> dict[str, Any]:
    """Solve LP, MILP, or VRP optimization problems using NVIDIA cuOpt.

    Requires CUOPT_DEPLOYMENT_ID environment variable pointing to a
    DataRobot deployment running the cuOpt solver.

    Set preview=True to validate the problem definition without solving.
    Returns the solution or validation results.
    """
    deployment_id = os.environ.get("CUOPT_DEPLOYMENT_ID")
    if not deployment_id:
        return {
            "error": (
                "CUOPT_DEPLOYMENT_ID not configured. "
                "Set this environment variable to the DataRobot deployment ID "
                "running the cuOpt solver."
            ),
            "solution": None,
        }

    import datarobot as dr

    client = dr.Client()

    if preview:
        # Validation only — check the problem definition without solving
        response = client.post(
            f"deployments/{deployment_id}/predictions/",
            json={"data": [{"problem": problem_definition, "mode": "validate"}]},
        )
        data = response.json()
        return {"preview": True, "validation": data}

    response = client.post(
        f"deployments/{deployment_id}/predictions/",
        json={"data": [{"problem": problem_definition, "mode": "solve"}]},
    )
    data = response.json()
    predictions = data.get("data", [])
    if not predictions:
        return {"error": "No solution returned from cuOpt deployment", "raw": data}

    result = predictions[0]
    return {
        "preview": False,
        "status": result.get("status"),
        "objective_value": result.get("objective_value"),
        "solution": result.get("solution"),
        "solver_info": result.get("solver_info"),
    }


register_tool(
    "cuopt_solve",
    cuopt_solve,
    "Solve LP, MILP, or VRP optimization problems using NVIDIA cuOpt via a DataRobot deployment.",
    "wren_tools",
)
