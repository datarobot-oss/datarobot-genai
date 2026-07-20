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

"""High-level cuOpt solve flow composing normalization, the deployment call
seam, and panel persistence.

This is the entry point BPA's wren-named facade delegates to until the
global-MCP cutover. It is deliberately *not* an MCP tool -- the cuOpt NIM is a
tool-tagged deployment surfaced through dynamic tool discovery, and this
module is the reusable glue around calling it.
"""

from __future__ import annotations

import logging
from typing import Any

from datarobot_genai.drmcputils.panels.access import _get_store
from datarobot_genai.drmcputils.panels.access import _require_mcp_sandbox
from datarobot_genai.drmcputils.panels.models import Json
from datarobot_genai.drtools.cuopt.deployment import get_cuopt_client
from datarobot_genai.drtools.cuopt.payload import infer_problem_type
from datarobot_genai.drtools.cuopt.payload import is_high_level_vrp_format
from datarobot_genai.drtools.cuopt.payload import is_native_vrp_format
from datarobot_genai.drtools.cuopt.payload import vrp_data_to_native
from datarobot_genai.drtools.cuopt.persistence import persist_cuopt_panels
from datarobot_genai.drtools.cuopt.tables import build_cuopt_error_response
from datarobot_genai.drtools.cuopt.tables import extract_solution_tables

logger = logging.getLogger(__name__)


async def solve_with_cuopt_deployment(  # noqa: PLR0911 - one return per actionable error shape
    data: dict[str, Any] | str,
    *,
    preview_only: bool = True,
) -> dict[str, Any]:
    """Solve LP/MILP/VRP optimization problems with the cuOpt NIM deployment.

    For full data lineage, store the problem in a Json panel first
    (``create_json_panel``) and pass that panel's id as ``data`` -- the
    resulting solution panels are then linked to the input. You can also pass
    the payload dict directly (no lineage link). LP vs MILP is determined by
    ``variable_types``: all "C" -> LP; any "I" -> MILP (for binary variables
    use "I" with variable_bounds [0, 1]).

    Args:
        data: cuOpt payload -- a Json panel id (str, recommended for lineage)
            or an inline dict. Use ``describe_panel_schema("cuopt.NativeMILPData")``
            for the format.
        preview_only: If True (default), validate the payload without solving.
            If False, run the optimization and persist the solution as panels.

    Returns
    -------
        With ``preview_only=True``: the validation result dict
        (``{"status": "valid", ...}`` or error details). With
        ``preview_only=False``: a dict with the summary ``comments`` and the
        created ``panels`` (summary Text, raw-solution Json, and
        solution-table Datasets), or an error dict on solver failure.
    """
    if data is None:
        return {
            "status": "error",
            "error": "data parameter is required",
            "hint": (
                "Provide a dict or a Json panel id matching the cuOpt schema. Use "
                "describe_panel_schema('cuopt.NativeMILPData') for the format. Passing a "
                "Json panel id is recommended for data lineage tracking."
            ),
        }

    # Resolve a Json panel id to its data dict (keeping the id for lineage), or
    # accept an inline dict payload directly.
    parent_id: str | None = None
    payload_dict: dict[str, Any]
    if isinstance(data, str):
        _require_mcp_sandbox()
        try:
            panel = await _get_store().get(data)
        except Exception as exc:  # noqa: BLE001 - surface as actionable output
            return {
                "status": "error",
                "error": f"Failed to load Json panel '{data}': {exc}",
                "hint": "Ensure the id points to an existing Json panel.",
            }
        if not isinstance(panel, Json):
            return {
                "status": "error",
                "error": (
                    f"Panel '{data}' is a {panel.type.value} panel; solving needs a Json panel."
                ),
                "hint": "Create the problem with create_json_panel and pass its id.",
            }
        parent_id = data
        payload_dict = dict(panel.data)
    elif isinstance(data, dict):
        payload_dict = data
    else:
        return {
            "status": "error",
            "error": f"Unsupported data type: {type(data).__name__}",
            "hint": "Pass a cuOpt payload dict or a Json panel id (str).",
        }

    # Convert high-level VRPData format to native cuOpt format if needed.
    if is_high_level_vrp_format(payload_dict) and not is_native_vrp_format(payload_dict):
        logger.info("Converting high-level VRPData to native cuOpt format")
        try:
            payload_dict = vrp_data_to_native(payload_dict)
        except Exception as e:  # noqa: BLE001 - surface as actionable output
            return {
                "status": "error",
                "error": f"Failed to convert VRPData to native format: {e}",
                "hint": (
                    "Ensure your data has valid depot and customers with coordinates "
                    "(x/y or lat/lng)."
                ),
            }

    client = get_cuopt_client()

    if preview_only:
        result = await client.validate(payload_dict)
        if result.get("status") == "valid":
            result["next_step"] = (
                "Re-run with preview_only=False to run the optimization and "
                "persist the solution as panels."
            )
        return result

    # Gate before solving (not just before persisting): the panel-id path
    # already gated at read time above, but the inline-dict path must not run
    # an unentitled solve just because persistence happens afterward.
    _require_mcp_sandbox()
    result = await client.solve(payload_dict)

    problem_type = infer_problem_type(payload_dict)

    if result.get("status") in {"error", "failed", "timeout"}:
        error_response = build_cuopt_error_response(result, problem_type)
        if error_response is not None:
            return error_response
        return result

    tables, summary_text = extract_solution_tables(problem_type, result)
    response = await persist_cuopt_panels(
        tables=tables,
        summary_text=summary_text,
        result_dict=result,
        problem_type=problem_type,
        parent_id=parent_id,
    )
    logger.info("Optimization completed with status: %s", result.get("status"))
    return response
