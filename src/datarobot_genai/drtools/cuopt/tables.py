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

"""cuOpt solution-table normalization (ported verbatim from wren-mcp).

Turns a raw cuOpt solve result into a list of titled dataframe tables plus a
human-readable summary, and builds structured error responses for failed
solves. pandas is used for the heterogeneous-shape normalization (available
via the ``datarobot-predict`` dependency); frames are converted to polars at
the Parquet persistence boundary in :mod:`persistence`.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def _list_to_dataframe(values: Any, *, value_label: str = "value") -> pd.DataFrame:
    if not isinstance(values, list):
        return pd.DataFrame()
    if not values:
        return pd.DataFrame()
    if all(isinstance(item, dict) for item in values):
        return pd.json_normalize(values, sep="_")
    return pd.DataFrame({value_label: values})


def _dict_to_dataframe(mapping: Any, *, key_label: str, value_label: str = "value") -> pd.DataFrame:
    if not isinstance(mapping, dict):
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    for key, value in mapping.items():
        if isinstance(value, dict):
            record = {key_label: key}
            record.update(value)
            records.append(record)
        else:
            records.append({key_label: key, value_label: value})
    return pd.DataFrame.from_records(records)


def _normalize_primal_solution(primal: Any) -> pd.DataFrame:
    if not isinstance(primal, dict):
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    for variable, values in primal.items():
        if isinstance(values, dict):
            for index, value in values.items():
                if isinstance(value, dict):
                    record = {"variable": variable, "index": index}
                    record.update(value)
                    records.append(record)
                else:
                    records.append({"variable": variable, "index": index, "value": value})
        else:
            records.append({"variable": variable, "index": None, "value": values})
    return pd.DataFrame.from_records(records)


def _normalize_routes(routes: Any) -> pd.DataFrame:
    if not isinstance(routes, list):
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    for route in routes:
        if isinstance(route, dict):
            vehicle_id = None
            for key in ("vehicle_id", "vehicle", "id", "name"):
                if key in route and route.get(key) is not None:
                    vehicle_id = route.get(key)
                    break
            stops = route.get("stops") or route.get("route") or route.get("visits") or []
            if isinstance(stops, list):
                for order, stop in enumerate(stops):
                    record: dict[str, Any] = {
                        "vehicle_id": vehicle_id,
                        "stop_order": order,
                    }
                    if isinstance(stop, dict):
                        record.update(stop)
                    else:
                        record["location"] = stop
                    records.append(record)
            else:
                records.append(
                    {
                        "vehicle_id": vehicle_id,
                        "stop_order": None,
                        "route": stops,
                    }
                )
        else:
            records.append(
                {
                    "vehicle_id": None,
                    "stop_order": None,
                    "route": route,
                }
            )
    return pd.DataFrame.from_records(records)


def _normalize_vehicle_data(vehicle_data: Any) -> pd.DataFrame:
    if not isinstance(vehicle_data, dict):
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    for vehicle_id, info in vehicle_data.items():
        if not isinstance(info, dict):
            continue
        list_fields = {k: v for k, v in info.items() if isinstance(v, list)}
        if not list_fields:
            continue
        max_len = max((len(v) for v in list_fields.values()), default=0)
        for idx in range(max_len):
            record: dict[str, Any] = {"vehicle_id": vehicle_id, "stop_order": idx}
            for key, values in list_fields.items():
                record[key] = values[idx] if idx < len(values) else None
            records.append(record)
    return pd.DataFrame.from_records(records)


def extract_solution_tables(
    problem_type: str, result_dict: dict[str, Any]
) -> tuple[list[dict[str, Any]], str]:
    """Extract titled dataframe tables + a summary text from a cuOpt solve result.

    Returns
    -------
        A ``(tables, summary_text)`` pair where each table is a dict with
        ``title``, ``description``, and a pandas ``dataframe``.
    """
    tables: list[dict[str, Any]] = []
    solution = result_dict.get("solution", {})
    performance_metrics = result_dict.get("performance_metrics", {})
    request_id = result_dict.get("request_id", "unknown")

    summary_lines = [
        f"Request ID: {request_id}",
        f"Detected problem type: {problem_type.upper()}",
        f"Status: {result_dict.get('status', 'unknown')}",
        f"Total cost: {result_dict.get('cost', 'N/A')}",
        f"Constraints satisfied: {result_dict.get('constraints_satisfied', 'N/A')}",
    ]

    if isinstance(performance_metrics, dict) and performance_metrics:
        summary_lines.append("Performance metrics:")
        for metric, value in performance_metrics.items():
            summary_lines.append(f"  - {metric}: {value}")

    if isinstance(solution, dict):
        if problem_type == "vrp":
            routes_df = _normalize_routes(solution.get("routes"))
            if not routes_df.empty:
                summary_lines.append(f"Vehicles with routes: {routes_df['vehicle_id'].nunique()}")
                tables.append(
                    {
                        "title": "cuOpt Vehicle Routes",
                        "description": "Stop-by-stop routing plan assigned to each vehicle.",
                        "dataframe": routes_df,
                    }
                )

            vehicle_data = solution.get("vehicle_data")
            vehicle_df = _dict_to_dataframe(vehicle_data, key_label="vehicle_id")
            if not vehicle_df.empty:
                summary_df = vehicle_df.copy()
                for col in list(summary_df.columns):
                    if summary_df[col].apply(lambda x: isinstance(x, list)).any():
                        summary_df[f"{col}_count"] = summary_df[col].apply(
                            lambda x: len(x) if isinstance(x, list) else 0
                        )
                        summary_df.drop(columns=[col], inplace=True)
                tables.append(
                    {
                        "title": "Vehicle Summary",
                        "description": "Per-vehicle metrics reported by the solver.",
                        "dataframe": summary_df,
                    }
                )

                vehicle_route_df = _normalize_vehicle_data(vehicle_data)
                if not vehicle_route_df.empty:
                    tables.append(
                        {
                            "title": "Vehicle Route Breakdown",
                            "description": "Expanded route details per vehicle and stop.",
                            "dataframe": vehicle_route_df,
                        }
                    )

            unassigned_df = _list_to_dataframe(
                solution.get("unassigned_tasks") or [], value_label="task_id"
            )
            if not unassigned_df.empty:
                tables.append(
                    {
                        "title": "Unassigned Tasks",
                        "description": "Tasks that could not be assigned by the solver.",
                        "dataframe": unassigned_df,
                    }
                )

        if problem_type in {"mip", "lp"}:
            solver_solution = solution.get("solution")
            if isinstance(solver_solution, dict):
                primal_df = _normalize_primal_solution(solver_solution.get("primal_solution"))
                if not primal_df.empty:
                    summary_lines.append(
                        f"Variables with assignments: {primal_df['variable'].nunique()}"
                    )
                    tables.append(
                        {
                            "title": "Primal Solution",
                            "description": "Decision variable assignments returned by the solver.",
                            "dataframe": primal_df,
                        }
                    )

                dual_df = _normalize_primal_solution(solver_solution.get("dual_solution"))
                if not dual_df.empty:
                    tables.append(
                        {
                            "title": "Dual Solution",
                            "description": "Dual values (shadow prices) for constraints.",
                            "dataframe": dual_df,
                        }
                    )

                scalar_metrics = {
                    key: value
                    for key, value in solver_solution.items()
                    if not isinstance(value, (dict, list))
                }
                if scalar_metrics:
                    tables.append(
                        {
                            "title": "Solution Metrics",
                            "description": "Scalar metrics reported by the solver.",
                            "dataframe": pd.DataFrame([scalar_metrics]),
                        }
                    )

        generic_tables_keys = {
            "task_summary": ("Task Summary", "Task-level summary returned by cuOpt."),
            "solution_summary": (
                "Solution Summary",
                "High-level summary of solver output.",
            ),
            "assignments": (
                "Assignments",
                "Assignments returned by the solver.",
            ),
        }

        for key, (title, description) in generic_tables_keys.items():
            value = solution.get(key)
            if isinstance(value, list):
                df = _list_to_dataframe(value)
            elif isinstance(value, dict):
                df = _dict_to_dataframe(value, key_label="key")
            else:
                df = pd.DataFrame()
            if not df.empty:
                tables.append(
                    {
                        "title": title,
                        "description": description,
                        "dataframe": df,
                    }
                )

    if isinstance(performance_metrics, dict) and performance_metrics:
        perf_df = pd.DataFrame(
            [{"metric": key, "value": value} for key, value in performance_metrics.items()]
        )
        tables.append(
            {
                "title": "Performance Metrics",
                "description": "Execution metrics reported during optimization.",
                "dataframe": perf_df,
            }
        )

    summary_text = "\n".join(summary_lines)
    return tables, summary_text


def build_cuopt_error_response(
    result_dict: dict[str, Any], problem_type: str
) -> dict[str, Any] | None:
    """Build a structured error dict for a failed solve, or ``None`` on success."""
    status = result_dict.get("status")
    solution = result_dict.get("solution")
    details = result_dict.get("details", {})

    # Extract error message from multiple possible locations:
    # 1. Top-level "error" field (from client HTTP error handling)
    # 2. "details.error" field (cuOpt DetailModel format)
    # 3. "solution.error" or "solution.message" (solver response)
    error_message = result_dict.get("error")

    if isinstance(details, dict):
        detail_error = details.get("error")
        if detail_error and detail_error != error_message:
            error_message = detail_error

    if isinstance(solution, dict):
        solution_error = solution.get("error") or solution.get("message")
        if solution_error and not error_message:
            error_message = solution_error

    is_failure = status in {"failed", "timeout", "error"} or bool(error_message)
    if not is_failure:
        return None

    hint = "Review cuOpt error details and adjust the problem payload before re-running."
    if status == "timeout":
        hint = "Request timed out. Consider increasing timeout or simplifying the problem."

    error_details: dict[str, Any] = {
        "error_message": error_message,
        "performance_metrics": result_dict.get("performance_metrics", {}),
    }

    if isinstance(details, dict) and details:
        error_details["cuopt_error"] = details

    if isinstance(solution, dict) and solution:
        error_details["solution"] = solution

    return {
        "status": "error",
        "error": error_message or "cuOpt optimization failed",
        "cuopt_status": status,
        "request_id": result_dict.get("request_id", "unknown"),
        "problem_analysis": result_dict.get(
            "problem_analysis", {"detected_type": problem_type.upper()}
        ),
        "details": error_details,
        "hint": hint,
    }
