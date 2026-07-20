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

"""Unit tests for cuOpt solution-table normalization and error-response building."""

from datarobot_genai.drtools.cuopt import tables as ct
from tests.drtools.unit.cuopt.conftest import lp_solve_result
from tests.drtools.unit.cuopt.conftest import vrp_solve_result

# --------------------------------------------------------------------------- #
# Normalization helpers
# --------------------------------------------------------------------------- #


def test_normalize_primal_solution_scalar_values() -> None:
    # GIVEN a primal solution mapping variables to scalar values
    # WHEN normalizing
    df = ct._normalize_primal_solution({"x1": 1.0, "x2": 0.0})

    # THEN one row per variable is produced
    assert df["variable"].tolist() == ["x1", "x2"]
    assert df["value"].tolist() == [1.0, 0.0]


def test_normalize_primal_solution_nested_values() -> None:
    # GIVEN an indexed (nested-dict) primal solution
    # WHEN normalizing
    df = ct._normalize_primal_solution({"x": {"0": 1.0, "1": 2.0}})

    # THEN the index level is expanded into rows
    assert df["variable"].tolist() == ["x", "x"]
    assert df["index"].tolist() == ["0", "1"]
    assert df["value"].tolist() == [1.0, 2.0]


def test_normalize_primal_solution_non_dict_is_empty() -> None:
    # GIVEN a non-dict primal solution WHEN normalizing THEN the frame is empty
    assert ct._normalize_primal_solution([1, 2]).empty


def test_normalize_routes_expands_stops() -> None:
    # GIVEN routes with per-vehicle stop lists
    # WHEN normalizing
    df = ct._normalize_routes(vrp_solve_result()["solution"]["routes"])

    # THEN each stop becomes a row with its order preserved
    assert df["vehicle_id"].tolist() == [0, 0, 0, 0, 1, 1, 1]
    assert df["stop_order"].tolist() == [0, 1, 2, 3, 0, 1, 2]
    assert df["location"].tolist() == [0, 1, 2, 0, 0, 2, 0]


def test_normalize_routes_preserves_zero_vehicle_id() -> None:
    # GIVEN a route whose vehicle_id is 0 (falsy -- wren regression)
    # WHEN normalizing
    df = ct._normalize_routes([{"vehicle_id": 0, "stops": [0, 1, 0]}])

    # THEN the zero id is kept, not dropped
    assert df["vehicle_id"].tolist() == [0, 0, 0]


def test_normalize_vehicle_data_zips_list_fields() -> None:
    # GIVEN vehicle data mixing list-valued and scalar fields
    # WHEN normalizing
    df = ct._normalize_vehicle_data(
        {"veh-0": {"task_id": ["Depot", "A", "Depot"], "total_cost": 12.5}}
    )

    # THEN list fields are zipped per stop and scalar fields are excluded
    assert df["vehicle_id"].tolist() == ["veh-0"] * 3
    assert df["task_id"].tolist() == ["Depot", "A", "Depot"]
    assert "total_cost" not in df.columns


def test_list_and_dict_to_dataframe() -> None:
    # GIVEN list and dict inputs
    # WHEN converting to frames
    # THEN lists become a labelled column, dicts a keyed record set, junk is empty
    assert ct._list_to_dataframe(["a", "b"], value_label="task_id")["task_id"].tolist() == [
        "a",
        "b",
    ]
    assert ct._list_to_dataframe("not-a-list").empty
    df = ct._dict_to_dataframe({"k": {"v": 1}}, key_label="key")
    assert df["key"].tolist() == ["k"] and df["v"].tolist() == [1]


# --------------------------------------------------------------------------- #
# Solution-table extraction (wren's canned LP + VRP shapes)
# --------------------------------------------------------------------------- #


def test_extract_solution_tables_lp() -> None:
    # GIVEN a successful LP solve result
    # WHEN extracting solution tables
    tables, summary = ct.extract_solution_tables("lp", lp_solve_result())

    # THEN primal/dual/metrics tables and a human summary are produced
    titles = [t["title"] for t in tables]
    assert titles == ["Primal Solution", "Dual Solution", "Solution Metrics"]
    primal = tables[0]["dataframe"]
    assert primal["variable"].tolist() == ["x1", "x2"]
    assert primal["value"].tolist() == [0.5, 0.5]
    metrics = tables[2]["dataframe"]
    assert metrics["primal_objective"].tolist() == [2.5]
    assert "Request ID: req-lp" in summary
    assert "Detected problem type: LP" in summary
    assert "Total cost: 2.5" in summary
    assert "Variables with assignments: 2" in summary


def test_extract_solution_tables_vrp() -> None:
    # GIVEN a successful VRP solve result with performance metrics
    result = vrp_solve_result()
    result["performance_metrics"] = {"execution_time": 1.25}

    # WHEN extracting solution tables
    tables, summary = ct.extract_solution_tables("vrp", result)

    # THEN routes/vehicle/unassigned/metrics tables are produced in order
    titles = [t["title"] for t in tables]
    assert titles == [
        "cuOpt Vehicle Routes",
        "Vehicle Summary",
        "Vehicle Route Breakdown",
        "Unassigned Tasks",
        "Performance Metrics",
    ]
    routes = tables[0]["dataframe"]
    assert routes["vehicle_id"].nunique() == 2
    vehicle_summary = tables[1]["dataframe"]
    # list-valued fields are summarized as counts
    assert vehicle_summary["task_id_count"].tolist() == [3]
    assert tables[3]["dataframe"]["task_id"].tolist() == ["B"]
    assert "Detected problem type: VRP" in summary
    assert "Vehicles with routes: 2" in summary
    assert "execution_time: 1.25" in summary


def test_extract_solution_tables_empty_solution() -> None:
    # GIVEN a solve result carrying no solution content
    # WHEN extracting solution tables
    tables, summary = ct.extract_solution_tables(
        "lp", {"request_id": "r", "status": "success", "solution": {}, "cost": 0.0}
    )

    # THEN no tables are produced but the summary still reports status
    assert tables == []
    assert "Status: success" in summary


# --------------------------------------------------------------------------- #
# Error-response building
# --------------------------------------------------------------------------- #


def test_build_error_response_from_failed_status() -> None:
    # GIVEN a failed solve result without an explicit error message
    # WHEN building the error response
    response = ct.build_cuopt_error_response(
        {"status": "failed", "request_id": "req-x", "solution": {}}, "lp"
    )

    # THEN a structured error dict carries status, fallback message, and analysis
    assert response is not None
    assert response["status"] == "error"
    assert response["cuopt_status"] == "failed"
    assert response["error"] == "cuOpt optimization failed"
    assert response["problem_analysis"] == {"detected_type": "LP"}


def test_build_error_response_prefers_details_error() -> None:
    # GIVEN an HTTP-level error whose details carry the real cuOpt message
    # WHEN building the error response
    response = ct.build_cuopt_error_response(
        {
            "status": "error",
            "error": "HTTP 400",
            "details": {"error": "Invalid constraint bounds"},
        },
        "mip",
    )

    # THEN the detail message wins over the generic HTTP error
    assert response is not None
    assert response["error"] == "Invalid constraint bounds"
    assert response["details"]["cuopt_error"] == {"error": "Invalid constraint bounds"}


def test_build_error_response_timeout_hint() -> None:
    # GIVEN a timed-out solve WHEN building the error response
    response = ct.build_cuopt_error_response({"status": "timeout"}, "vrp")

    # THEN the hint suggests increasing the timeout
    assert response is not None
    assert "timed out" in response["hint"]


def test_build_error_response_none_on_success() -> None:
    # GIVEN a successful result WHEN building THEN no error response is produced
    assert ct.build_cuopt_error_response({"status": "success", "solution": {}}, "lp") is None
