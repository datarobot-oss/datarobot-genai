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

"""Unit tests for cuOpt payload normalization (VRPData -> native, type inference)."""

import pytest

from datarobot_genai.drtools.cuopt import payload as cp
from tests.drtools.unit.cuopt.conftest import sample_lp_data
from tests.drtools.unit.cuopt.conftest import sample_milp_data


def test_vrp_data_to_native() -> None:
    # GIVEN a high-level VRP problem (depot + customers with x/y coordinates)
    # WHEN converting to the native cuOpt OptimizedRoutingData format
    native = cp.vrp_data_to_native(
        {
            "depot": {"x": 0, "y": 0},
            "customers": [
                {"id": "A", "x": 3, "y": 4, "demand": 2},
                {"id": "B", "x": 6, "y": 8},
            ],
            "num_vehicles": 2,
            "vehicle_capacity": 10,
        }
    )

    # THEN the euclidean cost matrix, task data, and fleet data are assembled
    matrix = native["cost_matrix_data"]["data"]["0"]
    assert len(matrix) == 3 and all(len(row) == 3 for row in matrix)
    assert matrix[0][1] == pytest.approx(5.0)  # depot -> A
    assert native["task_data"]["task_locations"] == [1, 2]
    assert native["task_data"]["task_ids"] == ["A", "B"]
    assert native["task_data"]["demand"] == [[2, 1]]  # missing demand defaults to 1
    assert native["fleet_data"]["capacities"] == [[10], [10]]
    assert native["fleet_data"]["vehicle_locations"] == [[0, 0], [0, 0]]


def test_vrp_data_to_native_latlng_coordinates() -> None:
    # GIVEN a VRP problem using lat/lng coordinates instead of x/y
    # WHEN converting to native format
    native = cp.vrp_data_to_native(
        {
            "depot": {"lat": 1.0, "lng": 2.0},
            "customers": [{"id": "A", "latitude": 4.0, "longitude": 6.0}],
        }
    )

    # THEN the cost matrix is computed over (lng, lat) pairs
    matrix = native["cost_matrix_data"]["data"]["0"]
    assert matrix[0][1] == pytest.approx(5.0)  # sqrt((6-2)^2 + (4-1)^2)


def test_vrp_format_detection() -> None:
    # GIVEN payloads in high-level and native VRP shapes
    # WHEN/THEN each detector recognizes only its own shape
    assert cp.is_high_level_vrp_format({"depot": {}, "customers": []})
    assert not cp.is_high_level_vrp_format({"fleet_data": {}, "task_data": {}})
    assert cp.is_native_vrp_format({"fleet_data": {}, "task_data": {}})
    assert not cp.is_native_vrp_format({"depot": {}, "customers": []})


def test_infer_problem_type_lp() -> None:
    # GIVEN a CSR payload with only continuous variables
    # WHEN inferring the problem type THEN it is LP
    assert cp.infer_problem_type(sample_lp_data()) == "lp"


def test_infer_problem_type_mip() -> None:
    # GIVEN a CSR payload with integer variables
    # WHEN inferring the problem type THEN it is MIP
    assert cp.infer_problem_type(sample_milp_data()) == "mip"


def test_infer_problem_type_vrp() -> None:
    # GIVEN a payload without a CSR constraint matrix
    # WHEN inferring the problem type THEN it defaults to VRP
    assert cp.infer_problem_type({"fleet_data": {}, "task_data": {}}) == "vrp"
