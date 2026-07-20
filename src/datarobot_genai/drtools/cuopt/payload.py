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

"""cuOpt payload normalization (ported from wren-mcp's ``cuopt_solve``).

Format detection, the high-level ``VRPData`` -> native ``OptimizedRoutingData``
conversion, and problem-type inference. Pure functions -- no I/O.
"""

from __future__ import annotations

import math
from typing import Any

from datarobot_genai.drtools.cuopt.schemas import Coordinate
from datarobot_genai.drtools.cuopt.schemas import VRPData
from datarobot_genai.drtools.cuopt.schemas import VRPNode

_DEFAULT_VEHICLE_CAPACITY = 999999
_DEFAULT_TIME_LIMIT_SECONDS = 30


def is_native_vrp_format(raw: dict[str, Any]) -> bool:
    """Check if data is in native cuOpt VRP format (OptimizedRoutingData)."""
    native_keys = {"fleet_data", "task_data"}
    return native_keys.issubset(raw.keys())


def is_high_level_vrp_format(raw: dict[str, Any]) -> bool:
    """Check if data is in high-level VRPData format (depot, customers)."""
    return "depot" in raw and "customers" in raw


def infer_problem_type(payload_dict: dict[str, Any]) -> str:
    """Infer the problem type (``lp`` / ``mip`` / ``vrp``) from payload contents."""
    if "csr_constraint_matrix" in payload_dict:
        variable_types = payload_dict.get("variable_types")
        if isinstance(variable_types, list) and variable_types:
            normalized_types = {
                var_type.upper() for var_type in variable_types if isinstance(var_type, str)
            }
            if normalized_types and normalized_types.issubset({"C", "CONTINUOUS"}):
                return "lp"
        return "mip"
    return "vrp"


def _compute_euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Compute Euclidean distance between two points."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def _get_coords(  # noqa: PLR0911 - one return per supported coordinate form
    node: dict[str, Any] | Coordinate | VRPNode,
) -> tuple[float, float]:
    """Extract x, y coordinates from a node (supports dict or model)."""
    if isinstance(node, dict):
        x = node.get("x")
        y = node.get("y")
        if x is None or y is None:
            lat = node.get("lat") or node.get("latitude")
            lng = node.get("lng") or node.get("longitude")
            if lat is not None and lng is not None:
                return (lng, lat)
            loc = node.get("location")
            if loc and len(loc) >= 2:  # noqa: PLR2004 - (x, y) pair
                return (loc[0], loc[1])
            raise ValueError(f"Cannot extract coordinates from node: {node}")
        return (x, y)
    else:
        if node.x is not None and node.y is not None:
            return (node.x, node.y)
        if node.lat is not None and node.lng is not None:
            return (node.lng, node.lat)
        if node.latitude is not None and node.longitude is not None:
            return (node.longitude, node.latitude)
        if node.location and len(node.location) >= 2:  # noqa: PLR2004 - (x, y) pair
            return (node.location[0], node.location[1])
        raise ValueError(f"Cannot extract coordinates from node: {node}")


def vrp_data_to_native(data: VRPData | dict[str, Any]) -> dict[str, Any]:
    """Convert high-level VRPData to native cuOpt OptimizedRoutingData format."""
    if isinstance(data, dict):
        vrp = VRPData.model_validate(data)
    else:
        vrp = data

    depot_coords = _get_coords(vrp.depot)

    all_coords: list[tuple[float, float]] = [depot_coords]
    customer_demands: list[int] = []
    task_ids: list[str | int] = []

    for i, customer in enumerate(vrp.customers):
        coords = _get_coords(customer)
        all_coords.append(coords)
        demand = customer.demand if isinstance(customer, VRPNode) else customer.get("demand", 1)
        customer_demands.append(int(demand or 1))
        cust_id = customer.id if isinstance(customer, VRPNode) else customer.get("id", i + 1)
        task_ids.append(str(cust_id) if cust_id is not None else str(i + 1))

    n_locations = len(all_coords)

    cost_matrix: list[list[float]] = []
    for i in range(n_locations):
        row: list[float] = []
        for j in range(n_locations):
            if i == j:
                row.append(0.0)
            else:
                dist = _compute_euclidean_distance(
                    all_coords[i][0],
                    all_coords[i][1],
                    all_coords[j][0],
                    all_coords[j][1],
                )
                row.append(dist)
        cost_matrix.append(row)

    num_vehicles = vrp.num_vehicles or 1

    vehicle_capacity = vrp.vehicle_capacity
    if vehicle_capacity is not None:
        capacities = [[int(vehicle_capacity)] for _ in range(num_vehicles)]
    else:
        capacities = [[_DEFAULT_VEHICLE_CAPACITY] for _ in range(num_vehicles)]

    task_locations = list(range(1, n_locations))

    native_payload: dict[str, Any] = {
        "cost_matrix_data": {"data": {"0": cost_matrix}},
        "fleet_data": {
            "vehicle_locations": [[0, 0] for _ in range(num_vehicles)],
            "capacities": capacities,
        },
        "task_data": {
            "task_locations": task_locations,
            "demand": [customer_demands],
            "task_ids": task_ids,
        },
        "solver_config": {
            "time_limit": _DEFAULT_TIME_LIMIT_SECONDS,
            "objectives": {
                "cost": 1,
            },
        },
    }

    return native_payload
