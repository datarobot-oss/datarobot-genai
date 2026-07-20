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

"""Shared fixtures for cuOpt glue unit tests.

The deployment HTTP boundary is mocked at the call seam (``get_cuopt_client``
returns canned ``validate``/``solve`` results) and panel writes go to an
in-memory ``FakeBlobStore``-backed ``PanelStore`` (reused from the panels test
suite). The normalization helpers, solution-table extraction, error-response
building, problem-type inference, and the VRPData -> native conversion run
unmocked -- they are pure functions ported from wren-mcp.
"""

from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

from datarobot_genai.drmcputils.panels.store import PanelStore
from datarobot_genai.drtools.cuopt import persistence as persistence_mod
from datarobot_genai.drtools.cuopt import solve as solve_mod
from tests.drtools.unit.panels.conftest import FakeBlobStore


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> PanelStore:
    """Inject an in-memory panel store and skip the entitlement guard."""
    blobs = FakeBlobStore()
    panel_store = PanelStore(blobs)
    monkeypatch.setattr(solve_mod, "_require_mcp_sandbox", lambda: None)
    monkeypatch.setattr(solve_mod, "_get_store", lambda: panel_store)
    monkeypatch.setattr(persistence_mod, "_get_store", lambda: panel_store)
    return panel_store


def patch_cuopt_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    solve_result: dict[str, Any] | None = None,
    validate_result: dict[str, Any] | None = None,
) -> MagicMock:
    """Mock the cuOpt deployment call seam with canned responses."""
    client = MagicMock()
    client.solve = AsyncMock(return_value=solve_result or {})
    client.validate = AsyncMock(return_value=validate_result or {})
    monkeypatch.setattr(solve_mod, "get_cuopt_client", lambda: client)
    return client


# Canned payloads/results -- ported from wren's cuopt test fixtures. --------- #


def sample_milp_data() -> dict[str, Any]:
    return {
        "csr_constraint_matrix": {
            "offsets": [0, 2],
            "indices": [0, 1],
            "values": [1.0, 1.0],
        },
        "constraint_bounds": {"upper_bounds": [1.0], "lower_bounds": [0.0]},
        "objective_data": {
            "coefficients": [4.0, 1.0],
            "scalability_factor": 1.0,
            "offset": 0.0,
        },
        "variable_bounds": {"upper_bounds": [1.0, 1.0], "lower_bounds": [0.0, 0.0]},
        "variable_names": ["x1", "x2"],
        "variable_types": ["I", "I"],
        "maximize": False,
    }


def sample_lp_data() -> dict[str, Any]:
    data = sample_milp_data()
    data["variable_types"] = ["C", "continuous"]
    return data


def lp_solve_result() -> dict[str, Any]:
    return {
        "request_id": "req-lp",
        "status": "success",
        "solution": {
            "solution": {
                "primal_solution": {"x1": 0.5, "x2": 0.5},
                "dual_solution": {"c1": 1.5},
                "primal_objective": 2.5,
            }
        },
        "cost": 2.5,
    }


def vrp_solve_result() -> dict[str, Any]:
    return {
        "request_id": "req-vrp",
        "status": "success",
        "solution": {
            "routes": [
                {"vehicle_id": 0, "stops": [0, 1, 2, 0]},
                {"vehicle_id": 1, "stops": [0, 2, 0]},
            ],
            "vehicle_data": {
                "veh-0": {"task_id": ["Depot", "A", "Depot"], "total_cost": 12.5},
            },
            "unassigned_tasks": ["B"],
        },
        "cost": 37.0,
    }
