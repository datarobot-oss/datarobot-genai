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

"""Unit tests for the cuOpt panel schemas and their registry integration."""

import pytest
from pydantic import ValidationError

from datarobot_genai.drtools.cuopt import schemas as cs
from datarobot_genai.drtools.panels.schema_registry import SchemaRegistry
from tests.drtools.unit.cuopt.conftest import sample_lp_data


def test_cuopt_schemas_registered_at_import() -> None:
    # GIVEN the cuopt schemas module has been imported (module-level registration)
    # WHEN listing the registry's cuopt namespace
    names = set(SchemaRegistry.list_schemas(namespace="cuopt"))

    # THEN all 11 cuopt.* panel schemas are discoverable
    assert {
        "cuopt.VRPData",
        "cuopt.NativeMILPData",
        "cuopt.Coordinate",
        "cuopt.VRPNode",
        "cuopt.CSRMatrix",
        "cuopt.ConstraintBounds",
        "cuopt.VariableBounds",
        "cuopt.ObjectiveData",
        "cuopt.NativeMILPSolverConfig",
        "cuopt.Constraints",
        "cuopt.SolverConfig",
    } <= names


def test_cuopt_schemas_describable_via_registry() -> None:
    # GIVEN the registered cuopt.NativeMILPData schema
    # WHEN describing it through the registry (backs describe_panel_schema)
    described = SchemaRegistry.describe("cuopt.NativeMILPData")

    # THEN the description carries the fields and JSON schema
    assert described["name"] == "cuopt.NativeMILPData"
    assert "variable_types" in described["fields"]
    assert described["json_schema"]["title"] == "NativeMILPData"


def test_csr_matrix_invariants() -> None:
    # GIVEN CSR payloads violating each structural invariant
    # WHEN validating them
    # THEN each raises a targeted validation error
    with pytest.raises(ValidationError, match="equal length"):
        cs.CSRMatrix.model_validate({"offsets": [0, 2], "indices": [0, 1], "values": [1.0]})
    with pytest.raises(ValidationError, match="monotonically non-decreasing"):
        cs.CSRMatrix.model_validate({"offsets": [0, 2, 1], "indices": [0, 1], "values": [1.0, 1.0]})
    with pytest.raises(ValidationError, match="last value must equal"):
        cs.CSRMatrix.model_validate({"offsets": [0, 1], "indices": [0, 1], "values": [1.0, 1.0]})


def test_none_bounds_coerced_to_infinity() -> None:
    # GIVEN bounds with nulls (LLM-generated payloads use null for unbounded)
    # WHEN validating constraint bounds
    validated = cs.ConstraintBounds.model_validate(
        {"upper_bounds": [None, 5.0], "lower_bounds": [0.0, None]}
    )

    # THEN None coerces to 'inf' on upper and 'ninf' on lower bounds
    assert validated.upper_bounds == ["inf", 5.0]
    assert validated.lower_bounds == [0.0, "ninf"]


def test_variable_bounds_none_coerced_to_infinity() -> None:
    # GIVEN variable bounds with nulls
    # WHEN validating variable bounds
    validated = cs.VariableBounds.model_validate({"upper_bounds": [None], "lower_bounds": [None]})

    # THEN None coerces to the matching unbounded sentinel
    assert validated.upper_bounds == ["inf"]
    assert validated.lower_bounds == ["ninf"]


def test_native_milp_variable_types_normalized() -> None:
    # GIVEN an LP payload using long-form type codes ('continuous')
    # WHEN validating the native MILP payload
    data = cs.NativeMILPData.model_validate(sample_lp_data())

    # THEN type codes normalize to single-letter form
    assert data.variable_types == ["C", "C"]


def test_native_milp_rejects_unknown_variable_type() -> None:
    # GIVEN a payload with an unsupported variable type code
    payload = sample_lp_data()
    payload["variable_types"] = ["B", "C"]

    # WHEN validating THEN a targeted error names the offending code
    with pytest.raises(ValidationError, match="Invalid variable type 'B'"):
        cs.NativeMILPData.model_validate(payload)


def test_native_milp_maximize_string_normalized() -> None:
    # GIVEN a payload carrying maximize as a string
    payload = sample_lp_data()
    payload["maximize"] = "True"

    # WHEN validating
    data = cs.NativeMILPData.model_validate(payload)

    # THEN maximize normalizes to a boolean
    assert data.maximize is True


def test_native_milp_length_mismatch_rejected() -> None:
    # GIVEN a payload whose objective coefficients disagree with variable_types
    payload = sample_lp_data()
    payload["objective_data"]["coefficients"] = [4.0]

    # WHEN validating THEN the length mismatch is reported
    with pytest.raises(ValidationError, match="coefficients length must match"):
        cs.NativeMILPData.model_validate(payload)


def test_coordinate_requires_some_coordinate_form() -> None:
    # GIVEN a node without any recognizable coordinate representation
    # WHEN validating THEN a coordinate-form error is raised
    with pytest.raises(ValidationError, match="Coordinate requires"):
        cs.Coordinate.model_validate({"id": "A"})


def test_vrp_data_requires_customers() -> None:
    # GIVEN a VRP payload with an empty customer list
    # WHEN validating THEN the empty list is rejected
    with pytest.raises(ValidationError, match="at least one entry"):
        cs.VRPData.model_validate({"depot": {"x": 0, "y": 0}, "customers": []})
