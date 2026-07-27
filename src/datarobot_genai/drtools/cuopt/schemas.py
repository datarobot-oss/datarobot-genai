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

"""cuOpt payload schemas (ported from wren-mcp's ``app/core/cuopt/schemas.py``).

Pydantic models for the cuOpt LP/MILP/VRP payload formats, registered with the
panels :class:`SchemaRegistry` under the ``cuopt.*`` namespace at import time
so the schema-discovery tools (``list_panel_schemas`` / ``describe_panel_schema``
/ ``validate_panel_data``) surface them.
"""

from __future__ import annotations

from typing import Any
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator

from datarobot_genai.drtools.panels.schema_registry import SchemaRegistry


class BaseSchema(BaseModel):
    """Base schema with permissive extras (cuOpt adds new fields frequently)."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)


class Coordinate(BaseSchema):
    """Flexible coordinate definition supporting cartesian and geographic formats."""

    x: float | None = None
    y: float | None = None
    lat: float | None = None
    lng: float | None = None
    latitude: float | None = None
    longitude: float | None = None
    location: list[float] | None = None

    @model_validator(mode="after")
    def ensure_coordinate(self) -> Coordinate:
        has_xy = self.x is not None and self.y is not None
        has_latlng = (self.lat is not None and self.lng is not None) or (
            self.latitude is not None and self.longitude is not None
        )
        has_location = (
            isinstance(self.location, (list, tuple))
            and len(self.location) >= 2  # noqa: PLR2004 - (x, y) pair
            and all(isinstance(item, (int, float)) for item in self.location[:2])
        )

        if not (has_xy or has_latlng or has_location):
            raise ValueError(
                "Coordinate requires either x/y, lat/lng, latitude/longitude, or a location array"
            )

        return self


class VRPNode(Coordinate):
    """Common fields shared by depot and customer nodes."""

    id: str | int | None = None
    demand: float | None = None


class VRPData(BaseSchema):
    """Vehicle Routing Problem payload."""

    depot: Coordinate
    customers: list[VRPNode]
    num_vehicles: int | None = Field(default=None, ge=1)
    vehicle_capacity: float | None = None

    @model_validator(mode="after")
    def ensure_customers(self) -> VRPData:
        if not self.customers:
            raise ValueError("'customers' must contain at least one entry")
        return self


# Type for bounds that can be numeric, "inf", or "ninf"
BoundValue = float | int | Literal["inf", "ninf", "-inf"]
# Input type that also accepts None (LLM-generated data uses null for unbounded)
BoundValueInput = float | int | Literal["inf", "ninf", "-inf"] | None


def coerce_bound_value(value: BoundValueInput, *, is_upper: bool) -> BoundValue:
    """Coerce a bound value, treating None as unbounded."""
    if value is None:
        return "inf" if is_upper else "ninf"
    return value


class CSRMatrix(BaseSchema):
    """Compressed Sparse Row matrix format for constraints."""

    offsets: list[int] = Field(description="Row pointer array (length = num_constraints + 1)")
    indices: list[int] = Field(description="Column indices of non-zero elements")
    values: list[float] = Field(description="Values of non-zero elements")

    @model_validator(mode="after")
    def validate_csr_invariants(self) -> CSRMatrix:
        if len(self.indices) != len(self.values):
            raise ValueError("CSR matrix requires indices and values to have equal length")

        if not self.offsets:
            raise ValueError("CSR matrix offsets must contain at least one entry")

        if any(self.offsets[i] > self.offsets[i + 1] for i in range(len(self.offsets) - 1)):
            raise ValueError("CSR matrix offsets must be monotonically non-decreasing")

        if self.offsets[-1] != len(self.indices):
            raise ValueError("CSR matrix offsets last value must equal the number of indices")

        return self


class ConstraintBounds(BaseSchema):
    """Bounds for constraint rows."""

    upper_bounds: list[BoundValue] = Field(
        description="Upper bounds for each constraint (use 'inf' for unbounded)"
    )
    lower_bounds: list[BoundValue] = Field(
        description="Lower bounds for each constraint (use 'ninf' or '-inf' for unbounded)"
    )

    @field_validator("upper_bounds", mode="before")
    @classmethod
    def coerce_upper_bounds(cls, value: Any) -> Any:
        if not isinstance(value, list):
            return value
        return [coerce_bound_value(v, is_upper=True) for v in value]

    @field_validator("lower_bounds", mode="before")
    @classmethod
    def coerce_lower_bounds(cls, value: Any) -> Any:
        if not isinstance(value, list):
            return value
        return [coerce_bound_value(v, is_upper=False) for v in value]


class VariableBounds(BaseSchema):
    """Bounds for decision variables."""

    upper_bounds: list[BoundValue] = Field(
        description="Upper bounds for each variable (use 'inf' for unbounded)"
    )
    lower_bounds: list[BoundValue] = Field(
        description="Lower bounds for each variable (use 0.0 for non-negative)"
    )

    @field_validator("upper_bounds", mode="before")
    @classmethod
    def coerce_upper_bounds(cls, value: Any) -> Any:
        if not isinstance(value, list):
            return value
        return [coerce_bound_value(v, is_upper=True) for v in value]

    @field_validator("lower_bounds", mode="before")
    @classmethod
    def coerce_lower_bounds(cls, value: Any) -> Any:
        if not isinstance(value, list):
            return value
        return [coerce_bound_value(v, is_upper=False) for v in value]


class ObjectiveData(BaseSchema):
    """Objective function coefficients."""

    coefficients: list[float] = Field(description="Linear coefficients for each variable")
    scalability_factor: float = Field(default=1.0, description="Scaling factor for objective")
    offset: float = Field(default=0.0, description="Constant offset in objective")


class NativeMILPSolverConfig(BaseSchema):
    """Solver configuration for native MILP format."""

    time_limit: int | None = Field(default=30, description="Maximum solve time in seconds")
    tolerances: dict[str, float] | None = Field(
        default=None, description="Solver tolerances (e.g., {'optimality': 0.0001})"
    )


class NativeMILPData(BaseSchema):
    """Native cuOpt LP/MILP format using CSR matrix representation.

    LP vs MILP is determined entirely by ``variable_types``: all "C"
    (continuous) is an LP; any "I" (integer) is a MILP. For binary variables,
    use "I" with variable_bounds [0, 1].
    """

    csr_constraint_matrix: CSRMatrix = Field(description="Constraint matrix in CSR format")
    constraint_bounds: ConstraintBounds = Field(
        description="Lower and upper bounds for constraints"
    )
    objective_data: ObjectiveData = Field(description="Objective function coefficients")
    variable_bounds: VariableBounds = Field(description="Bounds for each variable")
    variable_types: list[Literal["I", "C"]] = Field(
        description=(
            "REQUIRED: Type for each variable - THIS DETERMINES LP vs MILP! "
            "I=Integer (whole numbers: counts, yes/no decisions with 0-1 bounds), "
            "C=Continuous (fractions allowed). "
            "For binary variables, use I with variable_bounds [0, 1]. "
            "Use I for discrete decisions; all C = LP relaxation with fractional results!"
        )
    )
    variable_names: list[str] | None = Field(
        default=None, description="Optional names for variables"
    )
    maximize: bool | Literal["True", "False"] = Field(
        default=False, description="True to maximize, False to minimize"
    )
    solver_config: NativeMILPSolverConfig | None = Field(
        default=None, description="Solver configuration"
    )

    @field_validator("variable_types", mode="before")
    @classmethod
    def normalize_variable_types(cls, value: Any) -> list[str]:
        """Normalize variable type codes ('I'/'INTEGER' -> 'I', 'C'/'CONTINUOUS' -> 'C')."""
        if not isinstance(value, list):
            raise ValueError("variable_types must be a list")
        normalized = []
        for i, v in enumerate(value):
            if isinstance(v, str):
                v_upper = v.upper()
                if v_upper in {"I", "INTEGER"}:
                    normalized.append("I")
                elif v_upper in {"C", "CONTINUOUS"}:
                    normalized.append("C")
                else:
                    raise ValueError(
                        f"Invalid variable type '{v}' at index {i}. "
                        f"Only 'I' (integer) and 'C' (continuous) are supported. "
                        f"For binary variables, use 'I' with variable_bounds [0, 1]."
                    )
            else:
                raise ValueError("Variable types must be strings")
        return normalized

    @field_validator("maximize", mode="before")
    @classmethod
    def normalize_maximize(cls, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in {"true", "yes", "1"}
        return bool(value)

    @model_validator(mode="after")
    def validate_variable_lengths(self) -> NativeMILPData:
        num_vars = len(self.variable_types)

        if len(self.variable_bounds.upper_bounds) != num_vars:
            raise ValueError("variable_bounds.upper_bounds length must match variable_types length")
        if len(self.variable_bounds.lower_bounds) != num_vars:
            raise ValueError("variable_bounds.lower_bounds length must match variable_types length")
        if len(self.objective_data.coefficients) != num_vars:
            raise ValueError("objective_data.coefficients length must match variable_types length")
        if self.variable_names is not None and len(self.variable_names) != num_vars:
            raise ValueError("variable_names length must match variable_types length when provided")

        return self


class ConstraintsModel(BaseSchema):
    """Generic constraints model capturing common cuOpt fields."""

    vehicle_capacity: float | None = None
    time_windows: list[Any] | None = None
    distance_limit: float | None = None
    max_route_distance: float | None = None
    min_staff: int | None = None
    max_hours: float | None = None


class SolverConfigModel(BaseSchema):
    """Generic solver configuration."""

    timeout: int | float | None = None
    objectives: dict[str, float] = Field(default_factory=dict)
    max_iterations: int | None = None
    seed: int | None = None
    verbose: bool | None = None

    @field_validator("objectives", mode="before")
    @classmethod
    def coerce_objectives(cls, value: Any) -> dict[str, float]:
        if value is None:
            return {}
        if isinstance(value, dict):
            coerced: dict[str, float] = {}
            for key, weight in value.items():
                if not isinstance(weight, (int, float)):
                    raise ValueError("Objective weights must be numeric")
                coerced[str(key)] = float(weight)
            return coerced
        raise ValueError("Solver objectives must be a dictionary")


def register_cuopt_schemas() -> None:
    """Register the cuOpt schemas with the panels SchemaRegistry (matches wren's set)."""
    SchemaRegistry.register("cuopt.VRPData", VRPData)
    SchemaRegistry.register("cuopt.NativeMILPData", NativeMILPData)
    SchemaRegistry.register("cuopt.Coordinate", Coordinate)
    SchemaRegistry.register("cuopt.VRPNode", VRPNode)
    SchemaRegistry.register("cuopt.CSRMatrix", CSRMatrix)
    SchemaRegistry.register("cuopt.ConstraintBounds", ConstraintBounds)
    SchemaRegistry.register("cuopt.VariableBounds", VariableBounds)
    SchemaRegistry.register("cuopt.ObjectiveData", ObjectiveData)
    SchemaRegistry.register("cuopt.NativeMILPSolverConfig", NativeMILPSolverConfig)
    SchemaRegistry.register("cuopt.Constraints", ConstraintsModel)
    SchemaRegistry.register("cuopt.SolverConfig", SolverConfigModel)


# Register at import time so schema discovery surfaces the cuopt.* namespace as
# soon as any cuopt module (or a consumer facade) is imported.
register_cuopt_schemas()
