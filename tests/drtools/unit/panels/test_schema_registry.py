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

"""Unit tests for the panel SchemaRegistry and the schema tools."""

from typing import Any

import pytest
from pydantic import BaseModel
from pydantic import Field

from datarobot_genai.drtools.panels import tools as tools_mod
from datarobot_genai.drtools.panels.schema_registry import SchemaRegistry
from datarobot_genai.drtools.panels.schema_registry import SchemaValidationError


class Point(BaseModel):
    """A 2D point."""

    x: float
    y: float


class Route(BaseModel):
    """A demo route schema."""

    name: str = Field(description="Route label")
    stops: list[Point]
    max_length: int | None = None


@pytest.fixture
def registry() -> Any:
    SchemaRegistry.register("demo.Point", Point)
    SchemaRegistry.register("demo.Route", Route)
    yield SchemaRegistry
    SchemaRegistry.unregister("demo.Point")
    SchemaRegistry.unregister("demo.Route")


def test_register_rejects_non_model() -> None:
    with pytest.raises(TypeError):
        SchemaRegistry.register("bad", object)  # type: ignore[arg-type]


def test_list_schemas_filters_by_namespace(registry: Any) -> None:
    all_schemas = registry.list_schemas()
    assert {"demo.Point", "demo.Route"} <= set(all_schemas)
    only_demo = registry.list_schemas(namespace="demo")
    assert set(only_demo) == {"demo.Point", "demo.Route"}
    assert registry.list_schemas(namespace="nope") == {}
    assert only_demo["demo.Route"]["required_fields"] == ["name", "stops"]
    assert only_demo["demo.Route"]["optional_fields"] == ["max_length"]


def test_describe_includes_fields_json_schema_and_example(registry: Any) -> None:
    described = registry.describe("demo.Route")
    assert described["name"] == "demo.Route"
    assert described["fields"]["stops"]["items_schema"] == "Point"
    assert described["fields"]["name"]["description"] == "Route label"
    assert described["json_schema"]["title"] == "Route"
    assert described["example"]["stops"] == [{"x": 1.0, "y": 1.0}]


def test_validate_normalizes_and_raises(registry: Any) -> None:
    ok = registry.validate("demo.Point", {"x": 1, "y": 2})
    assert ok == {"x": 1.0, "y": 2.0}
    with pytest.raises(SchemaValidationError) as exc_info:
        registry.validate("demo.Route", {"name": "r"})
    assert any(err["loc"] == ("stops",) for err in exc_info.value.errors)
    with pytest.raises(KeyError):
        registry.validate("missing.Schema", {})


async def test_schema_tools_roundtrip(registry: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tools_mod, "_require_mcp_sandbox", lambda: None)

    listed = await tools_mod.list_panel_schemas(namespace="demo")
    assert listed["count"] == 2

    described = await tools_mod.describe_panel_schema("demo.Point")
    assert described["name"] == "demo.Point"

    missing = await tools_mod.describe_panel_schema("demo.Nope")
    assert "error" in missing
    assert "demo.Point" in missing["available_schemas"]

    valid = await tools_mod.validate_panel_data(schema_name="demo.Point", data={"x": 0, "y": 0})
    assert valid == {"valid": True, "normalized_data": {"x": 0.0, "y": 0.0}}

    invalid = await tools_mod.validate_panel_data(schema_name="demo.Point", data={"x": 0})
    assert invalid["valid"] is False
    assert invalid["errors"]

    unknown = await tools_mod.validate_panel_data(schema_name="nope.Nope", data={})
    assert unknown["valid"] is False
    assert "available_schemas" in unknown


async def test_list_panel_schemas_empty_namespace_hint(
    registry: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tools_mod, "_require_mcp_sandbox", lambda: None)
    out = await tools_mod.list_panel_schemas(namespace="absent")
    assert "No schemas found" in out["message"]
    assert "demo" in out["available_namespaces"]
