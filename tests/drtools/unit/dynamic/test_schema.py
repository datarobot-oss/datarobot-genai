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

"""Direct-import smoke tests for ``drtools.dynamic.schema``.

The bulk of schema behavior is covered by ``tests/drmcp/unit/dynamic_tools/test_schema.py``
(those tests import via the drmcp shim so they exercise the same drtools code).
These tests target the rarely-hit branches and assert that the package is
usable without going through the drmcp shim.
"""

from typing import Any

import pytest

from datarobot_genai.drtools.dynamic.schema import SchemaResolver
from datarobot_genai.drtools.dynamic.schema import SchemaValidationError
from datarobot_genai.drtools.dynamic.schema import create_input_schema_pydantic_model
from datarobot_genai.drtools.dynamic.schema import create_schema_model
from datarobot_genai.drtools.dynamic.schema import json_schema_to_python_type


class TestJsonSchemaTypeMapping:
    def test_basic_types(self) -> None:
        assert json_schema_to_python_type("string") is str
        assert json_schema_to_python_type("integer") is int
        assert json_schema_to_python_type("number") is float
        assert json_schema_to_python_type("boolean") is bool
        assert json_schema_to_python_type("array") is list
        assert json_schema_to_python_type("object") is dict
        assert json_schema_to_python_type("null") is type(None)

    def test_unknown_falls_back_to_any(self) -> None:
        assert json_schema_to_python_type("not-a-type") is Any  # type: ignore[comparison-overlap]


class TestSchemaResolver:
    def test_unsupported_ref_format_raises(self) -> None:
        resolver = SchemaResolver({})
        with pytest.raises(SchemaValidationError, match="Unsupported reference format"):
            resolver.resolve_ref("http://example.com/foo")

    def test_missing_definition_raises(self) -> None:
        resolver = SchemaResolver({})
        with pytest.raises(SchemaValidationError, match="not found in definitions"):
            resolver.resolve_ref("#/$defs/Missing")

    def test_complex_union_returned_as_is(self) -> None:
        resolver = SchemaResolver({})
        schema: dict[str, Any] = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
        # Two non-null variants -> complex union, returned unchanged.
        assert resolver.resolve_optional_union(schema) == schema


class TestCreateSchemaModel:
    def test_empty_schema_yields_empty_model(self) -> None:
        model = create_schema_model("Empty", {}, allow_nested=False)
        assert model.__name__ == "Empty"

    def test_complex_union_in_array_items_falls_back_to_list(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
                }
            },
        }
        model = create_schema_model("Items", schema, allow_nested=True)
        instance = model(items=["a", 1, True])  # type: ignore[arg-type]
        assert instance.items == ["a", 1, True]  # type: ignore[attr-defined]


class TestCreateInputSchemaPydanticModel:
    def test_unsupported_top_level_property_raises(self) -> None:
        with pytest.raises(SchemaValidationError, match="unsupported top-level"):
            create_input_schema_pydantic_model(
                {"type": "object", "properties": {"weird": {"type": "object"}}}
            )

    def test_empty_disallowed_by_default(self) -> None:
        with pytest.raises(SchemaValidationError, match="Empty schemas are disabled"):
            create_input_schema_pydantic_model({"type": "object", "properties": {}})

    def test_empty_allowed_when_flag_set(self) -> None:
        model = create_input_schema_pydantic_model(
            {"type": "object", "properties": {}}, allow_empty=True
        )
        # Should be instantiable with no args.
        assert model().model_dump() == {}

    def test_primitive_in_path_params_rejected(self) -> None:
        with pytest.raises(SchemaValidationError, match="does not support primitive type"):
            create_input_schema_pydantic_model(
                {"type": "object", "properties": {"path_params": {"type": "string"}}}
            )

    def test_data_supports_primitive_string(self) -> None:
        model = create_input_schema_pydantic_model(
            {"type": "object", "properties": {"data": {"type": "string"}}}
        )
        instance = model(data="raw-body")
        assert instance.model_dump()["data"] == "raw-body"

    def test_query_params_object(self) -> None:
        model = create_input_schema_pydantic_model(
            {
                "type": "object",
                "properties": {
                    "query_params": {
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                    }
                },
            }
        )
        instance = model(query_params={"q": "hi"})
        assert instance.model_dump()["query_params"]["q"] == "hi"
