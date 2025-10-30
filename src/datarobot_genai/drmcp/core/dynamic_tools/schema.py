# Copyright 2025 DataRobot, Inc.
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

from dataclasses import dataclass
from typing import Any
from typing import Union

from pydantic import BaseModel
from pydantic import Field
from pydantic import create_model
from pydantic.fields import FieldInfo


class SchemaValidationError(Exception):
    """Raised when a schema violates depth or complexity constraints."""

    pass


@dataclass(frozen=True)
class PropertyConfig:
    """Configuration for top-level input schema property types."""

    model_name: str
    allow_nested: bool
    allowed_primitive_types: set[str]
    default_description: str


# Type mapping from JSON Schema to Python types
_JSON_TYPE_MAPPING: dict[str, type[Any]] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}

# Property configurations for input schema validation
_PROPERTY_CONFIGS: dict[str, PropertyConfig] = {
    "path_params": PropertyConfig(
        model_name="PathParams",
        allow_nested=False,
        allowed_primitive_types=set(),
        default_description="Path params to substitute in endpoint.",
    ),
    "query_params": PropertyConfig(
        model_name="QueryParams",
        allow_nested=False,
        allowed_primitive_types=set(),
        default_description="Query parameters (?key=value).",
    ),
    "data": PropertyConfig(
        model_name="Data",
        allow_nested=True,
        allowed_primitive_types={"string"},
        default_description="Form or raw body data for POST requests.",
    ),
    "json": PropertyConfig(
        model_name="Json",
        allow_nested=True,
        allowed_primitive_types=set(),
        default_description="JSON body for POST requests.",
    ),
}


def json_schema_to_python_type(schema_type: str | list[str]) -> type[Any]:
    """Convert JSON schema type to Python type annotation.

    Args:
        schema_type: JSON schema type string or list of type strings

    Returns
    -------
        Python type annotation corresponding to the schema type
    """
    if isinstance(schema_type, list):
        types = [json_schema_to_python_type(t) for t in schema_type]
        return Union[tuple(types)]  # type: ignore  # noqa: UP007

    return _JSON_TYPE_MAPPING.get(schema_type, Any)


class SchemaResolver:
    """Helper class to resolve JSON schema references and handle unions."""

    def __init__(self, definitions: dict[str, Any] | None = None):
        """Initialize resolver with definitions.

        Args:
            definitions: Dictionary of reusable schema definitions ($defs)
        """
        self.definitions = definitions or {}

    def resolve_ref(self, ref_path: str) -> dict[str, Any]:
        """Resolve a JSON Schema $ref reference.

        Args:
            ref_path: The reference path (e.g., '#/$defs/ModelName')

        Returns
        -------
            The resolved schema definition

        Raises
        ------
            SchemaValidationError: If the reference cannot be resolved
        """
        if not ref_path.startswith("#/$defs/"):
            raise SchemaValidationError(
                f"Unsupported reference format: '{ref_path}'. "
                f"Only '#/$defs/...' references are supported."
            )

        def_name = ref_path.split("/")[-1]
        if def_name not in self.definitions:
            raise SchemaValidationError(f"Reference '{ref_path}' not found in definitions")

        return self.definitions[def_name]  # type: ignore[no-any-return]

    def resolve_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Resolve a schema, following $ref if present.

        Args:
            schema: Schema that may contain a $ref

        Returns
        -------
            Resolved schema
        """
        if "$ref" in schema:
            return self.resolve_ref(schema["$ref"])
        return schema

    def resolve_optional_union(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Resolve anyOf/oneOf unions for optional fields (Type | None).

        Args:
            schema: Schema that may contain anyOf/oneOf

        Returns
        -------
            Resolved schema for the non-null variant, or original schema
            if it's a complex union

        Raises
        ------
            SchemaValidationError: If union is too complex
        """
        if "anyOf" not in schema and "oneOf" not in schema:
            return schema

        variants = schema.get("anyOf") or schema.get("oneOf", [])
        non_null_variants = [v for v in variants if v.get("type") != "null"]

        # If there's exactly one non-null variant, it's a simple optional type (Type | None)
        if len(non_null_variants) == 1:
            return self.resolve_schema(non_null_variants[0])

        # If there are multiple non-null variants, it's a complex union
        # Return the schema as-is to be handled as Any type
        return schema


def _is_optional_field(field_spec: dict[str, Any]) -> bool:
    """Check if a field is optional (anyOf/oneOf containing null type).

    Args:
        field_spec: Field specification to check

    Returns
    -------
        True if field contains anyOf/oneOf with null type
    """
    if "anyOf" in field_spec or "oneOf" in field_spec:
        variants = field_spec.get("anyOf") or field_spec.get("oneOf", [])
        return any(v.get("type") == "null" for v in variants)
    return False


class FieldTypeResolver:
    """Resolves Python types for JSON schema fields."""

    def __init__(self, model_name: str, allow_nested: bool, resolver: SchemaResolver):
        """Initialize field type resolver.

        Args:
            model_name: Name of the parent model
            allow_nested: Whether nested objects/arrays are allowed
            resolver: Schema resolver for handling references
        """
        self.model_name = model_name
        self.allow_nested = allow_nested
        self.resolver = resolver

    def _validate_nested_allowed(self, field_name: str, structure_type: str) -> None:
        """Validate that nested structures are allowed.

        Args:
            field_name: Name of the field
            structure_type: Type of structure (e.g., "nested object", "array of objects")

        Raises
        ------
            SchemaValidationError: If nested structures not allowed
        """
        if not self.allow_nested:
            raise SchemaValidationError(
                f"The model '{self.model_name}' supports only flat structures. "
                f"Field '{field_name}' is a {structure_type}, which is not supported. "
                f"Please flatten the schema."
            )

    def _resolve_object_type(self, field_name: str, field_spec: dict[str, Any]) -> type[BaseModel]:
        """Resolve type for object fields.

        Args:
            field_name: Name of the field
            field_spec: Field specification

        Returns
        -------
            Pydantic model type
        """
        self._validate_nested_allowed(field_name, "nested object")
        return create_schema_model(
            name=f"{self.model_name}{field_name.capitalize()}",
            schema=field_spec,
            allow_nested=self.allow_nested,
            definitions=self.resolver.definitions,
        )

    def _resolve_array_type(self, field_name: str, field_spec: dict[str, Any]) -> type[Any]:
        """Resolve type for array fields.

        Args:
            field_name: Name of the field
            field_spec: Field specification

        Returns
        -------
            List type annotation
        """
        items_spec = field_spec.get("items", {})

        # Handle complex union types in array items
        if "anyOf" in items_spec or "oneOf" in items_spec:
            return list

        items_spec = self.resolver.resolve_schema(items_spec)
        items_type = items_spec.get("type") if isinstance(items_spec, dict) else None

        if items_type == "object":
            self._validate_nested_allowed(field_name, "array of objects")
            item_model = create_schema_model(
                name=f"{self.model_name}{field_name.capitalize()}Item",
                schema=items_spec,
                allow_nested=self.allow_nested,
                definitions=self.resolver.definitions,
            )
            return list[item_model]  # type: ignore[valid-type]

        return json_schema_to_python_type("array")

    def _resolve_union_type(self, field_name: str, schema_type: list[str]) -> type[Any]:
        """Resolve type for union fields.

        Args:
            field_name: Name of the field
            schema_type: List of type strings

        Returns
        -------
            Union type annotation

        Raises
        ------
            SchemaValidationError: If union contains complex types
        """
        if any(t in ("object", "array") for t in schema_type):
            raise SchemaValidationError(
                f"Field '{field_name}' contains complex types in a union. "
                "Complex types in unions are not supported. "
                "Use unions of primitive types only (e.g., ['string', 'null'])."
            )
        return json_schema_to_python_type(schema_type)

    def resolve(self, field_name: str, field_spec: dict[str, Any]) -> type[Any]:
        """Resolve the Python type for a JSON schema field.

        Args:
            field_name: Name of the field
            field_spec: JSON schema specification for the field

        Returns
        -------
            Python type annotation for the field
        """
        field_spec = self.resolver.resolve_schema(field_spec)
        schema_type = field_spec.get("type")

        if schema_type == "object":
            return self._resolve_object_type(field_name, field_spec)

        if schema_type == "array":
            return self._resolve_array_type(field_name, field_spec)

        if isinstance(schema_type, list):
            return self._resolve_union_type(field_name, schema_type)

        return json_schema_to_python_type(schema_type) if schema_type else Any


def create_schema_model(
    name: str,
    schema: dict[str, Any],
    allow_nested: bool,
    definitions: dict[str, Any] | None = None,
) -> type[BaseModel]:
    """Create a Pydantic model from a JSON schema, supporting nested objects.

    Args:
        name: Name for the generated model
        schema: JSON schema defining the model structure
        allow_nested: Whether to allow nested objects and arrays in
                the schema. If False, raises error on complex types.
        definitions: Dictionary of reusable schema definitions ($defs)

    Returns
    -------
        A Pydantic BaseModel class
    """
    if not schema or not schema.get("properties"):
        return create_model(name)

    # Extract definitions if present at this level
    definitions = definitions or schema.get("$defs", {})

    resolver = SchemaResolver(definitions)
    field_resolver = FieldTypeResolver(name, allow_nested, resolver)

    properties = schema.get("properties", {})
    required_fields = set(schema.get("required", []))

    fields: dict[str, Any] = {}
    for field_name, field_spec in properties.items():
        # Check if field is optional (has anyOf/oneOf with null)
        is_optional = _is_optional_field(field_spec)
        is_required = field_name in required_fields

        # Resolve anyOf/oneOf unions to get the actual field spec
        resolved_spec = resolver.resolve_optional_union(field_spec) if is_optional else field_spec

        # Resolve the field type from the resolved spec
        field_type = field_resolver.resolve(field_name, resolved_spec)

        # Get default value and description
        default = (
            ... if is_required else (resolved_spec.get("default") or field_spec.get("default"))
        )
        description = resolved_spec.get("description") or field_spec.get("description")

        # Wrap in Optional if field has anyOf/oneOf with null
        if is_optional and not is_required:
            field_type = field_type | None  # type: ignore[assignment]

        fields[field_name] = (field_type, Field(default, description=description))

    return create_model(name, **fields)


class InputSchemaPropertyHandler:
    """Handles processing of input schema properties with validation."""

    def __init__(self, config: PropertyConfig, resolver: SchemaResolver):
        """Initialize property handler.

        Args:
            config: Configuration for this property type
            resolver: Schema resolver for handling references
        """
        self.config = config
        self.resolver = resolver

    def _validate_primitive_type(self, property_name: str, schema_type: str) -> None:
        """Validate that a primitive type is allowed for this property.

        Args:
            property_name: Name of the property
            schema_type: The schema type to validate

        Raises
        ------
            SchemaValidationError: If the primitive type is not allowed
        """
        if schema_type not in self.config.allowed_primitive_types:
            allowed_types_str = (
                ", ".join(sorted(self.config.allowed_primitive_types))
                if self.config.allowed_primitive_types
                else "none"
            )
            raise SchemaValidationError(
                f"Property '{property_name}' does not support primitive type '{schema_type}'. "
                f"Allowed primitive types: {allowed_types_str}. "
                f"Use an object schema with properties instead."
            )

    def process_property(
        self, property_name: str, property_schema: dict[str, Any]
    ) -> tuple[type[Any], FieldInfo]:
        """Process a property schema and return its field definition.

        Args:
            property_name: Name of the property
            property_schema: Schema definition for the property

        Returns
        -------
            Tuple of (type, FieldInfo) for the property
        """
        # Resolve unions and references
        property_schema = self.resolver.resolve_optional_union(property_schema)
        property_schema = self.resolver.resolve_schema(property_schema)

        description = property_schema.get("description", self.config.default_description)
        schema_type = property_schema.get("type")

        # Handle primitive types
        if schema_type and schema_type != "object":
            self._validate_primitive_type(property_name, schema_type)
            python_type = json_schema_to_python_type(schema_type)
            return python_type | None, Field(None, description=description)  # type: ignore[return-value]

        # Handle object types
        property_definitions = property_schema.get("$defs", self.resolver.definitions)
        model = create_schema_model(
            self.config.model_name,
            property_schema,
            self.config.allow_nested,
            property_definitions,
        )
        return model | None, Field(None, description=description)  # type: ignore[return-value]


def _validate_input_schema_properties(
    properties: dict[str, Any], expected: set[str], allow_empty: bool
) -> None:
    """Validate input schema properties.

    Args:
        properties: Properties from the input schema
        expected: Expected property names
        allow_empty: Whether empty schemas are allowed

    Raises
    ------
        SchemaValidationError: If validation fails
    """
    if properties:
        unexpected = set(properties.keys()) - expected
        if unexpected:
            raise SchemaValidationError(
                f"Input schema contains unsupported top-level properties: {unexpected}. "
                f"Please note that top-level properties organize parameters within groups "
                f"corresponding to the HTTP request structure: {expected}. "
                f"Please define parameters within one of these top-level keys."
            )
    elif not allow_empty:
        raise SchemaValidationError(
            f"Input schema must define 'properties' with at least one of: {expected}. "
            f"Empty schemas are disabled by default. "
            f"To enable registration of tools with no input parameters "
            f"(e.g., static endpoints), set environment variable "
            f"MCP_SERVER_TOOL_REGISTRATION_ALLOW_EMPTY_SCHEMA='true'."
        )


def create_input_schema_pydantic_model(
    input_schema: dict[str, Any],
    model_name: str = "InputSchema",
    allow_empty: bool = False,
) -> type[BaseModel]:
    """Create a properly typed ExternalToolRegistrationConfig with validated sub-schemas.

    Args:
        input_schema: JSON schema for input parameters
        model_name: Name for the generated Pydantic model
        allow_empty: Whether to allow empty schema (no properties)

    Returns
    -------
        A Pydantic BaseModel class with properly typed fields

    Raises
    ------
        SchemaValidationError: If schema validation fails
    """
    properties = input_schema.get("properties", {})
    expected_properties = set(_PROPERTY_CONFIGS.keys())
    definitions = input_schema.get("$defs", {})

    # Validate properties
    _validate_input_schema_properties(properties, expected_properties, allow_empty)

    if not properties:
        return create_model(model_name)

    # Build field definitions using configuration
    resolver = SchemaResolver(definitions)
    fields: dict[str, Any] = {}

    for property_name, config in _PROPERTY_CONFIGS.items():
        if property_schema := properties.get(property_name):
            handler = InputSchemaPropertyHandler(config, resolver)
            field_type, field = handler.process_property(property_name, property_schema)
            fields[property_name] = (field_type, field)

    return create_model(model_name, **fields)
