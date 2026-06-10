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

"""Schema registry for Json panel validation (ported from wren-mcp).

A process-wide registry of namespaced Pydantic schemas (e.g. ``"cuopt.VRPData"``)
that agents can discover (:meth:`SchemaRegistry.list_schemas`), introspect
(:meth:`SchemaRegistry.describe`), and validate data against
(:meth:`SchemaRegistry.validate`) before storing it on a Json panel. Domains
register their schemas at import time::

    from datarobot_genai.drtools.panels.schema_registry import SchemaRegistry

    SchemaRegistry.register("myapp.MyModel", MyModel)
"""

from __future__ import annotations

import json
import logging
from typing import Any
from typing import Union
from typing import get_args
from typing import get_origin

from pydantic import BaseModel
from pydantic import ValidationError
from pydantic_core import PydanticUndefined

logger = logging.getLogger(__name__)


class SchemaValidationError(ValueError):
    """Raised when data fails schema validation."""

    def __init__(self, schema_name: str, data: dict[str, Any], errors: list[dict[str, Any]]):
        self.schema_name = schema_name
        self.data = data
        self.errors = errors

        error_lines = [f"Schema validation failed for '{schema_name}':"]
        for err in errors:
            loc = " -> ".join(str(x) for x in err.get("loc", []))
            msg = err.get("msg", "Unknown error")
            err_type = err.get("type", "")
            error_lines.append(f"  • {loc}: {msg}" if loc else f"  • {msg}")
            if err_type:
                error_lines.append(f"    (error type: {err_type})")

        super().__init__("\n".join(error_lines))


class SchemaRegistry:
    """Central registry of Pydantic schemas for Json panel validation."""

    _schemas: dict[str, type[BaseModel]] = {}

    @classmethod
    def register(cls, name: str, schema: type[BaseModel]) -> None:
        """Register a Pydantic model class under a namespaced ``name``."""
        if not isinstance(schema, type) or not issubclass(schema, BaseModel):
            raise TypeError(f"Schema must be a Pydantic BaseModel class, got {type(schema)}")
        cls._schemas[name] = schema
        logger.debug("Registered schema: %s", name)

    @classmethod
    def unregister(cls, name: str) -> bool:
        """Remove a schema from the registry. Returns True if it existed."""
        return cls._schemas.pop(name, None) is not None

    @classmethod
    def get(cls, name: str) -> type[BaseModel]:
        """Get a schema by name; raises ``KeyError`` if not registered."""
        if name not in cls._schemas:
            available = ", ".join(sorted(cls._schemas.keys()))
            raise KeyError(
                f"Schema '{name}' not found. Available schemas: {available or '(none registered)'}"
            )
        return cls._schemas[name]

    @classmethod
    def exists(cls, name: str) -> bool:
        """Check if a schema is registered."""
        return name in cls._schemas

    @classmethod
    def list_schemas(cls, namespace: str | None = None) -> dict[str, dict[str, Any]]:
        """List registered schemas (optionally filtered by ``namespace``) with summaries."""
        result = {}
        for name, schema in sorted(cls._schemas.items()):
            if namespace and not name.startswith(f"{namespace}."):
                continue
            result[name] = {
                "name": name,
                "description": (schema.__doc__ or "").strip().split("\n")[0],
                "required_fields": cls._get_required_fields(schema),
                "optional_fields": cls._get_optional_fields(schema),
            }
        return result

    @classmethod
    def describe(cls, name: str) -> dict[str, Any]:
        """Detailed schema description: fields, full JSON Schema, and an example."""
        schema = cls.get(name)
        return {
            "name": name,
            "description": (schema.__doc__ or "").strip(),
            "fields": cls._describe_fields(schema),
            "json_schema": schema.model_json_schema(),
            "example": cls._generate_example(schema),
        }

    @classmethod
    def validate(cls, name: str, data: dict[str, Any]) -> dict[str, Any]:
        """Validate ``data`` and return it normalized; raises :class:`SchemaValidationError`."""
        schema = cls.get(name)
        try:
            validated = schema.model_validate(data)
            return validated.model_dump(mode="python", exclude_none=True)
        except ValidationError as e:
            raise SchemaValidationError(
                schema_name=name,
                data=data,
                errors=[dict(err) for err in e.errors()],
            ) from e

    @classmethod
    def _describe_fields(cls, schema: type[BaseModel]) -> dict[str, dict[str, Any]]:
        fields = {}
        for name, field_info in schema.model_fields.items():
            annotation = field_info.annotation
            field_desc: dict[str, Any] = {
                "type": cls._type_to_string(annotation),
                "required": field_info.is_required(),
            }

            if field_info.default is not None and field_info.default is not PydanticUndefined:
                default_val = field_info.default
                try:
                    json.dumps(default_val)
                    field_desc["default"] = default_val
                except (TypeError, ValueError):
                    field_desc["default"] = str(default_val)

            if field_info.description:
                field_desc["description"] = field_info.description

            origin = get_origin(annotation)
            if origin is list:
                args = get_args(annotation)
                if args and isinstance(args[0], type) and issubclass(args[0], BaseModel):
                    field_desc["items_schema"] = args[0].__name__
            elif isinstance(annotation, type) and issubclass(annotation, BaseModel):
                field_desc["nested_schema"] = annotation.__name__

            fields[name] = field_desc

        return fields

    @classmethod
    def _get_required_fields(cls, schema: type[BaseModel]) -> list[str]:
        return [name for name, info in schema.model_fields.items() if info.is_required()]

    @classmethod
    def _get_optional_fields(cls, schema: type[BaseModel]) -> list[str]:
        return [name for name, info in schema.model_fields.items() if not info.is_required()]

    @classmethod
    def _type_to_string(cls, annotation: Any) -> str:
        if annotation is None:
            return "None"

        origin = get_origin(annotation)
        args = get_args(annotation)

        if origin is None:
            if hasattr(annotation, "__name__"):
                return str(annotation.__name__)
            return str(annotation)

        origin_name = getattr(origin, "__name__", str(origin))
        if args:
            args_str = ", ".join(cls._type_to_string(arg) for arg in args)
            return f"{origin_name}[{args_str}]"
        return origin_name

    @classmethod
    def _generate_example(cls, schema: type[BaseModel]) -> dict[str, Any] | None:
        try:
            return {
                name: cls._example_for_type(info.annotation, name)
                for name, info in schema.model_fields.items()
                if info.is_required()
            }
        except Exception:  # noqa: BLE001 - examples are best-effort decoration
            return None

    @classmethod
    def _example_for_type(cls, annotation: Any, field_name: str = "") -> Any:
        origin = get_origin(annotation)
        args = get_args(annotation)

        if origin is type(None):
            return None

        if origin is Union:
            for arg in args:
                if arg is not type(None):
                    return cls._example_for_type(arg, field_name)
            return None

        if origin is list:
            return [cls._example_for_type(args[0], field_name)] if args else []

        if origin is dict:
            return {}

        if origin is tuple:
            if args:
                return tuple(
                    cls._example_for_type(arg, f"{field_name}[{i}]") for i, arg in enumerate(args)
                )
            return ()

        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return cls._generate_example(annotation)

        if annotation is str:
            return f"example_{field_name}" if field_name else "example"
        if annotation is int:
            return 1
        if annotation is float:
            return 1.0
        if annotation is bool:
            return True

        return None
