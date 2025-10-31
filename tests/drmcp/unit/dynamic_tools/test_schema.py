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

"""Tests for schema conversion and validation."""

from datetime import date

import pytest
from openai.types.chat.completion_create_params import CompletionCreateParamsBase
from pydantic import BaseModel
from pydantic import Field
from pydantic import TypeAdapter
from pydantic import ValidationError

from datarobot_genai.drmcp.core.dynamic_tools.deployment.adapters.drum import DrumTargetType
from datarobot_genai.drmcp.core.dynamic_tools.deployment.adapters.drum import get_default_schema
from datarobot_genai.drmcp.core.dynamic_tools.schema import SchemaValidationError
from datarobot_genai.drmcp.core.dynamic_tools.schema import create_input_schema_pydantic_model
from datarobot_genai.drmcp.core.dynamic_tools.schema import create_schema_model

TEMPERATURE_DEFAULT = 0.7
ITEM_VALUE = 2.5


class TestCreateInputSchemaPydanticModel:
    """Tests for create_input_schema_pydantic_model function - happy path."""

    def test_create_model_with_query_params(self):
        """Test creating model with query parameters."""
        schema = {
            "type": "object",
            "properties": {
                "query_params": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "days": {"type": "integer"},
                    },
                    "required": ["city"],
                }
            },
            "required": ["query_params"],
        }

        model = create_input_schema_pydantic_model(schema)

        assert issubclass(model, BaseModel)
        input_model = model(query_params={"city": "NYC", "days": 3})
        assert (input_model.query_params.city, input_model.query_params.days) == (
            "NYC",
            3,
        )

    def test_pydanic_schema_with_optional_subschema(self):
        class WeatherRequest(BaseModel):
            class PathParams(BaseModel):
                city: str = Field(description="City name to get weather for")

            class QueryParams(BaseModel):
                units: str = Field(default="metric", description="metric or imperial")
                lang: str = Field(default="en", description="Language code")

            path_params: PathParams
            query_params: QueryParams | None = None

        model = create_input_schema_pydantic_model(WeatherRequest.model_json_schema())
        input_model = model(path_params={"city": "London"}, query_params={"units": "imperial"})

        assert input_model.path_params.city == "London"
        assert input_model.query_params.units == "imperial"
        assert input_model.query_params.lang == "en"

    def test_pydantic_multi_object_schema_with_optional_primitive_fields(self):
        class DateRange(BaseModel):
            start: date = Field(description="Start date in YYYY-MM-DD format")
            end: date | None = Field(default=None, description="End date in YYYY-MM-DD format")

        class Filters(BaseModel):
            status: str | None = None
            date_range: DateRange | None = None

        class QueryRequest(BaseModel):
            """Query a dataset with filters."""

            class JsonBody(BaseModel):
                filters: Filters | None = None
                limit: int = Field(default=100, description="Max results")

            json: JsonBody

        model = create_input_schema_pydantic_model(QueryRequest.model_json_schema())

        input_model = model(
            json={
                "limit": 50,
                "filters": {
                    "status": "active",
                    "date_range": {"start": "2024-01-01", "end": "2024-01-31"},
                },
            }
        )

        assert input_model.json.limit == 50
        assert input_model.json.filters.status == "active"
        assert input_model.json.filters.date_range.start == "2024-01-01"
        assert input_model.json.filters.date_range.end == "2024-01-31"

    def test_create_model_with_json_body(self):
        """Test creating model with JSON body."""
        schema = {
            "type": "object",
            "properties": {
                "json": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                    },
                    "required": ["name"],
                }
            },
            "required": ["json"],
        }

        model = create_input_schema_pydantic_model(schema)

        input_model = model(json={"name": "Alice", "age": 30})
        assert input_model.json.name == "Alice"
        assert input_model.json.age == 30

    def test_create_model_with_path_params(self):
        """Test creating model with path parameters."""
        schema = {
            "type": "object",
            "properties": {
                "path_params": {
                    "type": "object",
                    "properties": {"user_id": {"type": "string"}},
                    "required": ["user_id"],
                }
            },
            "required": ["path_params"],
        }

        model = create_input_schema_pydantic_model(schema)

        input_model = model(path_params={"user_id": "123"})
        assert input_model.path_params.user_id == "123"

    def test_create_model_with_all_param_types(self):
        """Test creating model with all parameter types (path, query, and JSON body)."""
        schema = {
            "type": "object",
            "properties": {
                "path_params": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}},
                },
                "query_params": {
                    "type": "object",
                    "properties": {"filter": {"type": "string"}},
                },
                "json": {
                    "type": "object",
                    "properties": {"data": {"type": "string"}},
                },
            },
        }

        model = create_input_schema_pydantic_model(schema)

        input_model = model(
            path_params={"id": "1"},
            query_params={"filter": "active"},
            json={"data": "value"},
        )
        assert input_model.path_params.id == "1"
        assert input_model.query_params.filter == "active"
        assert input_model.json.data == "value"

    def test_create_model_validates_required_fields(self):
        """Test that model validates required fields."""
        schema = {
            "type": "object",
            "properties": {
                "json": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                }
            },
            "required": ["json"],
        }

        model = create_input_schema_pydantic_model(schema)

        with pytest.raises(ValidationError):
            model(json={})

    def test_create_model_with_chat_completions_schema(self):
        schema = {
            "type": "object",
            "properties": {
                "json": {
                    "$defs": {
                        "Message": {
                            "properties": {
                                "content": {"title": "Content", "type": "string"},
                                "role": {
                                    "enum": ["system", "user", "assistant"],
                                    "title": "Role",
                                    "type": "string",
                                },
                            },
                            "required": ["role", "content"],
                            "title": "Message",
                            "type": "object",
                        }
                    },
                    "type": "object",
                    "properties": {
                        "messages": {
                            "items": {"$ref": "#/$defs/Message"},
                            "title": "Messages",
                            "type": "array",
                        }
                    },
                    "required": ["messages"],
                    "title": "ChatRequest",
                }
            },
            "required": ["json"],
        }

        model = create_input_schema_pydantic_model(schema)
        instance = model(json={"messages": [{"role": "user", "content": "Hello"}]})
        assert len(instance.json.messages) == 1
        assert instance.json.messages[0].role == "user"

    def test_entire_chat_completions_schema(self):
        """Test with the real OpenAI chat completions schema."""
        # Get the real schema from OpenAI library for CompletionCreateParamsBase
        adapter = TypeAdapter(CompletionCreateParamsBase)
        input_schema = {
            "type": "object",
            "properties": {"json": adapter.json_schema()},
            "required": ["json"],
        }

        model = create_input_schema_pydantic_model(input_schema)

        # Test with a complete example
        instance = model(
            json={
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello!"},
                ],
                "model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 100,
            }
        )

        assert len(instance.json.messages) == 2
        assert instance.json.messages[0]["role"] == "system"
        assert instance.json.messages[0]["content"] == "You are a helpful assistant."
        assert instance.json.messages[1]["role"] == "user"
        assert instance.json.messages[1]["content"] == "Hello!"
        assert instance.json.model == "gpt-4"
        assert instance.json.temperature == TEMPERATURE_DEFAULT
        assert instance.json.max_tokens == 100

    def test_create_model_with_optional_fields(self):
        """Test creating model with optional fields."""
        schema = {
            "type": "object",
            "properties": {
                "query_params": {
                    "type": "object",
                    "properties": {
                        "required_field": {"type": "string"},
                        "optional_field": {"type": "string"},
                    },
                    "required": ["required_field"],
                }
            },
        }

        model = create_input_schema_pydantic_model(schema)

        input_model = model(query_params={"required_field": "value"})
        assert input_model.query_params.required_field == "value"

    def test_create_model_with_data_field_for_csv(self):
        """Test creating model with data field for text/csv upload with raw CSV bytes."""
        schema = {
            "type": "object",
            "properties": {
                "data": {
                    "type": "string",
                    "description": "Raw CSV file content as text/csv bytes",
                },
            },
            "required": ["data"],
        }

        model = create_input_schema_pydantic_model(schema)

        # Simulate raw CSV content that would come from a text/csv prediction
        # this is common for predictive models.
        csv_content = "name,age,city\nAlice,30,NYC\nBob,25,LA\nCharlie,35,SF"
        input_model = model(data=csv_content)

        assert input_model.data == csv_content
        # Verify it's the raw CSV string that can be parsed
        lines = input_model.data.split("\n")
        assert lines[0] == "name,age,city"
        assert len(lines) == 4  # header + 3 data rows

    def test_create_model_with_data_field_nested_structure(self):
        """Test creating model with nested data structure for form uploads."""
        schema = {
            "type": "object",
            "properties": {
                "data": {
                    "type": "object",
                    "properties": {
                        "metadata": {
                            "type": "object",
                            "properties": {
                                "author": {"type": "string"},
                                "timestamp": {"type": "integer"},
                            },
                        },
                        "content": {"type": "string"},
                    },
                }
            },
        }

        model = create_input_schema_pydantic_model(schema)

        input_model = model(
            data={
                "metadata": {"author": "Alice", "timestamp": 1234567890},
                "content": "file content",
            }
        )

        assert input_model.data.metadata.author == "Alice"
        assert input_model.data.metadata.timestamp == 1234567890
        assert input_model.data.content == "file content"

    def test_create_model_with_path_and_query_params(self):
        """Test creating model with both path and query parameters."""
        schema = {
            "type": "object",
            "properties": {
                "path_params": {
                    "type": "object",
                    "properties": {
                        "deployment_id": {"type": "string"},
                    },
                    "required": ["deployment_id"],
                },
                "query_params": {
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer"},
                        "offset": {"type": "integer"},
                    },
                },
            },
        }

        model = create_input_schema_pydantic_model(schema)

        input_model = model(
            path_params={"deployment_id": "abc123"},
            query_params={"limit": 10, "offset": 0},
        )

        assert input_model.path_params.deployment_id == "abc123"
        assert input_model.query_params.limit == 10
        assert input_model.query_params.offset == 0

    def test_create_model_with_json_and_query_params(self):
        """Test creating model with JSON body and query parameters."""
        schema = {
            "type": "object",
            "properties": {
                "query_params": {
                    "type": "object",
                    "properties": {"format": {"type": "string"}},
                },
                "json": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
        }

        model = create_input_schema_pydantic_model(schema)

        input_model = model(
            query_params={"format": "json"},
            json={"data": ["item1", "item2", "item3"]},
        )

        assert input_model.query_params.format == "json"
        assert len(input_model.json.data) == 3

    def test_create_model_with_all_four_param_types(self):
        """Test creating model with path, query, data, and json parameters."""
        schema = {
            "type": "object",
            "properties": {
                "path_params": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}},
                },
                "query_params": {
                    "type": "object",
                    "properties": {"verbose": {"type": "boolean"}},
                },
                "data": {
                    "type": "object",
                    "properties": {"file": {"type": "string"}},
                },
                "json": {
                    "type": "object",
                    "properties": {
                        "metadata": {
                            "type": "object",
                            "properties": {"key": {"type": "string"}},
                        },
                    },
                },
            },
        }

        model = create_input_schema_pydantic_model(schema)

        input_model = model(
            path_params={"id": "123"},
            query_params={"verbose": True},
            data={"file": "content"},
            json={"metadata": {"key": "value"}},
        )

        assert input_model.path_params.id == "123"
        assert input_model.query_params.verbose is True
        assert input_model.data.file == "content"
        assert input_model.json.metadata.key == "value"

    def test_create_model_rejects_nested_in_query_params(self):
        """Test that nested objects in query_params are rejected."""
        schema = {
            "type": "object",
            "properties": {
                "query_params": {
                    "type": "object",
                    "properties": {
                        "filter": {
                            "type": "object",
                            "properties": {"status": {"type": "string"}},
                        },
                    },
                },
            },
        }

        with pytest.raises(SchemaValidationError, match="supports only flat structures"):
            create_input_schema_pydantic_model(schema)

    def test_create_model_rejects_nested_in_path_params(self):
        """Test that nested objects in path_params are rejected."""
        schema = {
            "type": "object",
            "properties": {
                "path_params": {
                    "type": "object",
                    "properties": {
                        "user": {
                            "type": "object",
                            "properties": {"id": {"type": "string"}},
                        },
                    },
                },
            },
        }

        with pytest.raises(SchemaValidationError, match="supports only flat structures"):
            create_input_schema_pydantic_model(schema)

    def test_create_model_allows_nested_in_json(self):
        """Test that nested objects are allowed in json field."""
        schema = {
            "type": "object",
            "properties": {
                "json": {
                    "type": "object",
                    "properties": {
                        "user": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "profile": {
                                    "type": "object",
                                    "properties": {"bio": {"type": "string"}},
                                },
                            },
                        },
                    },
                },
            },
        }

        model = create_input_schema_pydantic_model(schema)

        input_model = model(json={"user": {"name": "Alice", "profile": {"bio": "Developer"}}})

        assert input_model.json.user.name == "Alice"
        assert input_model.json.user.profile.bio == "Developer"

    def test_create_model_allows_nested_in_data(self):
        """Test that nested objects are allowed in data field."""
        schema = {
            "type": "object",
            "properties": {
                "data": {
                    "type": "object",
                    "properties": {
                        "upload": {
                            "type": "object",
                            "properties": {
                                "file": {"type": "string"},
                                "metadata": {
                                    "type": "object",
                                    "properties": {"size": {"type": "integer"}},
                                },
                            },
                        },
                    },
                },
            },
        }

        model = create_input_schema_pydantic_model(schema)

        input_model = model(data={"upload": {"file": "content", "metadata": {"size": 1024}}})

        assert input_model.data.upload.file == "content"
        assert input_model.data.upload.metadata.size == 1024

    def test_create_model_with_array_in_json(self):
        """Test creating model with array fields in JSON body."""
        schema = {
            "type": "object",
            "properties": {
                "json": {
                    "type": "object",
                    "properties": {
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "value": {"type": "number"},
                                },
                            },
                        },
                    },
                },
            },
        }

        model = create_input_schema_pydantic_model(schema)

        input_model = model(
            json={
                "items": [
                    {"id": "a", "value": 1.5},
                    {"id": "b", "value": 2.5},
                ]
            }
        )

        assert len(input_model.json.items) == 2
        assert input_model.json.items[0].id == "a"
        assert input_model.json.items[1].value == ITEM_VALUE

    def test_create_model_with_empty_schema_when_allowed(self):
        """Test creating model with empty schema when explicitly allowed."""
        schema = {"type": "object", "properties": {}}

        model = create_input_schema_pydantic_model(schema, allow_empty=True)

        instance = model()
        assert instance is not None

    def test_create_model_rejects_empty_schema_by_default(self):
        """Test that empty schema is rejected by default."""
        schema = {"type": "object", "properties": {}}

        with pytest.raises(SchemaValidationError, match="Empty schemas are disabled"):
            create_input_schema_pydantic_model(schema, allow_empty=False)

    def test_create_model_rejects_unsupported_top_level_property(self):
        """Test that unsupported top-level properties are rejected."""
        schema = {
            "type": "object",
            "properties": {
                "headers": {  # Invalid top-level property
                    "type": "object",
                    "properties": {"Authorization": {"type": "string"}},
                },
            },
        }

        with pytest.raises(SchemaValidationError, match="unsupported top-level properties"):
            create_input_schema_pydantic_model(schema)

    def test_create_model_with_descriptions_on_properties(self):
        """Test that property descriptions are preserved."""
        schema = {
            "type": "object",
            "properties": {
                "query_params": {
                    "type": "object",
                    "description": "Query string parameters",
                    "properties": {"search": {"type": "string"}},
                },
                "json": {
                    "type": "object",
                    "description": "Request body",
                    "properties": {"data": {"type": "string"}},
                },
            },
        }

        model = create_input_schema_pydantic_model(schema)

        input_model = model(
            query_params={"search": "test"},
            json={"data": "value"},
        )

        assert input_model.query_params.search == "test"
        assert input_model.json.data == "value"

    def test_create_model_with_optional_all_fields(self):
        """Test that all parameter types are optional when not in required."""
        schema = {
            "type": "object",
            "properties": {
                "path_params": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}},
                },
                "query_params": {
                    "type": "object",
                    "properties": {"filter": {"type": "string"}},
                },
            },
        }

        model = create_input_schema_pydantic_model(schema)

        # Should work with no parameters
        input_model = model()
        assert input_model.path_params is None
        assert input_model.query_params is None

        # Should work with only one parameter
        input_model2 = model(path_params={"id": "123"})
        assert input_model2.path_params.id == "123"
        assert input_model2.query_params is None

    def test_create_model_with_ref_in_json_body(self):
        """Test model with $ref in JSON body definition."""
        schema = {
            "type": "object",
            "$defs": {
                "User": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "email": {"type": "string"},
                    },
                    "required": ["name"],
                }
            },
            "properties": {
                "json": {
                    "type": "object",
                    "properties": {
                        "user": {"$ref": "#/$defs/User"},
                    },
                },
            },
        }

        model = create_input_schema_pydantic_model(schema)

        input_model = model(json={"user": {"name": "Alice", "email": "alice@example.com"}})

        assert input_model.json.user.name == "Alice"
        assert input_model.json.user.email == "alice@example.com"

    def test_create_model_with_multiple_refs(self):
        """Test model with multiple $ref definitions."""
        schema = {
            "type": "object",
            "$defs": {
                "Address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"},
                    },
                },
                "Contact": {
                    "type": "object",
                    "properties": {
                        "phone": {"type": "string"},
                        "email": {"type": "string"},
                    },
                },
            },
            "properties": {
                "json": {
                    "type": "object",
                    "properties": {
                        "address": {"$ref": "#/$defs/Address"},
                        "contact": {"$ref": "#/$defs/Contact"},
                    },
                },
            },
        }

        model = create_input_schema_pydantic_model(schema)

        input_model = model(
            json={
                "address": {"street": "123 Main", "city": "NYC"},
                "contact": {"phone": "555-1234", "email": "test@example.com"},
            }
        )

        assert input_model.json.address.city == "NYC"
        assert input_model.json.contact.email == "test@example.com"

    def test_create_model_rejects_primitive_type_for_json(self):
        """Test that primitive types are rejected for json field."""
        schema = {
            "type": "object",
            "properties": {"json": {"type": "string", "description": "This should fail"}},
        }

        with pytest.raises(SchemaValidationError, match="does not support primitive type"):
            create_input_schema_pydantic_model(schema)

    def test_create_model_rejects_primitive_type_for_query_params(self):
        """Test that primitive types are rejected for query_params field."""
        schema = {
            "type": "object",
            "properties": {"query_params": {"type": "string"}},
        }

        with pytest.raises(SchemaValidationError, match="does not support primitive type"):
            create_input_schema_pydantic_model(schema)

    def test_create_model_rejects_primitive_type_for_path_params(self):
        """Test that primitive types are rejected for path_params field."""
        schema = {
            "type": "object",
            "properties": {"path_params": {"type": "integer"}},
        }

        with pytest.raises(SchemaValidationError, match="does not support primitive type"):
            create_input_schema_pydantic_model(schema)


class TestCreateSchemaModel:
    """Tests for create_schema_model function - happy path."""

    def test_create_simple_model(self):
        """Test creating a simple flat model."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "active": {"type": "boolean"},
            },
            "required": ["name"],
        }

        model = create_schema_model(name="SimpleModel", schema=schema, allow_nested=False)

        assert issubclass(model, BaseModel)
        instance = model(name="Alice", age=30, active=True)
        assert (instance.name, instance.age, instance.active) == ("Alice", 30, True)

    @pytest.mark.parametrize(
        ("field_type", "python_type", "test_value"),
        [
            ("string", str, "test"),
            ("integer", int, 42),
            ("number", float, 3.14),
            ("boolean", bool, True),
        ],
    )
    def test_create_model_with_basic_types(self, field_type, python_type, test_value):
        """Test model creation with basic JSON schema types."""
        schema = {
            "type": "object",
            "properties": {"field": {"type": field_type}},
        }

        model = create_schema_model(name="TestModel", schema=schema, allow_nested=False)

        instance = model(field=test_value)
        assert isinstance(instance.field, python_type)
        assert instance.field == test_value

    def test_create_model_with_array_of_strings(self):
        """Test creating model with array of strings."""
        schema = {
            "type": "object",
            "properties": {
                "tags": {"type": "array", "items": {"type": "string"}},
            },
        }

        model = create_schema_model(name="ArrayModel", schema=schema, allow_nested=False)

        instance = model(tags=["tag1", "tag2", "tag3"])
        assert len(instance.tags) == 3
        assert instance.tags[0] == "tag1"

    def test_create_model_with_array_of_integers(self):
        """Test creating model with array of integers."""
        schema = {
            "type": "object",
            "properties": {
                "numbers": {"type": "array", "items": {"type": "integer"}},
            },
        }

        model = create_schema_model(name="NumbersModel", schema=schema, allow_nested=False)

        instance = model(numbers=[1, 2, 3, 4, 5])
        assert (len(instance.numbers), sum(instance.numbers)) == (5, 15)

    def test_create_model_with_nested_object_when_allowed(self):
        """Test creating model with nested object when allowed."""
        schema = {
            "type": "object",
            "properties": {
                "address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"},
                    },
                }
            },
        }

        model = create_schema_model(name="NestedModel", schema=schema, allow_nested=True)

        instance = model(address={"street": "123 Main St", "city": "NYC"})
        assert (instance.address.street, instance.address.city) == (
            "123 Main St",
            "NYC",
        )

    def test_create_model_rejects_nested_when_not_allowed(self):
        """Test that nested objects raise error when not allowed."""
        schema = {
            "type": "object",
            "properties": {
                "address": {
                    "type": "object",
                    "properties": {"street": {"type": "string"}},
                }
            },
        }

        with pytest.raises(SchemaValidationError, match="supports only flat structures"):
            create_schema_model(name="FlatModel", schema=schema, allow_nested=False)

    def test_create_model_with_field_descriptions(self):
        """Test that field descriptions are preserved in model."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "User's full name"},
                "email": {"type": "string", "description": "User's email address"},
            },
        }

        model = create_schema_model(name="DescribedModel", schema=schema, allow_nested=False)

        instance = model(name="Alice", email="alice@example.com")
        assert instance.name == "Alice"
        assert instance.email == "alice@example.com"

    def test_create_model_validates_types(self):
        """Test that model validates field types."""
        schema = {
            "type": "object",
            "properties": {
                "age": {"type": "integer"},
            },
        }

        model = create_schema_model(name="TypedModel", schema=schema, allow_nested=False)

        with pytest.raises(ValidationError):
            model(age="not a number")

    def test_create_model_with_default_values(self):
        """Test creating model with default values."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "status": {"type": "string", "default": "active"},
            },
        }

        model = create_schema_model(name="DefaultModel", schema=schema, allow_nested=False)

        instance = model(name="Alice")
        assert instance.name == "Alice"

    def test_create_model_empty_schema(self):
        """Test creating model with minimal schema."""
        schema = {"type": "object", "properties": {}}

        model = create_schema_model(name="EmptyModel", schema=schema, allow_nested=False)

        instance = model()
        assert instance is not None


class TestSchemaValidationError:
    """Tests for SchemaValidationError exception."""

    def test_schema_validation_error_is_exception(self):
        """Test that SchemaValidationError is an Exception."""
        assert issubclass(SchemaValidationError, Exception)

    def test_schema_validation_error_can_be_raised(self):
        """Test that SchemaValidationError can be raised and caught."""
        with pytest.raises(SchemaValidationError, match="Test error"):
            raise SchemaValidationError("Test error")

    def test_schema_validation_error_message(self):
        """Test that SchemaValidationError preserves error message."""
        error_msg = "Schema depth exceeds maximum allowed"

        with pytest.raises(SchemaValidationError, match=error_msg):
            raise SchemaValidationError(error_msg)

    @pytest.mark.parametrize("target_type", list(DrumTargetType))
    def test_fallback_schemas_can_be_parsed(self, target_type):
        fallback_schema = get_default_schema(target_type)
        if fallback_schema:
            _ = create_input_schema_pydantic_model(fallback_schema)
