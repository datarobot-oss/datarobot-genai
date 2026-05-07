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

"""Direct-import tests for ``drtools.dynamic.adapters``."""

import pytest

from datarobot_genai.drtools.dynamic.adapters import DrumMetadataAdapter
from datarobot_genai.drtools.dynamic.adapters import DrumTargetType
from datarobot_genai.drtools.dynamic.adapters import Metadata
from datarobot_genai.drtools.dynamic.adapters import get_default_schema
from datarobot_genai.drtools.dynamic.adapters import is_drum


class TestIsDrum:
    def test_true_when_drum_server_present(self) -> None:
        assert is_drum({"drum_server": "x"}) is True

    def test_true_when_drum_version_present(self) -> None:
        assert is_drum({"drum_version": "v1"}) is True

    def test_false_for_plain_dict(self) -> None:
        assert is_drum({}) is False


class TestDefaultSchemaSelection:
    def test_agentic_loads_agentic_schema(self) -> None:
        schema = get_default_schema(DrumTargetType.AGENTIC_WORKFLOW)
        assert isinstance(schema, dict)
        assert "properties" in schema

    def test_prediction_types_load_prediction_schema(self) -> None:
        schema = get_default_schema(DrumTargetType.BINARY)
        assert isinstance(schema, dict)
        assert "properties" in schema

    def test_unstructured_returns_empty(self) -> None:
        # UNSTRUCTURED is supported as a target type but has no fallback schema.
        assert get_default_schema(DrumTargetType.UNSTRUCTURED) == {}


class TestDefaultMetadataAdapter:
    def test_required_fields_round_trip(self) -> None:
        meta = Metadata(
            {
                "name": "tool",
                "description": "desc",
                "base_url": "https://api/",
                "endpoint": "/x",
                "input_schema": {
                    "type": "object",
                    "properties": {"path_params": {"type": "object", "properties": {}}},
                },
                "method": "post",
                "headers": {"X-A": "v"},
            }
        )
        assert meta.name == "tool"
        assert meta.description == "desc"
        assert meta.base_url == "https://api/"
        assert meta.endpoint == "/x"
        assert meta.method == "POST"
        assert meta.input_schema["type"] == "object"
        assert meta.headers == {"X-A": "v"}

    @pytest.mark.parametrize(
        "missing_field",
        ["base_url", "endpoint", "input_schema", "method"],
    )
    def test_missing_required_field_raises(self, missing_field: str) -> None:
        full = {
            "base_url": "https://api/",
            "endpoint": "/x",
            "input_schema": {"type": "object", "properties": {}},
            "method": "GET",
        }
        full.pop(missing_field)
        with pytest.raises(ValueError):
            getattr(Metadata(full), missing_field)

    def test_invalid_method_raises(self) -> None:
        meta = Metadata(
            {
                "base_url": "https://api/",
                "endpoint": "/x",
                "input_schema": {"type": "object", "properties": {}},
                "method": "OPTIONS",
            }
        )
        with pytest.raises(ValueError, match="unsupported `method`"):
            _ = meta.method

    def test_invalid_headers_type_raises(self) -> None:
        meta = Metadata(
            {
                "base_url": "https://api/",
                "endpoint": "/x",
                "input_schema": {"type": "object", "properties": {}},
                "method": "GET",
                "headers": "not-a-dict",
            }
        )
        with pytest.raises(ValueError, match="must be a dictionary"):
            _ = meta.headers


class TestDrumAdapter:
    def test_unsupported_target_type_raises(self) -> None:
        with pytest.raises(ValueError, match="is not supported"):
            DrumMetadataAdapter({"target_type": "nope"})

    def test_from_target_type_minimal(self) -> None:
        adapter = DrumMetadataAdapter.from_target_type("BINARY")
        assert adapter.target_type == DrumTargetType.BINARY
        # Falls back to default schema since model_metadata is empty.
        assert "properties" in adapter.input_schema

    def test_from_deployment_metadata_rejects_non_drum(self) -> None:
        with pytest.raises(ValueError, match="not from a DRUM deployment"):
            DrumMetadataAdapter.from_deployment_metadata({"target_type": "binary"})

    def test_from_deployment_metadata_passes_drum(self) -> None:
        adapter = DrumMetadataAdapter.from_deployment_metadata(
            {
                "target_type": "binary",
                "drum_version": "v1",
                "model_metadata": {"name": "m", "description": "d"},
            }
        )
        assert adapter.name == "m"
        assert adapter.description == "d"
        assert adapter.method == "POST"

    def test_endpoint_for_unstructured(self) -> None:
        adapter = DrumMetadataAdapter.from_target_type("unstructured")
        assert adapter.endpoint == "/predictionsUnstructured"

    def test_endpoint_for_agentic(self) -> None:
        adapter = DrumMetadataAdapter.from_target_type("agenticworkflow")
        assert adapter.endpoint == "/chat/completions"

    def test_prediction_headers_include_csv_content_type(self) -> None:
        adapter = DrumMetadataAdapter.from_target_type("binary")
        assert adapter.headers == {"Content-Type": "text/csv"}

    def test_agentic_headers_empty(self) -> None:
        adapter = DrumMetadataAdapter.from_target_type("agenticworkflow")
        assert adapter.headers == {}

    def test_input_schema_invalid_type_raises(self) -> None:
        adapter = DrumMetadataAdapter.from_deployment_metadata(
            {
                "target_type": "binary",
                "drum_version": "v1",
                "model_metadata": {"input_schema": "not-a-dict"},
            }
        )
        with pytest.raises(ValueError, match="missing a valid input schema"):
            _ = adapter.input_schema
