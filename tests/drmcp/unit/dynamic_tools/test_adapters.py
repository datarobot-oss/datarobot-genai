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

"""Tests for metadata adapters (DRUM and Default)."""

import pytest

from datarobot_genai.drmcp.core.dynamic_tools.deployment.adapters.default import Metadata
from datarobot_genai.drmcp.core.dynamic_tools.deployment.adapters.drum import DrumMetadataAdapter
from datarobot_genai.drmcp.core.dynamic_tools.deployment.adapters.drum import is_drum


# Fixtures for DRUM metadata
@pytest.fixture
def drum_metadata_unstructured():
    """DRUM metadata for unstructured target type."""
    return {
        "drum_server": "flask",
        "drum_version": "1.18.0",
        "target_type": "unstructured",
        "model_metadata": {
            "name": "text_processor",
            "description": "Process text data",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                    }
                },
                "required": ["data"],
            },
        },
    }


@pytest.fixture
def drum_metadata_binary():
    """DRUM metadata for binary target type."""
    return {
        "drum_server": "flask",
        "drum_version": "1.17.2",
        "target_type": "binary",
        "model_metadata": {
            "name": "classifier",
            "description": "Binary classifier",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data": {
                        "type": "string",
                        "description": "csv with rows for predictions",
                    },
                    "json": {
                        "type": "object",
                        "properties": {
                            "param1": {"type": "number"},
                            "param2": {"type": "string"},
                        },
                    },
                },
            },
        },
    }


@pytest.fixture
def drum_adapter(drum_metadata_unstructured):
    """Create DRUM adapter with unstructured metadata."""
    return DrumMetadataAdapter.from_deployment_metadata(drum_metadata_unstructured)


# Fixtures for default metadata
@pytest.fixture
def valid_metadata():
    """Return valid external deployment metadata."""
    return {
        "name": "weather_api",
        "description": "Weather forecast API",
        "base_url": "https://api.weather.com",
        "endpoint": "/forecast",
        "method": "POST",
        "headers": {"x-custom-key": "value"},
        "input_schema": {
            "type": "object",
            "properties": {
                "json": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                }
            },
        },
    }


@pytest.fixture
def default_adapter(valid_metadata):
    """Create default adapter with valid metadata."""
    return Metadata(valid_metadata)


class TestDrumMetadataAdapter:
    """Tests for DRUM metadata adapter - happy path."""

    def test_drum_adapter_initialization_success(self, drum_metadata_unstructured):
        """Test successful DRUM adapter initialization."""
        adapter = DrumMetadataAdapter.from_deployment_metadata(drum_metadata_unstructured)

        assert adapter.metadata == drum_metadata_unstructured
        assert adapter.target_type == "unstructured"

    def test_drum_adapter_raises_on_non_drum_metadata(self):
        """Test that adapter raises ValueError for non-DRUM metadata."""
        non_drum_metadata = {"endpoint": "/predict", "method": "POST"}

        with pytest.raises(ValueError, match="not from a DRUM deployment"):
            DrumMetadataAdapter.from_deployment_metadata(non_drum_metadata)

    def test_drum_adapter_basic_properties(self, drum_adapter):
        """Test basic adapter properties."""
        assert (drum_adapter.name, drum_adapter.description, drum_adapter.method) == (
            "text_processor",
            "Process text data",
            "POST",
        )

    @pytest.mark.parametrize(
        "target_type,expected_endpoint",
        [
            pytest.param("binary", "/predictions", id="binary"),
            pytest.param("regression", "/predictions", id="regression"),
            pytest.param("multiclass", "/predictions", id="multiclass"),
            pytest.param("textgeneration", "/predictions", id="text_generation"),
            pytest.param("unstructured", "/predictionsUnstructured", id="unstructured"),
            pytest.param("vectordatabase", "/predictions", id="vector_database"),
            pytest.param("agenticworkflow", "/chat/completions", id="agentic_workflow"),
            pytest.param("anomaly", "/predictions", id="anomaly"),
            pytest.param("geopoint", "/predictions", id="geopoint"),
        ],
    )
    def test_drum_adapter_endpoint_by_target_type(
        self, drum_metadata_unstructured, target_type, expected_endpoint
    ):
        """Test endpoint selection based on target type."""
        drum_metadata_unstructured["target_type"] = target_type
        adapter = DrumMetadataAdapter.from_deployment_metadata(drum_metadata_unstructured)

        assert adapter.endpoint == expected_endpoint

    def test_drum_adapter_input_schema_property(self, drum_adapter, drum_metadata_unstructured):
        """Test input_schema property extraction."""
        expected_schema = drum_metadata_unstructured["model_metadata"]["input_schema"]

        assert drum_adapter.input_schema == expected_schema
        assert "type" in drum_adapter.input_schema
        assert "properties" in drum_adapter.input_schema

    def test_drum_adapter_headers_are_empty(self, drum_adapter):
        """Test that DRUM deployments have Content-Type header."""
        assert drum_adapter.headers == {}

    @pytest.mark.parametrize(
        "missing_field,expected_value",
        [
            pytest.param("name", "", id="missing_name"),
            pytest.param("description", "", id="missing_description"),
        ],
    )
    def test_drum_adapter_handles_missing_model_metadata_fields(
        self, drum_metadata_unstructured, missing_field, expected_value
    ):
        """Test adapter handles missing fields in model_metadata."""
        del drum_metadata_unstructured["model_metadata"][missing_field]
        adapter = DrumMetadataAdapter.from_deployment_metadata(drum_metadata_unstructured)

        assert getattr(adapter, missing_field) == expected_value

    def test_drum_adapter_raises_on_unsupported_target_type(self, drum_metadata_unstructured):
        """Test that adapter raises ValueError for unsupported target_type."""
        drum_metadata_unstructured["target_type"] = "MCP"

        with pytest.raises(
            ValueError,
            match="The deployment target_type: MCP is not supported, to be registered as MCP Tool.",
        ):
            DrumMetadataAdapter.from_deployment_metadata(drum_metadata_unstructured)


class TestDefaultMetadataAdapter:
    """Tests for default metadata adapter - happy path."""

    def test_default_adapter_initialization(self, valid_metadata):
        """Test successful adapter initialization."""
        adapter = Metadata(valid_metadata)
        assert adapter.metadata == valid_metadata

    def test_default_adapter_basic_properties(self, default_adapter):
        """Test basic adapter properties."""
        assert (
            default_adapter.name,
            default_adapter.description,
            default_adapter.base_url,
            default_adapter.endpoint,
        ) == (
            "weather_api",
            "Weather forecast API",
            "https://api.weather.com",
            "/forecast",
        )

    @pytest.mark.parametrize(
        "method",
        [
            pytest.param("GET", id="get_method"),
            pytest.param("POST", id="post_method"),
        ],
    )
    def test_default_adapter_method_property(self, valid_metadata, method):
        """Test method property extraction for valid methods."""
        valid_metadata["method"] = method
        adapter = Metadata(valid_metadata)

        assert adapter.method == method

    @pytest.mark.parametrize(
        "method",
        [
            pytest.param("HEAD", id="head_method"),
            pytest.param("OPTIONS", id="options_method"),
        ],
    )
    def test_default_adapter_unsupported_methods_raise_error(self, valid_metadata, method):
        """Test that currently unsupported HTTP methods raise ValueError.

        This test documents the current behavior and should be removed or updated
        when these methods are officially supported.
        """
        valid_metadata["method"] = method
        adapter = Metadata(valid_metadata)

        with pytest.raises(ValueError, match=f"unsupported `method`: {method}"):
            _ = adapter.method

    def test_default_adapter_headers_property(self, default_adapter):
        """Test headers property extraction."""
        assert default_adapter.headers == {"x-custom-key": "value"}

    def test_default_adapter_input_schema_property(self, default_adapter, valid_metadata):
        """Test input_schema property extraction."""
        assert default_adapter.input_schema == valid_metadata["input_schema"]

    @pytest.mark.parametrize(
        "missing_field,expected_value",
        [
            pytest.param("name", "", id="missing_name"),
            pytest.param("description", "", id="missing_description"),
        ],
    )
    def test_default_adapter_missing_optional_fields_return_empty(
        self, valid_metadata, missing_field, expected_value
    ):
        """Test that missing optional fields return empty string."""
        del valid_metadata[missing_field]
        adapter = Metadata(valid_metadata)

        assert getattr(adapter, missing_field) == expected_value

    @pytest.mark.parametrize(
        "missing_field,error_match",
        [
            pytest.param("base_url", "missing required 'base_url'", id="missing_base_url"),
            pytest.param("endpoint", "missing required 'endpoint'", id="missing_endpoint"),
            pytest.param("method", "missing required 'method'", id="missing_method"),
            pytest.param(
                "input_schema",
                "missing required 'inputSchema'",
                id="missing_input_schema",
            ),
        ],
    )
    def test_default_adapter_raises_on_missing_required_fields(
        self, valid_metadata, missing_field, error_match
    ):
        """Test that missing required fields raise ValueError."""
        del valid_metadata[missing_field]
        adapter = Metadata(valid_metadata)

        with pytest.raises(ValueError, match=error_match):
            _ = getattr(adapter, missing_field)

    def test_default_adapter_raises_on_invalid_method(self, valid_metadata):
        """Test that unsupported method raises ValueError."""
        valid_metadata["method"] = "XYZ"
        adapter = Metadata(valid_metadata)

        with pytest.raises(ValueError, match="unsupported `method`: XYZ"):
            _ = adapter.method

    def test_default_adapter_empty_headers_when_missing(self, valid_metadata):
        """Test that missing headers returns empty dict."""
        del valid_metadata["headers"]
        adapter = Metadata(valid_metadata)

        assert adapter.headers == {}

    def test_default_adapter_raises_on_invalid_headers_type(self, valid_metadata):
        """Test that non-dict headers raises ValueError."""
        valid_metadata["headers"] = "invalid"
        adapter = Metadata(valid_metadata)

        with pytest.raises(ValueError, match="'headers' field must be a dictionary"):
            _ = adapter.headers


@pytest.mark.parametrize(
    "metadata,expected",
    [
        pytest.param(
            {"drum_server": "flask", "drum_version": "1.18.0"},
            True,
            id="valid_drum_flask",
        ),
        pytest.param(
            {"drum_server": "gunicorn", "drum_version": "1.17.2"},
            True,
            id="valid_drum_gunicorn",
        ),
        pytest.param({"drum_server": "flask"}, True, id="missing_version"),
        pytest.param({"drum_version": "1.18.0"}, True, id="missing_server"),
        pytest.param({}, False, id="empty_metadata"),
        pytest.param({"endpoint": "/predict"}, False, id="non_drum_metadata"),
    ],
)
def test_is_drum_detection(metadata, expected):
    """Test DRUM deployment detection."""
    assert is_drum(metadata) == expected
