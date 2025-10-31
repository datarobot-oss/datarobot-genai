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

from unittest.mock import Mock
from unittest.mock import patch

import pytest

from datarobot_genai.drmcp.core.dynamic_tools.deployment.adapters.drum import is_drum
from datarobot_genai.drmcp.core.dynamic_tools.deployment.metadata import _fetch_deployment_metadata
from datarobot_genai.drmcp.core.dynamic_tools.deployment.metadata import _get_model_attribute
from datarobot_genai.drmcp.core.dynamic_tools.deployment.metadata import (
    _is_datarobot_structured_prediction,
)
from datarobot_genai.drmcp.core.dynamic_tools.deployment.metadata import _normalize_api_response
from datarobot_genai.drmcp.core.dynamic_tools.deployment.metadata import get_mcp_tool_metadata


@pytest.fixture
def drum_metadata():
    return {
        "drum_server": "server",
        "drum_version": "1.18.0",
        "target_type": "unstructured",
        "model_metadata": {
            "name": "model-name",
            "description": "model-description",
            "input_schema": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    }


@pytest.fixture
def non_drum_metadata():
    return {
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        "endpoint": "/predict",
    }


def test_is_drum_true(drum_metadata):
    assert is_drum(drum_metadata) is True


def test_is_drum_false(non_drum_metadata):
    assert is_drum(non_drum_metadata) is False


class TestNormalizeApiResponse:
    """Test cases for _normalize_api_response function."""

    def test_list_response_takes_first_element(self):
        """Test that list response returns first element."""
        mock_response = Mock()
        mock_response.json.return_value = [{"key": "value1"}, {"key": "value2"}]

        with patch(
            "datarobot_genai.drmcp.core.dynamic_tools.deployment.metadata.from_api"
        ) as mock_from_api:
            mock_from_api.return_value = [{"key": "value1"}, {"key": "value2"}]
            result = _normalize_api_response(mock_response)
            assert result == {"key": "value1"}

    def test_empty_list_response_returns_empty_dict(self):
        """Test that empty list response returns empty dict."""
        mock_response = Mock()
        mock_response.json.return_value = []

        with patch(
            "datarobot_genai.drmcp.core.dynamic_tools.deployment.metadata.from_api"
        ) as mock_from_api:
            mock_from_api.return_value = []
            result = _normalize_api_response(mock_response)
            assert result == {}

    def test_dict_response_returns_dict(self):
        """Test that dict response returns the dict."""
        mock_response = Mock()
        mock_response.json.return_value = {"key": "value"}

        with patch(
            "datarobot_genai.drmcp.core.dynamic_tools.deployment.metadata.from_api"
        ) as mock_from_api:
            mock_from_api.return_value = {"key": "value"}
            result = _normalize_api_response(mock_response)
            assert result == {"key": "value"}

    def test_non_list_non_dict_response_returns_empty_dict(self):
        """Test that non-list, non-dict response returns empty dict."""
        mock_response = Mock()
        mock_response.json.return_value = "string"

        with patch(
            "datarobot_genai.drmcp.core.dynamic_tools.deployment.metadata.from_api"
        ) as mock_from_api:
            mock_from_api.return_value = "string"
            result = _normalize_api_response(mock_response)
            assert result == {}


class TestGetModelAttribute:
    """Test cases for _get_model_attribute function."""

    def test_existing_key_returns_value(self):
        """Test that existing key returns the value."""
        model = {"target_type": "binary"}
        result = _get_model_attribute(model, "target_type")
        assert result == "binary"

    def test_missing_key_returns_default(self):
        """Test that missing key returns default value."""
        model = {"other_key": "value"}
        result = _get_model_attribute(model, "target_type", "default")
        assert result == "default"

    def test_none_value_returns_default(self):
        """Test that None value returns default."""
        model = {"target_type": None}
        result = _get_model_attribute(model, "target_type", "default")
        assert result == "default"

    def test_value_converted_to_lowercase(self):
        """Test that value is converted to lowercase."""
        model = {"target_type": "BINARY"}
        result = _get_model_attribute(model, "target_type")
        assert result == "binary"

    def test_empty_string_returns_default(self):
        """Test that empty string returns default."""
        model = {"target_type": ""}
        result = _get_model_attribute(model, "target_type", "default")
        assert result == "default"


class TestIsDataRobotStructuredPrediction:
    """Test cases for _is_datarobot_structured_prediction function."""

    def test_none_model_returns_none(self):
        """Test that None model returns None."""
        deployment = Mock()
        deployment.model = None
        result = _is_datarobot_structured_prediction(deployment)
        assert result is None

    def test_datarobot_build_env_with_valid_target_type(self):
        """Test datarobot build env with valid target type returns target type."""
        deployment = Mock()
        deployment.model = {"target_type": "binary", "build_environment_type": "datarobot"}
        result = _is_datarobot_structured_prediction(deployment)
        assert result == "binary"

    def test_datarobot_build_env_with_invalid_target_type(self):
        """Test datarobot build env with invalid target type returns None."""
        deployment = Mock()
        deployment.model = {"target_type": "invalid_type", "build_environment_type": "datarobot"}
        result = _is_datarobot_structured_prediction(deployment)
        assert result is None

    def test_non_datarobot_build_env_returns_none(self):
        """Test non-datarobot build env returns None."""
        deployment = Mock()
        deployment.model = {"target_type": "binary", "build_environment_type": "other"}
        result = _is_datarobot_structured_prediction(deployment)
        assert result is None

    def test_missing_target_type_returns_none(self):
        """Test missing target type returns None."""
        deployment = Mock()
        deployment.model = {"build_environment_type": "datarobot"}
        result = _is_datarobot_structured_prediction(deployment)
        assert result is None

    def test_missing_build_env_returns_none(self):
        """Test missing build env returns None."""
        deployment = Mock()
        deployment.model = {"target_type": "binary"}
        result = _is_datarobot_structured_prediction(deployment)
        assert result is None

    def test_empty_target_type_returns_none(self):
        """Test empty target type returns None."""
        deployment = Mock()
        deployment.model = {"target_type": "", "build_environment_type": "datarobot"}
        result = _is_datarobot_structured_prediction(deployment)
        assert result is None

    def test_empty_build_env_returns_none(self):
        """Test empty build env returns None."""
        deployment = Mock()
        deployment.model = {"target_type": "binary", "build_environment_type": ""}
        result = _is_datarobot_structured_prediction(deployment)
        assert result is None


class TestFetchDeploymentMetadata:
    """Test cases for _fetch_deployment_metadata function."""

    @patch("datarobot_genai.drmcp.core.dynamic_tools.deployment.metadata.get_api_client")
    @patch("datarobot_genai.drmcp.core.dynamic_tools.deployment.metadata._normalize_api_response")
    def test_successful_fetch(self, mock_normalize, mock_get_client):
        """Test successful metadata fetch."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_client.get.return_value = mock_response
        mock_get_client.return_value = mock_client
        mock_normalize.return_value = {"key": "value"}

        deployment = Mock()
        deployment.id = "test-deployment-id"

        result = _fetch_deployment_metadata(deployment)

        assert result == {"key": "value"}
        mock_client.get.assert_called_once_with(
            url="deployments/test-deployment-id/directAccess/info/"
        )
        mock_normalize.assert_called_once_with(mock_response)

    @patch("datarobot_genai.drmcp.core.dynamic_tools.deployment.metadata.get_api_client")
    def test_api_call_failure_raises_runtime_error(self, mock_get_client):
        """Test that API call failure raises RuntimeError."""
        mock_client = Mock()
        mock_client.get.side_effect = Exception("API Error")
        mock_get_client.return_value = mock_client

        deployment = Mock()
        deployment.id = "test-deployment-id"

        with pytest.raises(
            RuntimeError, match="Could not retrieve metadata for deployment test-deployment-id"
        ):
            _fetch_deployment_metadata(deployment)


class TestGetMcpToolMetadata:
    """Test cases for get_mcp_tool_metadata function."""

    @patch(
        "datarobot_genai.drmcp.core.dynamic_tools.deployment.metadata._is_datarobot_structured_prediction"
    )
    @patch(
        "datarobot_genai.drmcp.core.dynamic_tools.deployment.adapters.drum.DrumMetadataAdapter.from_target_type"
    )
    def test_datarobot_structured_prediction_returns_drum_adapter(
        self, mock_from_target_type, mock_is_structured
    ):
        """Test that DataRobot structured prediction returns DrumMetadataAdapter."""
        mock_is_structured.return_value = "binary"
        mock_adapter = Mock()
        mock_from_target_type.return_value = mock_adapter

        deployment = Mock()
        result = get_mcp_tool_metadata(deployment)

        assert result == mock_adapter
        mock_is_structured.assert_called_once_with(deployment)
        mock_from_target_type.assert_called_once_with("binary")

    @patch(
        "datarobot_genai.drmcp.core.dynamic_tools.deployment.metadata._is_datarobot_structured_prediction"
    )
    @patch(
        "datarobot_genai.drmcp.core.dynamic_tools.deployment.metadata._fetch_deployment_metadata"
    )
    @patch("datarobot_genai.drmcp.core.dynamic_tools.deployment.metadata.is_drum")
    @patch("datarobot_genai.drmcp.core.dynamic_tools.deployment.metadata.DrumMetadataAdapter")
    def test_drum_metadata_returns_drum_adapter(
        self, mock_drum_adapter, mock_is_drum, mock_fetch, mock_is_structured
    ):
        """Test that DRUM metadata returns DrumMetadataAdapter."""
        mock_is_structured.return_value = None
        mock_fetch.return_value = {"server": "drum"}
        mock_is_drum.return_value = True
        mock_adapter = Mock()
        mock_drum_adapter.from_deployment_metadata.return_value = mock_adapter

        deployment = Mock()
        result = get_mcp_tool_metadata(deployment)

        assert result == mock_adapter
        mock_fetch.assert_called_once_with(deployment)
        mock_is_drum.assert_called_once_with({"server": "drum"})
        mock_drum_adapter.from_deployment_metadata.assert_called_once_with({"server": "drum"})

    @patch(
        "datarobot_genai.drmcp.core.dynamic_tools.deployment.metadata._is_datarobot_structured_prediction"
    )
    @patch(
        "datarobot_genai.drmcp.core.dynamic_tools.deployment.metadata._fetch_deployment_metadata"
    )
    @patch("datarobot_genai.drmcp.core.dynamic_tools.deployment.metadata.is_drum")
    @patch("datarobot_genai.drmcp.core.dynamic_tools.deployment.metadata.Metadata")
    def test_default_metadata_returns_metadata_adapter(
        self, mock_metadata, mock_is_drum, mock_fetch, mock_is_structured
    ):
        """Test that default metadata returns Metadata adapter."""
        mock_is_structured.return_value = None
        mock_fetch.return_value = {"server": "other"}
        mock_is_drum.return_value = False
        mock_adapter = Mock()
        mock_metadata.return_value = mock_adapter

        deployment = Mock()
        result = get_mcp_tool_metadata(deployment)

        assert result == mock_adapter
        mock_fetch.assert_called_once_with(deployment)
        mock_is_drum.assert_called_once_with({"server": "other"})
        mock_metadata.assert_called_once_with({"server": "other"})
