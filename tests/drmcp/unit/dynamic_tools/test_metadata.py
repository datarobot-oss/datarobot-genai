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

from datarobot_genai.drmcp.core.dynamic_tools.deployment.metadata import _fetch_deployment_metadata
from datarobot_genai.drmcp.core.dynamic_tools.deployment.metadata import _fetch_supports_chat_api
from datarobot_genai.drmcp.core.dynamic_tools.deployment.metadata import _normalize_api_response
from datarobot_genai.drmcp.core.dynamic_tools.deployment.metadata import get_mcp_tool_metadata
from datarobot_genai.drmcpbase.dynamic_tools.deployment.adapters.drum import is_drum
from datarobot_genai.drmcpbase.dynamic_tools.deployment.metadata import _get_model_attribute
from datarobot_genai.drmcpbase.dynamic_tools.deployment.metadata import (
    _is_datarobot_structured_prediction,
)


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

    @patch("datarobot_genai.drmcp.core.dynamic_tools.deployment.metadata.request_user_dr_client")
    @patch("datarobot_genai.drmcp.core.dynamic_tools.deployment.metadata._normalize_api_response")
    def test_successful_fetch(self, mock_normalize, mock_get_client):
        """Test successful metadata fetch."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_client.get.return_value = mock_response
        mock_get_client.return_value.__enter__.return_value = mock_client
        mock_get_client.return_value.__exit__.return_value = False
        mock_normalize.return_value = {"key": "value"}

        deployment = Mock()
        deployment.id = "test-deployment-id"

        result = _fetch_deployment_metadata(deployment)

        assert result == {"key": "value"}
        mock_client.get.assert_called_once_with(
            url="deployments/test-deployment-id/directAccess/info/"
        )
        mock_normalize.assert_called_once_with(mock_response)

    @patch("datarobot_genai.drmcp.core.dynamic_tools.deployment.metadata.request_user_dr_client")
    def test_api_call_failure_raises_runtime_error(self, mock_get_client):
        """Test that API call failure raises RuntimeError."""
        mock_client = Mock()
        mock_client.get.side_effect = Exception("API Error")
        mock_get_client.return_value.__enter__.return_value = mock_client
        mock_get_client.return_value.__exit__.return_value = False

        deployment = Mock()
        deployment.id = "test-deployment-id"

        with pytest.raises(
            RuntimeError, match="Could not retrieve metadata for deployment test-deployment-id"
        ):
            _fetch_deployment_metadata(deployment)


class TestGetMcpToolMetadata:
    """Test cases for the get_mcp_tool_metadata orchestration in drmcp.

    The adapter-selection logic is now inside drmcpbase.build_mcp_tool_metadata
    (tested separately in drmcpbase tests).  These tests verify that drmcp's
    orchestration layer:
      - calls _is_datarobot_structured_prediction to decide whether to fetch
      - calls _fetch_deployment_metadata / _fetch_supports_chat_api only for
        non-structured deployments
      - delegates to build_mcp_tool_metadata with the right arguments
    """

    _MODULE = "datarobot_genai.drmcp.core.dynamic_tools.deployment.metadata"

    @patch(
        "datarobot_genai.drmcp.core.dynamic_tools.deployment.metadata._is_datarobot_structured_prediction",
        return_value="binary",
    )
    @patch("datarobot_genai.drmcp.core.dynamic_tools.deployment.metadata.build_mcp_tool_metadata")
    def test_datarobot_structured_prediction_calls_build_with_none_info_payload(
        self, mock_build, mock_is_structured
    ):
        """Structured prediction short-circuits: build_mcp_tool_metadata is called
        with info_payload=None and supports_chat_api=False.
        """
        mock_adapter = Mock()
        mock_build.return_value = mock_adapter

        deployment = Mock()
        result = get_mcp_tool_metadata(deployment)

        assert result == mock_adapter
        mock_is_structured.assert_called_once_with(deployment)
        mock_build.assert_called_once_with(deployment, None, False)

    @patch(
        "datarobot_genai.drmcp.core.dynamic_tools.deployment.metadata._is_datarobot_structured_prediction",
        return_value=None,
    )
    @patch(
        "datarobot_genai.drmcp.core.dynamic_tools.deployment.metadata._fetch_deployment_metadata",
        return_value={"server": "drum"},
    )
    @patch(
        "datarobot_genai.drmcp.core.dynamic_tools.deployment.metadata._fetch_supports_chat_api",
        return_value=False,
    )
    @patch("datarobot_genai.drmcp.core.dynamic_tools.deployment.metadata.build_mcp_tool_metadata")
    def test_non_structured_deployment_fetches_and_delegates(
        self, mock_build, mock_fetch_chat, mock_fetch, mock_is_structured
    ):
        """Non-structured deployments fetch info + capabilities then delegate."""
        deployment = Mock()
        result = get_mcp_tool_metadata(deployment)

        mock_fetch.assert_called_once_with(deployment)
        mock_fetch_chat.assert_called_once_with(deployment)
        mock_build.assert_called_once_with(deployment, {"server": "drum"}, False)
        assert result == mock_build.return_value

    @patch(
        "datarobot_genai.drmcp.core.dynamic_tools.deployment.metadata._is_datarobot_structured_prediction",
        return_value="regression",
    )
    @patch(
        "datarobot_genai.drmcp.core.dynamic_tools.deployment.metadata._fetch_deployment_metadata"
    )
    @patch("datarobot_genai.drmcp.core.dynamic_tools.deployment.metadata._fetch_supports_chat_api")
    @patch("datarobot_genai.drmcp.core.dynamic_tools.deployment.metadata.build_mcp_tool_metadata")
    def test_structured_prediction_short_circuit_skips_fetches(
        self, mock_build, mock_fetch_chat, mock_fetch, mock_is_structured
    ):
        """Native DR predictive models skip both the info fetch and the
        capabilities fetch — they are guaranteed non-chat structured models.
        """
        deployment = Mock()
        get_mcp_tool_metadata(deployment)

        mock_fetch.assert_not_called()
        mock_fetch_chat.assert_not_called()


class TestFetchSupportsChatApi:
    """Test cases for _fetch_supports_chat_api."""

    @patch("datarobot_genai.drmcp.core.dynamic_tools.deployment.metadata.request_user_dr_client")
    def test_returns_true_when_capability_present_and_supported(self, mock_get_client):
        mock_client = Mock()
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "data": [
                {"name": "supports_target_drift_tracking", "supported": False},
                {"name": "supports_chat_api", "supported": True},
            ]
        }
        mock_client.get.return_value = mock_response
        mock_get_client.return_value.__enter__.return_value = mock_client
        mock_get_client.return_value.__exit__.return_value = False

        deployment = Mock()
        deployment.id = "dep-123"

        assert _fetch_supports_chat_api(deployment) is True
        mock_client.get.assert_called_once_with(url="deployments/dep-123/capabilities/")

    @patch("datarobot_genai.drmcp.core.dynamic_tools.deployment.metadata.request_user_dr_client")
    def test_returns_false_when_capability_present_and_unsupported(self, mock_get_client):
        mock_client = Mock()
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "data": [{"name": "supports_chat_api", "supported": False}]
        }
        mock_client.get.return_value = mock_response
        mock_get_client.return_value.__enter__.return_value = mock_client
        mock_get_client.return_value.__exit__.return_value = False

        deployment = Mock()
        deployment.id = "dep-123"

        assert _fetch_supports_chat_api(deployment) is False

    @patch("datarobot_genai.drmcp.core.dynamic_tools.deployment.metadata.request_user_dr_client")
    def test_returns_false_when_capability_absent(self, mock_get_client):
        mock_client = Mock()
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"data": []}
        mock_client.get.return_value = mock_response
        mock_get_client.return_value.__enter__.return_value = mock_client
        mock_get_client.return_value.__exit__.return_value = False

        deployment = Mock()
        deployment.id = "dep-123"

        assert _fetch_supports_chat_api(deployment) is False

    @patch("datarobot_genai.drmcp.core.dynamic_tools.deployment.metadata.request_user_dr_client")
    def test_returns_false_on_api_error(self, mock_get_client):
        """Older clusters without /capabilities/ should not break tool
        registration — fail closed to /predictions routing.
        """
        mock_client = Mock()
        mock_client.get.side_effect = Exception("404 capabilities not found")
        mock_get_client.return_value.__enter__.return_value = mock_client
        mock_get_client.return_value.__exit__.return_value = False

        deployment = Mock()
        deployment.id = "dep-123"

        assert _fetch_supports_chat_api(deployment) is False
