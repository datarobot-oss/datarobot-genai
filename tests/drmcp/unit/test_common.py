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

from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

from datarobot_genai.drmcp.core.common import dr
from datarobot_genai.drmcp.core.common import get_api_client
from datarobot_genai.drmcp.core.common import get_s3_bucket_info
from datarobot_genai.drmcp.core.common import get_sdk_client
from datarobot_genai.drmcp.core.common import prefix_mount_path


def test_get_sdk_client_returns_dr() -> None:
    mock_creds = MagicMock()
    mock_creds.datarobot.application_api_token = "token"
    mock_creds.datarobot.endpoint = "url"
    with (
        patch("datarobot_genai.drmcp.core.common.dr.Client") as mock_client,
        patch("datarobot_genai.drmcp.core.common.get_credentials", return_value=mock_creds),
    ):
        result = get_sdk_client()
        mock_client.assert_called_once_with(token="token", endpoint="url")
        assert result is dr


def test_get_s3_bucket_info() -> None:
    mock_creds = MagicMock()
    mock_creds.aws_predictions_s3_bucket = "bucket"
    mock_creds.aws_predictions_s3_prefix = "prefix"
    with patch("datarobot_genai.drmcp.core.common.get_credentials", return_value=mock_creds):
        result = get_s3_bucket_info()
        assert result == {"bucket": "bucket", "prefix": "prefix"}


class TestGetApiClient:
    """Test cases for get_api_client function."""

    @patch("datarobot_genai.drmcp.core.common.get_sdk_client")
    def test_get_api_client_returns_client(self, mock_get_sdk_client):
        """Test that get_api_client returns the REST client."""
        mock_dr = Mock()
        mock_client = Mock()
        mock_rest_client = Mock()
        mock_client.get_client.return_value = mock_rest_client
        mock_dr.client = mock_client
        mock_get_sdk_client.return_value = mock_dr

        result = get_api_client()

        assert result == mock_rest_client
        mock_get_sdk_client.assert_called_once()
        mock_client.get_client.assert_called_once()


class TestPrefixMountPath:
    """Test cases for prefix_mount_path function."""

    @patch("datarobot_genai.drmcp.core.common.get_config")
    def test_mount_path_root(self, mock_get_config):
        """Test with mount_path as root."""
        mock_config = Mock()
        mock_config.mount_path = "/"
        mock_get_config.return_value = mock_config

        result = prefix_mount_path("/test/endpoint")
        assert result == "/test/endpoint"

    @patch("datarobot_genai.drmcp.core.common.get_config")
    def test_mount_path_with_trailing_slash(self, mock_get_config):
        """Test with mount_path ending with slash."""
        mock_config = Mock()
        mock_config.mount_path = "/api/"
        mock_get_config.return_value = mock_config

        result = prefix_mount_path("/test/endpoint")
        assert result == "/api/test/endpoint"

    @patch("datarobot_genai.drmcp.core.common.get_config")
    def test_mount_path_without_trailing_slash(self, mock_get_config):
        """Test with mount_path not ending with slash."""
        mock_config = Mock()
        mock_config.mount_path = "/api"
        mock_get_config.return_value = mock_config

        result = prefix_mount_path("/test/endpoint")
        assert result == "/api/test/endpoint"

    @patch("datarobot_genai.drmcp.core.common.get_config")
    def test_endpoint_without_leading_slash(self, mock_get_config):
        """Test with endpoint not starting with slash."""
        mock_config = Mock()
        mock_config.mount_path = "/api"
        mock_get_config.return_value = mock_config

        result = prefix_mount_path("test/endpoint")
        assert result == "/api/test/endpoint"

    @patch("datarobot_genai.drmcp.core.common.get_config")
    def test_endpoint_with_leading_slash(self, mock_get_config):
        """Test with endpoint starting with slash."""
        mock_config = Mock()
        mock_config.mount_path = "/api"
        mock_get_config.return_value = mock_config

        result = prefix_mount_path("/test/endpoint")
        assert result == "/api/test/endpoint"

    @patch("datarobot_genai.drmcp.core.common.get_config")
    def test_empty_endpoint(self, mock_get_config):
        """Test with empty endpoint."""
        mock_config = Mock()
        mock_config.mount_path = "/api"
        mock_get_config.return_value = mock_config

        result = prefix_mount_path("")
        assert result == "/api/"

    @patch("datarobot_genai.drmcp.core.common.get_config")
    def test_root_mount_path_with_empty_endpoint(self, mock_get_config):
        """Test with root mount_path and empty endpoint."""
        mock_config = Mock()
        mock_config.mount_path = "/"
        mock_get_config.return_value = mock_config

        result = prefix_mount_path("")
        assert result == ""
