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

from datarobot_genai.drmcp.core.utils import generate_presigned_url


class TestUtilsAdditional:
    """Additional test cases for utils functions."""

    @patch("datarobot_genai.drmcp.core.utils.boto3.client")
    def test_generate_presigned_url_success(self, mock_boto3_client):
        """Test successful generation of presigned URL."""
        mock_s3_client = Mock()
        mock_s3_client.generate_presigned_url.return_value = "https://example.com/presigned-url"
        mock_boto3_client.return_value = mock_s3_client

        result = generate_presigned_url("test-bucket", "test-key")

        assert result == "https://example.com/presigned-url"
        mock_boto3_client.assert_called_once_with("s3")
        mock_s3_client.generate_presigned_url.assert_called_once_with(
            "get_object", Params={"Bucket": "test-bucket", "Key": "test-key"}, ExpiresIn=2592000
        )

    @patch("datarobot_genai.drmcp.core.utils.boto3.client")
    def test_generate_presigned_url_with_custom_expires(self, mock_boto3_client):
        """Test generation of presigned URL with custom expiration."""
        mock_s3_client = Mock()
        mock_s3_client.generate_presigned_url.return_value = "https://example.com/presigned-url"
        mock_boto3_client.return_value = mock_s3_client

        result = generate_presigned_url("test-bucket", "test-key", expires_in=7200)

        assert result == "https://example.com/presigned-url"
        mock_s3_client.generate_presigned_url.assert_called_once_with(
            "get_object", Params={"Bucket": "test-bucket", "Key": "test-key"}, ExpiresIn=7200
        )

    @patch("datarobot_genai.drmcp.core.utils.boto3.client")
    def test_generate_presigned_url_returns_string(self, mock_boto3_client):
        """Test that generate_presigned_url always returns a string."""
        mock_s3_client = Mock()
        # Simulate boto3 returning a non-string type
        mock_s3_client.generate_presigned_url.return_value = 12345
        mock_boto3_client.return_value = mock_s3_client

        result = generate_presigned_url("test-bucket", "test-key")

        assert result == "12345"
        assert isinstance(result, str)
