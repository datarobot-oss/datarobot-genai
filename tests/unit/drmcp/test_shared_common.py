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
from unittest.mock import patch

from datarobot_genai.drmcp.core import common


def test_get_sdk_client_returns_dr() -> None:
    mock_creds = MagicMock()
    mock_creds.datarobot.application_api_token = "token"
    mock_creds.datarobot.endpoint = "url"
    with (
        patch("datarobot_genai.drmcp.core.common.dr.Client") as mock_client,
        patch("datarobot_genai.drmcp.core.common.get_credentials", return_value=mock_creds),
    ):
        result = common.get_sdk_client()
        mock_client.assert_called_once_with(token="token", endpoint="url")
        assert result is common.dr


def test_get_s3_bucket_info() -> None:
    mock_creds = MagicMock()
    mock_creds.aws_predictions_s3_bucket = "bucket"
    mock_creds.aws_predictions_s3_prefix = "prefix"
    with patch("datarobot_genai.drmcp.core.common.get_credentials", return_value=mock_creds):
        result = common.get_s3_bucket_info()
        assert result == {"bucket": "bucket", "prefix": "prefix"}
