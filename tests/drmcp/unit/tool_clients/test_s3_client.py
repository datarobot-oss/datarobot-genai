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

from datarobot_genai.drtools.clients.s3 import get_s3_bucket_info


def test_get_s3_bucket_info() -> None:
    mock_creds = MagicMock()
    mock_creds.aws_predictions_s3_bucket = "bucket"
    mock_creds.aws_predictions_s3_prefix = "prefix"
    with patch("datarobot_genai.drtools.clients.s3.get_credentials", return_value=mock_creds):
        result = get_s3_bucket_info()
        assert result == {"bucket": "bucket", "prefix": "prefix"}
