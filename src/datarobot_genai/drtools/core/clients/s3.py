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

import io
import logging
from typing import Any

import boto3

from datarobot_genai.drtools.core.credentials import get_credentials

logger = logging.getLogger(__name__)


def get_s3_bucket_info() -> dict[str, str]:
    """Get S3 bucket configuration."""
    credentials = get_credentials()
    return {
        "bucket": credentials.aws_predictions_s3_bucket,
        "prefix": credentials.aws_predictions_s3_prefix,
    }


def generate_presigned_url(bucket: str, key: str, expires_in: int = 2592000) -> str:
    """
    Generate a presigned S3 URL for the given bucket and key.
    Args:
        bucket (str): S3 bucket name.
        key (str): S3 object key.
        expires_in (int): Expiration in seconds (default 30 days).

    Returns
    -------
        str: Presigned S3 URL for get_object.
    """
    s3 = boto3.client("s3")
    result = s3.generate_presigned_url(
        "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=expires_in
    )
    return str(result)


def upload_dataframe_to_s3(df: Any, bucket: str, key: str) -> str:
    """Upload a pandas DataFrame to S3 as CSV and return the presigned URL."""
    # Convert DataFrame to CSV
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)

    # Upload to S3
    s3 = boto3.client("s3")
    s3.put_object(Bucket=bucket, Key=key, Body=csv_buffer.getvalue())

    # Generate presigned URL
    return generate_presigned_url(bucket, key)
