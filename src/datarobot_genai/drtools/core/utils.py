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
"""Shared utilities for drtools and drmcp."""

from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel

from .clients.s3 import upload_dataframe_to_s3
from .constants import MAX_INLINE_SIZE


def is_valid_url(url: str) -> bool:
    """Check if a string is a valid URL."""
    try:
        result = urlparse(url)
        return bool(result.scheme and result.netloc)
    except Exception:
        return False


class PredictionResponse(BaseModel):
    type: str
    data: str | None = None
    resource_id: str | None = None
    s3_url: str | None = None
    show_explanations: bool | None = None


def predictions_result_response(
    df: Any, bucket: str, key: str, resource_name: str, show_explanations: bool = False
) -> PredictionResponse:
    csv_str = df.to_csv(index=False)
    if len(csv_str.encode("utf-8")) < MAX_INLINE_SIZE:
        return PredictionResponse(type="inline", data=csv_str, show_explanations=show_explanations)
    else:
        s3_url = upload_dataframe_to_s3(df, bucket, key)
        return PredictionResponse(
            type="resource",
            s3_url=s3_url,
            show_explanations=show_explanations,
        )
