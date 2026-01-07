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
import base64
import uuid
from typing import Any
from urllib.parse import urlparse

import boto3
from fastmcp.resources import HttpResource
from fastmcp.tools.tool import ToolResult
from pydantic import BaseModel

from .constants import MAX_INLINE_SIZE
from .mcp_instance import mcp


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
        resource = save_df_to_s3_and_register_resource(df, bucket, key, resource_name)
        return PredictionResponse(
            type="resource",
            resource_id=str(resource.uri),
            s3_url=resource.url,
            show_explanations=show_explanations,
        )


def save_df_to_s3_and_register_resource(
    df: Any, bucket: str, key: str, resource_name: str, mime_type: str = "text/csv"
) -> HttpResource:
    """
    Save a DataFrame to a temp CSV, upload to S3, register as a resource, and return the
    presigned URL.
    Args:
        df (pd.DataFrame): DataFrame to save and upload.
        bucket (str): S3 bucket name.
        key (str): S3 object key.
        resource_name (str): Name for the registered resource.
        mime_type (str): MIME type for the resource (default 'text/csv').

    Returns
    -------
        str: Presigned S3 URL for the uploaded file.
    """
    temp_csv = f"/tmp/{uuid.uuid4()}.csv"
    df.to_csv(temp_csv, index=False)
    s3 = boto3.client("s3")
    s3.upload_file(temp_csv, bucket, key)
    s3_url = generate_presigned_url(bucket, key)
    resource = HttpResource(
        uri="predictions://" + uuid.uuid4().hex,  # type: ignore[arg-type]
        url=s3_url,
        name=resource_name,
        mime_type=mime_type,
    )
    mcp.add_resource(resource)
    return resource


def format_response_as_tool_result(data: bytes, content_type: str, charset: str) -> ToolResult:
    """Format the deployment response into a ToolResult.

    Using structured_content, to return as much information about
    the response as possible, for LLMs to correctly interpret the
    response.
    """
    charset = charset or "utf-8"
    content_type = content_type.lower() if content_type else ""

    if content_type.startswith("text/") or content_type == "application/json":
        payload = {
            "type": "text",
            "mime_type": content_type,
            "data": data.decode(charset),
        }
    elif content_type.startswith("image/"):
        payload = {
            "type": "image",
            "mime_type": content_type,
            "data_base64": base64.b64encode(data).decode(charset),
        }
    else:
        payload = {
            "type": "binary",
            "mime_type": content_type,
            "data_base64": base64.b64encode(data).decode(charset),
        }

    return ToolResult(structured_content=payload)


def is_valid_url(url: str) -> bool:
    """Check if a URL is valid."""
    result = urlparse(url)
    return all([result.scheme, result.netloc])
