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
from fastmcp.tools import Tool
from fastmcp.tools.tool import ToolResult
from mcp.types import Prompt as MCPPrompt
from mcp.types import Resource as MCPResource
from mcp.types import Tool as MCPTool
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


def get_prompt_tags(prompt: MCPPrompt) -> set[str]:
    """
    Extract tags from a prompt.

    Args:
        prompt: MCP protocol Prompt

    Returns
    -------
        Set of tag strings, empty set if no tags found
    """
    # MCPPrompt has tags in meta._fastmcp.tags (as a list)
    if not (prompt.meta and isinstance(prompt.meta, dict)):
        return set()

    fastmcp_meta = prompt.meta.get("_fastmcp")
    if not (fastmcp_meta and isinstance(fastmcp_meta, dict)):
        return set()

    tags = fastmcp_meta.get("tags")
    return set(tags) if tags else set()


def get_resource_tags(resource: MCPResource) -> set[str]:
    """
    Extract tags from a resource.

    Args:
        resource: MCP protocol Resource

    Returns
    -------
        Set of tag strings, empty set if no tags found
    """
    # MCPResource has tags in meta._fastmcp.tags (as a list)
    if not (resource.meta and isinstance(resource.meta, dict)):
        return set()

    fastmcp_meta = resource.meta.get("_fastmcp")
    if not (fastmcp_meta and isinstance(fastmcp_meta, dict)):
        return set()

    tags = fastmcp_meta.get("tags")
    return set(tags) if tags else set()


def get_tool_tags(tool: Tool | MCPTool) -> set[str]:
    """
    Extract tags from a tool, handling both FastMCP Tool and MCP protocol Tool types.

    Args:
        tool: Either a FastMCP Tool or MCP protocol Tool

    Returns
    -------
        Set of tag strings, empty set if no tags found
    """
    if isinstance(tool, Tool):
        # FastMCP Tool has tags directly as a set
        return getattr(tool, "tags", None) or set()

    # MCPTool has tags in meta._fastmcp.tags (as a list)
    if not (tool.meta and isinstance(tool.meta, dict)):
        return set()

    fastmcp_meta = tool.meta.get("_fastmcp")
    if not (fastmcp_meta and isinstance(fastmcp_meta, dict)):
        return set()

    tags = fastmcp_meta.get("tags")
    return set(tags) if tags else set()


def filter_tools_by_tags(
    *,
    tools: list[Tool | MCPTool],
    tags: list[str] | None = None,
    match_all: bool = False,
) -> list[Tool | MCPTool]:
    """
    Filter tools by tags.

    Args:
        tools: List of tools to filter
        tags: List of tags to filter by. If None, returns all tools
        match_all: If True, tool must have all specified tags. If False, tool must have at least
            one tag.

    Returns
    -------
        List of tools that match the tag criteria
    """
    if not tags:
        return tools

    # Convert tags to set for O(1) lookup instead of O(n)
    tags_set = set(tags)

    return [
        tool
        for tool in tools
        if (tool_tags := get_tool_tags(tool))
        and (tags_set.issubset(tool_tags) if match_all else tags_set & tool_tags)
    ]
