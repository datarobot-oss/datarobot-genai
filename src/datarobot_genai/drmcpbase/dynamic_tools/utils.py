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

import base64

from fastmcp.tools.tool import ToolResult


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
