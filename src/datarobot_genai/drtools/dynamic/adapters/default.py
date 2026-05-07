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
from typing import Any
from typing import Literal
from typing import cast

from .base import MetadataBase


class Metadata(MetadataBase):
    """Default adapter for external deployment metadata."""

    def __init__(self, metadata: dict[str, Any]) -> None:
        self.metadata = metadata

    @property
    def name(self) -> str:
        return str(self.metadata.get("name", ""))

    @property
    def description(self) -> str:
        return str(self.metadata.get("description", ""))

    @property
    def base_url(self) -> str:
        base_url = self.metadata.get("base_url")
        if not base_url or not isinstance(base_url, str):
            raise ValueError(
                "Deployment missing required 'base_url' field in /info/ metadata. "
                "This field is required for MCP Server to route tool requests correctly."
            )
        return str(base_url)

    @property
    def endpoint(self) -> str:
        endpoint = self.metadata.get("endpoint")
        if not endpoint or not isinstance(endpoint, str):
            raise ValueError(
                "Deployment missing required 'endpoint' field in /info/ metadata. "
                "This field is required for MCP Server to route tool requests correctly."
            )
        return str(endpoint)

    @property
    def input_schema(self) -> dict[str, Any]:
        input_schema = self.metadata.get("input_schema")
        if not input_schema or not isinstance(input_schema, dict):
            raise ValueError(
                "Deployment missing required 'inputSchema' field in /info/ metadata. "
                "This field is required for MCP Server to route tool requests correctly."
            )
        return dict(input_schema)

    @property
    def method(self) -> Literal["GET", "POST", "PATCH", "PUT", "DELETE"]:
        method = self.metadata.get("method", "").upper()
        if not method or not isinstance(method, str):
            raise ValueError(
                "Deployment missing required 'method' field in /info/ metadata. "
                "This field is required for MCP Server to route tool requests correctly."
            )
        if method not in ("GET", "POST", "PATCH", "PUT", "DELETE"):
            raise ValueError(f"Deployment metadata is invalid, unsupported `method`: {method}.")
        return cast(Literal["GET", "POST", "PATCH", "PUT", "DELETE"], method)

    @property
    def headers(self) -> dict[str, str]:
        headers = self.metadata.get("headers", {})
        if not isinstance(headers, dict):
            raise ValueError("Deployment metadata 'headers' field must be a dictionary.")
        return dict(headers)
