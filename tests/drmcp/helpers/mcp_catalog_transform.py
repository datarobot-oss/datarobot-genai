# Copyright 2026 DataRobot, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared helpers for MCP catalog transform integration / acceptance / e2e tests."""

from datarobot_genai.drmcpbase.fastmcp_transforms.utils import MCP_MODE_HEADER
from datarobot_genai.drmcpbase.fastmcp_transforms.utils import MCP_TOOLS_HEADER

CODE_MODE_TOOL_NAMES = frozenset({"search", "get_schema", "execute"})

# Tools verified to succeed when invoked with an empty argument dict (see integration tests).
EMPTY_ARGS_CALLABLE_TOOL_NAMES = frozenset(
    {
        "workload_bundle_list",
        "catalog_list_datasets",
        "catalog_list_datastores",
        "datarobot_usecases_list",
        "deployment_get_list",
        "file_list",
        "get_auth_context_user_info",
        "get_user_greeting",
        "modeling_list_projects",
        "vdb_list",
        "workload_list",
    }
)


def catalog_transform_headers(
    *,
    mode: str | None = None,
    tools: str | None = None,
) -> dict[str, str]:
    headers: dict[str, str] = {}
    if mode is not None:
        headers[MCP_MODE_HEADER] = mode
    if tools is not None:
        headers[MCP_TOOLS_HEADER] = tools
    return headers


def tool_names_from_list_tools_result(tools_result: object) -> set[str]:
    return {tool.name for tool in tools_result.tools}  # type: ignore[attr-defined]


def tool_name_callable_with_empty_args(tools_result: object) -> str | None:
    """Return the lexicographically first tool known to accept an empty argument dict."""
    baseline = tool_names_from_list_tools_result(tools_result)
    candidates = sorted(baseline & EMPTY_ARGS_CALLABLE_TOOL_NAMES)
    return candidates[0] if candidates else None
