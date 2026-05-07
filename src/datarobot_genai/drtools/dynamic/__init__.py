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

"""Framework-agnostic building blocks for dynamic external tool registration.

This subpackage exposes the JSON-schema-to-Pydantic-model converter, deployment
metadata adapters and the HTTP request callable factory needed to register
external HTTP endpoints as MCP tools.

It is intentionally independent of FastMCP and DrMCP so it can be reused by
other MCP hosts (for example, ``datarobot/global-mcp``) that cannot import
``drmcp`` because of transitive ``datarobot.context`` imports.
"""

from datarobot_genai.drtools.dynamic.external_tool import REQUEST_CONNECT_TIMEOUT
from datarobot_genai.drtools.dynamic.external_tool import REQUEST_FORWARDED_HEADERS
from datarobot_genai.drtools.dynamic.external_tool import REQUEST_MAX_RETRY
from datarobot_genai.drtools.dynamic.external_tool import REQUEST_RETRY_SLEEP
from datarobot_genai.drtools.dynamic.external_tool import REQUEST_RETRYABLE_STATUS_CODES
from datarobot_genai.drtools.dynamic.external_tool import REQUEST_TOTAL_TIMEOUT
from datarobot_genai.drtools.dynamic.external_tool import ExternalToolRegistrationConfig
from datarobot_genai.drtools.dynamic.external_tool import build_external_tool_callable
from datarobot_genai.drtools.dynamic.external_tool import get_outbound_headers
from datarobot_genai.drtools.dynamic.schema import SchemaValidationError
from datarobot_genai.drtools.dynamic.schema import create_input_schema_pydantic_model
from datarobot_genai.drtools.dynamic.schema import create_schema_model

__all__ = [
    "REQUEST_CONNECT_TIMEOUT",
    "REQUEST_FORWARDED_HEADERS",
    "REQUEST_MAX_RETRY",
    "REQUEST_RETRYABLE_STATUS_CODES",
    "REQUEST_RETRY_SLEEP",
    "REQUEST_TOTAL_TIMEOUT",
    "ExternalToolRegistrationConfig",
    "SchemaValidationError",
    "build_external_tool_callable",
    "create_input_schema_pydantic_model",
    "create_schema_model",
    "get_outbound_headers",
]
