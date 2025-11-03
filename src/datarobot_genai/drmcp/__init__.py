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

"""
DataRobot MCP Server Library.

A reusable library for building Model Context Protocol (MCP) servers with DataRobot integration.
"""

# Export main server components
from datarobot_genai.drmcp.test_utils.mcp_utils_ete import ete_test_mcp_session
from datarobot_genai.drmcp.test_utils.mcp_utils_ete import get_dr_mcp_server_url
from datarobot_genai.drmcp.test_utils.mcp_utils_ete import get_headers
from datarobot_genai.drmcp.test_utils.mcp_utils_integration import integration_test_mcp_session
from datarobot_genai.drmcp.test_utils.openai_llm_mcp_client import LLMMCPClient
from datarobot_genai.drmcp.test_utils.tool_base_ete import ETETestExpectations
from datarobot_genai.drmcp.test_utils.tool_base_ete import ToolBaseE2E
from datarobot_genai.drmcp.test_utils.tool_base_ete import ToolCallTestExpectations

from .core.clients import get_sdk_client
from .core.config import MCPServerConfig
from .core.config import get_config
from .core.config_utils import extract_datarobot_credential_runtime_param_payload
from .core.config_utils import extract_datarobot_dict_runtime_param_payload
from .core.config_utils import extract_datarobot_runtime_param_payload
from .core.constants import RUNTIME_PARAM_ENV_VAR_NAME_PREFIX
from .core.credentials import MCPServerCredentials
from .core.credentials import get_credentials
from .core.dr_mcp_server import BaseServerLifecycle
from .core.dr_mcp_server import DataRobotMCPServer
from .core.dr_mcp_server import create_mcp_server
from .core.logging import MCPLogging
from .core.mcp_instance import dr_mcp_tool
from .core.mcp_instance import register_tools

__all__ = [
    # Main server
    "DataRobotMCPServer",
    "create_mcp_server",
    "BaseServerLifecycle",
    # Configuration
    "get_config",
    "MCPServerConfig",
    # Credentials
    "get_credentials",
    "MCPServerCredentials",
    # Constants
    "RUNTIME_PARAM_ENV_VAR_NAME_PREFIX",
    # User extensibility
    "get_sdk_client",
    "dr_mcp_tool",
    "register_tools",
    # Utilities
    "MCPLogging",
    "extract_datarobot_runtime_param_payload",
    "extract_datarobot_dict_runtime_param_payload",
    "extract_datarobot_credential_runtime_param_payload",
    # Test utilities
    "get_dr_mcp_server_url",
    "get_headers",
    "ete_test_mcp_session",
    "LLMMCPClient",
    "ETETestExpectations",
    "ToolBaseE2E",
    "ToolCallTestExpectations",
    "integration_test_mcp_session",
]
