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

import logging
from typing import Any

from fastmcp.tools.tool import Tool

from datarobot_genai.drmcp.core.config import get_config
from datarobot_genai.drmcp.core.mcp_instance import register_tools
from datarobot_genai.drmcpbase.dynamic_tools.external_tool import ExternalToolRegistrationConfig
from datarobot_genai.drmcpbase.dynamic_tools.external_tool import _external_tool_callable_factory

logger = logging.getLogger(__name__)


async def register_external_tool(config: ExternalToolRegistrationConfig, **kwargs: Any) -> Tool:
    """Create and register a generic HTTP tool in the MCP server.

    Args:
        config: ExternalToolRegistrationConfig object containing all tool parameters.
        **kwargs: Additional keyword arguments to pass to tools registration.

    Returns
    -------
        The registered Tool instance with full MCP integration.
    """
    external_tool_callable = _external_tool_callable_factory(
        config,
        allow_empty=get_config().tool_registration_allow_empty_schema,
    )

    registered_tool = await register_tools(
        fn=external_tool_callable,
        name=config.name,
        title=config.title,
        description=config.description,
        tags=config.tags,
        **kwargs,
    )

    return registered_tool
