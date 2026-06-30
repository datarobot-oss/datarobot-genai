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

"""Integration tests for the panels MCP surface.

Boots the DRMCP server with ``ENABLE_PANELS_TOOLS=true`` (client stubs) and
verifies that the panel **tools** and the ``panels://`` **resource templates**
are registered and visible over the MCP session — i.e. the drtools→drmcp
registration wiring works end to end.
"""

import pytest

from datarobot_genai.drmcp.test_utils.mcp_utils_integration import integration_test_mcp_session
from datarobot_genai.drmcp.test_utils.mcp_utils_integration import (
    integration_test_server_params_with_env,
)

_EXPECTED_TOOLS = {
    "list_panels",
    "get_panel",
    "create_text_panel",
    "create_json_panel",
    "delete_panel",
    "create_dataset_panel_from_connector",
    "transform_panel",
    "filter_panel",
}

_EXPECTED_RESOURCE_TEMPLATES = {
    "panels://{source}",
    "panels://{source}/{panel_id}",
    "panels://{source}/{panel_id}/content",
}


def _panels_server_params():
    return integration_test_server_params_with_env({"ENABLE_PANELS_TOOLS": "true"})


@pytest.mark.asyncio
class TestMCPPanelsIntegration:
    async def test_panel_tools_registered(self) -> None:
        async with integration_test_mcp_session(server_params=_panels_server_params()) as session:
            result = await session.list_tools()
            tool_names = {t.name for t in result.tools}
            missing = _EXPECTED_TOOLS - tool_names
            assert not missing, f"panel tools not registered: {missing} (have {sorted(tool_names)})"

    async def test_panel_resource_templates_registered(self) -> None:
        async with integration_test_mcp_session(server_params=_panels_server_params()) as session:
            result = await session.list_resource_templates()
            uris = {t.uriTemplate for t in result.resourceTemplates}
            missing = _EXPECTED_RESOURCE_TEMPLATES - uris
            assert not missing, f"panel resources not registered: {missing} (have {sorted(uris)})"
