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

from unittest.mock import patch

from datarobot_genai.drmcp.core.dynamic_tools.deployment.register import (
    get_datarobot_tool_deployments,
)


@patch("datarobot_genai.drmcp.core.dynamic_tools.deployment.register.get_api_client")
@patch("datarobot_genai.drmcp.core.dynamic_tools.deployment.register.dr")
def test_get_datarobot_tool_deployments_filters_tags_correctly(mock_dr, mock_get_api_client):
    """Test get_datarobot_tool_deployments accurately filters tags with AND logic."""
    mock_deployments_data = [
        # Should be included (has name="tool" and value="tool")
        {
            "id": "deployment_1",
            "tags": [{"name": "tool", "value": "tool"}, {"name": "other", "value": "other"}],
        },
        # Should be excluded (has name="tool", but value="MCP")
        {
            "id": "deployment_2",
            "tags": [{"name": "tool", "value": "MCP"}],
        },
        # Should be excluded (no tags)
        {
            "id": "deployment_3",
            "tags": [],
        },
        # Should be excluded (no 'tags' key)
        {
            "id": "deployment_4",
        },
        # Should be included (has name="tool" and value="tool" in a different position)
        {
            "id": "deployment_5",
            "tags": [{"name": "something", "value": "else"}, {"name": "tool", "value": "tool"}],
        },
        # Should be excluded (name="other", value="tool")
        {
            "id": "deployment_6",
            "tags": [{"name": "other", "value": "tool"}],
        },
    ]

    mock_dr.utils.pagination.unpaginate.return_value = mock_deployments_data

    result = get_datarobot_tool_deployments()

    # It calls dr.utils.pagination.unpaginate with exactly these parameters
    mock_dr.utils.pagination.unpaginate.assert_called_once_with(
        initial_url="deployments/",
        initial_params={"tag_values": "tool", "tag_keys": "tool"},
        client=mock_get_api_client(),
    )

    assert result == ["deployment_1", "deployment_5"]
