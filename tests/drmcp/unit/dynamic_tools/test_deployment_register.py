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
import inspect
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from aioresponses import aioresponses

from datarobot_genai.drmcp.core.dynamic_tools.deployment.register import (
    get_datarobot_tool_deployments,
)
from datarobot_genai.drmcp.core.dynamic_tools.deployment.register import (
    register_tool_of_datarobot_deployment,
)
from datarobot_genai.drmcpbase.dynamic_tools.external_tool import ExternalToolRegistrationConfig
from datarobot_genai.drmcputils.auth import set_request_headers


@patch("datarobot_genai.drmcp.core.dynamic_tools.deployment.register.request_user_dr_client")
@patch("datarobot_genai.drmcp.core.dynamic_tools.deployment.register.dr")
def test_get_datarobot_tool_deployments_filters_tags_correctly(
    mock_dr, mock_request_user_dr_client
):
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
    mock_rest_client = mock_request_user_dr_client.return_value.__enter__.return_value

    result = get_datarobot_tool_deployments()

    # It calls dr.utils.pagination.unpaginate with exactly these parameters
    mock_dr.utils.pagination.unpaginate.assert_called_once_with(
        initial_url="deployments/",
        initial_params={"tag_values": "tool", "tag_keys": "tool"},
        client=mock_rest_client,
    )
    mock_request_user_dr_client.assert_called_once_with(headers_auth_only=False)

    assert result == ["deployment_1", "deployment_5"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("request_headers", "authorization"),
    [
        ({"x-datarobot-external-access-token": "caller-token"}, "Bearer caller-token"),
        ({}, "Bearer registration-token"),
    ],
)
@patch("datarobot_genai.drmcpbase.dynamic_tools.external_tool.get_http_headers", return_value={})
@patch("datarobot_genai.drmcp.core.dynamic_tools.register.register_tools")
@patch("datarobot_genai.drmcp.core.dynamic_tools.deployment.register.create_deployment_tool_config")
async def test_deployment_tool_authenticates_with_the_request_token(
    mock_create_config, mock_register_tools, _mock_http_headers, request_headers, authorization
):
    # GIVEN a deployment tool whose config holds the token from registration
    mock_create_config.return_value = ExternalToolRegistrationConfig(
        name="tool",
        method="POST",
        base_url="https://pred.example.com/",
        endpoint="predict",
        headers={"Authorization": "Bearer registration-token"},
        input_schema={
            "type": "object",
            "properties": {"query_params": {"type": "object", "properties": {}}},
        },
    )
    deployment = MagicMock(prediction_environment={"platform": "datarobotServerless"})
    await register_tool_of_datarobot_deployment(deployment)
    call_tool = mock_register_tools.call_args.kwargs["fn"]
    input_model = inspect.signature(call_tool).parameters["inputs"].annotation

    # WHEN the tool is called
    set_request_headers(request_headers)
    with aioresponses() as mocked:
        mocked.post("https://pred.example.com/predict", payload={})
        await call_tool(input_model())

    # THEN the request's token is sent, and the registration token only without one
    (request,) = next(iter(mocked.requests.values()))
    assert request.kwargs["headers"]["Authorization"] == authorization
