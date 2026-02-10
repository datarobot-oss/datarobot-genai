# Copyright 2025 DataRobot, Inc. and its affiliates.
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

from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest

from datarobot_genai.nat.datarobot_auth_provider import DataRobotMCPAuthProviderConfig
from datarobot_genai.nat.helpers import add_headers_to_datarobot_llm_deployment
from datarobot_genai.nat.helpers import add_headers_to_datarobot_mcp_auth
from datarobot_genai.nat.helpers import load_config
from datarobot_genai.nat.helpers import load_workflow


@pytest.mark.parametrize(
    "config, headers, expected",
    [
        ({}, None, {}),
        (
            {"authentication": {"some_auth_name": {"_type": "datarobot_mcp_auth"}}},
            None,
            {"authentication": {"some_auth_name": {"_type": "datarobot_mcp_auth"}}},
        ),
        (
            {"authentication": {"some_auth_name": {"_type": "datarobot_mcp_auth"}}},
            {},
            {"authentication": {"some_auth_name": {"_type": "datarobot_mcp_auth"}}},
        ),
        (
            {"authentication": {"some_auth_name": {"_type": "datarobot_mcp_auth"}}},
            {"h1": "v1"},
            {
                "authentication": {
                    "some_auth_name": {"_type": "datarobot_mcp_auth", "headers": {"h1": "v1"}}
                }
            },
        ),
        (
            {"authentication": {"some_auth_name": {"_type": "not_datarobot_mcp_auth"}}},
            {"h1": "v1"},
            {"authentication": {"some_auth_name": {"_type": "not_datarobot_mcp_auth"}}},
        ),
        (
            {"not_authentication": {"some_auth_name": {"_type": "datarobot_mcp_auth"}}},
            {"h1": "v1"},
            {"not_authentication": {"some_auth_name": {"_type": "datarobot_mcp_auth"}}},
        ),
    ],
)
def test_add_headers_to_datarobot_mcp_auth(config, headers, expected):
    add_headers_to_datarobot_mcp_auth(config, headers)
    assert config == expected


@pytest.mark.parametrize(
    "config, headers, expected",
    [
        ({}, None, {}),
        (
            {"llms": {"datarobot_llm": {"_type": "datarobot-llm-component"}}},
            None,
            {"llms": {"datarobot_llm": {"_type": "datarobot-llm-component"}}},
        ),
        (
            {"llms": {"datarobot_llm": {"_type": "datarobot-llm-component"}}},
            {"X-DataRobot-Identity-Token": "identity-123"},
            {
                "llms": {
                    "datarobot_llm": {
                        "_type": "datarobot-llm-component",
                        "headers": {"X-DataRobot-Identity-Token": "identity-123"},
                    }
                }
            },
        ),
        (
            {"llms": {"datarobot_llm": {"_type": "datarobot-llm-deployment"}}},
            {"X-DataRobot-Identity-Token": "identity-123"},
            {
                "llms": {
                    "datarobot_llm": {
                        "_type": "datarobot-llm-deployment",
                        "headers": {"X-DataRobot-Identity-Token": "identity-123"},
                    }
                }
            },
        ),
        (
            {
                "llms": {
                    "datarobot_llm_1": {"_type": "datarobot-llm-deployment"},
                    "datarobot_llm_2": {"_type": "datarobot-llm-component"},
                }
            },
            {"X-DataRobot-Identity-Token": "identity-123"},
            {
                "llms": {
                    "datarobot_llm_1": {
                        "_type": "datarobot-llm-deployment",
                        "headers": {"X-DataRobot-Identity-Token": "identity-123"},
                    },
                    "datarobot_llm_2": {
                        "_type": "datarobot-llm-component",
                        "headers": {"X-DataRobot-Identity-Token": "identity-123"},
                    },
                }
            },
        ),
        (
            {"llms": {"datarobot_llm": {"_type": "datarobot-llm-abc"}}},
            {"X-DataRobot-Identity-Token": "identity-123"},
            {
                "llms": {
                    "datarobot_llm": {
                        "_type": "datarobot-llm-abc",
                    },
                }
            },
        ),
    ],
)
def test_add_headers_to_datarobot_llm_deployment(config, headers, expected):
    add_headers_to_datarobot_llm_deployment(config, headers)
    assert config == expected


@pytest.mark.parametrize(
    "config_yaml, headers, should_have_headers",
    [
        ({"authentication": {"some_auth_name": {"_type": "datarobot_mcp_auth"}}}, None, False),
        ({"authentication": {"some_auth_name": {"_type": "datarobot_mcp_auth"}}}, {}, False),
        (
            {"authentication": {"some_auth_name": {"_type": "datarobot_mcp_auth"}}},
            {"h1": "v1"},
            True,
        ),
    ],
)
def test_load_config(config_yaml, headers, should_have_headers):
    with patch("datarobot_genai.nat.helpers.yaml_load", return_value=config_yaml):
        config = load_config("some_path", headers)
    dr_auth_config = config.authentication["some_auth_name"]
    assert isinstance(dr_auth_config, DataRobotMCPAuthProviderConfig)
    if should_have_headers and headers:
        assert dr_auth_config.headers == headers
    else:
        assert not dr_auth_config.headers


async def test_load_workflow():
    with patch("datarobot_genai.nat.helpers.WorkflowBuilder"):
        with patch("datarobot_genai.nat.helpers.SessionManager") as mock_session_manager:
            mock_session_manager.create = AsyncMock()
            with patch("datarobot_genai.nat.helpers.load_config") as mock_load_config:
                path = "some_path"
                headers = {"h1": "v1"}
                async with load_workflow(path, headers=headers) as workflow:
                    assert workflow
                    mock_load_config.assert_called_once_with("some_path", headers=headers)
