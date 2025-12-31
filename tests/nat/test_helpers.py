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

import pytest

from datarobot_genai.nat.helpers import add_headers_to_datarobot_mcp_auth


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
