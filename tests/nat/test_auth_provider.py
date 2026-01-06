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

from nat.authentication.api_key.api_key_auth_provider import APIKeyAuthProvider
from nat.builder.workflow_builder import WorkflowBuilder

from datarobot_genai.nat.datarobot_auth_provider import DataRobotAPIKeyAuthProviderConfig
from datarobot_genai.nat.datarobot_auth_provider import DataRobotMCPAuthProvider
from datarobot_genai.nat.datarobot_auth_provider import DataRobotMCPAuthProviderConfig


async def test_datarobot_auth_provider():
    config = DataRobotAPIKeyAuthProviderConfig(raw_key="some_token")
    async with WorkflowBuilder() as builder:
        await builder.add_auth_provider("datarobot_api_key", config)
        auth_provider = await builder.get_auth_provider("datarobot_api_key")
        assert isinstance(auth_provider, APIKeyAuthProvider)


async def test_datarobot_mcp_auth_provider():
    config = DataRobotMCPAuthProviderConfig()
    async with WorkflowBuilder() as builder:
        await builder.add_auth_provider("datarobot_mcp_auth", config)
        auth_provider = await builder.get_auth_provider("datarobot_mcp_auth")
        assert isinstance(auth_provider, DataRobotMCPAuthProvider)
