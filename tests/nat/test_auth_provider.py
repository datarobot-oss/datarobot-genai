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
import os
from unittest.mock import patch

from nat.authentication.api_key.api_key_auth_provider import APIKeyAuthProvider
from nat.builder.workflow_builder import WorkflowBuilder
from pydantic import SecretStr

from datarobot_genai.nat.datarobot_auth_provider import DataRobotAPIKeyAuthProviderConfig
from datarobot_genai.nat.datarobot_auth_provider import DataRobotMCPAuthProvider
from datarobot_genai.nat.datarobot_auth_provider import DataRobotMCPAuthProviderConfig


async def test_datarobot_auth_provider():
    config = DataRobotAPIKeyAuthProviderConfig(raw_key="some_token")
    async with WorkflowBuilder() as builder:
        await builder.add_auth_provider("datarobot_api_key", config)
        auth_provider = await builder.get_auth_provider("datarobot_api_key")
        assert isinstance(auth_provider, APIKeyAuthProvider)


def test_datarobot_auth_provider_config_serialization():
    """Test that config serializes properly when raw_key is provided directly."""
    config = DataRobotAPIKeyAuthProviderConfig(raw_key="test_token_abc")

    # Verify raw_key is a SecretStr, not a plain string
    assert isinstance(config.raw_key, SecretStr), (
        f"raw_key should be SecretStr, got {type(config.raw_key)}"
    )

    # Serialize to dict - this is how NAT uses the config
    dumped = config.model_dump()
    assert "raw_key" in dumped
    assert dumped["raw_key"] == "test_token_abc"

    # Serialize to JSON
    json_str = config.model_dump_json()
    assert "test_token_abc" in json_str


def test_datarobot_auth_provider_config_serialization_from_env():
    """Test config serializes properly when raw_key comes from environment."""
    with patch.dict(os.environ, {"DATAROBOT_API_TOKEN": "test_token_123"}):
        config = DataRobotAPIKeyAuthProviderConfig()

    # Verify raw_key is a SecretStr, not a plain string
    assert isinstance(config.raw_key, SecretStr), (
        f"raw_key should be SecretStr when loaded from env, got {type(config.raw_key)}"
    )

    # Serialize to dict - this is how NAT uses the config
    # This would fail if raw_key is a plain string
    dumped = config.model_dump()
    assert "raw_key" in dumped
    assert dumped["raw_key"] == "test_token_123"

    # Serialize to JSON - another common NAT operation
    json_str = config.model_dump_json()
    assert "test_token_123" in json_str


async def test_datarobot_mcp_auth_provider():
    config = DataRobotMCPAuthProviderConfig()
    async with WorkflowBuilder() as builder:
        await builder.add_auth_provider("datarobot_mcp_auth", config)
        auth_provider = await builder.get_auth_provider("datarobot_mcp_auth")
        assert isinstance(auth_provider, DataRobotMCPAuthProvider)
