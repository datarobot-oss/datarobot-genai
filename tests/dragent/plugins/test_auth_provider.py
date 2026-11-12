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

import pytest
from nat.authentication.api_key.api_key_auth_provider import APIKeyAuthProvider
from nat.builder.workflow_builder import WorkflowBuilder
from pydantic import SecretStr

from datarobot_genai.core.auth.workload_token import WorkloadTokenError
from datarobot_genai.dragent.plugins.datarobot_auth_provider import (
    DataRobotAPIKeyAuthProviderConfig,
)
from datarobot_genai.dragent.plugins.datarobot_auth_provider import DataRobotMCPAuthProvider
from datarobot_genai.dragent.plugins.datarobot_auth_provider import DataRobotMCPAuthProviderConfig
from datarobot_genai.dragent.plugins.datarobot_auth_provider import (
    DataRobotWorkloadIdentityAuthProvider,
)
from datarobot_genai.dragent.plugins.datarobot_auth_provider import (
    DataRobotWorkloadIdentityAuthProviderConfig,
)


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


class _FakeTokenSource:
    def __init__(self, token):
        self._token = token

    def token(self):
        if self._token is None:
            raise WorkloadTokenError("No valid workload token: down")
        return self._token


_PROVIDER_MODULE = "datarobot_genai.dragent.plugins.datarobot_auth_provider"


async def test_workload_identity_auth_provider_registers():
    config = DataRobotWorkloadIdentityAuthProviderConfig()
    async with WorkflowBuilder() as builder:
        await builder.add_auth_provider("datarobot_workload_identity", config)
        provider = await builder.get_auth_provider("datarobot_workload_identity")
        assert isinstance(provider, DataRobotWorkloadIdentityAuthProvider)


async def test_workload_identity_auth_provider_sends_workload_token():
    provider = DataRobotWorkloadIdentityAuthProvider(DataRobotWorkloadIdentityAuthProviderConfig())
    with patch(
        f"{_PROVIDER_MODULE}.get_workload_token_source", return_value=_FakeTokenSource("wl")
    ):
        result = await provider.authenticate()
    (cred,) = result.credentials
    assert cred.header_name == "Authorization"
    assert cred.scheme == "Bearer"
    assert cred.token.get_secret_value() == "wl"


async def test_workload_identity_auth_provider_falls_back_to_api_key():
    provider = DataRobotWorkloadIdentityAuthProvider(DataRobotWorkloadIdentityAuthProviderConfig())
    with (
        patch(f"{_PROVIDER_MODULE}.get_workload_token_source", return_value=None),
        patch(f"{_PROVIDER_MODULE}.default_api_key", return_value="api-tok"),
    ):
        result = await provider.authenticate()
    assert result.credentials[0].token.get_secret_value() == "api-tok"


async def test_workload_identity_auth_provider_without_fallback_raises():
    config = DataRobotWorkloadIdentityAuthProviderConfig(fallback_to_api_key=False)
    provider = DataRobotWorkloadIdentityAuthProvider(config)
    with (
        patch(f"{_PROVIDER_MODULE}.get_workload_token_source", return_value=_FakeTokenSource(None)),
        patch(f"{_PROVIDER_MODULE}.default_api_key", return_value="api-tok"),
        pytest.raises(WorkloadTokenError, match="No valid workload token: down"),
    ):
        await provider.authenticate()


async def test_workload_identity_auth_provider_falls_back_when_token_fails():
    provider = DataRobotWorkloadIdentityAuthProvider(DataRobotWorkloadIdentityAuthProviderConfig())
    with (
        patch(f"{_PROVIDER_MODULE}.get_workload_token_source", return_value=_FakeTokenSource(None)),
        patch(f"{_PROVIDER_MODULE}.default_api_key", return_value="api-tok"),
    ):
        result = await provider.authenticate()
    assert result.credentials[0].token.get_secret_value() == "api-tok"


async def test_workload_identity_auth_provider_unconfigured_without_fallback_raises():
    config = DataRobotWorkloadIdentityAuthProviderConfig(fallback_to_api_key=False)
    provider = DataRobotWorkloadIdentityAuthProvider(config)
    with (
        patch(f"{_PROVIDER_MODULE}.get_workload_token_source", return_value=None),
        pytest.raises(WorkloadTokenError, match="not configured"),
    ):
        await provider.authenticate()
