# Copyright 2026 DataRobot, Inc. and its affiliates.
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

from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from nat.runtime.loader import load_config

import datarobot_genai.dragent.plugins.auth_a2a_client  # noqa: F401
from datarobot_genai.dragent.agent_card_registry import reset_default_registry
from datarobot_genai.dragent.plugins.auth_a2a_client import AgentCardRegistryLookup
from datarobot_genai.dragent.plugins.auth_a2a_client import AuthenticatedA2AClientConfig
from datarobot_genai.dragent.registry_warmup import collect_registry_lookup_ids
from datarobot_genai.dragent.registry_warmup import is_registry_warm
from datarobot_genai.dragent.registry_warmup import register_registry_lookup_ids
from datarobot_genai.dragent.registry_warmup import reset_registry_warm_state
from datarobot_genai.dragent.registry_warmup import warmup_registry_from_config

_MODULE = "datarobot_genai.dragent.registry_warmup"
_REGISTRY_SETTINGS_PATCH = "datarobot_genai.dragent.agent_card_registry._resolve_settings"
_TEST_REGISTRY_CREDENTIALS = ("test-token", "https://app.datarobot.com/api/v2")


@pytest.fixture(autouse=True)
def _reset_warm_state():
    reset_registry_warm_state()
    reset_default_registry()
    with patch(_REGISTRY_SETTINGS_PATCH, return_value=_TEST_REGISTRY_CREDENTIALS):
        yield
    reset_registry_warm_state()
    reset_default_registry()


@pytest.fixture
def workflow_path() -> Path:
    return Path(__file__).parent / "plugins" / "fixtures" / "workflow_with_a2a.yaml"


@pytest.fixture
def nat_config(
    workflow_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _reset_warm_state,
):
    """Parse workflow_with_a2a.yaml; registry credentials must be resolved at parse time."""
    monkeypatch.setenv("DATAROBOT_API_TOKEN", _TEST_REGISTRY_CREDENTIALS[0])
    monkeypatch.setenv("DATAROBOT_ENDPOINT", _TEST_REGISTRY_CREDENTIALS[1])
    with patch(_REGISTRY_SETTINGS_PATCH, return_value=_TEST_REGISTRY_CREDENTIALS):
        return load_config(workflow_path)


class TestCollectRegistryLookupIds:
    def test_collects_ids_from_yaml_config(self, nat_config):
        collected = collect_registry_lookup_ids(nat_config)
        assert collected.deployment_ids == ["1234"]
        assert collected.external_ids == ["abcd"]
        assert collected.workload_ids == ["wl-5678"]

    def test_deduplicates_ids(self):
        config = MagicMock()
        shared_registry = AgentCardRegistryLookup(deployment_id="dep-1")
        config.function_groups = {
            "a": AuthenticatedA2AClientConfig(registry=shared_registry, auth_provider="x"),
            "b": AuthenticatedA2AClientConfig(registry=shared_registry, auth_provider="x"),
        }
        collected = collect_registry_lookup_ids(config)
        assert collected.deployment_ids == ["dep-1"]
        assert collected.external_ids == []
        assert collected.workload_ids == []

    def test_deduplicates_workload_ids(self):
        """GIVEN two groups sharing a workload ID WHEN collected THEN it appears once."""
        config = MagicMock()
        shared_registry = AgentCardRegistryLookup(workload_id="wl-1")
        config.function_groups = {
            "a": AuthenticatedA2AClientConfig(registry=shared_registry, auth_provider="x"),
            "b": AuthenticatedA2AClientConfig(registry=shared_registry, auth_provider="x"),
        }
        collected = collect_registry_lookup_ids(config)
        assert collected.workload_ids == ["wl-1"]
        assert collected.deployment_ids == []

    def test_skips_url_only_a2a_clients(self, nat_config):
        collected = collect_registry_lookup_ids(nat_config)
        # workflow also has a2a_agent with url only — must not appear
        assert "http://agent.example.com:8080" not in collected.deployment_ids
        assert len(collected.deployment_ids) == 1

    def test_empty_when_no_registry_groups(self):
        config = MagicMock()
        config.function_groups = {
            "mcp": MagicMock(),
            "a2a": AuthenticatedA2AClientConfig(
                url="http://agent.example.com:8080",
                auth_provider="auth",
            ),
        }
        collected = collect_registry_lookup_ids(config)
        assert collected.deployment_ids == []
        assert collected.external_ids == []
        assert collected.workload_ids == []
        assert collected.is_empty() is True


class TestRegisterRegistryLookupIds:
    def test_registers_all_id_kinds(self):
        mock_registry = MagicMock()
        collected = collect_registry_lookup_ids(
            MagicMock(
                function_groups={
                    "a": AuthenticatedA2AClientConfig(
                        registry=AgentCardRegistryLookup(deployment_id="dep-1"),
                        auth_provider="x",
                    ),
                    "b": AuthenticatedA2AClientConfig(
                        registry=AgentCardRegistryLookup(external_id="ext-1"),
                        auth_provider="x",
                    ),
                    "c": AuthenticatedA2AClientConfig(
                        registry=AgentCardRegistryLookup(workload_id="wl-1"),
                        auth_provider="x",
                    ),
                }
            )
        )

        register_registry_lookup_ids(mock_registry, collected)

        mock_registry.register.assert_any_call(deployment_id="dep-1")
        mock_registry.register.assert_any_call(external_id="ext-1")
        mock_registry.register.assert_any_call(workload_id="wl-1")
        assert mock_registry.register.call_count == 3


class TestWarmupRegistryFromConfig:
    async def test_prefetch_called_for_registry_ids(self, nat_config):
        mock_registry = AsyncMock()
        with patch(f"{_MODULE}.get_default_registry", AsyncMock(return_value=mock_registry)):
            await warmup_registry_from_config(nat_config)

        mock_registry.register.assert_any_call(deployment_id="1234")
        mock_registry.register.assert_any_call(external_id="abcd")
        mock_registry.register.assert_any_call(workload_id="wl-5678")
        mock_registry.prefetch.assert_awaited_once_with(
            deployment_ids=["1234"],
            external_ids=["abcd"],
            workload_ids=["wl-5678"],
        )
        assert is_registry_warm() is True

    async def test_no_op_when_no_registry_groups(self):
        config = MagicMock()
        config.function_groups = {
            "a2a": AuthenticatedA2AClientConfig(
                url="http://agent.example.com:8080",
                auth_provider="auth",
            ),
        }
        mock_get = AsyncMock()
        with patch(f"{_MODULE}.get_default_registry", mock_get):
            await warmup_registry_from_config(config)

        mock_get.assert_not_awaited()
        assert is_registry_warm() is True

    async def test_warm_false_on_prefetch_failure(self, nat_config):
        mock_registry = AsyncMock()
        mock_registry.prefetch.side_effect = RuntimeError("registry down")
        with patch(f"{_MODULE}.get_default_registry", AsyncMock(return_value=mock_registry)):
            await warmup_registry_from_config(nat_config)

        assert is_registry_warm() is False
