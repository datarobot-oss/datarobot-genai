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
from datarobot_genai.dragent.plugins.auth_a2a_client import AgentCardRegistryLookup
from datarobot_genai.dragent.plugins.auth_a2a_client import AuthenticatedA2AClientConfig
from datarobot_genai.dragent.registry_warmup import collect_registry_lookup_ids
from datarobot_genai.dragent.registry_warmup import is_registry_warm
from datarobot_genai.dragent.registry_warmup import reset_registry_warm_state
from datarobot_genai.dragent.registry_warmup import warmup_registry_from_config

_MODULE = "datarobot_genai.dragent.registry_warmup"


@pytest.fixture(autouse=True)
def _reset_warm_state():
    reset_registry_warm_state()
    yield
    reset_registry_warm_state()


@pytest.fixture
def workflow_path() -> Path:
    return Path(__file__).parent / "plugins" / "fixtures" / "workflow_with_a2a.yaml"


@pytest.fixture
def nat_config(workflow_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DATAROBOT_API_TOKEN", "test-token")
    return load_config(workflow_path)


class TestCollectRegistryLookupIds:
    def test_collects_ids_from_yaml_config(self, nat_config):
        dep_ids, ext_ids = collect_registry_lookup_ids(nat_config)
        assert dep_ids == ["1234"]
        assert ext_ids == ["abcd"]

    def test_deduplicates_ids(self):
        config = MagicMock()
        shared_registry = AgentCardRegistryLookup(deployment_id="dep-1")
        config.function_groups = {
            "a": AuthenticatedA2AClientConfig(registry=shared_registry, auth_provider="x"),
            "b": AuthenticatedA2AClientConfig(registry=shared_registry, auth_provider="x"),
        }
        dep_ids, ext_ids = collect_registry_lookup_ids(config)
        assert dep_ids == ["dep-1"]
        assert ext_ids == []

    def test_skips_url_only_a2a_clients(self, nat_config):
        dep_ids, ext_ids = collect_registry_lookup_ids(nat_config)
        # workflow also has a2a_agent with url only — must not appear
        assert "http://agent.example.com:8080" not in dep_ids
        assert len(dep_ids) == 1

    def test_empty_when_no_registry_groups(self):
        config = MagicMock()
        config.function_groups = {
            "mcp": MagicMock(),
            "a2a": AuthenticatedA2AClientConfig(
                url="http://agent.example.com:8080",
                auth_provider="auth",
            ),
        }
        dep_ids, ext_ids = collect_registry_lookup_ids(config)
        assert dep_ids == []
        assert ext_ids == []


class TestWarmupRegistryFromConfig:
    async def test_prefetch_called_for_registry_ids(self, nat_config):
        mock_registry = AsyncMock()
        with (
            patch(f"{_MODULE}.is_prefetch_on_startup_enabled", return_value=True),
            patch(f"{_MODULE}.get_default_registry", AsyncMock(return_value=mock_registry)),
        ):
            await warmup_registry_from_config(nat_config)

        mock_registry.prefetch.assert_awaited_once_with(
            deployment_ids=["1234"],
            external_ids=["abcd"],
        )
        assert is_registry_warm() is True

    async def test_skips_when_disabled(self, nat_config):
        mock_get = AsyncMock()
        with (
            patch(f"{_MODULE}.is_prefetch_on_startup_enabled", return_value=False),
            patch(f"{_MODULE}.get_default_registry", mock_get),
        ):
            await warmup_registry_from_config(nat_config)

        mock_get.assert_not_awaited()
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
        with (
            patch(f"{_MODULE}.is_prefetch_on_startup_enabled", return_value=True),
            patch(f"{_MODULE}.get_default_registry", mock_get),
        ):
            await warmup_registry_from_config(config)

        mock_get.assert_not_awaited()
        assert is_registry_warm() is True

    async def test_warm_false_on_prefetch_failure(self, nat_config):
        mock_registry = AsyncMock()
        mock_registry.prefetch.side_effect = RuntimeError("registry down")
        with (
            patch(f"{_MODULE}.is_prefetch_on_startup_enabled", return_value=True),
            patch(f"{_MODULE}.get_default_registry", AsyncMock(return_value=mock_registry)),
        ):
            await warmup_registry_from_config(nat_config)

        assert is_registry_warm() is False
