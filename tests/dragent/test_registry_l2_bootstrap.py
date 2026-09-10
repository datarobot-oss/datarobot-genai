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

from __future__ import annotations

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import datarobot_genai.dragent.registry_l2_bootstrap as bootstrap
from datarobot_genai.dragent.registry_warmup import warmup_registry_from_config


def test_ensure_registry_l2_cache_provisioned_resets_singleton_on_success() -> None:
    with (
        patch.object(
            bootstrap,
            "_registry_l2_gate_status",
            return_value={
                "enclave_host_set": True,
                "enclave_prefix_set": True,
                "enclave_api_endpoint": "https://example.com/api/v2",
                "workload_id": "wl-123",
                "enclave_l2_workload": True,
                "api_token_set": True,
            },
        ),
        patch.object(
            bootstrap,
            "try_resolve_memory_space_id",
            return_value="space-abc",
        ) as resolve_mock,
        patch.object(bootstrap, "reset_default_registry") as reset_mock,
    ):
        assert bootstrap.ensure_registry_l2_cache_provisioned(phase="test") == "space-abc"

    resolve_mock.assert_called_once()
    reset_mock.assert_called_once()


def test_ensure_registry_l2_cache_provisioned_skips_without_enclave_gateway() -> None:
    with (
        patch.object(
            bootstrap,
            "_registry_l2_gate_status",
            return_value={
                "enclave_host_set": False,
                "enclave_prefix_set": False,
                "enclave_api_endpoint": None,
                "workload_id": None,
                "enclave_l2_workload": False,
                "api_token_set": False,
            },
        ),
        patch.object(
            bootstrap,
            "try_resolve_memory_space_id",
        ) as resolve_mock,
    ):
        assert bootstrap.ensure_registry_l2_cache_provisioned(phase="test") is None

    resolve_mock.assert_not_called()


async def test_warmup_retries_provision_before_prefetch() -> None:
    config = MagicMock()
    config.function_groups = {}

    with (
        patch(
            "datarobot_genai.dragent.registry_warmup.ensure_registry_l2_cache_provisioned_async",
            new_callable=AsyncMock,
            return_value="space-abc",
        ) as ensure_mock,
        patch(
            "datarobot_genai.dragent.registry_warmup.collect_registry_lookup_ids",
            return_value=MagicMock(is_empty=lambda: True),
        ),
    ):
        await warmup_registry_from_config(config)

    ensure_mock.assert_awaited_once_with(phase="lifespan-warmup")


async def test_ensure_registry_l2_cache_provisioned_async_resets_singleton_on_success() -> None:
    """GIVEN an enclave workload WHEN async ensure runs on a loop THEN the singleton is reset."""
    with (
        patch.object(
            bootstrap,
            "_registry_l2_gate_status",
            return_value={
                "enclave_host_set": True,
                "enclave_prefix_set": True,
                "enclave_api_endpoint": "https://example.com/api/v2",
                "workload_id": "wl-123",
                "enclave_l2_workload": True,
                "api_token_set": True,
            },
        ),
        patch.object(
            bootstrap,
            "try_resolve_memory_space_id_async",
            new_callable=AsyncMock,
            return_value="space-abc",
        ) as resolve_mock,
        patch.object(bootstrap, "reset_default_registry") as reset_mock,
    ):
        result = await bootstrap.ensure_registry_l2_cache_provisioned_async(phase="test")
        assert result == "space-abc"

    resolve_mock.assert_awaited_once()
    reset_mock.assert_called_once()


def test_bootstrap_provisions_once() -> None:
    bootstrap._BootstrapState.bootstrapped = False
    with patch.object(
        bootstrap,
        "ensure_registry_l2_cache_provisioned",
    ) as ensure_mock:
        bootstrap.bootstrap_registry_l2_cache()
        bootstrap.bootstrap_registry_l2_cache()

    ensure_mock.assert_called_once_with(phase="import")
    bootstrap._BootstrapState.bootstrapped = False


def test_ensure_registry_l2_cache_provisioned_logs_probe_on_failure() -> None:
    health_response = MagicMock()
    health_response.status_code = 200
    health_response.headers = {"content-type": "application/json"}
    health_response.text = '{"status":"ready"}'

    create_response = MagicMock()
    create_response.status_code = 404
    create_response.headers = {}
    create_response.text = "not found"

    with (
        patch.object(
            bootstrap,
            "_registry_l2_gate_status",
            return_value={
                "enclave_host_set": True,
                "enclave_prefix_set": True,
                "enclave_api_endpoint": "https://enclave.example.com/api/v2",
                "workload_id": "wl-123",
                "enclave_l2_workload": True,
                "api_token_set": True,
            },
        ),
        patch.object(
            bootstrap,
            "try_resolve_memory_space_id",
            return_value=None,
        ),
        patch.object(bootstrap, "_resolve_api_token", return_value="token-123"),
        patch.object(bootstrap.requests, "get", return_value=health_response) as get_mock,
        patch.object(bootstrap.requests, "post", return_value=create_response) as post_mock,
        patch.object(bootstrap, "reset_default_registry") as reset_mock,
    ):
        assert bootstrap.ensure_registry_l2_cache_provisioned(phase="test") is None

    get_mock.assert_called_once_with(
        "https://enclave.example.com/api/v2/memory/health/ready/",
        headers={"Authorization": "Token token-123"},
        timeout=30,
    )
    post_mock.assert_called_once()
    reset_mock.assert_not_called()
