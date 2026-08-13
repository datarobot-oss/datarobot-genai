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
"""Resolving this server's own externally reachable URL."""

import os
from unittest.mock import patch

import pytest

from datarobot_genai.drmcp.core.runtime_identity import build_deployment_url
from datarobot_genai.drmcp.core.runtime_identity import build_workload_url
from datarobot_genai.drmcp.core.runtime_identity import get_deployment_id
from datarobot_genai.drmcp.core.runtime_identity import get_workload_id
from datarobot_genai.drmcp.core.runtime_identity import resolve_datarobot_endpoint
from datarobot_genai.drmcp.core.runtime_identity import resolve_self_url

PUBLIC = "https://public.datarobot.com/api/v2"
INTERNAL = "https://internal.k8s.local/api/v2"


class TestBuildDeploymentUrl:
    @pytest.mark.parametrize("endpoint", [PUBLIC, PUBLIC + "/"])
    def test_trailing_slash_on_the_endpoint_is_ignored(self, endpoint: str) -> None:
        assert (
            build_deployment_url(endpoint, "dep1", "mcp")
            == f"{PUBLIC}/deployments/dep1/directAccess/mcp"
        )

    def test_id_appears_verbatim(self) -> None:
        assert "/deployments/abc-123/" in build_deployment_url(PUBLIC, "abc-123", "mcp")


class TestBuildWorkloadUrl:
    @pytest.mark.parametrize("endpoint", [PUBLIC, PUBLIC + "/"])
    def test_trailing_slash_on_the_endpoint_is_ignored(self, endpoint: str) -> None:
        assert build_workload_url(endpoint, "wl1", "mcp") == f"{PUBLIC}/endpoints/workloads/wl1/mcp"

    def test_workload_is_routed_through_the_endpoints_prefix(self) -> None:
        """A workload is reached through the API host, so no lookup call is needed."""
        assert "/endpoints/workloads/wl1/" in build_workload_url(PUBLIC, "wl1", "mcp")


class TestResolveDataRobotEndpoint:
    def test_public_endpoint_wins_over_the_internal_one(self) -> None:
        """On-prem points DATAROBOT_ENDPOINT at a cluster address no client can reach."""
        env = {"DATAROBOT_PUBLIC_API_ENDPOINT": PUBLIC, "DATAROBOT_ENDPOINT": INTERNAL}
        with patch.dict(os.environ, env, clear=True):
            assert resolve_datarobot_endpoint() == PUBLIC

    def test_falls_back_to_the_standard_variable(self) -> None:
        with patch.dict(os.environ, {"DATAROBOT_ENDPOINT": INTERNAL}, clear=True):
            assert resolve_datarobot_endpoint() == INTERNAL

    def test_blank_is_treated_as_unset(self) -> None:
        env = {"DATAROBOT_PUBLIC_API_ENDPOINT": "  ", "DATAROBOT_ENDPOINT": INTERNAL}
        with patch.dict(os.environ, env, clear=True):
            assert resolve_datarobot_endpoint() == INTERNAL

    def test_returns_none_rather_than_guessing_a_host(self) -> None:
        """The value is published as the server's identity; a wrong one breaks discovery."""
        with patch.dict(os.environ, {}, clear=True):
            assert resolve_datarobot_endpoint() is None


class TestPlatformIds:
    def test_ids_are_read_from_the_injected_variables(self) -> None:
        env = {"MLOPS_DEPLOYMENT_ID": "dep1", "WORKLOAD_ID": "wl1"}
        with patch.dict(os.environ, env, clear=True):
            assert get_deployment_id() == "dep1"
            assert get_workload_id() == "wl1"

    def test_blank_is_treated_as_unset(self) -> None:
        with patch.dict(os.environ, {"MLOPS_DEPLOYMENT_ID": "   "}, clear=True):
            assert get_deployment_id() is None


class TestResolveSelfUrl:
    def test_deployment_mode(self) -> None:
        env = {"DATAROBOT_ENDPOINT": PUBLIC, "MLOPS_DEPLOYMENT_ID": "dep1"}
        with patch.dict(os.environ, env, clear=True):
            assert resolve_self_url() == f"{PUBLIC}/deployments/dep1/directAccess/mcp"

    def test_workload_mode(self) -> None:
        env = {"DATAROBOT_ENDPOINT": PUBLIC, "WORKLOAD_ID": "wl1"}
        with patch.dict(os.environ, env, clear=True):
            assert resolve_self_url() == f"{PUBLIC}/endpoints/workloads/wl1/mcp"

    def test_deployment_wins_when_both_ids_are_present(self) -> None:
        env = {"DATAROBOT_ENDPOINT": PUBLIC, "MLOPS_DEPLOYMENT_ID": "dep1", "WORKLOAD_ID": "wl1"}
        with patch.dict(os.environ, env, clear=True):
            assert "/deployments/dep1/" in str(resolve_self_url())

    def test_uses_the_public_endpoint_for_the_published_identity(self) -> None:
        env = {
            "DATAROBOT_PUBLIC_API_ENDPOINT": PUBLIC,
            "DATAROBOT_ENDPOINT": INTERNAL,
            "MLOPS_DEPLOYMENT_ID": "dep1",
        }
        with patch.dict(os.environ, env, clear=True):
            assert str(resolve_self_url()).startswith(PUBLIC)

    def test_local_development_resolves_to_nothing(self) -> None:
        """GIVEN no platform id, THEN the caller decides what to publish instead."""
        with patch.dict(os.environ, {"DATAROBOT_ENDPOINT": PUBLIC}, clear=True):
            assert resolve_self_url() is None

    def test_no_endpoint_resolves_to_nothing(self) -> None:
        with patch.dict(os.environ, {"MLOPS_DEPLOYMENT_ID": "dep1"}, clear=True):
            assert resolve_self_url() is None

    def test_path_separators_are_not_doubled(self) -> None:
        env = {"DATAROBOT_ENDPOINT": PUBLIC, "MLOPS_DEPLOYMENT_ID": "dep1"}
        with patch.dict(os.environ, env, clear=True):
            assert resolve_self_url("/mcp/") == f"{PUBLIC}/deployments/dep1/directAccess/mcp"
