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
from collections.abc import Iterator
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from datarobot_genai.drmcp.core.runtime_identity import DeploymentEndpointResolver
from datarobot_genai.drmcp.core.runtime_identity import DeploymentRelatedConfig
from datarobot_genai.drmcp.core.runtime_identity import build_deployment_url
from datarobot_genai.drmcp.core.runtime_identity import build_workload_url
from datarobot_genai.drmcp.core.runtime_identity import get_deployment_id
from datarobot_genai.drmcp.core.runtime_identity import get_workload_id
from datarobot_genai.drmcp.core.runtime_identity import resolve_datarobot_endpoint

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


class TestDeploymentEndpointResolver:
    @pytest.fixture
    def mock_mcp_url_segment(self) -> str:
        return "mcp"

    @pytest.fixture
    def mock_get_datarobot_public_api_endpoint(self) -> Iterator[Mock]:
        with patch.object(
            DeploymentRelatedConfig, "get_datarobot_public_api_endpoint"
        ) as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_is_workload_deployment(self) -> Iterator[Mock]:
        with patch.object(DeploymentEndpointResolver, "is_workload_deployment") as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_is_workload_deployment_behind_non_public_api_gateway(self) -> Iterator[Mock]:
        with patch.object(
            DeploymentEndpointResolver, "is_workload_deployment_behind_non_public_api_gateway"
        ) as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_build_url_of_mcp_workload_behind_non_public_api_based_gateway(
        self,
    ) -> Iterator[Mock]:
        with patch.object(
            DeploymentEndpointResolver,
            "build_url_of_mcp_workload_deployment_behind_non_public_api_based_gateway",
        ) as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_build_url_of_mcp_workload_behind_public_api_based_gateway(
        self,
    ) -> Iterator[Mock]:
        with patch.object(
            DeploymentEndpointResolver,
            "build_url_of_mcp_workload_deployment_behind_public_api_based_gateway",
        ) as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_build_url_of_mcp_mlops_deployment_behind_public_api_based_gateway(
        self,
    ) -> Iterator[Mock]:
        with patch.object(
            DeploymentEndpointResolver,
            "build_url_of_mcp_mlops_deployment_behind_public_api_based_gateway",
        ) as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_get_gateway_url(self) -> Iterator[Mock]:
        with patch.object(
            DeploymentEndpointResolver,
            "get_gateway_url",
        ) as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_get_deployment_id(self) -> Iterator[Mock]:
        with patch.object(
            DeploymentEndpointResolver,
            "get_deployment_id",
        ) as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_get_non_public_api_based_gateway_url(self) -> Iterator[Mock]:
        with patch.object(
            DeploymentEndpointResolver,
            "get_non_public_api_based_gateway_url",
        ) as mock_func:
            yield mock_func

    @pytest.mark.parametrize(
        "mcp_path, output",
        [("/mcp/", "mcp"), ("mcp", "mcp"), ("mcp/", "mcp")],
        ids=str,
    )
    def test_init_strip_mcp_path_segment(self, mcp_path: str, output: str) -> None:
        resolver = DeploymentEndpointResolver(mcp_path)
        assert resolver.mcp_path_segment == output

    def test_get_workload_deployment_id(self) -> None:
        with patch.dict(os.environ, {"WORKLOAD_ID": "wl1"}, clear=True):
            assert DeploymentEndpointResolver.get_workload_deployment_id() == "wl1"

    def test_get_mlops_deployment_id(self) -> None:
        with patch.dict(os.environ, {"MLOPS_DEPLOYMENT_ID": "dep1"}, clear=True):
            assert DeploymentEndpointResolver.get_mlops_deployment_id() == "dep1"

    def test_is_workload_deployment_true(self) -> None:
        with patch.dict(os.environ, {"WORKLOAD_ID": "wl1"}, clear=True):
            assert DeploymentEndpointResolver.is_workload_deployment() is True

    def test_is_workload_deployment_false(self) -> None:
        with patch.dict(os.environ, {"MLOPS_DEPLOYMENT_ID": "dep1"}, clear=True):
            assert DeploymentEndpointResolver.is_workload_deployment() is False

    def test_get_gateway_url_prefers_non_public_over_public(self) -> None:
        env = {"DR_WORKLOAD_EXTERNAL_URL_HOST": "host", "DATAROBOT_PUBLIC_API_ENDPOINT": "afdafds"}
        with patch.dict(os.environ, env, clear=True):
            assert DeploymentEndpointResolver.get_gateway_url() == "host"

    def test_get_gateway_url_falls_back_to_public(self) -> None:
        with patch.dict(os.environ, {"DATAROBOT_PUBLIC_API_ENDPOINT": "endpoint"}, clear=True):
            assert DeploymentEndpointResolver.get_gateway_url() == "endpoint"

    def test_get_gateway_url_none_when_nothing_configured(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            assert DeploymentEndpointResolver.get_gateway_url() is None

    def test_get_deployment_id_prefers_workload_over_mlops(self) -> None:
        env = {"WORKLOAD_ID": "wl1", "MLOPS_DEPLOYMENT_ID": "dep1"}
        with patch.dict(os.environ, env, clear=True):
            assert DeploymentEndpointResolver.get_deployment_id() == "wl1"

    def test_get_deployment_id_falls_back_to_mlops(self) -> None:
        with patch.dict(os.environ, {"MLOPS_DEPLOYMENT_ID": "dep1"}, clear=True):
            assert DeploymentEndpointResolver.get_deployment_id() == "dep1"

    def test_get_deployment_id_none_when_neither_set(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            assert DeploymentEndpointResolver.get_deployment_id() is None

    def test_get_non_public_api_based_gateway_url(self) -> None:
        with patch.dict(os.environ, {"DR_WORKLOAD_EXTERNAL_URL_HOST": "host"}, clear=True):
            assert DeploymentEndpointResolver.get_non_public_api_based_gateway_url() == "host"

    def test_get_public_api_based_gateway_url(
        self, mock_get_datarobot_public_api_endpoint: Mock
    ) -> None:
        output = DeploymentEndpointResolver.get_public_api_based_gateway_url()

        mock_get_datarobot_public_api_endpoint.assert_called_once_with()
        assert output == mock_get_datarobot_public_api_endpoint.return_value

    @pytest.mark.parametrize(
        "gateway_url, output",
        [("dsafa", True), (None, False)],
        ids=str,
    )
    def test_is_deployed_behind_non_public_api_based_gateway(
        self,
        gateway_url: str | None,
        output: bool,
        mock_get_non_public_api_based_gateway_url: Mock,
    ) -> None:
        mock_get_non_public_api_based_gateway_url.return_value = gateway_url

        assert (
            DeploymentEndpointResolver.is_deployed_behind_non_public_api_based_gateway() is output
        )

    def test_build_url_of_mcp_mlops_deployment_behind_public_api_based_gateway(
        self, mock_mcp_url_segment: str
    ) -> None:
        resolver = DeploymentEndpointResolver(mock_mcp_url_segment)
        gateway_url = "http://localhost/foo/bar"
        deployment_id = "1234"
        output = resolver.build_url_of_mcp_mlops_deployment_behind_public_api_based_gateway(
            gateway_url, deployment_id
        )

        assert (
            output
            == f"{gateway_url}/deployments/{deployment_id}/directAccess/{mock_mcp_url_segment}"
        )

    def test_build_url_of_mcp_workload_deployment_behind_public_api_based_gateway(
        self, mock_mcp_url_segment: str
    ) -> None:
        resolver = DeploymentEndpointResolver(mock_mcp_url_segment)
        gateway_url = "http://localhost/foo/bar"
        deployment_id = "1234"
        output = resolver.build_url_of_mcp_workload_deployment_behind_public_api_based_gateway(
            gateway_url, deployment_id
        )

        assert output == f"{gateway_url}/endpoints/workloads/{deployment_id}/{mock_mcp_url_segment}"

    def test_build_url_of_mcp_workload_deployment_behind_non_public_api_based_gateway(
        self, mock_mcp_url_segment: str
    ) -> None:
        resolver = DeploymentEndpointResolver(mock_mcp_url_segment)
        gateway_url = "http://localhost/foo/bar"
        workload_deployment_segment = "workload/1234"
        output = resolver.build_url_of_mcp_workload_deployment_behind_non_public_api_based_gateway(
            gateway_url,
            workload_deployment_segment,
        )

        assert output == f"{gateway_url}/{workload_deployment_segment}/{mock_mcp_url_segment}"

    @pytest.mark.usefixtures("mock_get_deployment_id")
    def test_return_url_of_mcp_workload_deployment_behind_public_api_based_gateway(
        self,
        mock_is_workload_deployment_behind_non_public_api_gateway: Mock,
        mock_build_url_of_mcp_workload_behind_non_public_api_based_gateway: Mock,
        mock_get_gateway_url: Mock,
    ) -> None:
        expected_mcp_workload_deployment_segment = "dsafda"
        with patch.dict(
            os.environ,
            {"DR_WORKLOAD_EXTERNAL_URL_PREFIX": expected_mcp_workload_deployment_segment},
            clear=True,
        ):
            mock_is_workload_deployment_behind_non_public_api_gateway.return_value = True

            resolver = DeploymentEndpointResolver(Mock())
            output = resolver.get_deployment_url()

            mock_get_gateway_url.assert_called_once_with()
            mock_build_url_of_mcp_workload_behind_non_public_api_based_gateway.assert_called_once_with(
                mock_get_gateway_url.return_value,
                expected_mcp_workload_deployment_segment,
            )
            assert (
                output
                == mock_build_url_of_mcp_workload_behind_non_public_api_based_gateway.return_value
            )

    def test_return_url_of_mcp_workload_deployment_behind_non_public_api_based_gateway(
        self,
        mock_get_gateway_url: Mock,
        mock_is_workload_deployment: Mock,
        mock_get_deployment_id: Mock,
        mock_is_workload_deployment_behind_non_public_api_gateway: Mock,
        mock_build_url_of_mcp_workload_behind_public_api_based_gateway: Mock,
    ) -> None:
        mock_is_workload_deployment_behind_non_public_api_gateway.return_value = False

        resolver = DeploymentEndpointResolver(Mock())
        output = resolver.get_deployment_url()

        mock_get_gateway_url.assert_called_once_with()
        mock_get_deployment_id.assert_called_once_with()
        mock_build_url_of_mcp_workload_behind_public_api_based_gateway.assert_called_once_with(
            mock_get_gateway_url.return_value,
            mock_get_deployment_id.return_value,
        )
        assert output == mock_build_url_of_mcp_workload_behind_public_api_based_gateway.return_value

    def test_return_url_of_mcp_mlops_deployment_behind_public_api_based_gateway(
        self,
        mock_get_gateway_url: Mock,
        mock_is_workload_deployment: Mock,
        mock_get_deployment_id: Mock,
        mock_is_workload_deployment_behind_non_public_api_gateway: Mock,
        mock_build_url_of_mcp_mlops_deployment_behind_public_api_based_gateway: Mock,
    ) -> None:
        mock_is_workload_deployment_behind_non_public_api_gateway.return_value = False
        mock_is_workload_deployment.return_value = False

        resolver = DeploymentEndpointResolver(Mock())
        output = resolver.get_deployment_url()

        mock_get_gateway_url.assert_called_once_with()
        mock_get_deployment_id.assert_called_once_with()
        mock_build_url_of_mcp_mlops_deployment_behind_public_api_based_gateway.assert_called_once_with(
            mock_get_gateway_url.return_value,
            mock_get_deployment_id.return_value,
        )
        assert (
            output
            == mock_build_url_of_mcp_mlops_deployment_behind_public_api_based_gateway.return_value
        )
