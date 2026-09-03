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
from datarobot_genai.drmcp.core.runtime_identity import GatewayType
from datarobot_genai.drmcp.core.runtime_identity import MCPDeploymentType
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


class TestDeploymentRelatedConfig:
    @pytest.mark.parametrize(
        "enum_value",
        [enum_value for enum_value in DeploymentRelatedConfig],
        ids=str,
    )
    def test_get_from_os_env_strip_surrounding_spaces(
        self, enum_value: DeploymentRelatedConfig
    ) -> None:
        expected_value = "value_without_surrounding_space"
        with patch.dict(
            os.environ,
            {enum_value.name: f" {expected_value} "},
            clear=True,
        ):
            assert enum_value.get_from_os_env() == expected_value

    def test_get_public_api_endpoint_prefer_env_var_datarobot_public_api_endpoint(self) -> None:
        expected_value = "https://foo/bar"
        with patch.dict(
            os.environ,
            {"DATAROBOT_PUBLIC_API_ENDPOINT": expected_value, "DATAROBOT_ENDPOINT": "dsafas"},
            clear=True,
        ):
            assert DeploymentRelatedConfig.get_datarobot_public_api_endpoint() == expected_value

    def test_get_public_api_endpoint_fallback_env_var_datarobot_endpoint(self) -> None:
        expected_value = "https://foo/bar"
        with patch.dict(
            os.environ,
            {"DATAROBOT_ENDPOINT": expected_value},
            clear=True,
        ):
            assert DeploymentRelatedConfig.get_datarobot_public_api_endpoint() == expected_value


class TestGatewayType:
    def test_get_url_segment_of_workload_deployment_deployed_behind_non_public_api_based_gateway(
        self,
    ) -> None:
        mock_deployment_id = "123"
        output = GatewayType.NON_PUBLIC_API_BASED_GATEWAY.get_workload_deployment_url_segment(
            mock_deployment_id
        )

        assert output == f"workloads/{mock_deployment_id}"

    def test_get_url_segment_of_workload_deployment_deployed_behind_public_api_based_gateway(
        self,
    ) -> None:
        mock_deployment_id = "123"
        output = GatewayType.PUBLIC_API_BASED_GATEWAY.get_workload_deployment_url_segment(
            mock_deployment_id,
        )

        assert output == f"endpoints/workloads/{mock_deployment_id}"

    def test_get_url_segment_of_mlops_deployment_deployed_behind_non_public_api_based_gateway(
        self,
    ) -> None:
        with pytest.raises(ValueError):
            GatewayType.NON_PUBLIC_API_BASED_GATEWAY.get_mlops_deployment_url_segment(Mock())

    def test_get_url_segment_of_mlops_deployment_deployed_behind_public_api_based_gateway(
        self,
    ) -> None:
        mock_deployment_id = "123"
        output = GatewayType.PUBLIC_API_BASED_GATEWAY.get_mlops_deployment_url_segment(
            mock_deployment_id,
        )

        assert output == f"deployments/{mock_deployment_id}/directAccess"


class TestDeploymentEndpointResolver:
    @pytest.fixture
    def mock_get_gateway_url(self) -> Iterator[Mock]:
        with patch.object(DeploymentEndpointResolver, "get_gateway_url") as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_get_gateway_type(self) -> Iterator[Mock]:
        with patch.object(DeploymentEndpointResolver, "get_gateway_type") as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_get_deployment_id(self) -> Iterator[Mock]:
        with patch.object(DeploymentEndpointResolver, "get_deployment_id") as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_get_deployment_type(self) -> Iterator[Mock]:
        with patch.object(DeploymentEndpointResolver, "get_deployment_type") as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_get_deployment_segment_url(self) -> Iterator[Mock]:
        with patch.object(DeploymentEndpointResolver, "get_deployment_segment_url") as mock_func:
            yield mock_func

    def test_init(
        self,
        mock_get_gateway_url: Mock,
        mock_get_gateway_type: Mock,
        mock_get_deployment_id: Mock,
        mock_get_deployment_type: Mock,
    ) -> None:
        resolver = DeploymentEndpointResolver()

        assert resolver.mcp_path_suffix == "mcp"
        assert resolver.gateway_url == mock_get_gateway_url.return_value
        assert resolver.gateway_type == mock_get_gateway_type.return_value
        assert resolver.deployment_id == mock_get_deployment_id.return_value
        assert resolver.deployment_type == mock_get_deployment_type.return_value

    def test_get_deployment_url_return_none_when_gateway_url_is_none(
        self,
        mock_get_gateway_url: Mock,
    ) -> None:
        mock_get_gateway_url.return_value = None

        resolver = DeploymentEndpointResolver()
        assert resolver.get_deployment_url() is None

    def test_get_deployment_url_return_none_when_deployment_id_is_none(
        self,
        mock_get_deployment_id: Mock,
    ) -> None:
        mock_get_deployment_id.return_value = None

        resolver = DeploymentEndpointResolver()
        assert resolver.get_deployment_url() is None

    @pytest.mark.usefixtures(
        "mock_get_gateway_url",
        "mock_get_deployment_id",
    )
    def test_get_deployment_url(
        self,
        mock_get_gateway_url: Mock,
        mock_get_deployment_segment_url: Mock,
    ) -> None:
        expected_gateway_url = "https://gateway_url"
        mock_get_gateway_url.return_value = expected_gateway_url
        expected_deployment_segment_url = "mcp/deployment"
        mock_get_deployment_segment_url.return_value = expected_deployment_segment_url

        resolver = DeploymentEndpointResolver()
        output = resolver.get_deployment_url()

        mock_get_deployment_segment_url.assert_called_once_with(resolver.deployment_id)
        assert output == f"{expected_gateway_url}/{expected_deployment_segment_url}/mcp"

    def test_get_well_known_url_return_none_when_gateway_url_is_none(
        self,
        mock_get_gateway_url: Mock,
    ) -> None:
        mock_get_gateway_url.return_value = None

        resolver = DeploymentEndpointResolver()
        assert resolver.get_well_known_protected_resource_metadata_url() is None

    def test_get_well_known_url_return_none_when_deployment_id_is_none(
        self,
        mock_get_deployment_id: Mock,
    ) -> None:
        mock_get_deployment_id.return_value = None

        resolver = DeploymentEndpointResolver()
        assert resolver.get_well_known_protected_resource_metadata_url() is None

    @pytest.mark.usefixtures(
        "mock_get_gateway_url",
        "mock_get_deployment_id",
    )
    def test_get_well_known_url(
        self,
        mock_get_gateway_url: Mock,
        mock_get_deployment_segment_url: Mock,
    ) -> None:
        expected_gateway_url = "https://gateway_url"
        mock_get_gateway_url.return_value = expected_gateway_url
        expected_deployment_segment_url = "mcp/deployment"
        mock_get_deployment_segment_url.return_value = expected_deployment_segment_url

        resolver = DeploymentEndpointResolver()
        output = resolver.get_well_known_protected_resource_metadata_url()

        mock_get_deployment_segment_url.assert_called_once_with(resolver.deployment_id)
        assert output == (
            f"{expected_gateway_url}/{expected_deployment_segment_url}"
            "/.well-known/oauth-protected-resource"
        )

    def test_get_segment_url_of_mlops_deployment(
        self,
        mock_get_gateway_type: Mock,
    ) -> None:
        mock_gateway_type = mock_get_gateway_type.return_value

        resolver = DeploymentEndpointResolver()
        resolver.deployment_type = MCPDeploymentType.MLOPS

        mock_deployment_id = Mock()
        output = resolver.get_deployment_segment_url(mock_deployment_id)
        mock_gateway_type.get_mlops_deployment_url_segment.assert_called_once_with(
            mock_deployment_id
        )
        assert output == mock_gateway_type.get_mlops_deployment_url_segment.return_value

    def test_get_segment_url_of_workload_deployment(
        self,
        mock_get_gateway_type: Mock,
    ) -> None:
        mock_gateway_type = mock_get_gateway_type.return_value

        resolver = DeploymentEndpointResolver()
        resolver.deployment_type = MCPDeploymentType.WORKLOAD

        mock_deployment_id = Mock()
        output = resolver.get_deployment_segment_url(mock_deployment_id)
        mock_gateway_type.get_workload_deployment_url_segment.assert_called_once_with(
            mock_deployment_id
        )
        assert output == mock_gateway_type.get_workload_deployment_url_segment.return_value

    def test_get_deployment_type_return_mlops_type(self) -> None:
        with patch.dict(os.environ, {"MLOPS": "adfsa"}, clear=True):
            assert DeploymentEndpointResolver.get_deployment_type() == MCPDeploymentType.MLOPS

    def test_get_deployment_type_return_workload_type(self) -> None:
        with patch.dict(os.environ, {"WORKLOAD_ID": "adfsa"}, clear=True):
            assert DeploymentEndpointResolver.get_deployment_type() == MCPDeploymentType.WORKLOAD

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

    def test_get_gateway_url_prefers_non_public_api_based_gateway(self) -> None:
        env = {
            "DR_WORKLOAD_EXTERNAL_URL_HOST": "https://aaa/bbb",
            "DATAROBOT_PUBLIC_API_ENDPOINT": "afdafds",
        }
        with patch.dict(os.environ, env, clear=True):
            assert DeploymentEndpointResolver.get_gateway_url() == "https://aaa/bbb"

    def test_get_gateway_url_falls_back_to_non_public_api_based_gateway(self) -> None:
        with patch.dict(
            os.environ, {"DATAROBOT_PUBLIC_API_ENDPOINT": "https://aaa/bbb"}, clear=True
        ):
            assert DeploymentEndpointResolver.get_gateway_url() == "https://aaa/bbb"

    def test_get_gateway_url_none_when_nothing_configured(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            assert DeploymentEndpointResolver.get_gateway_url() is None

    @pytest.mark.parametrize(
        "url_host_env_var",
        ["https://aaa/bbb", "https://aaa/bbb/", "aaa/bbb"],
        ids=str,
    )
    def get_get_gateway_url(self, url_host_env_var: str) -> None:
        env = {"DR_WORKLOAD_EXTERNAL_URL_HOST": url_host_env_var}
        with patch.dict(os.environ, env, clear=True):
            assert DeploymentEndpointResolver.get_gateway_url() == "https://aaa/bbb"
