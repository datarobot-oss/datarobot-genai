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

import os
from unittest.mock import patch

import pytest

from datarobot_genai.dragent.deployment_urls import _DEFAULT_DATAROBOT_ENDPOINT
from datarobot_genai.dragent.deployment_urls import build_deployment_a2a_url
from datarobot_genai.dragent.deployment_urls import build_deployment_agent_card_url
from datarobot_genai.dragent.deployment_urls import build_workload_a2a_url
from datarobot_genai.dragent.deployment_urls import build_workload_agent_card_url
from datarobot_genai.dragent.deployment_urls import get_deployment_id
from datarobot_genai.dragent.deployment_urls import get_workload_id
from datarobot_genai.dragent.deployment_urls import is_deployment_mode
from datarobot_genai.dragent.deployment_urls import is_hosted_runtime
from datarobot_genai.dragent.deployment_urls import is_workload_mode
from datarobot_genai.dragent.deployment_urls import resolve_datarobot_endpoint


class TestBuildDeploymentA2aUrl:
    @pytest.mark.parametrize(
        "endpoint,dep_id,expected",
        [
            (
                "https://app.datarobot.com/api/v2",
                "abc123",
                "https://app.datarobot.com/api/v2/deployments/abc123/directAccess/a2a/",
            ),
            (
                "https://app.datarobot.com/api/v2/",
                "abc123",
                "https://app.datarobot.com/api/v2/deployments/abc123/directAccess/a2a/",
            ),
            (
                "https://acme.internal/api/v2",
                "dep-999",
                "https://acme.internal/api/v2/deployments/dep-999/directAccess/a2a/",
            ),
        ],
    )
    def test_builds_correct_url(self, endpoint, dep_id, expected):
        assert build_deployment_a2a_url(endpoint, dep_id) == expected

    @pytest.mark.parametrize("deployment_id", ["dep1", "abc-123", "0" * 24])
    def test_deployment_id_appears_verbatim_in_path(self, deployment_id):
        url = build_deployment_a2a_url("https://app.datarobot.com/api/v2", deployment_id)
        assert f"/deployments/{deployment_id}/" in url


class TestBuildDeploymentAgentCardUrl:
    @pytest.mark.parametrize(
        "endpoint,dep_id,expected",
        [
            (
                "https://app.datarobot.com/api/v2",
                "abc123",
                "https://app.datarobot.com/api/v2/deployments/abc123/agentCard/",
            ),
            (
                "https://app.datarobot.com/api/v2/",
                "abc123",
                "https://app.datarobot.com/api/v2/deployments/abc123/agentCard/",
            ),
            (
                "https://acme.internal/api/v2",
                "dep-999",
                "https://acme.internal/api/v2/deployments/dep-999/agentCard/",
            ),
        ],
    )
    def test_builds_correct_url(self, endpoint, dep_id, expected):
        assert build_deployment_agent_card_url(endpoint, dep_id) == expected


class TestBuildWorkloadA2aUrl:
    @pytest.mark.parametrize(
        "endpoint,workload_id,expected",
        [
            (
                "https://app.datarobot.com/api/v2",
                "abc123",
                "https://app.datarobot.com/api/v2/endpoints/workloads/abc123/a2a/",
            ),
            (
                "https://app.datarobot.com/api/v2/",
                "abc123",
                "https://app.datarobot.com/api/v2/endpoints/workloads/abc123/a2a/",
            ),
            (
                "https://acme.internal/api/v2",
                "wl-999",
                "https://acme.internal/api/v2/endpoints/workloads/wl-999/a2a/",
            ),
        ],
    )
    def test_builds_correct_url(self, endpoint, workload_id, expected):
        assert build_workload_a2a_url(endpoint, workload_id) == expected

    @pytest.mark.parametrize("workload_id", ["wl1", "abc-123", "0" * 24])
    def test_workload_id_appears_verbatim_in_path(self, workload_id):
        url = build_workload_a2a_url("https://app.datarobot.com/api/v2", workload_id)
        assert f"/endpoints/workloads/{workload_id}/" in url


class TestBuildWorkloadAgentCardUrl:
    @pytest.mark.parametrize(
        "endpoint,workload_id,expected",
        [
            (
                "https://app.datarobot.com/api/v2",
                "abc123",
                "https://app.datarobot.com/api/v2/workloads/abc123/agentCard/",
            ),
            (
                "https://app.datarobot.com/api/v2/",
                "abc123",
                "https://app.datarobot.com/api/v2/workloads/abc123/agentCard/",
            ),
            (
                "https://acme.internal/api/v2",
                "wl-999",
                "https://acme.internal/api/v2/workloads/wl-999/agentCard/",
            ),
        ],
    )
    def test_builds_correct_url(self, endpoint, workload_id, expected):
        assert build_workload_agent_card_url(endpoint, workload_id) == expected


class TestResolveDataRobotEndpoint:
    def test_prefers_public_api_endpoint_over_endpoint(self):
        env = {
            "DATAROBOT_PUBLIC_API_ENDPOINT": "https://public.datarobot.com/api/v2",
            "DATAROBOT_ENDPOINT": "https://internal.k8s.local/api/v2",
        }
        with patch.dict(os.environ, env, clear=True):
            assert resolve_datarobot_endpoint() == "https://public.datarobot.com/api/v2"

    def test_falls_back_to_endpoint_when_public_absent(self):
        env = {"DATAROBOT_ENDPOINT": "https://app.datarobot.com/api/v2"}
        with patch.dict(os.environ, env, clear=True):
            assert resolve_datarobot_endpoint() == "https://app.datarobot.com/api/v2"

    def test_returns_default_when_neither_set_and_require_false(self):
        with patch.dict(os.environ, {}, clear=True):
            assert resolve_datarobot_endpoint(require=False) == _DEFAULT_DATAROBOT_ENDPOINT

    def test_raises_when_neither_set_and_require_true(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                ValueError,
                match="DATAROBOT_PUBLIC_API_ENDPOINT or DATAROBOT_ENDPOINT must be set",
            ):
                resolve_datarobot_endpoint(require=True)

    def test_empty_string_env_var_is_ignored(self):
        env = {
            "DATAROBOT_PUBLIC_API_ENDPOINT": "",
            "DATAROBOT_ENDPOINT": "https://app.datarobot.com/api/v2",
        }
        with patch.dict(os.environ, env, clear=True):
            assert resolve_datarobot_endpoint() == "https://app.datarobot.com/api/v2"


class TestUrlConsistency:
    def test_a2a_url_and_agent_card_url_share_deployment_path_prefix(self):
        endpoint = "https://app.datarobot.com/api/v2"
        dep_id = "abc123"
        a2a = build_deployment_a2a_url(endpoint, dep_id)
        card = build_deployment_agent_card_url(endpoint, dep_id)
        expected_prefix = f"{endpoint}/deployments/{dep_id}/"
        assert a2a.startswith(expected_prefix)
        assert card.startswith(expected_prefix)
        assert a2a != card

    def test_workload_a2a_and_agent_card_urls_differ_and_include_workload_id(self):
        endpoint = "https://app.datarobot.com/api/v2"
        workload_id = "abc123"
        a2a = build_workload_a2a_url(endpoint, workload_id)
        card = build_workload_agent_card_url(endpoint, workload_id)
        assert a2a == f"{endpoint}/endpoints/workloads/{workload_id}/a2a/"
        assert card == f"{endpoint}/workloads/{workload_id}/agentCard/"
        assert a2a != card


class TestGetWorkloadId:
    @pytest.mark.parametrize("value", ["", "   ", "\t"])
    def test_returns_none_when_unset_or_blank(self, value):
        with patch.dict(os.environ, {"WORKLOAD_ID": value}, clear=True):
            assert get_workload_id() is None

    def test_returns_none_when_env_absent(self):
        with patch.dict(os.environ, {}, clear=True):
            assert get_workload_id() is None

    @pytest.mark.parametrize("value,expected", [("wl-123", "wl-123"), ("  wl-9  ", "wl-9")])
    def test_returns_stripped_value(self, value, expected):
        with patch.dict(os.environ, {"WORKLOAD_ID": value}, clear=True):
            assert get_workload_id() == expected


class TestGetDeploymentId:
    @pytest.mark.parametrize("value", ["", "   ", "\t"])
    def test_returns_none_when_unset_or_blank(self, value):
        with patch.dict(os.environ, {"MLOPS_DEPLOYMENT_ID": value}, clear=True):
            assert get_deployment_id() is None

    def test_returns_none_when_env_absent(self):
        with patch.dict(os.environ, {}, clear=True):
            assert get_deployment_id() is None

    @pytest.mark.parametrize("value,expected", [("dep-123", "dep-123"), ("  dep-9  ", "dep-9")])
    def test_returns_stripped_value(self, value, expected):
        with patch.dict(os.environ, {"MLOPS_DEPLOYMENT_ID": value}, clear=True):
            assert get_deployment_id() == expected


class TestIsWorkloadMode:
    def test_true_when_workload_id_set(self):
        with patch.dict(os.environ, {"WORKLOAD_ID": "wl-1"}, clear=True):
            assert is_workload_mode() is True

    @pytest.mark.parametrize("env", [{}, {"WORKLOAD_ID": ""}, {"WORKLOAD_ID": "  "}])
    def test_false_when_absent_or_blank(self, env):
        with patch.dict(os.environ, env, clear=True):
            assert is_workload_mode() is False


class TestIsDeploymentMode:
    def test_true_when_deployment_id_set(self):
        with patch.dict(os.environ, {"MLOPS_DEPLOYMENT_ID": "dep-1"}, clear=True):
            assert is_deployment_mode() is True

    @pytest.mark.parametrize(
        "env", [{}, {"MLOPS_DEPLOYMENT_ID": ""}, {"MLOPS_DEPLOYMENT_ID": "  "}]
    )
    def test_false_when_absent_or_blank(self, env):
        with patch.dict(os.environ, env, clear=True):
            assert is_deployment_mode() is False


class TestIsHostedRuntime:
    def test_false_when_neither_set(self):
        with patch.dict(os.environ, {}, clear=True):
            assert is_hosted_runtime() is False

    def test_true_when_deployment_only(self):
        with patch.dict(os.environ, {"MLOPS_DEPLOYMENT_ID": "dep-1"}, clear=True):
            assert is_hosted_runtime() is True

    def test_true_when_workload_only(self):
        with patch.dict(os.environ, {"WORKLOAD_ID": "wl-1"}, clear=True):
            assert is_hosted_runtime() is True
