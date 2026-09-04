# Copyright 2025 DataRobot, Inc. and its affiliates.
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

import httpx
import pytest
import respx
from datarobot.models.genai.agent.auth import set_authorization_context

from datarobot_genai.core.config import Config
from datarobot_genai.core.mcp import MCPServerRef
from datarobot_genai.core.mcp import MCPTarget
from datarobot_genai.core.mcp import MCPTargetKind
from datarobot_genai.core.mcp import build_headers
from datarobot_genai.core.mcp import build_server_config
from datarobot_genai.core.mcp import build_target
from datarobot_genai.core.mcp import clear_workload_endpoint_cache
from datarobot_genai.core.mcp import lookup_workload_endpoint

WORKLOAD_ID = "6a6b3d359e6b2c11158c2a13"
DEPLOYMENT_ID = "69331f1f30548f83b668d9dc"
API_ENDPOINT = "https://test.datarobot.com/api/v2"
LOOKUP_URL = f"{API_ENDPOINT}/workloads/{WORKLOAD_ID}/"
WORKLOAD_ENDPOINT = f"https://test.datarobot.com/workloads/{WORKLOAD_ID}"


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_workload_endpoint_cache()
    yield
    clear_workload_endpoint_cache()


@pytest.fixture(autouse=True)
def _empty_agent_auth_context():
    set_authorization_context({})


def deployment_target(name="analytics", **kwargs):
    return build_target(
        MCPServerRef(name=name, deployment_id=DEPLOYMENT_ID),
        datarobot_endpoint=API_ENDPOINT,
        datarobot_api_token=kwargs.get("token", "tok"),
    )


class TestMCPServerRef:
    """A declared server: a name, and exactly one address, validated where it is written."""

    def test_a_single_address_resolves_to_its_kind(self):
        assert MCPServerRef(deployment_id=DEPLOYMENT_ID).kind is MCPTargetKind.DEPLOYMENT
        assert MCPServerRef(workload_id=WORKLOAD_ID).kind is MCPTargetKind.WORKLOAD
        assert MCPServerRef(local_port=9001).kind is MCPTargetKind.LOCAL
        assert MCPServerRef(url="https://partner.example.com/mcp").kind is MCPTargetKind.EXTERNAL

    def test_the_name_defaults_so_a_single_server_need_not_be_named(self):
        assert MCPServerRef(deployment_id=DEPLOYMENT_ID).name == "default"

    def test_no_address_is_an_error_rather_than_a_server_that_resolves_to_nothing(self):
        with pytest.raises(ValueError, match="exactly one of"):
            MCPServerRef(name="analytics")

    def test_two_addresses_are_an_error_rather_than_a_silent_precedence(self):
        # GIVEN a server given both a deployment and a workload -- the case that used to
        # discard one of them with no error and no warning
        with pytest.raises(ValueError, match="exactly one of"):
            MCPServerRef(name="analytics", deployment_id=DEPLOYMENT_ID, workload_id=WORKLOAD_ID)

    @pytest.mark.parametrize(
        "host", ["localhost", "127.0.0.1", "[::1]"], ids=["localhost", "ipv4", "ipv6"]
    )
    def test_a_loopback_url_is_rejected_because_it_would_send_no_credentials(self, host):
        # GIVEN a local server addressed as though it were third-party. It would work on
        # the laptop where it was written, because a local server rarely enforces auth,
        # and fail only once deployed.
        with pytest.raises(ValueError, match="loopback"):
            MCPServerRef(name="local", url=f"http://{host}:9001/mcp")

    @pytest.mark.parametrize(
        "field", ["deployment_id", "workload_id"], ids=["deployment", "workload"]
    )
    def test_a_malformed_id_is_a_load_time_error_not_a_none(self, field):
        # GIVEN an ID that is not 24 hex characters. This used to log a warning and
        # resolve to None, which downstream reads as "no server configured".
        with pytest.raises(ValueError):
            MCPServerRef(name="analytics", **{field: "not-a-hex-id"})

    def test_a_name_that_could_not_prefix_a_tool_is_rejected(self):
        with pytest.raises(ValueError):
            MCPServerRef(name="Not Valid", deployment_id=DEPLOYMENT_ID)

    def test_it_is_frozen_so_it_can_be_shared_safely(self):
        # Immutability is what makes "resolve once, share across every request" safe.
        # Not hashability: the static-headers dict rules that out.
        ref = MCPServerRef(deployment_id=DEPLOYMENT_ID)
        with pytest.raises(ValueError):
            ref.name = "other"
        assert ref == MCPServerRef(deployment_id=DEPLOYMENT_ID)


class TestResolveMCPServers:
    """The fleet, as the application's config reports it."""

    def test_many_servers_of_every_kind_at_once(self):
        # GIVEN one variable carrying servers of all four kinds
        fleet = (
            '[{"name":"analytics","deployment_id":"69331f1f30548f83b668d9dc"},'
            '{"name":"catalog","deployment_id":"7a4402ab30548f83b668e1fe"},'
            '{"name":"search","workload_id":"6a6b3d359e6b2c11158c2a13"},'
            '{"name":"docs","local_port":9001},'
            '{"name":"partner","url":"https://partner.example.com/mcp"}]'
        )
        with patch.dict(os.environ, {"MCP_SERVERS": fleet}, clear=True):
            servers = Config().resolve_mcp_servers()
        # THEN every one of them is reachable, and kind is a per-server property
        assert [s.name for s in servers] == ["analytics", "catalog", "search", "docs", "partner"]
        assert [s.kind.value for s in servers] == [
            "deployment",
            "deployment",
            "workload",
            "local",
            "external",
        ]

    def test_two_servers_may_share_a_deployment_but_not_a_name(self):
        fleet = (
            '[{"name":"one","deployment_id":"69331f1f30548f83b668d9dc"},'
            '{"name":"one","local_port":9001}]'
        )
        with patch.dict(os.environ, {"MCP_SERVERS": fleet}, clear=True):
            with pytest.raises(ValueError, match="unique"):
                Config().resolve_mcp_servers()

    def test_an_unknown_name_fails_rather_than_looking_like_an_unconfigured_server(self):
        # This is the guard that catches both a typo and a config provider registered
        # too late: a fallback config contains none of the app's server names.
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(LookupError, match="No MCP server named 'analytics'"):
                Config().resolve_mcp_server("analytics")

    @pytest.mark.parametrize(
        ("env", "expected_kind"),
        [
            pytest.param({"MCP_DEPLOYMENT_ID": DEPLOYMENT_ID}, "deployment", id="deployment"),
            pytest.param({"MCP_WORKLOAD_ID": WORKLOAD_ID}, "workload", id="workload"),
            pytest.param({"MCP_SERVER_PORT": "9001"}, "local", id="local"),
            pytest.param(
                {"EXTERNAL_MCP_URL": "https://partner.example.com/mcp"}, "external", id="external"
            ),
        ],
    )
    def test_todays_singular_variables_keep_working_as_the_default_server(self, env, expected_kind):
        # GIVEN an existing .env written before MCP_SERVERS existed
        with patch.dict(os.environ, env, clear=True):
            servers = Config().resolve_mcp_servers()
        # THEN it resolves, unchanged, as the server named `default`
        assert [(s.name, s.kind.value) for s in servers] == [("default", expected_kind)]

    def test_a_bundled_servers_port_does_not_collide_with_a_remote_address(self):
        # MCP_SERVER_PORT names the port an MCP *server* process binds, and the
        # application templates set it unconditionally for their bundled server. It is
        # therefore present alongside a remote address in a stock configuration, and
        # must stay a fallback rather than a competing client address.
        with patch.dict(
            os.environ,
            {"MCP_SERVER_PORT": "9000", "MCP_WORKLOAD_ID": WORKLOAD_ID},
            clear=True,
        ):
            servers = Config().resolve_mcp_servers()
        assert [(s.name, s.kind.value) for s in servers] == [("default", "workload")]

    def test_two_singular_variables_now_raise_instead_of_discarding_one(self):
        # GIVEN both a workload and a deployment, which resolved by precedence before --
        # the deployment silently dropped, with no error and no warning
        with patch.dict(
            os.environ,
            {"MCP_WORKLOAD_ID": WORKLOAD_ID, "MCP_DEPLOYMENT_ID": DEPLOYMENT_ID},
            clear=True,
        ):
            with pytest.raises(ValueError, match="mutually exclusive"):
                Config().resolve_mcp_servers()

    def test_a_declared_fleet_supersedes_the_singular_variables_wholesale(self):
        # Not merged per name: the list you declare is the fleet you get. Merging would
        # silently add a `default` from MCP_SERVER_PORT -- which the application
        # templates set unconditionally for their bundled server -- so a declared fleet
        # would gain a server nobody asked for, whose tools do not change when
        # MCP_SERVERS does.
        with patch.dict(
            os.environ,
            {
                "MCP_DEPLOYMENT_ID": DEPLOYMENT_ID,
                "MCP_SERVER_PORT": "9000",
                "MCP_SERVERS": '[{"name":"analytics","local_port":9001}]',
            },
            clear=True,
        ):
            servers = Config().resolve_mcp_servers()
        assert [(s.name, s.kind.value) for s in servers] == [("analytics", "local")]

    def test_an_empty_variable_does_not_shadow_the_runtime_parameter(self):
        # GIVEN an empty MCP_DEPLOYMENT_ID left in a container image, and the runtime
        # parameter infra actually set. Without env_ignore_empty the empty one wins,
        # because a set-but-empty environment variable outranks a runtime parameter.
        with patch.dict(
            os.environ,
            {
                "MCP_DEPLOYMENT_ID": "",
                "MLOPS_RUNTIME_PARAM_MCP_DEPLOYMENT_ID": (
                    '{"type":"string","payload":"69331f1f30548f83b668d9dc"}'
                ),
            },
            clear=True,
        ):
            servers = Config().resolve_mcp_servers()
        assert [s.deployment_id for s in servers] == [DEPLOYMENT_ID]

    def test_no_configuration_at_all_is_a_legitimate_empty_fleet(self):
        with patch.dict(os.environ, {}, clear=True):
            assert Config().resolve_mcp_servers() == []


class TestBuildTarget:
    """Resolution: config to target, where the kind decides the URL."""

    def test_a_deployment_url_is_composed_with_no_network_call(self):
        target = build_target(
            MCPServerRef(name="analytics", deployment_id=DEPLOYMENT_ID),
            datarobot_endpoint=API_ENDPOINT,
            datarobot_api_token="tok",
        )
        assert target.url == f"{API_ENDPOINT}/deployments/{DEPLOYMENT_ID}/directAccess/mcp"
        assert target.kind is MCPTargetKind.DEPLOYMENT
        assert target.api_token == "tok"

    @respx.mock
    def test_a_workload_url_is_read_from_the_platform(self):
        respx.get(LOOKUP_URL).mock(
            return_value=httpx.Response(
                200, json={"status": "running", "endpoint": WORKLOAD_ENDPOINT}
            )
        )
        target = build_target(
            MCPServerRef(name="search", workload_id=WORKLOAD_ID),
            datarobot_endpoint=API_ENDPOINT,
            datarobot_api_token="tok",
        )
        assert target.url == f"{WORKLOAD_ENDPOINT}/mcp"
        assert target.kind is MCPTargetKind.WORKLOAD

    @respx.mock
    def test_an_unreadable_workload_raises_naming_the_server(self):
        # GIVEN a workload whose endpoint cannot be read. This used to yield no server
        # at all, which is indistinguishable from "no MCP configured".
        respx.get(LOOKUP_URL).mock(return_value=httpx.Response(403, json={"message": "nope"}))
        with pytest.raises(LookupError, match="'search'"):
            build_target(
                MCPServerRef(name="search", workload_id=WORKLOAD_ID),
                datarobot_endpoint=API_ENDPOINT,
                datarobot_api_token="tok",
            )

    def test_a_local_server_uses_its_own_host_and_port(self):
        target = build_target(
            MCPServerRef(name="docs", local_port=9003, local_host="mcp-docs"),
            datarobot_api_token="tok",
        )
        assert target.url == "http://mcp-docs:9003/mcp"
        assert target.kind is MCPTargetKind.LOCAL

    def test_a_third_party_server_never_carries_a_datarobot_token(self):
        # The invariant the loopback validator protects: `external` implies no token,
        # enforced at construction so no call site has to remember it.
        target = build_target(
            MCPServerRef(name="partner", url="https://partner.example.com/mcp/"),
            datarobot_endpoint=API_ENDPOINT,
            datarobot_api_token="tok",
        )
        assert target.url == "https://partner.example.com/mcp"
        assert target.api_token is None

    @pytest.mark.parametrize(
        ("endpoint", "token", "missing"),
        [
            pytest.param(None, "tok", "DATAROBOT_ENDPOINT", id="no-endpoint"),
            pytest.param(API_ENDPOINT, None, "DATAROBOT_API_TOKEN", id="no-token"),
        ],
    )
    def test_a_datarobot_server_without_credentials_raises(self, endpoint, token, missing):
        with pytest.raises(ValueError, match=missing):
            build_target(
                MCPServerRef(name="analytics", deployment_id=DEPLOYMENT_ID),
                datarobot_endpoint=endpoint,
                datarobot_api_token=token,
            )

    def test_the_target_keeps_the_source_id_for_telemetry(self):
        # Holding the ref rather than copying out of it is what keeps "which deployment"
        # answerable after the URL has been composed.
        target = deployment_target()
        assert target.ref.deployment_id == DEPLOYMENT_ID
        assert target.name == "analytics"


class TestBuildHeaders:
    """Consumption: target to headers, where the same kind decides the credentials.

    Table-driven over (kind x forwarded x auth-context x extra) rather than a handful of
    examples, because every credential path in the system converges here.
    """

    def _target(self, kind, token="tok"):
        refs = {
            "deployment": MCPServerRef(name="s", deployment_id=DEPLOYMENT_ID),
            "local": MCPServerRef(name="s", local_port=9001),
            "external": MCPServerRef(name="s", url="https://partner.example.com/mcp"),
        }
        if kind == "workload":
            return MCPTarget(
                ref=MCPServerRef(name="s", workload_id=WORKLOAD_ID),
                url=f"{WORKLOAD_ENDPOINT}/mcp",
                api_token=token,
            )
        return build_target(refs[kind], datarobot_endpoint=API_ENDPOINT, datarobot_api_token=token)

    @pytest.mark.parametrize(
        ("kind", "expect_api_key"),
        [
            pytest.param("workload", True, id="workload-gets-the-api-key-header"),
            pytest.param("deployment", False, id="deployment-does-not"),
            pytest.param("local", False, id="local-does-not"),
        ],
    )
    def test_only_workloads_receive_the_extra_api_key_header(self, kind, expect_api_key):
        # This single line is why credentials cannot be decided once for a whole fleet.
        headers = build_headers(self._target(kind))
        assert headers["Authorization"] == "Bearer tok"
        assert ("x-datarobot-api-key" in headers) is expect_api_key

    def test_a_third_party_server_receives_no_datarobot_credentials(self):
        headers = build_headers(self._target("external"))
        assert "Authorization" not in headers
        assert "x-datarobot-api-key" not in headers

    def test_a_third_party_servers_static_headers_are_sent(self):
        target = build_target(
            MCPServerRef(name="s", url="https://partner.example.com/mcp", headers={"x-key": "abc"})
        )
        assert build_headers(target) == {"x-key": "abc"}

    def test_a_bearer_prefixed_token_is_not_prefixed_twice(self):
        headers = build_headers(self._target("deployment", token="Bearer tok"))
        assert headers["Authorization"] == "Bearer tok"

    def test_the_workload_api_key_header_carries_the_bare_token(self):
        headers = build_headers(self._target("workload", token="Bearer  tok "))
        assert headers["Authorization"] == "Bearer  tok "
        assert headers["x-datarobot-api-key"] == "tok"

    def test_forwarded_headers_are_sent(self):
        headers = build_headers(
            self._target("deployment"), forwarded={"x-datarobot-entity-id": "e1"}
        )
        assert headers["x-datarobot-entity-id"] == "e1"

    def test_a_forwarded_api_key_outranks_the_service_one(self):
        # The caller's own scoped token wins. DO NOT let a simplification of the bearer
        # step overwrite it.
        headers = build_headers(
            self._target("workload"), forwarded={"x-datarobot-api-key": "callers-own"}
        )
        assert headers["x-datarobot-api-key"] == "callers-own"

    def test_a_forwarded_api_key_is_matched_case_insensitively(self):
        headers = build_headers(
            self._target("workload"), forwarded={"X-DataRobot-API-Key": "callers-own"}
        )
        assert headers["X-DataRobot-API-Key"] == "callers-own"
        assert "x-datarobot-api-key" not in headers

    def test_explicit_headers_are_merged_last_and_win(self):
        headers = build_headers(
            self._target("deployment"),
            forwarded={"x-datarobot-entity-id": "forwarded"},
            extra={"x-datarobot-entity-id": "explicit", "Authorization": "Bearer override"},
        )
        assert headers["x-datarobot-entity-id"] == "explicit"
        assert headers["Authorization"] == "Bearer override"

    def test_the_authorization_context_is_encoded_into_its_own_header(self):
        set_authorization_context({})
        headers = build_headers(self._target("deployment"), auth_context={"user_id": "u1"})
        assert "X-DataRobot-Authorization-Context" in headers

    def test_a_missing_authorization_context_is_not_fatal(self):
        headers = build_headers(self._target("deployment"), auth_context=None)
        assert headers["Authorization"] == "Bearer tok"

    def test_each_call_builds_a_fresh_dict_so_requests_cannot_share_headers(self):
        # The memoised `server_config` property this replaces was copied between
        # requests, which is how one request's headers reached another's connection.
        target = self._target("deployment")
        first = build_headers(target, forwarded={"x-datarobot-entity-id": "one"})
        second = build_headers(target, forwarded={"x-datarobot-entity-id": "two"})
        assert first["x-datarobot-entity-id"] == "one"
        assert second["x-datarobot-entity-id"] == "two"

    def test_build_server_config_renders_the_connection_dict(self):
        config = build_server_config(self._target("deployment"))
        assert config["url"] == f"{API_ENDPOINT}/deployments/{DEPLOYMENT_ID}/directAccess/mcp"
        assert config["transport"] == "streamable-http"
        assert config["headers"]["Authorization"] == "Bearer tok"

    def test_a_third_party_server_may_use_sse(self):
        target = build_target(
            MCPServerRef(name="s", url="https://partner.example.com/mcp", transport="sse")
        )
        assert build_server_config(target)["transport"] == "sse"


class TestAMixedFleetGetsPerServerCredentials:
    """The defect a single-server test cannot catch.

    NAT hands ONE auth provider instance to every block that names it. If the target
    lived on the provider, the last block to build would win and every server would
    receive that block's credentials -- which is wrong for at least one server under
    every possible global resolution, because `x-datarobot-api-key` depends on the kind.
    """

    @respx.mock
    def test_three_servers_of_three_kinds_each_get_their_own_credentials(self):
        respx.get(LOOKUP_URL).mock(
            return_value=httpx.Response(
                200, json={"status": "running", "endpoint": WORKLOAD_ENDPOINT}
            )
        )
        fleet = (
            f'[{{"name":"analytics","deployment_id":"{DEPLOYMENT_ID}"}},'
            f'{{"name":"search","workload_id":"{WORKLOAD_ID}"}},'
            f'{{"name":"partner","url":"https://partner.example.com/mcp"}}]'
        )
        with patch.dict(
            os.environ,
            {
                "MCP_SERVERS": fleet,
                "DATAROBOT_ENDPOINT": API_ENDPOINT,
                "DATAROBOT_API_TOKEN": "tok",
            },
            clear=True,
        ):
            config = Config()
            headers = {
                ref.name: build_headers(
                    build_target(
                        ref,
                        datarobot_endpoint=config.resolve_datarobot_endpoint(),
                        datarobot_api_token=config.resolve_datarobot_api_token(),
                    )
                )
                for ref in config.resolve_mcp_servers()
            }

        # The workload needs the extra key header; the deployment must not get it; and
        # the third-party server must receive no DataRobot identity at all.
        assert "x-datarobot-api-key" in headers["search"]
        assert "x-datarobot-api-key" not in headers["analytics"]
        assert headers["analytics"]["Authorization"] == "Bearer tok"
        assert headers["partner"] == {}


class TestLookupWorkloadEndpoint:
    @respx.mock
    def test_returns_the_endpoint_the_platform_reports(self):
        # GIVEN a cluster that serves workloads, so the
        # workload's host differs from the API host
        route = respx.get(LOOKUP_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "status": "running",
                    "endpoint": "https://test.datarobot.com/workloads/{WORKLOAD_ID}/",
                },
            )
        )
        # WHEN the endpoint is looked up
        resolved = lookup_workload_endpoint(WORKLOAD_ID, endpoint=API_ENDPOINT, token="tok")
        # THEN the platform's answer is used verbatim, host and prefix included
        assert resolved == "https://test.datarobot.com/workloads/{WORKLOAD_ID}/"
        assert route.called

    @respx.mock
    def test_endpoint_is_normalized_before_the_lookup(self):
        # GIVEN DATAROBOT_ENDPOINT spelled without /api/v2
        route = respx.get(LOOKUP_URL).mock(
            return_value=httpx.Response(
                200, json={"endpoint": "https://test.datarobot.com/workloads/{WORKLOAD_ID}/"}
            )
        )
        # WHEN the endpoint is looked up
        lookup_workload_endpoint(
            WORKLOAD_ID,
            endpoint="https://test.datarobot.com/",
            token="tok",
        )
        # THEN the request still goes to /api/v2/workloads/<id>/
        assert route.called

    @respx.mock
    def test_bearer_token_is_sent_once(self):
        # GIVEN a token that already carries the Bearer prefix
        route = respx.get(LOOKUP_URL).mock(
            return_value=httpx.Response(
                200, json={"endpoint": "https://test.datarobot.com/workloads/{WORKLOAD_ID}/"}
            )
        )
        # WHEN the endpoint is looked up
        lookup_workload_endpoint(WORKLOAD_ID, endpoint=API_ENDPOINT, token="Bearer tok")
        # THEN the prefix is not doubled
        assert route.calls.last.request.headers["Authorization"] == "Bearer tok"

    @respx.mock
    def test_running_workloads_answer_is_cached(self):
        # GIVEN a running workload whose endpoint has been resolved once
        route = respx.get(LOOKUP_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "status": "running",
                    "endpoint": "https://test.datarobot.com/workloads/{WORKLOAD_ID}/",
                },
            )
        )
        lookup_workload_endpoint(WORKLOAD_ID, endpoint=API_ENDPOINT, token="tok")
        # WHEN it is looked up again
        second_lookup = lookup_workload_endpoint(WORKLOAD_ID, endpoint=API_ENDPOINT, token="tok")
        # THEN the platform is asked only once — a running workload's route is settled
        assert second_lookup == "https://test.datarobot.com/workloads/{WORKLOAD_ID}/"
        assert route.call_count == 1

    @pytest.mark.parametrize("status", ["submitted", "provisioning", "launching", "suspended"])
    @respx.mock
    def test_a_workload_that_is_not_running_is_not_cached(self, status):
        """GIVEN a workload that has not been scheduled yet.

        On a cluster that advertises the Covalent-reported inference endpoint, the
        API answers with the prediction-gateway URL until the workload is scheduled
        — the wrong route there. Remembering it would pin the agent to it for the
        life of the process, so the answer is used but not cached.
        """
        gateway_url = f"https://app.datarobot.com/api/v2/endpoints/workloads/{WORKLOAD_ID}/"
        route = respx.get(LOOKUP_URL).mock(
            side_effect=[
                httpx.Response(200, json={"status": status, "endpoint": gateway_url}),
                httpx.Response(
                    200,
                    json={
                        "status": "running",
                        "endpoint": "https://test.datarobot.com/workloads/{WORKLOAD_ID}/",
                    },
                ),
            ]
        )
        # WHEN it is looked up while starting, and again once it is running
        assert lookup_workload_endpoint(WORKLOAD_ID, endpoint=API_ENDPOINT, token="tok") == (
            gateway_url
        )
        assert lookup_workload_endpoint(WORKLOAD_ID, endpoint=API_ENDPOINT, token="tok") == (
            "https://test.datarobot.com/workloads/{WORKLOAD_ID}/"
        )
        # THEN the stale answer was re-resolved rather than kept
        assert route.call_count == 2

    @respx.mock
    def test_a_different_api_endpoint_is_looked_up_separately(self):
        # GIVEN the same workload ID resolved against two clusters
        respx.get(LOOKUP_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "status": "running",
                    "endpoint": f"https://test.datarobot.com/workloads/{WORKLOAD_ID}/",
                },
            )
        )
        other = respx.get(f"https://other.datarobot.com/api/v2/workloads/{WORKLOAD_ID}/").mock(
            return_value=httpx.Response(
                200,
                json={
                    "status": "running",
                    "endpoint": (
                        f"https://other.datarobot.com/api/v2/endpoints/workloads/{WORKLOAD_ID}/"
                    ),
                },
            )
        )
        lookup_workload_endpoint(WORKLOAD_ID, endpoint=API_ENDPOINT, token="tok")
        # WHEN the second cluster is asked
        resolved = lookup_workload_endpoint(
            WORKLOAD_ID, endpoint="https://other.datarobot.com/api/v2", token="tok"
        )
        # THEN the cache does not leak one cluster's answer into the other
        assert other.called
        assert resolved == f"https://other.datarobot.com/api/v2/endpoints/workloads/{WORKLOAD_ID}/"

    @pytest.mark.parametrize(
        "response",
        [
            pytest.param(httpx.Response(403, json={"message": "no permission"}), id="forbidden"),
            pytest.param(httpx.Response(404, json={"message": "not found"}), id="not-found"),
            pytest.param(httpx.Response(500, text="boom"), id="server-error"),
            pytest.param(httpx.Response(200, text="not json"), id="non-json-body"),
        ],
    )
    @respx.mock
    def test_unreadable_workload_yields_no_answer(self, response, caplog):
        # GIVEN a lookup the platform will not answer
        respx.get(LOOKUP_URL).mock(return_value=response)
        # WHEN the endpoint is looked up
        with caplog.at_level("WARNING"):
            resolved = lookup_workload_endpoint(WORKLOAD_ID, endpoint=API_ENDPOINT, token="tok")
        # THEN the caller is told nothing was resolved, and what to do about it
        assert resolved is None
        assert "may read the workload" in caplog.text

    @respx.mock
    def test_transport_error_yields_no_answer(self, caplog):
        # GIVEN an unreachable API host
        respx.get(LOOKUP_URL).mock(side_effect=httpx.ConnectError("unreachable"))
        # WHEN the endpoint is looked up
        with caplog.at_level("WARNING"):
            assert lookup_workload_endpoint(WORKLOAD_ID, endpoint=API_ENDPOINT, token="tok") is None
        assert "Could not read the endpoint of workload" in caplog.text

    @pytest.mark.parametrize(
        "payload",
        [
            pytest.param({"status": "stopped"}, id="missing"),
            pytest.param({"status": "stopped", "endpoint": None}, id="null"),
            pytest.param({"status": "stopped", "endpoint": "   "}, id="blank"),
        ],
    )
    @respx.mock
    def test_workload_without_an_endpoint_yields_no_answer(self, payload, caplog):
        # GIVEN a workload that is not serving yet, so it has no endpoint
        respx.get(LOOKUP_URL).mock(return_value=httpx.Response(200, json=payload))
        # WHEN the endpoint is looked up
        with caplog.at_level("WARNING"):
            resolved = lookup_workload_endpoint(WORKLOAD_ID, endpoint=API_ENDPOINT, token="tok")
        # THEN nothing is invented, and the log says it may not be running
        assert resolved is None
        assert "may not be running yet" in caplog.text

    @respx.mock
    def test_a_failed_lookup_is_not_cached(self):
        # GIVEN a lookup that fails once and then succeeds
        route = respx.get(LOOKUP_URL).mock(
            side_effect=[
                httpx.Response(503, text="unavailable"),
                httpx.Response(
                    200, json={"endpoint": "https://test.datarobot.com/workloads/{WORKLOAD_ID}/"}
                ),
            ]
        )
        assert lookup_workload_endpoint(WORKLOAD_ID, endpoint=API_ENDPOINT, token="tok") is None
        # WHEN it is retried
        resolved = lookup_workload_endpoint(WORKLOAD_ID, endpoint=API_ENDPOINT, token="tok")
        # THEN the transient failure was not remembered
        assert resolved == "https://test.datarobot.com/workloads/{WORKLOAD_ID}/"
        assert route.call_count == 2


class TestCredentialsAreDeclaredNotInferred:
    """The #29 decoupling: the address decides the URL, `auth_provider` the credentials.

    The point of every case here is that the *derived defaults* reproduce the previous
    behaviour exactly, so a configuration that never sets `auth_provider` cannot notice
    this change -- while a configuration that does set it can reach cases that were
    previously unexpressible.
    """

    ENDPOINT = "https://app.datarobot.com/api/v2"

    def test_a_url_server_can_now_carry_datarobot_credentials(self):
        """Previously impossible: `if kind == "external": headers = {}` was unreachable.

        This is what makes the platform's global MCP reachable at all -- pasting its URL
        into EXTERNAL_MCP_URL got a DataRobot-hosted server treated as a third party.
        """
        ref = MCPServerRef(
            name="global",
            url=f"{self.ENDPOINT}/genai/globalmcp/mcp",
            auth_provider="datarobot_mcp_auth",
        )
        target = build_target(ref, datarobot_endpoint=self.ENDPOINT, datarobot_api_token="tok")
        assert build_headers(target)["Authorization"] == "Bearer tok"

    def test_credentials_to_a_foreign_host_fail_the_build(self):
        """#30. Without this, declarable auth is a service-token leak waiting to be typed."""
        ref = MCPServerRef(
            name="partner",
            url="https://partner.example.com/mcp",
            auth_provider="datarobot_mcp_auth",
        )
        with pytest.raises(ValueError, match="would send DataRobot credentials"):
            build_target(ref, datarobot_endpoint=self.ENDPOINT, datarobot_api_token="tok")

    def test_trust_host_is_the_deliberate_admission(self):
        ref = MCPServerRef(
            name="partner",
            url="https://partner.example.com/mcp",
            auth_provider="datarobot_mcp_auth",
            trust_host=True,
        )
        target = build_target(ref, datarobot_endpoint=self.ENDPOINT, datarobot_api_token="tok")
        assert build_headers(target)["Authorization"] == "Bearer tok"

    def test_a_datarobot_hosted_server_can_be_reached_anonymously(self):
        """The other direction, also previously unexpressible."""
        ref = MCPServerRef(name="docs", local_port=9001, auth_provider="none")
        target = build_target(ref, datarobot_endpoint=self.ENDPOINT, datarobot_api_token="tok")
        assert "Authorization" not in build_headers(target)

    def test_static_headers_survive_on_an_authenticated_server(self):
        """They used to be reachable only on the `external` branch."""
        ref = MCPServerRef(name="docs", local_port=9001, headers={"x-team": "search"})
        target = build_target(ref, datarobot_endpoint=self.ENDPOINT, datarobot_api_token="tok")
        headers = build_headers(target)
        assert headers["x-team"] == "search"
        assert headers["Authorization"] == "Bearer tok"

    def test_api_key_header_follows_the_route_not_the_identity(self):
        """A url server behind the workload gateway can ask for the header."""
        ref = MCPServerRef(
            name="wl",
            url="https://app.datarobot.com/wl/mcp",
            auth_provider="datarobot_mcp_auth",
            api_key_header=True,
        )
        target = build_target(ref, datarobot_endpoint=self.ENDPOINT, datarobot_api_token="tok")
        assert build_headers(target)["x-datarobot-api-key"] == "tok"

    def test_a_forwarded_key_still_outranks_the_service_one(self):
        """The guard the docstring says not to simplify, re-asserted after the rewrite."""
        ref = MCPServerRef(name="wl", workload_id="a" * 24)
        with patch(
            "datarobot_genai.core.mcp.target.lookup_workload_endpoint",
            return_value="https://wl.example.com",
        ):
            target = build_target(ref, datarobot_endpoint=self.ENDPOINT, datarobot_api_token="tok")
        headers = build_headers(target, forwarded={"x-datarobot-api-key": "caller-key"})
        assert headers["x-datarobot-api-key"] == "caller-key"

    def test_a_per_server_token_overrides_the_service_one(self):
        ref = MCPServerRef(name="docs", local_port=9001, api_token="per-server")
        target = build_target(ref, datarobot_endpoint=self.ENDPOINT, datarobot_api_token="service")
        assert build_headers(target)["Authorization"] == "Bearer per-server"
