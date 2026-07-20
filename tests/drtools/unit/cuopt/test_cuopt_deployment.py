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

"""Unit tests for cuOpt deployment resolution and the deployment call seam."""

import json
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
import respx

from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drtools.cuopt import deployment as cd

# --------------------------------------------------------------------------- #
# CUOPT_DEPLOYMENT_ID resolution (plain env + MLOPS runtime param + tag lookup)
# --------------------------------------------------------------------------- #


def _clear_cuopt_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CUOPT_DEPLOYMENT_ID", raising=False)
    monkeypatch.delenv("MLOPS_RUNTIME_PARAM_CUOPT_DEPLOYMENT_ID", raising=False)


def test_resolve_deployment_id_plain_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # GIVEN CUOPT_DEPLOYMENT_ID set as a plain env var
    _clear_cuopt_env(monkeypatch)
    monkeypatch.setenv("CUOPT_DEPLOYMENT_ID", "dep-plain")

    # WHEN resolving THEN the env value wins without any deployment lookup
    assert cd.resolve_cuopt_deployment_id() == "dep-plain"


def test_resolve_deployment_id_mlops_runtime_param(monkeypatch: pytest.MonkeyPatch) -> None:
    # GIVEN the id surfaced as a DataRobot MLOPS_RUNTIME_PARAM_* JSON payload
    _clear_cuopt_env(monkeypatch)
    monkeypatch.setenv(
        "MLOPS_RUNTIME_PARAM_CUOPT_DEPLOYMENT_ID",
        json.dumps({"type": "string", "payload": "dep-runtime"}),
    )

    # WHEN resolving THEN the runtime-param payload is unwrapped
    assert cd.resolve_cuopt_deployment_id() == "dep-runtime"


def test_resolve_deployment_id_falls_back_to_tool_tag_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # GIVEN no env configuration but a tool-tagged cuOpt deployment
    _clear_cuopt_env(monkeypatch)
    monkeypatch.setattr(cd, "find_cuopt_tool_deployment_id", lambda: "dep-tagged")

    # WHEN resolving THEN the tag lookup provides the id
    assert cd.resolve_cuopt_deployment_id() == "dep-tagged"


def test_resolve_deployment_id_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    # GIVEN no env configuration and no tool-tagged cuOpt deployment
    _clear_cuopt_env(monkeypatch)
    monkeypatch.setattr(cd, "find_cuopt_tool_deployment_id", lambda: None)

    # WHEN resolving THEN the empty string signals "unconfigured"
    assert cd.resolve_cuopt_deployment_id() == ""


# --------------------------------------------------------------------------- #
# Tool-tagged deployment discovery
# --------------------------------------------------------------------------- #


def _tool_tagged(deployment_id: str, label: str) -> dict[str, Any]:
    return {
        "id": deployment_id,
        "label": label,
        "tags": [{"name": "tool", "value": "tool"}],
    }


def test_find_cuopt_tool_deployment_matches_label(monkeypatch: pytest.MonkeyPatch) -> None:
    # GIVEN tool-tagged deployments where exactly one is a cuOpt NIM
    monkeypatch.setattr(
        cd,
        "_list_tool_tagged_deployments",
        lambda: [
            _tool_tagged("dep-other", "Churn Model"),
            _tool_tagged("dep-cuopt", "cuOpt NIM Solver"),
        ],
    )

    # WHEN discovering THEN the cuOpt-labelled deployment id is returned
    assert cd.find_cuopt_tool_deployment_id() == "dep-cuopt"


def test_find_cuopt_tool_deployment_none_when_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    # GIVEN tool-tagged deployments with no cuOpt candidate
    monkeypatch.setattr(
        cd,
        "_list_tool_tagged_deployments",
        lambda: [_tool_tagged("dep-other", "Churn Model")],
    )

    # WHEN discovering THEN None is returned (caller decides how to error)
    assert cd.find_cuopt_tool_deployment_id() is None


def test_find_cuopt_tool_deployment_ambiguous_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # GIVEN multiple tool-tagged cuOpt deployments
    monkeypatch.setattr(
        cd,
        "_list_tool_tagged_deployments",
        lambda: [
            _tool_tagged("dep-a", "cuOpt East"),
            _tool_tagged("dep-b", "cuopt west"),
        ],
    )

    # WHEN discovering THEN the ambiguity is surfaced with both ids
    with pytest.raises(ToolError, match="dep-a.*dep-b"):
        cd.find_cuopt_tool_deployment_id()


def test_find_cuopt_tool_deployment_requires_exact_tool_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # GIVEN a cuOpt deployment whose tags only partially match (API filters use
    # OR semantics, so tag_keys/tag_values matches must be re-checked with AND)
    monkeypatch.setattr(
        cd,
        "_list_tool_tagged_deployments",
        lambda: [
            {
                "id": "dep-half",
                "label": "cuOpt NIM",
                "tags": [{"name": "tool", "value": "not-tool"}],
            }
        ],
    )

    # WHEN discovering THEN the half-tagged deployment is not a candidate
    assert cd.find_cuopt_tool_deployment_id() is None


# --------------------------------------------------------------------------- #
# get_cuopt_client -- URL + auth built through the shared deployment seam
# --------------------------------------------------------------------------- #


def _fake_serverless_deployment(deployment_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=deployment_id,
        prediction_environment={"platform": "datarobotServerless"},
        default_prediction_server=None,
    )


def _patch_dr_seam(
    monkeypatch: pytest.MonkeyPatch,
    *,
    endpoint: str = "https://dr.example/api/v2",
    token: str = "tok",
    deployment_id: str = "dep-42",
) -> None:
    fake_api_client = SimpleNamespace(endpoint=endpoint, token=token)

    @contextmanager
    def _fake_request_user_client(**_: Any) -> Any:
        yield fake_api_client

    fake_tsc = MagicMock()
    fake_tsc.return_value.request_user_client = _fake_request_user_client
    monkeypatch.setattr(cd, "ThreadSafeDataRobotClient", fake_tsc)
    monkeypatch.setattr(
        cd.dr.Deployment,
        "get",
        classmethod(lambda cls, _id: _fake_serverless_deployment(deployment_id)),
    )


def test_get_cuopt_client_requires_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    # GIVEN no configured or discoverable cuOpt deployment
    monkeypatch.setattr(cd, "resolve_cuopt_deployment_id", lambda: "")

    # WHEN building the client THEN a validation ToolError names the env var
    with pytest.raises(ToolError, match="CUOPT_DEPLOYMENT_ID"):
        cd.get_cuopt_client()


def test_get_cuopt_client_builds_direct_access_url(monkeypatch: pytest.MonkeyPatch) -> None:
    # GIVEN a configured serverless cuOpt deployment
    monkeypatch.setenv("CUOPT_DEPLOYMENT_ID", "dep-42")
    _patch_dr_seam(monkeypatch)

    # WHEN building the client
    client = cd.get_cuopt_client()

    # THEN the base URL is the deployment's directAccess NIM route and auth
    # headers come from the shared deployment seam
    assert client.base_url == "https://dr.example/api/v2/deployments/dep-42/directAccess/nim/cuopt"
    assert client.timeout == cd.CUOPT_TIMEOUT_SECONDS
    assert client.headers["Authorization"] == "Bearer tok"
    assert client.headers["Content-Type"] == "application/json"


def test_get_cuopt_client_resolves_mlops_runtime_param(monkeypatch: pytest.MonkeyPatch) -> None:
    # GIVEN the deployment id only via the MLOPS runtime param
    monkeypatch.delenv("CUOPT_DEPLOYMENT_ID", raising=False)
    monkeypatch.setenv(
        "MLOPS_RUNTIME_PARAM_CUOPT_DEPLOYMENT_ID",
        json.dumps({"type": "string", "payload": "dep-runtime"}),
    )
    _patch_dr_seam(monkeypatch, deployment_id="dep-runtime")

    # WHEN building the client THEN the runtime-param deployment is addressed
    client = cd.get_cuopt_client()
    assert (
        client.base_url
        == "https://dr.example/api/v2/deployments/dep-runtime/directAccess/nim/cuopt"
    )


# --------------------------------------------------------------------------- #
# CuOptDeploymentClient -- submit/poll protocol over the seam-built URL
# --------------------------------------------------------------------------- #

_BASE = "https://dr.example/api/v2/deployments/dep-42/directAccess/nim/cuopt"


def _client(timeout: int = 5) -> Any:
    return cd.CuOptDeploymentClient(
        base_url=_BASE,
        headers={"Authorization": "Bearer tok", "Content-Type": "application/json"},
        timeout=timeout,
    )


@respx.mock
async def test_client_validate_success() -> None:
    # GIVEN the NIM accepts the payload in validation-only mode
    route = respx.post(f"{_BASE}/request", params={"validation_only": "true"}).mock(
        return_value=httpx.Response(200, json={"msg": "ok"})
    )

    # WHEN validating
    result = await _client().validate({"some": "payload"})

    # THEN a valid status is returned and auth headers were sent
    assert result["status"] == "valid"
    assert route.calls.last.request.headers["Authorization"] == "Bearer tok"


@respx.mock
async def test_client_validate_invalid_payload() -> None:
    # GIVEN the NIM rejects the payload
    respx.post(f"{_BASE}/request", params={"validation_only": "true"}).mock(
        return_value=httpx.Response(422, json={"error": "bad bounds"})
    )

    # WHEN validating THEN an invalid status with details is returned
    result = await _client().validate({"some": "payload"})
    assert result["status"] == "invalid"
    assert "HTTP 422" in result["error"]
    assert result["details"] == {"error": "bad bounds"}


@respx.mock
async def test_client_solve_polls_until_solution() -> None:
    # GIVEN a submit that returns a request id and a solution ready on poll
    respx.post(f"{_BASE}/request").mock(return_value=httpx.Response(200, json={"reqId": "r1"}))
    respx.get(f"{_BASE}/solution/r1").mock(
        return_value=httpx.Response(
            200, json={"response": {"solver_response": {"solution_cost": 5.0}}}
        )
    )

    # WHEN solving
    result = await _client().solve({"some": "payload"})

    # THEN the solver response is unwrapped into a success result
    assert result == {
        "request_id": "r1",
        "status": "success",
        "solution": {"solution_cost": 5.0},
        "cost": 5.0,
    }


@respx.mock
async def test_client_solve_infeasible() -> None:
    # GIVEN the solver reports the problem infeasible
    respx.post(f"{_BASE}/request").mock(return_value=httpx.Response(200, json={"reqId": "r2"}))
    respx.get(f"{_BASE}/solution/r2").mock(
        return_value=httpx.Response(
            200, json={"response": {"solver_infeasible_response": {"solution_cost": 0.0}}}
        )
    )

    # WHEN solving THEN the infeasible status is surfaced
    result = await _client().solve({"some": "payload"})
    assert result["status"] == "infeasible"


@respx.mock
async def test_client_solve_submit_http_error() -> None:
    # GIVEN the submit itself fails with an HTTP error
    respx.post(f"{_BASE}/request").mock(return_value=httpx.Response(500, json={"error": "boom"}))

    # WHEN solving THEN a structured error result is returned (no raise)
    result = await _client().solve({"some": "payload"})
    assert result["status"] == "error"
    assert result["error"] == "HTTP 500"


@respx.mock
async def test_client_solve_poll_validation_error() -> None:
    # GIVEN the poll reports a cuOpt validation failure
    respx.post(f"{_BASE}/request").mock(return_value=httpx.Response(200, json={"reqId": "r3"}))
    respx.get(f"{_BASE}/solution/r3").mock(
        return_value=httpx.Response(422, json={"error": "Invalid constraint bounds"})
    )

    # WHEN solving THEN the validation message is surfaced as an error result
    result = await _client().solve({"some": "payload"})
    assert result["status"] == "error"
    assert result["error"] == "Invalid constraint bounds"


@respx.mock
async def test_client_solve_missing_request_id() -> None:
    # GIVEN a submit response without a request id
    respx.post(f"{_BASE}/request").mock(return_value=httpx.Response(200, json={"oops": True}))

    # WHEN solving THEN the missing id is reported as an error
    result = await _client().solve({"some": "payload"})
    assert result["status"] == "error"
    assert "No request ID" in result["error"]
