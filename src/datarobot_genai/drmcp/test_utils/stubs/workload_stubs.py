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

"""REST stubs for DataRobot Workload API tools (integration tests)."""

from __future__ import annotations

from typing import Any

from datarobot_genai.drmcp.test_utils.stubs.stub_rest_response import StubRestResponse

STUB_WORKLOAD_ID = "stub_workload_id"
STUB_ARTIFACT_ID = "stub_artifact_id"
STUB_PROTON_ID = "stub_proton_id"
STUB_BUILD_ID = "stub_build_id"
STUB_REPOSITORY_ID = "stub_repository_id"
STUB_BUNDLE_ID = "cpu.small"

_WORKLOAD: dict[str, Any] = {
    "id": STUB_WORKLOAD_ID,
    "name": "Stub Workload Alpha",
    "status": "running",
    "artifactId": STUB_ARTIFACT_ID,
    "importance": "low",
}

_ARTIFACT: dict[str, Any] = {
    "id": STUB_ARTIFACT_ID,
    "name": "Stub Service Artifact",
    "status": "draft",
    "type": "service",
}

_REPOSITORY: dict[str, Any] = {
    "id": STUB_REPOSITORY_ID,
    "name": "Stub Artifact Repository",
    "type": "service",
}

_BUILD: dict[str, Any] = {"id": STUB_BUILD_ID, "status": "success"}

_PROTON: dict[str, Any] = {"id": STUB_PROTON_ID, "status": "running", "replicaCount": 1}

_LOG: dict[str, Any] = {
    "body": "Server started on port 8080",
    "level": "info",
    "timestamp": "2026-01-01T00:00:00Z",
}


def _params_dict(params: dict[str, Any] | list[tuple[str, Any]] | None) -> dict[str, Any]:
    if params is None:
        return {}
    if isinstance(params, dict):
        return params
    out: dict[str, Any] = {}
    for key, value in params:
        if key in out:
            existing = out[key]
            out[key] = [existing, value] if not isinstance(existing, list) else [*existing, value]
        else:
            out[key] = value
    return out


def _paginate(items: list[dict[str, Any]], params: dict[str, Any]) -> dict[str, Any]:
    offset = int(params.get("offset", 0))
    limit = int(params.get("limit", 100))
    page = items[offset : offset + limit]
    return {
        "data": page,
        "count": len(page),
        "totalCount": len(items),
        "next": None,
        "previous": None,
    }


def _workload_get_response(url: str, p: dict[str, Any]) -> StubRestResponse | None:
    response: StubRestResponse | None = None
    if url.rstrip("/") == "workloads":
        items = [_WORKLOAD]
        search = p.get("search")
        if search:
            needle = str(search).lower()
            items = [w for w in items if needle in w["name"].lower()]
        response = StubRestResponse(_paginate(items, p))
    elif url.startswith("workloads/") and url.count("/") == 1:
        response = StubRestResponse(_WORKLOAD)
    elif "/settings" in url:
        response = StubRestResponse(
            {"runtime": {"containerGroups": [{"name": "default", "replicaCount": 1}]}}
        )
    elif "/stats" in url:
        response = StubRestResponse(
            {
                "requestCount": 120,
                "errorRate": 0.01,
                "responseTimeQuantileMs": 45,
                "slowRequestsCount": 2,
            }
        )
    elif "/history" in url:
        response = StubRestResponse(
            _paginate([{"artifactId": STUB_ARTIFACT_ID, "deployedAt": "2026-01-01T00:00:00Z"}], p)
        )
    elif "/events" in url:
        response = StubRestResponse(_paginate([{"type": "status_change", "status": "running"}], p))
    elif "/related" in url:
        response = StubRestResponse({"artifacts": [{"id": STUB_ARTIFACT_ID}]})
    elif url.endswith("/protons") or url.endswith("/protons/"):
        response = StubRestResponse(_paginate([_PROTON], p))
    elif "/protons/" in url and url.endswith("/statusDetails"):
        response = StubRestResponse(
            {"replicas": [{"name": "pod-0", "phase": "Running", "ready": True, "restartCount": 0}]}
        )
    elif "/protons/" in url:
        response = StubRestResponse(_PROTON)
    elif "/replacement" in url:
        response = StubRestResponse(
            {"candidateArtifactId": STUB_ARTIFACT_ID, "strategy": "rolling"}
        )
    elif url.rstrip("/") == "mlops/compute/bundles":
        response = StubRestResponse(
            {
                "data": [
                    {
                        "id": STUB_BUNDLE_ID,
                        "cpuCount": 1,
                        "memoryBytes": 134217728,
                        "gpuType": None,
                    }
                ]
            }
        )
    return response


def _artifact_get_response(url: str, p: dict[str, Any]) -> StubRestResponse | None:
    response: StubRestResponse | None = None
    if url.rstrip("/") == "artifacts":
        response = StubRestResponse(_paginate([_ARTIFACT], p))
    elif url.startswith("artifacts/") and url.count("/") == 1:
        response = StubRestResponse(_ARTIFACT)
    elif url.endswith("/builds") or url.endswith("/builds/"):
        response = StubRestResponse(_paginate([_BUILD], p))
    elif "/builds/" in url and url.endswith("/logs"):
        response = StubRestResponse(text="Step 1/3: FROM python:3.11\nBuild complete.\n")
    elif "/builds/" in url:
        response = StubRestResponse(_BUILD)
    return response


def _repository_get_response(url: str, p: dict[str, Any]) -> StubRestResponse | None:
    response: StubRestResponse | None = None
    if url.rstrip("/") == "artifactRepositories":
        response = StubRestResponse(_paginate([_REPOSITORY], p))
    elif url.startswith("artifactRepositories/"):
        response = StubRestResponse(_REPOSITORY)
    return response


def workload_stub_get(
    url: str, params: dict[str, Any] | list[tuple[str, Any]] | None = None, **kwargs: Any
) -> StubRestResponse | None:
    del kwargs
    p = _params_dict(params)
    for handler in (_workload_get_response, _artifact_get_response, _repository_get_response):
        response = handler(url, p)
        if response is not None:
            return response
    if url.startswith("otel/workload/") and "/logs/" in url:
        return StubRestResponse(_paginate([_LOG], p))
    return None


def workload_stub_post(
    url: str, json: dict[str, Any] | None = None, **kwargs: Any
) -> StubRestResponse | None:
    del kwargs
    payload = json or {}
    response: StubRestResponse | None = None
    if url.rstrip("/") == "workloads":
        response = StubRestResponse(
            {
                "id": STUB_WORKLOAD_ID,
                "name": payload.get("name", "new-workload"),
                "status": "initializing",
                "artifactId": payload.get("artifactId"),
            }
        )
    elif url.endswith("/start") or url.endswith("/stop"):
        response = StubRestResponse({"accepted": True})
    elif url.endswith("/replacement"):
        response = StubRestResponse(
            {
                "candidateArtifactId": payload.get("artifactId", STUB_ARTIFACT_ID),
                "strategy": payload.get("strategy", "rolling"),
            }
        )
    elif url.rstrip("/") == "artifacts":
        response = StubRestResponse(
            {
                "id": "stub_artifact_new",
                "name": payload.get("name", "new-artifact"),
                "status": "draft",
            }
        )
    elif url.endswith("/clone"):
        response = StubRestResponse(
            {
                "id": "stub_artifact_clone",
                "name": payload.get("name", "cloned-artifact"),
                "status": "draft",
            }
        )
    elif url.endswith("/builds"):
        response = StubRestResponse({"buildIds": [STUB_BUILD_ID]})
    return response


def workload_stub_patch(
    url: str, json: dict[str, Any] | None = None, **kwargs: Any
) -> StubRestResponse | None:
    del kwargs
    payload = json or {}
    response: StubRestResponse | None = None
    if url.startswith("workloads/") and url.count("/") == 1:
        response = StubRestResponse({"id": url.split("/", 1)[1], **payload})
    elif "/settings" in url:
        runtime = payload.get("runtime", payload)
        response = StubRestResponse({"id": "repl-1", "status": "in_progress", "runtime": runtime})
    elif url.startswith("artifacts/"):
        aid = url.split("/", 1)[1]
        response = StubRestResponse({"id": aid, **payload})
    return response


def workload_stub_delete(url: str, **kwargs: Any) -> StubRestResponse | None:
    del kwargs
    if url.startswith("workloads/"):
        return StubRestResponse({})
    if url.startswith("artifactRepositories/"):
        return StubRestResponse({})
    return None
