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

import asyncio
import logging
import time
from typing import Any

from datarobot_genai.drmcputils.clients.datarobot import request_user_dr_client

logger = logging.getLogger(__name__)


class WorkloadApiClient:
    """Workload API methods backed by the per-request DataRobot REST client.

    Each call runs inside :func:`request_user_dr_client` so credentials come from
    the requesting user's headers::

        client = WorkloadApiClient()
        result = client.list_workloads(limit=50)
    """

    # ------------------------------------------------------------------ #
    # Workloads — read                                                     #
    # ------------------------------------------------------------------ #

    def list_workloads(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        search: str | None = None,
    ) -> dict[str, Any]:
        """GET /workloads/ with optional server-side search and pagination."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if search:
            params["search"] = search
        with request_user_dr_client() as client:
            return client.get("workloads/", params=params).json()

    def get_workload(self, workload_id: str) -> dict[str, Any]:
        """GET /workloads/{workload_id}."""
        with request_user_dr_client() as client:
            return client.get(f"workloads/{workload_id}").json()

    # ------------------------------------------------------------------ #
    # Compute bundles                                                      #
    # ------------------------------------------------------------------ #

    def list_bundles(self) -> dict[str, Any]:
        """GET /mlops/compute/bundles — available CPU/GPU resource bundles."""
        with request_user_dr_client() as client:
            return client.get("mlops/compute/bundles").json()

    # ------------------------------------------------------------------ #
    # Workloads — write                                                    #
    # ------------------------------------------------------------------ #

    def create_workload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST /workloads/ — returns the created WorkloadFormatted (201)."""
        with request_user_dr_client() as client:
            return client.post("workloads/", json=payload).json()

    def start_workload(self, workload_id: str) -> dict[str, Any]:
        """POST /workloads/{id}/start — returns WorkloadOperationResponse (202)."""
        with request_user_dr_client() as client:
            resp = client.post(f"workloads/{workload_id}/start")
            return resp.json() if resp.content else {}

    def stop_workload(self, workload_id: str) -> dict[str, Any]:
        """POST /workloads/{id}/stop — returns WorkloadOperationResponse (202)."""
        with request_user_dr_client() as client:
            resp = client.post(f"workloads/{workload_id}/stop")
            return resp.json() if resp.content else {}

    def delete_workload(self, workload_id: str) -> None:
        """DELETE /workloads/{id} — 204 No Content on success."""
        with request_user_dr_client() as client:
            client.delete(f"workloads/{workload_id}")

    def patch_workload(self, workload_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """PATCH /workloads/{id} — partial update of name / description / importance."""
        with request_user_dr_client() as client:
            return client.patch(f"workloads/{workload_id}", json=payload).json()

    # ------------------------------------------------------------------ #
    # Workloads — polling                                                  #
    # ------------------------------------------------------------------ #

    async def wait_for_workload_status(
        self,
        workload_id: str,
        target_status: str,
        *,
        timeout_seconds: int = 600,
        poll_interval_seconds: int = 1,
    ) -> dict[str, Any]:
        """Poll until the workload reaches *target_status*, enters errored, or times out.

        Uses :func:`asyncio.sleep` between polls so async callers do not block the event loop.

        Raises
        ------
        RuntimeError
            When the workload enters ``errored`` before reaching *target_status*.
        TimeoutError
            When *timeout_seconds* elapses before reaching *target_status*.
        """
        deadline = time.monotonic() + timeout_seconds
        last_status: str | None = None
        while True:
            obj = self.get_workload(workload_id)
            status = obj.get("status")
            if status != last_status:
                logger.info("Workload %s status: %s", workload_id, status)
                last_status = status
            if status == target_status:
                return obj
            if status == "errored":
                raise RuntimeError(
                    f"Workload {workload_id} errored while waiting for '{target_status}'"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timeout waiting for workload {workload_id} to reach "
                    f"'{target_status}'. Last status: {status}"
                )
            await asyncio.sleep(poll_interval_seconds)

    # ------------------------------------------------------------------ #
    # Workloads — settings                                                 #
    # ------------------------------------------------------------------ #

    def get_workload_settings(self, workload_id: str) -> dict[str, Any]:
        """GET /workloads/{id}/settings — returns WorkloadSettingsResponse."""
        with request_user_dr_client() as client:
            return client.get(f"workloads/{workload_id}/settings").json()

    def update_workload_settings(self, workload_id: str, runtime: dict[str, Any]) -> dict[str, Any]:
        """PATCH /workloads/{id}/settings — triggers rolling replacement (202)."""
        with request_user_dr_client() as client:
            return client.patch(
                f"workloads/{workload_id}/settings", json={"runtime": runtime}
            ).json()

    # ------------------------------------------------------------------ #
    # Workloads — observability                                            #
    # ------------------------------------------------------------------ #

    def get_workload_stats(
        self,
        workload_id: str,
        *,
        proton_id: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        response_time_quantile: float = 0.5,
        slow_requests_threshold: int = 2000,
    ) -> dict[str, Any]:
        """GET /workloads/{id}/stats — aggregated performance statistics."""
        params: dict[str, Any] = {
            "responseTimeQuantile": response_time_quantile,
            "slowRequestsThreshold": slow_requests_threshold,
        }
        if proton_id:
            params["protonId"] = proton_id
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        with request_user_dr_client() as client:
            return client.get(f"workloads/{workload_id}/stats", params=params).json()

    def list_workload_history(
        self, workload_id: str, *, limit: int = 10, offset: int = 0
    ) -> dict[str, Any]:
        """GET /workloads/{id}/history — artifact deployment history."""
        with request_user_dr_client() as client:
            return client.get(
                f"workloads/{workload_id}/history",
                params={"limit": limit, "offset": offset},
            ).json()

    def list_workload_events(
        self, workload_id: str, *, limit: int = 10, offset: int = 0
    ) -> dict[str, Any]:
        """GET /workloads/{id}/events — status-change and error events."""
        with request_user_dr_client() as client:
            return client.get(
                f"workloads/{workload_id}/events",
                params={"limit": limit, "offset": offset},
            ).json()

    def promote_workload_artifact(self, workload_id: str) -> dict[str, Any]:
        """POST /workloads/{id}/promote — lock the running draft artifact (202)."""
        with request_user_dr_client() as client:
            resp = client.post(f"workloads/{workload_id}/promote")
            return resp.json() if resp.content else {}

    def get_workload_related(self, workload_id: str) -> dict[str, Any]:
        """GET /workloads/{id}/related — linked artifacts and related entities."""
        with request_user_dr_client() as client:
            return client.get(f"workloads/{workload_id}/related").json()

    # ------------------------------------------------------------------ #
    # Protons                                                              #
    # ------------------------------------------------------------------ #

    def list_protons(self, workload_id: str, *, limit: int = 10, offset: int = 0) -> dict[str, Any]:
        """GET /workloads/{id}/protons — deployed proton instances for a workload."""
        with request_user_dr_client() as client:
            return client.get(
                f"workloads/{workload_id}/protons",
                params={"limit": limit, "offset": offset},
            ).json()

    def get_proton(self, workload_id: str, proton_id: str) -> dict[str, Any]:
        """GET /workloads/{id}/protons/{proton_id} — single proton record."""
        with request_user_dr_client() as client:
            return client.get(f"workloads/{workload_id}/protons/{proton_id}").json()

    def get_proton_status_details(self, workload_id: str, proton_id: str) -> dict[str, Any] | None:
        """GET /workloads/{id}/protons/{proton_id}/statusDetails.

        Returns the ReplicaStatusesSnapshot when available (200), or
        ``None`` when no status has been received yet (204).
        """
        with request_user_dr_client() as client:
            resp = client.get(f"workloads/{workload_id}/protons/{proton_id}/statusDetails")
            return resp.json() if resp.content else None

    # ------------------------------------------------------------------ #
    # OTel logs                                                            #
    # ------------------------------------------------------------------ #

    def list_workload_logs(
        self,
        workload_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
        level: str = "debug",
        start_time: str | None = None,
        end_time: str | None = None,
        includes: list[str] | None = None,
        excludes: list[str] | None = None,
        span_id: str | None = None,
        trace_id: str | None = None,
    ) -> dict[str, Any]:
        """GET /otel/workload/{id}/logs/ — OTel log lines for a workload."""
        params: list[tuple[str, Any]] = [
            ("limit", limit),
            ("offset", offset),
            ("level", level),
        ]
        if start_time:
            params.append(("startTime", start_time))
        if end_time:
            params.append(("endTime", end_time))
        for v in includes or []:
            params.append(("includes", v))
        for v in excludes or []:
            params.append(("excludes", v))
        if span_id:
            params.append(("spanId", span_id))
        if trace_id:
            params.append(("traceId", trace_id))
        with request_user_dr_client() as client:
            return client.get(f"otel/workload/{workload_id}/logs/", params=params).json()

    # ------------------------------------------------------------------ #
    # Artifacts                                                            #
    # ------------------------------------------------------------------ #

    def list_artifacts(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        search: str | None = None,
        status: str | None = None,
        artifact_type: str | None = None,
        repository_id: str | None = None,
    ) -> dict[str, Any]:
        """GET /artifacts/ with optional filters and pagination."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if search:
            params["search"] = search
        if status:
            params["status"] = status
        if artifact_type:
            params["type"] = artifact_type
        if repository_id:
            params["repositoryId"] = repository_id
        with request_user_dr_client() as client:
            return client.get("artifacts/", params=params).json()

    def get_artifact(self, artifact_id: str) -> dict[str, Any]:
        """GET /artifacts/{id}."""
        with request_user_dr_client() as client:
            return client.get(f"artifacts/{artifact_id}").json()

    def create_artifact(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST /artifacts/ — returns the created ArtifactFormatted (201)."""
        with request_user_dr_client() as client:
            return client.post("artifacts/", json=payload).json()

    def put_artifact(self, artifact_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """PUT /artifacts/{id} — full replacement with InputArtifact payload."""
        with request_user_dr_client() as client:
            return client.put(f"artifacts/{artifact_id}", json=payload).json()

    def patch_artifact(self, artifact_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """PATCH /artifacts/{id} — partial update via UpdateArtifactRequest."""
        with request_user_dr_client() as client:
            return client.patch(f"artifacts/{artifact_id}", json=payload).json()

    def delete_artifact(self, artifact_id: str) -> None:
        """DELETE /artifacts/{id} — 204 No Content on success."""
        with request_user_dr_client() as client:
            client.delete(f"artifacts/{artifact_id}")

    def clone_artifact(self, artifact_id: str, name: str) -> dict[str, Any]:
        """POST /artifacts/{id}/clone — duplicate with a new name."""
        with request_user_dr_client() as client:
            return client.post(f"artifacts/{artifact_id}/clone", json={"name": name}).json()

    # ------------------------------------------------------------------ #
    # Artifact builds                                                      #
    # ------------------------------------------------------------------ #

    def list_artifact_builds(
        self,
        artifact_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """GET /artifacts/{id}/builds."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        with request_user_dr_client() as client:
            return client.get(f"artifacts/{artifact_id}/builds", params=params).json()

    def trigger_artifact_build(self, artifact_id: str) -> dict[str, Any]:
        """POST /artifacts/{id}/builds — start image build(s) for a draft service artifact."""
        with request_user_dr_client() as client:
            return client.post(f"artifacts/{artifact_id}/builds").json()

    def get_artifact_build(self, artifact_id: str, build_id: str) -> dict[str, Any]:
        """GET /artifacts/{id}/builds/{build_id}."""
        with request_user_dr_client() as client:
            return client.get(f"artifacts/{artifact_id}/builds/{build_id}").json()

    def get_artifact_build_logs(self, artifact_id: str, build_id: str) -> str:
        """GET /artifacts/{id}/builds/{build_id}/logs — returns raw log text."""
        with request_user_dr_client() as client:
            return client.get(f"artifacts/{artifact_id}/builds/{build_id}/logs").text

    def delete_artifact_build(self, artifact_id: str, build_id: str) -> None:
        """DELETE /artifacts/{id}/builds/{build_id} — cancel or delete a build (204)."""
        with request_user_dr_client() as client:
            client.delete(f"artifacts/{artifact_id}/builds/{build_id}")

    # ------------------------------------------------------------------ #
    # Artifact repositories                                                #
    # ------------------------------------------------------------------ #

    def list_artifact_repositories(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        search: str | None = None,
        artifact_type: str | None = None,
    ) -> dict[str, Any]:
        """GET /artifactRepositories."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if search:
            params["search"] = search
        if artifact_type:
            params["type"] = artifact_type
        with request_user_dr_client() as client:
            return client.get("artifactRepositories", params=params).json()

    def get_artifact_repository(self, repository_id: str) -> dict[str, Any]:
        """GET /artifactRepositories/{id}."""
        with request_user_dr_client() as client:
            return client.get(f"artifactRepositories/{repository_id}").json()

    def delete_artifact_repository(self, repository_id: str) -> None:
        """DELETE /artifactRepositories/{id} — 204 No Content on success."""
        with request_user_dr_client() as client:
            client.delete(f"artifactRepositories/{repository_id}")
