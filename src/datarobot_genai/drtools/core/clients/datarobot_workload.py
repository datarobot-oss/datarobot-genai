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

from datarobot_genai.drtools.core.clients.datarobot import request_user_dr_client

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
            return client.post(f"workloads/{workload_id}/promote").json()

    def get_workload_related(self, workload_id: str) -> dict[str, Any]:
        """GET /workloads/{id}/related — linked artifacts and related entities."""
        with request_user_dr_client() as client:
            return client.get(f"workloads/{workload_id}/related").json()
