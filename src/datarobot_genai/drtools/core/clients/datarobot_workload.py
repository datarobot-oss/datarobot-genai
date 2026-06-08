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

import logging
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
        return self._client.get(f"workloads/{workload_id}").json()

    # ------------------------------------------------------------------ #
    # Compute bundles                                                      #
    # ------------------------------------------------------------------ #

    def list_bundles(self) -> dict[str, Any]:
        """GET /mlops/compute/bundles — available CPU/GPU resource bundles."""
        with request_user_dr_client() as client:
            return client.get("mlops/compute/bundles").json()
