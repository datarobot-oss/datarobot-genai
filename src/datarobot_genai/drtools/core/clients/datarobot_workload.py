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

from datarobot.rest import RESTClientObject

logger = logging.getLogger(__name__)


class WorkloadApiClient:
    """Workload API methods backed by the DataRobot REST client.

    Instantiate inside a
    :func:`~datarobot_genai.drtools.core.clients.datarobot.request_user_dr_client`
    (or :meth:`ThreadSafeDataRobotClient.request_user_client`) context so the
    underlying client carries the correct per-user credentials::

        with ThreadSafeDataRobotClient().request_user_client():
            rest_client = dr.client.get_client()
            client = WorkloadApiClient(rest_client)
            result = client.list_workloads(limit=50)
    """

    def __init__(self, rest_client: RESTClientObject) -> None:
        self._client = rest_client

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
        return self._client.get("workloads/", params=params).json()

    def get_workload(self, workload_id: str) -> dict[str, Any]:
        """GET /workloads/{workload_id}."""
        return self._client.get(f"workloads/{workload_id}").json()

    # ------------------------------------------------------------------ #
    # Compute bundles                                                      #
    # ------------------------------------------------------------------ #

    def list_bundles(self) -> dict[str, Any]:
        """GET /mlops/compute/bundles — available CPU/GPU resource bundles."""
        return self._client.get("mlops/compute/bundles").json()
