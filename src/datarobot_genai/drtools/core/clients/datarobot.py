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

# TODO: This needs to be moved to drmcputils.
# Also use file api with openapi tools. Also add tool configs in drmcp.

from typing import Any

from datarobot_genai.drmcputils.clients.datarobot import request_user_dr_client


class DataRobotApiClient:
    """Core DataRobot API methods backed by the per-request REST client.

    Each call runs inside :func:`request_user_dr_client` so credentials come from
    the requesting user's headers::

        client = DataRobotApiClient()
        result = client.list_credentials(limit=50)
    """

    def list_credentials(self, *, limit: int = 100, offset: int = 0) -> dict[str, Any]:
        """GET /credentials/ — list all accessible credentials (id, name, type)."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        with request_user_dr_client() as client:
            return client.get("credentials/", params=params).json()

    def get_credential(self, credential_id: str) -> dict[str, Any]:
        """GET /credentials/{id}/ — fetch a single credential record."""
        with request_user_dr_client() as client:
            return client.get(f"credentials/{credential_id}/").json()

    def get_openapi_spec(self, path: str = "openapi.json") -> dict[str, Any]:
        """GET {path} relative to the configured DR endpoint — returns the JSON spec."""
        with request_user_dr_client() as client:
            return client.get(path).json()
