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

"""DataRobot API client for tools."""

import logging
from collections.abc import AsyncIterator
from types import TracebackType
from typing import Any
from urllib.parse import urlparse
from urllib.parse import urlunparse

import datarobot as dr
from datarobot.context import Context as DRContext

from datarobot_genai.drtools.core.auth import resolve_token_from_headers
from datarobot_genai.drtools.core.clients.utils import get_async_https_retry_client
from datarobot_genai.drtools.core.clients.utils import get_async_https_session
from datarobot_genai.drtools.core.credentials import get_credentials
from datarobot_genai.drtools.core.exceptions import ToolError
from datarobot_genai.drtools.core.exceptions import ToolErrorKind

logger = logging.getLogger(__name__)


async def get_datarobot_access_token() -> str:
    """
    Get DataRobot API token from HTTP headers.

    Uses the same token extraction as core (auth headers and authorization
    context metadata). For use in tools only; core modules use get_sdk_client()
    from drmcp.core.clients.

    Returns
    -------
        API token string

    Raises
    ------
        ToolError: If no API token is found in headers
    """
    token = resolve_token_from_headers()
    if not token:
        logger.warning("DataRobot API token not found in headers")
        raise ToolError(
            "DataRobot API token not found in headers. "
            "Please provide it via 'Authorization' (Bearer), 'x-datarobot-api-token' headers.",
            kind=ToolErrorKind.AUTHENTICATION,
        )
    return token


class DataRobotClient:
    """Client for interacting with DataRobot API in tools.

    Wraps the DataRobot Python SDK (datarobot package). Obtain the token
    via get_datarobot_access_token() and pass it to the constructor.
    """

    def __init__(self, token: str) -> None:
        self._token = token

    def get_client(self) -> Any:
        """
        Configure the DataRobot SDK with this client's token and return the dr module.

        The returned value is the global datarobot module (dr) after
        dr.Client(token=..., endpoint=...) has been called. Use it as
        client.Project.list(), client.Deployment.get(...), etc.

        Returns
        -------
            The datarobot module (dr) configured for the current token and endpoint.
        """
        credentials = get_credentials()
        dr.Client(token=self._token, endpoint=credentials.datarobot.endpoint)
        # Avoid use-case context from trafaret affecting tool calls
        DRContext.use_case = None
        return dr


class DataRobotClientWithAsyncAPI:
    """A async wrapper of DR RESTFul APIs."""

    def __init__(self, dr_host: str) -> None:
        self._dr_host = dr_host
        self._session = get_async_https_session()
        self._retry_client = get_async_https_retry_client(self._session)

    async def clean_up(self) -> None:
        await self._retry_client.close()

    async def __aenter__(self) -> "DataRobotClientWithAsyncAPI":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.clean_up()

    @staticmethod
    def get_api_v2_endpoint(host_url: str, v2_path: str) -> str:
        parsed_url = urlparse(host_url)
        full_path = f"api/v2/{v2_path}/"
        full_path = full_path.replace("//", "/")
        new_parsed_url = parsed_url._replace(path=full_path)
        return urlunparse(new_parsed_url)

    async def _unpaginate(
        self,
        initial_url: str,
        headers: dict[str, str],
    ) -> AsyncIterator[Any]:
        url: str | None = initial_url
        while url is not None:
            async with self._retry_client.get(url, headers=headers) as resp:
                resp.raise_for_status()
                resp_data = await resp.json()
            for item in resp_data.get("data", []):
                yield item
            url = resp_data.get("next")

    async def get_feature_entitlement_evaluate_result(
        self,
        feature_flag_name: str,
        dr_bearer_token: str,
    ) -> dict[str, Any]:
        url = self.get_api_v2_endpoint(self._dr_host, "/entitlements/evaluate/")
        async with self._retry_client.post(
            url,
            json={"entitlements": [{"name": feature_flag_name}]},
            headers={"Authorization": f"Bearer {dr_bearer_token}"},
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def is_feature_flag_enabled(self, feature_flag_name: str, dr_bearer_token: str) -> bool:
        response = await self.get_feature_entitlement_evaluate_result(
            feature_flag_name, dr_bearer_token
        )
        feature_flag_info = response["entitlements"][0]
        return bool(feature_flag_info["value"])

    async def list_mcp_deployment_ids(self, dr_bearer_token: str) -> list[str]:
        url = self.get_api_v2_endpoint(self._dr_host, "/deployments/")
        headers = {"Authorization": f"Bearer {dr_bearer_token}"}
        ids: list[str] = []
        async for deployment in self._unpaginate(url, headers):
            model = deployment.get("model") or {}
            if model.get("targetType") == "MCP":
                ids.append(deployment["id"])
        return ids

    async def list_mcp_tool_custom_model_deployment_ids(self, dr_bearer_token: str) -> list[str]:
        url = self.get_api_v2_endpoint(self._dr_host, "/deployments")
        url = f"{url}?tagValues=tool&tagKeys=tool"
        headers = {"Authorization": f"Bearer {dr_bearer_token}"}
        ids: list[str] = []
        async for deployment in self._unpaginate(url, headers):
            for tag in deployment.get("tags", []):
                if tag.get("name") == "tool" and tag.get("value") == "tool":
                    ids.append(deployment["id"])
        return ids

    async def get_datarobot_deployment(
        self, deployment_id: str, dr_bearer_token: str
    ) -> dr.Deployment:
        url = self.get_api_v2_endpoint(self._dr_host, f"/deployments/{deployment_id}/")
        headers = {"Authorization": f"Bearer {dr_bearer_token}"}
        async with self._retry_client.get(
            url,
            headers=headers,
        ) as resp:
            resp.raise_for_status()
            deployment_json_response = await resp.json()
            return dr.Deployment.from_server_data(deployment_json_response)
