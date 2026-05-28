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
from collections.abc import AsyncIterator
from enum import Enum
from enum import auto
from http import HTTPMethod
from ssl import SSLContext
from ssl import create_default_context as create_default_ssl_context
from types import TracebackType
from typing import Any
from urllib.parse import urlparse
from urllib.parse import urlunparse

import datarobot as dr
from aiohttp import ClientSession
from aiohttp import ClientTimeout
from aiohttp import TCPConnector
from aiohttp_retry import ExponentialRetry
from aiohttp_retry import RetryClient


class TimeMeasurement(Enum):
    HOUR = auto()
    MINUTE = auto()
    SECOND = auto()

    def to_numeric_value_in_second(self) -> int:
        return {
            TimeMeasurement.HOUR: 3600,
            TimeMeasurement.MINUTE: 60,
            TimeMeasurement.SECOND: 1,
        }[self]


def get_ssl_context_from_ca_file(ca_path: str) -> SSLContext:
    ctx = create_default_ssl_context()
    ctx.load_verify_locations(cafile=ca_path)
    return ctx


def get_connect_timeout_in_second() -> int:
    return TimeMeasurement.SECOND.to_numeric_value_in_second() * 30


def get_read_timeout_in_second() -> int:
    return TimeMeasurement.SECOND.to_numeric_value_in_second() * 60


def get_async_https_session(root_ca: str | None = None) -> ClientSession:
    headers = {"User-Agent": "global-mcp"}

    ssl_arg = get_ssl_context_from_ca_file(root_ca) if root_ca is not None else True
    connector = TCPConnector(ssl=ssl_arg)
    timeout = ClientTimeout(
        connect=get_connect_timeout_in_second(),
        sock_read=get_read_timeout_in_second(),
    )

    return ClientSession(
        headers=headers,
        connector=connector,
        timeout=timeout,
    )


def get_async_https_retry_client(session: ClientSession) -> RetryClient:
    retry_options = ExponentialRetry(
        attempts=3,
        start_timeout=0.1,
        methods=[HTTPMethod.GET, HTTPMethod.POST, HTTPMethod.PUT],  # type: ignore[arg-type]
    )
    return RetryClient(client_session=session, retry_options=retry_options)


class DataRobotClientWithAsyncAPI:
    """A async wrapper of DR RESTFul APIs.

    This client was designed for internal use only (Some methods are created as private methods).
    It is the interim solution before async support is added in
    https://github.com/datarobot/public_api_client.
    """

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

    async def _get_feature_entitlement_evaluate_result(
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

    async def _is_feature_flag_enabled(self, feature_flag_name: str, dr_bearer_token: str) -> bool:
        response = await self._get_feature_entitlement_evaluate_result(
            feature_flag_name, dr_bearer_token
        )
        feature_flag_info = response["entitlements"][0]
        return bool(feature_flag_info["value"])

    async def _list_mcp_deployment_ids(self, dr_bearer_token: str) -> list[str]:
        url = self.get_api_v2_endpoint(self._dr_host, "/deployments/")
        headers = {"Authorization": f"Bearer {dr_bearer_token}"}
        ids: list[str] = []
        async for deployment in self._unpaginate(url, headers):
            model = deployment.get("model") or {}
            if model.get("targetType") == "MCP":
                ids.append(deployment["id"])
        return ids

    async def _list_mcp_tool_custom_model_deployment_ids(self, dr_bearer_token: str) -> list[str]:
        url = self.get_api_v2_endpoint(self._dr_host, "/deployments")
        url = f"{url}?tagValues=tool&tagKeys=tool"
        headers = {"Authorization": f"Bearer {dr_bearer_token}"}
        ids: list[str] = []
        async for deployment in self._unpaginate(url, headers):
            for tag in deployment.get("tags", []):
                if tag.get("name") == "tool" and tag.get("value") == "tool":
                    ids.append(deployment["id"])
        return ids

    async def _get_datarobot_deployment(
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
