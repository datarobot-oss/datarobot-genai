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
import logging
from collections.abc import Generator
from dataclasses import dataclass

from httpx import AsyncClient
from httpx import Auth
from httpx import Request
from httpx import Response
from httpx import Timeout
from httpx_retries import Retry
from httpx_retries import RetryTransport

logger = logging.getLogger(__name__)


class AuthWithInjectedHeaders(Auth):
    def __init__(self, headers: dict[str, str]) -> None:
        super().__init__()
        self.headers = headers

    def auth_flow(self, request: Request) -> Generator[Request, Response, None]:
        request.headers.update(self.headers)
        yield request


@dataclass
class AsyncClientConfig:
    max_retries: int = 5
    retry_backoff: float = 2.0
    timeout_in_seconds: float = 30.0

    def create_retry_policy(self) -> Retry:
        return Retry(total=self.max_retries, backoff_factor=self.retry_backoff)

    def create_httpx_retry_transport(self) -> RetryTransport:
        return RetryTransport(retry=self.create_retry_policy())

    def create_httpx_timeout(self) -> Timeout:
        return Timeout(self.timeout_in_seconds)


def get_retriable_async_http_client(
    auth_with_headers: AuthWithInjectedHeaders | None = None,
    async_client_config: AsyncClientConfig | None = None,
) -> AsyncClient:
    async_client_config = async_client_config or AsyncClientConfig()
    return AsyncClient(
        auth=auth_with_headers,
        timeout=async_client_config.create_httpx_timeout(),
        transport=async_client_config.create_httpx_retry_transport(),
    )
