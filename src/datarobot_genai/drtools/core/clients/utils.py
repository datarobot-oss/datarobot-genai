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
from enum import Enum
from enum import auto
from http import HTTPMethod
from ssl import SSLContext
from ssl import create_default_context as create_default_ssl_context

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
