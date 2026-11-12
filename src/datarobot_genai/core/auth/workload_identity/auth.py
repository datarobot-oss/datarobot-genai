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

"""``requests`` and ``httpx`` auth adapters that send the workload token as a bearer."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from collections.abc import Generator

import httpx
import requests

from .source import WorkloadTokenSource


class WorkloadTokenAuth(requests.auth.AuthBase):
    """Sets ``Authorization: Bearer <workload token>``; raises ``WorkloadTokenError`` if none."""

    def __init__(self, source: WorkloadTokenSource) -> None:
        self._source = source

    def __call__(self, r: requests.PreparedRequest) -> requests.PreparedRequest:
        r.headers["Authorization"] = f"Bearer {self._source.token()}"
        return r


class WorkloadTokenHttpxAuth(httpx.Auth):
    """httpx ``Authorization: Bearer <workload token>`` for sync and async clients."""

    def __init__(self, source: WorkloadTokenSource) -> None:
        self._source = source

    def sync_auth_flow(
        self, request: httpx.Request
    ) -> Generator[httpx.Request, httpx.Response, None]:
        request.headers["Authorization"] = f"Bearer {self._source.token()}"
        yield request

    async def async_auth_flow(
        self, request: httpx.Request
    ) -> AsyncGenerator[httpx.Request, httpx.Response]:
        # token() may block on the first fetch; keep it off the event loop.
        token = await asyncio.to_thread(self._source.token)
        request.headers["Authorization"] = f"Bearer {token}"
        yield request
