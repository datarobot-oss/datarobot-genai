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

import hashlib
import os

import httpx
from mem0 import AsyncMemoryClient
from mem0.client.project import AsyncProject
from mem0.memory.telemetry import capture_client_event


class DataRobotMemoryClient(AsyncMemoryClient):
    def __init__(
        self,
        api_key: str | None = None,
        host: str | None = None,
        org_id: str | None = None,
        project_id: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("MEM0_API_KEY")
        self.host = host or "https://api.mem0.ai"
        self.org_id = org_id
        self.project_id = project_id

        if not self.api_key:
            raise ValueError("Mem0 API Key not provided. Please provide an API Key.")

        self.user_id = hashlib.md5(self.api_key.encode(), usedforsecurity=False).hexdigest()

        if client is not None:
            self.async_client = client
            self.async_client.base_url = httpx.URL(self.host)
            self.async_client.headers.update(
                {
                    "Authorization": f"Token {self.api_key}",
                    "Mem0-User-ID": self.user_id,
                }
            )
        else:
            self.async_client = httpx.AsyncClient(
                base_url=self.host,
                headers={
                    "Authorization": f"Token {self.api_key}",
                    "Mem0-User-ID": self.user_id,
                },
                timeout=300,
            )

        self.user_email = self._validate_api_key()

        self.project = AsyncProject(
            client=self.async_client,
            org_id=self.org_id,
            project_id=self.project_id,
            user_email=self.user_email,
        )

        capture_client_event("client.init", self, {"sync_type": "async"})
