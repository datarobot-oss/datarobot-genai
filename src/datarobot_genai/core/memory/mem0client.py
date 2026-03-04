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

from typing import Any

import httpx

from .base import BaseMemoryClient
from .datarobot_memory_client import DataRobotMemoryClient


class Mem0Client(BaseMemoryClient):
    def __init__(
        self,
        api_key: str | None = None,
        host: str | None = None,
        org_id: str | None = None,
        project_id: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._memory = DataRobotMemoryClient(
            api_key=api_key,
            host=host,
            org_id=org_id,
            project_id=project_id,
            client=client,
        )

    async def retrieve(
        self,
        user_id: str,
        prompt: str,
        attributes: dict[str, Any] | None = None,
    ) -> str:
        conditions = [{"user_id": user_id}]

        if attributes:
            conditions.extend({k: v} for k, v in attributes.items())

        filters = {"AND": conditions}

        return await self._memory.search(
            query=prompt, filters=filters, version="v2", output_format="v1.1"
        )

    async def store(
        self,
        user_id: str,
        user_message: str,
        attributes: dict[str, Any] | None = None,
    ) -> None:

        messages = [{"role": "user", "content": user_message}]

        kwargs: dict[str, Any] = {
            "user_id": user_id,
            "version": "v2",
            "output_format": "v1.1",
        }

        if attributes:
            kwargs.update(attributes)

        await self._memory.add(messages, **kwargs)
