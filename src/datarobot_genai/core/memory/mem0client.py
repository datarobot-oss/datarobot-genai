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
        prompt: str,
        run_id: str | None = None,
        agent_id: str | None = None,
        app_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> str:
        conditions: list[dict[str, Any]] = [{"user_id": self._memory.user_id}]
        for key, value in (
            ("run_id", run_id),
            ("agent_id", agent_id),
            ("app_id", app_id),
        ):
            if value:
                conditions.append({key: value})

        if attributes:
            conditions.append({"metadata": attributes})

        filters = {"AND": conditions}

        result = await self._memory.search(query=prompt, filters=filters)
        return self._format_search_result(result)

    async def store(
        self,
        user_message: str,
        run_id: str | None = None,
        agent_id: str | None = None,
        app_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> None:

        messages = [{"role": "user", "content": user_message}]

        kwargs: dict[str, Any] = {
            "version": "v1",
            "output_format": "v1.1",
        }
        if attributes:
            kwargs["metadata"] = attributes

        kwargs["user_id"] = self._memory.user_id
        if run_id:
            kwargs["run_id"] = run_id
        if agent_id:
            kwargs["agent_id"] = agent_id
        if app_id:
            kwargs["app_id"] = app_id

        await self._memory.add(messages, **kwargs)

    @staticmethod
    def _format_search_result(result: Any) -> str:
        if isinstance(result, str):
            return result

        items = result.get("results", []) if isinstance(result, dict) else result
        if not isinstance(items, list):
            return str(items)

        texts: list[str] = []
        for item in items:
            if not isinstance(item, dict):
                texts.append(str(item))
                continue

            for key in ("memory", "text", "content"):
                value = item.get(key)
                if value:
                    texts.append(str(value))
                    break

        return "\n".join(dict.fromkeys(texts))
