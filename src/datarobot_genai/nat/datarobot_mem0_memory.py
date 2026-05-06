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

"""NAT memory provider backed by DataRobot's Mem0 client.

This provider wires ``datarobot-genai[memory]`` into NAT's ``MemoryEditor``
interface so ``auto_memory_agent`` can store and retrieve long-term memory
without relying on the upstream ``nvidia-nat-mem0ai`` plugin.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

from nat.builder.builder import Builder
from nat.cli.register_workflow import register_memory
from nat.data_models.memory import MemoryBaseConfig
from nat.data_models.retry_mixin import RetryMixin
from nat.memory.interfaces import MemoryEditor
from nat.memory.models import MemoryItem
from nat.utils.exception_handlers.automatic_retries import patch_with_retry
from pydantic import Field


class DRMem0MemoryClientConfig(  # type: ignore[call-arg]
    MemoryBaseConfig,  # type: ignore[misc]
    RetryMixin,  # type: ignore[misc]
    name="dr_mem0_memory",
):
    """A NAT memory backend backed by ``datarobot-genai``'s Mem0 client."""

    api_key: str = Field(description="Mem0 API key used by the memory backend.")
    host: str | None = None
    org_id: str | None = None
    project_id: str | None = None


class DRMem0Editor(MemoryEditor):  # type: ignore[misc]
    """Adapt ``Mem0Client`` to NAT's ``MemoryEditor`` interface."""

    def __init__(self, client: Any) -> None:
        self._client = client
        self._mem0 = client._memory

    async def add_items(self, items: list[MemoryItem], **kwargs: Any) -> None:
        coroutines = []
        for item in items:
            metadata = dict(item.metadata or {})
            run_id = metadata.pop("run_id", None)
            coroutines.append(
                self._mem0.add(
                    item.conversation,
                    user_id=item.user_id,
                    run_id=run_id,
                    tags=item.tags,
                    metadata=metadata,
                    output_format="v1.1",
                    **kwargs,
                )
            )

        if coroutines:
            await asyncio.gather(*coroutines)

    async def search(self, query: str, top_k: int = 5, **kwargs: Any) -> list[MemoryItem]:
        # Mem0's v2 search endpoint expects filters instead of top-level user/run/app ids.
        user_id = kwargs.pop("user_id")
        conditions: list[dict[str, Any]] = [{"user_id": user_id}]
        for key in ("run_id", "agent_id", "app_id"):
            value = kwargs.pop(key, None)
            if value:
                conditions.append({key: value})

        metadata_filter = kwargs.pop("metadata", None)
        if metadata_filter:
            conditions.append({"metadata": metadata_filter})

        filters = kwargs.pop("filters", None) or {"AND": conditions}
        result = await self._mem0.search(
            query,
            filters=filters,
            top_k=top_k,
            output_format="v1.1",
            **kwargs,
        )

        memories: list[MemoryItem] = []
        for raw_result in result.get("results", []) or []:
            if not isinstance(raw_result, dict):
                continue
            item_meta = raw_result.get("metadata") or {}
            memories.append(
                MemoryItem(
                    conversation=raw_result.get("input") or [],
                    user_id=user_id,
                    memory=raw_result.get("memory"),
                    tags=raw_result.get("categories") or [],
                    metadata=item_meta,
                )
            )
        return memories

    async def remove_items(self, **kwargs: Any) -> None:
        if "memory_id" in kwargs:
            await self._mem0.delete(kwargs.pop("memory_id"))
        elif "user_id" in kwargs:
            await self._mem0.delete_all(user_id=kwargs.pop("user_id"))


def _create_mem0_client(config: DRMem0MemoryClientConfig, api_key: str) -> Any:
    try:
        from datarobot_genai.core.memory.mem0client import Mem0Client
    except ImportError as exc:
        raise RuntimeError(
            "The DataRobot Mem0 NAT memory provider requires the memory extra. "
            'Install it with `pip install "datarobot-genai[nat,memory]"`.'
        ) from exc

    return Mem0Client(
        api_key=api_key,
        host=config.host,
        org_id=config.org_id,
        project_id=config.project_id,
    )


@register_memory(config_type=DRMem0MemoryClientConfig)
async def dr_mem0_memory_client(
    config: DRMem0MemoryClientConfig, builder: Builder
) -> AsyncGenerator[MemoryEditor]:
    editor: MemoryEditor = DRMem0Editor(_create_mem0_client(config, config.api_key))
    if isinstance(config, RetryMixin):
        editor = patch_with_retry(
            editor,
            retries=config.num_retries,
            retry_codes=config.retry_on_status_codes,
            retry_on_messages=config.retry_on_errors,
        )

    yield editor
