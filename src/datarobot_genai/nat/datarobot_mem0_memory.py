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

from datarobot.core.config import DataRobotAppFrameworkBaseSettings
from nat.builder.builder import Builder
from nat.builder.context import Context
from nat.cli.register_workflow import register_memory
from nat.data_models.memory import MemoryBaseConfig
from nat.data_models.retry_mixin import RetryMixin
from nat.memory.interfaces import MemoryEditor
from nat.memory.models import MemoryItem
from nat.utils.exception_handlers.automatic_retries import patch_with_retry
from pydantic import Field


class Config(DataRobotAppFrameworkBaseSettings):
    """
    Finds variables in the priority order of: env
    variables (including Runtime Parameters), .env, file_secrets, then
    Pulumi output variables.
    """

    mem0_api_key: str | None = None


def _get_default_mem0_api_key() -> str | None:
    return Config().mem0_api_key


class _UserManagerShim:
    """Stub that prevents ``AttributeError`` on NAT 1.6.

    ``nvidia-nat-langchain`` 1.6.0's ``auto_memory_wrapper`` accesses
    ``Context.user_manager.get_id()`` to resolve a user_id, but
    ``Context.user_manager`` only exists on newer NAT versions.
    Returning ``None`` lets the wrapper's fallback chain run without
    raising; the value it eventually picks is overridden by
    :class:`DRMem0Editor`, which pins every add and search to
    ``DataRobotMemoryClient.user_id`` (= ``sha256(MEM0_API_KEY)``).
    """

    def get_id(self) -> str | None:
        return None


if not hasattr(Context, "user_manager"):
    Context.user_manager = property(  # type: ignore[attr-defined]
        lambda self: _UserManagerShim()
    )


class DRMem0MemoryClientConfig(  # type: ignore[call-arg]
    MemoryBaseConfig,  # type: ignore[misc]
    RetryMixin,  # type: ignore[misc]
    name="dr_mem0_memory",
):
    """A NAT memory backend backed by ``datarobot-genai``'s Mem0 client."""

    api_key: str | None = Field(
        default_factory=_get_default_mem0_api_key,
        description="Mem0 API key used by the memory backend.",
    )
    host: str | None = None
    org_id: str | None = None
    project_id: str | None = None


class DRMem0Editor(MemoryEditor):  # type: ignore[misc]
    """Adapt ``Mem0Client`` to NAT's ``MemoryEditor`` interface."""

    def __init__(self, client: Any) -> None:
        self._client = client
        self._mem0 = client._memory

    async def add_items(self, items: list[MemoryItem], **kwargs: Any) -> None:
        add_kwargs = dict(kwargs)
        output_format = add_kwargs.pop("output_format", "v1.1")
        configured_run_id = add_kwargs.pop("run_id", None)
        configured_tags = add_kwargs.pop("tags", None)
        configured_metadata = dict(add_kwargs.pop("metadata", None) or {})
        add_kwargs.pop("user_id", None)

        # Pin every write to the API-key owner so add and search share scope.
        # See ``DataRobotMemoryClient.user_id`` (= ``sha256(api_key)``).
        user_id = self._mem0.user_id

        coroutines = []
        for item in items:
            metadata = configured_metadata | dict(item.metadata or {})
            run_id = metadata.pop("run_id", configured_run_id)
            item_kwargs = add_kwargs | {
                "user_id": user_id,
                "tags": item.tags or configured_tags or [],
                "metadata": metadata,
                "output_format": output_format,
            }
            if run_id:
                item_kwargs["run_id"] = run_id
            coroutines.append(self._mem0.add(item.conversation, **item_kwargs))

        if coroutines:
            await asyncio.gather(*coroutines)

    async def search(self, query: str, top_k: int = 5, **kwargs: Any) -> list[MemoryItem]:
        # Mem0's v2 search endpoint expects filters instead of top-level user/run/app ids.
        # Pin to the API-key owner; see ``add_items`` for rationale.
        kwargs.pop("user_id", None)
        user_id = self._mem0.user_id
        conditions: list[dict[str, Any]] = [{"user_id": user_id}]
        for key in ("run_id", "agent_id", "app_id"):
            value = kwargs.pop(key, None)
            if value:
                conditions.append({key: value})

        metadata_filter = kwargs.pop("metadata", None)
        if metadata_filter:
            conditions.append({"metadata": metadata_filter})

        filters = kwargs.pop("filters", None) or {"AND": conditions}
        output_format = kwargs.pop("output_format", "v1.1")
        result = await self._mem0.search(
            query,
            filters=filters,
            top_k=top_k,
            output_format=output_format,
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
            kwargs.pop("user_id")
            await self._mem0.delete_all(user_id=self._mem0.user_id)


def _create_mem0_client(config: DRMem0MemoryClientConfig, api_key: str | None) -> Any:
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
    if not config.api_key:
        raise RuntimeError(
            "Mem0 API key is not set. Please configure memory.api_key or MEM0_API_KEY."
        )

    editor: MemoryEditor = DRMem0Editor(_create_mem0_client(config, config.api_key))
    if isinstance(config, RetryMixin):
        editor = patch_with_retry(
            editor,
            retries=config.num_retries,
            retry_codes=config.retry_on_status_codes,
            retry_on_messages=config.retry_on_errors,
        )

    yield editor
