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
interface so ``auto_memory_agent`` can store and retrieve long-term memory.

Backend selection is driven by config:

* ``memory_space_id`` → the DataRobot Memory Service's mem0-compatible API,
  reached at ``{DATAROBOT_ENDPOINT}/memory/{memory_space_id}/`` and
  authenticated with the DataRobot API token. See PBMP-7431 ("Agentic Memory
  Service"), section "Connect to DataRobot memory" / "API Layout".
* ``api_key`` (or ``MEM0_API_KEY``) → Mem0's hosted SaaS at
  ``https://api.mem0.ai``. Used when no ``memory_space_id`` is configured.

Both routes share the same ``MemoryEditor`` adapter because the DR endpoint
is API-compatible with mem0 — only the host and auth token differ.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncGenerator
from datetime import UTC
from datetime import datetime
from datetime import timedelta
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

logger = logging.getLogger(__name__)


class Config(DataRobotAppFrameworkBaseSettings):
    """
    Finds variables in the priority order of: env
    variables (including Runtime Parameters), .env, file_secrets, then
    Pulumi output variables.
    """

    mem0_api_key: str | None = None
    agent_memory_ttl_seconds: int | None = None


def _get_default_mem0_api_key() -> str | None:
    return Config().mem0_api_key


def _get_default_ttl_seconds() -> int | None:
    return Config().agent_memory_ttl_seconds


def _ttl_to_expiration_date(ttl_seconds: int) -> str:
    """Translate a TTL in seconds into Mem0's ``expiration_date`` format.

    Mem0's REST ``add`` endpoint expects ``expiration_date`` as a
    ``YYYY-MM-DD`` calendar date (not a timestamp); the platform's
    expiration sweep deletes memories on or after that date. Sub-day TTLs
    are rounded up to "today" by the calendar floor — callers needing
    finer granularity should pass ``expiration_date`` explicitly.
    """
    return (datetime.now(UTC) + timedelta(seconds=ttl_seconds)).strftime("%Y-%m-%d")


class _UserManagerShim:
    """Bridge ``Context.user_manager.get_id()`` to ``Context.user_id`` on NAT 1.6.

    ``nvidia-nat-langchain`` 1.6.0's ``auto_memory_wrapper`` reads
    ``Context.user_manager.get_id()``, but NAT 1.6 doesn't expose
    ``user_manager`` on ``Context``. Identity resolution itself already happens
    upstream in :class:`DRAgentAGUISessionManager` (decodes
    ``X-DataRobot-Authorization-Context`` via ``DRAgentUserManager`` and stores
    the resolved id on ``ContextState.user_id``), so this shim just forwards
    that value. Falls through to the per-user-workflow ``default-user``
    constant when no identity is available.
    """

    def __init__(self, context: Context) -> None:
        self._context = context

    def get_id(self) -> str | None:
        return self._context.user_id


if not hasattr(Context, "user_manager"):
    Context.user_manager = property(_UserManagerShim)  # type: ignore[attr-defined]


class DRMem0MemoryClientConfig(  # type: ignore[call-arg]
    MemoryBaseConfig,  # type: ignore[misc]
    RetryMixin,  # type: ignore[misc]
    name="dr_mem0_memory",
):
    """A NAT memory backend backed by ``datarobot-genai``'s Mem0 client.

    Backend selection:

    * If ``memory_space_id`` is set, requests go to the DataRobot Memory
      Service's mem0-compatible endpoint at
      ``{datarobot_endpoint}/memory/{memory_space_id}/`` authenticated with
      ``datarobot_api_token`` (or ``DATAROBOT_API_TOKEN``).
    * Otherwise, ``api_key`` (or ``MEM0_API_KEY``) is used against Mem0's
      hosted SaaS (``host`` defaults to ``https://api.mem0.ai``).
    """

    api_key: str | None = Field(
        default_factory=_get_default_mem0_api_key,
        description="Mem0 API key used when targeting Mem0's hosted SaaS.",
    )
    host: str | None = Field(
        default=None,
        description=(
            "Mem0 base URL for the SaaS backend. Ignored when ``memory_space_id`` is set."
        ),
    )
    org_id: str | None = None
    project_id: str | None = None
    memory_space_id: str | None = Field(
        default=None,
        description=(
            "DataRobot MemorySpace ID. When set, the editor uses the DataRobot "
            "Memory Service's mem0-compatible endpoint instead of Mem0 SaaS. "
            "The endpoint is built as ``{datarobot_endpoint}/memory/{id}/``."
        ),
    )
    datarobot_endpoint: str | None = Field(
        default=None,
        description=(
            "DataRobot API base URL used to build the mem0 endpoint when "
            "``memory_space_id`` is set (e.g. ``https://app.datarobot.com/api/v2``). "
            "Defaults to the ``DATAROBOT_ENDPOINT`` env var."
        ),
    )
    datarobot_api_token: str | None = Field(
        default=None,
        description=(
            "DataRobot API token used when ``memory_space_id`` is set. "
            "Defaults to the ``DATAROBOT_API_TOKEN`` env var."
        ),
    )
    default_ttl_seconds: int | None = Field(
        default_factory=_get_default_ttl_seconds,
        ge=0,
        description=(
            "Default TTL in seconds for stored memories. When set to a "
            "positive value, the editor passes "
            "``expiration_date = today + default_ttl_seconds`` (UTC, "
            "``YYYY-MM-DD``) through to Mem0's ``add`` API so memories "
            "auto-expire. Callers may override per-call by passing "
            "``expiration_date`` in ``add_params``. ``None`` or ``0`` "
            "leaves the field unset (no expiration). Defaults from the "
            "``AGENT_MEMORY_TTL_SECONDS`` env var."
        ),
    )


class DRMem0Editor(MemoryEditor):  # type: ignore[misc]
    """Adapt ``Mem0Client`` to NAT's ``MemoryEditor`` interface."""

    def __init__(self, client: Any, ttl_seconds: int | None = None) -> None:
        self._client = client
        self._mem0 = client._memory
        self._ttl_seconds = ttl_seconds

    async def add_items(self, items: list[MemoryItem], **kwargs: Any) -> None:
        add_kwargs = dict(kwargs)
        output_format = add_kwargs.pop("output_format", "v1.1")
        configured_run_id = add_kwargs.pop("run_id", None)
        configured_tags = add_kwargs.pop("tags", None)
        configured_metadata = dict(add_kwargs.pop("metadata", None) or {})
        configured_user_id = add_kwargs.pop("user_id", None)

        # Inject the configured TTL as Mem0's ``expiration_date`` only when
        # the caller hasn't supplied one. Per-call overrides (e.g. via
        # ``add_params: {expiration_date: ...}`` in workflow.yaml) win so
        # special-case memories can opt out of or extend the default.
        if "expiration_date" not in add_kwargs and self._ttl_seconds:
            add_kwargs["expiration_date"] = _ttl_to_expiration_date(self._ttl_seconds)

        coroutines = []
        for item in items:
            metadata = configured_metadata | dict(item.metadata or {})
            run_id = metadata.pop("run_id", configured_run_id)
            user_id = item.user_id or configured_user_id or self._mem0.user_id
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
        user_id = kwargs.pop("user_id", None) or self._mem0.user_id
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
            return
        user_id = kwargs.pop("user_id", None)
        if user_id:
            await self._mem0.delete_all(user_id=user_id)


def _dr_mem0_endpoint(config: DRMem0MemoryClientConfig) -> str:
    """Build the DataRobot Memory Service mem0 endpoint for a memory space.

    Per PBMP-7431 ("API Layout"), the DR memory service exposes a
    mem0-compatible API at ``{DATAROBOT_ENDPOINT}/memory/{memory_space_id}``.

    No trailing slash: mem0's ``AsyncMemoryClient._validate_api_key`` builds
    its ping URL as ``f"{host}/v1/ping/"`` via raw string concat (it does not
    use httpx's base-url joining for that call). A trailing slash on the
    host would produce a double slash there.
    """
    base = config.datarobot_endpoint or os.getenv("DATAROBOT_ENDPOINT")
    if not base:
        raise RuntimeError(
            "DataRobot endpoint is not set. Configure memory.datarobot_endpoint "
            "or DATAROBOT_ENDPOINT when using memory_space_id."
        )
    return f"{base.rstrip('/')}/memory/{config.memory_space_id}"


def _create_mem0_client(config: DRMem0MemoryClientConfig, api_key: str | None) -> Any:
    try:
        from datarobot_genai.core.memory.mem0client import Mem0Client
    except ImportError as exc:
        raise RuntimeError(
            "The DataRobot Mem0 NAT memory provider requires the memory extra. "
            'Install it with `pip install "datarobot-genai[nat,memory]"`.'
        ) from exc

    host = _dr_mem0_endpoint(config) if config.memory_space_id else config.host
    return Mem0Client(
        api_key=api_key,
        host=host,
        org_id=config.org_id,
        project_id=config.project_id,
    )


@register_memory(config_type=DRMem0MemoryClientConfig)
async def dr_mem0_memory_client(
    config: DRMem0MemoryClientConfig, builder: Builder
) -> AsyncGenerator[MemoryEditor]:
    if config.memory_space_id and config.api_key:
        # These point at different services with different tokens. Silently
        # picking one masks misconfiguration — e.g. a config copied from a
        # Mem0-SaaS deployment that left ``api_key`` populated, or a stray
        # ``MEM0_API_KEY`` in env hydrating ``api_key`` via its default
        # factory. Force the caller to disambiguate.
        raise RuntimeError(
            "memory_space_id and api_key are mutually exclusive: they target "
            "different services (DataRobot Memory Service vs. Mem0 SaaS) with "
            "different auth tokens. Set exactly one. If MEM0_API_KEY is in env, "
            "either unset it or pass api_key=None explicitly when using memory_space_id."
        )

    if config.memory_space_id:
        api_key = config.datarobot_api_token or os.getenv("DATAROBOT_API_TOKEN")
        if not api_key:
            raise RuntimeError(
                "DataRobot API token is not set. Configure memory.datarobot_api_token "
                "or DATAROBOT_API_TOKEN when using memory_space_id."
            )
    elif config.api_key:
        api_key = config.api_key
    else:
        raise RuntimeError(
            "Mem0 API key is not set. Please configure memory.api_key or MEM0_API_KEY, "
            "or set memory_space_id to target the DataRobot mem0 endpoint."
        )

    editor: MemoryEditor = DRMem0Editor(
        _create_mem0_client(config, api_key),
        ttl_seconds=config.default_ttl_seconds,
    )
    if isinstance(config, RetryMixin):
        editor = patch_with_retry(
            editor,
            retries=config.num_retries,
            retry_codes=config.retry_on_status_codes,
            retry_on_messages=config.retry_on_errors,
        )

    yield editor
