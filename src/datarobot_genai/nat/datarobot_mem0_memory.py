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

* ``agent_memory_space_id`` → the DataRobot Memory Service's mem0-compatible API,
  reached at ``{DATAROBOT_ENDPOINT}/memory/{agent_memory_space_id}/`` and
  authenticated with the DataRobot API token. See PBMP-7431 ("Agentic Memory
  Service"), section "Connect to DataRobot memory" / "API Layout".
* ``api_key`` (or ``MEM0_API_KEY``) → Mem0's hosted SaaS at
  ``https://api.mem0.ai``. Used when no ``agent_memory_space_id`` is configured.

Both routes share the same ``MemoryEditor`` adapter because the DR endpoint
is API-compatible with mem0 — only the host and auth token differ.

When neither route is configured (no ``agent_memory_space_id`` + token and no
``api_key`` / ``MEM0_API_KEY``), the provider yields an
:class:`UnconfiguredMemoryEditor` so workflows can declare ``dr_mem0_memory``
unconditionally and enable memory later via runtime parameters or env vars.
"""

import asyncio
import logging
import os
from collections.abc import AsyncGenerator
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from typing import Any

import httpx
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
from pydantic import model_validator

from datarobot_genai.core.telemetry_memory import trace_memory_operation
from datarobot_genai.core.telemetry_memory import truncate_memory_text

logger = logging.getLogger(__name__)


class Config(DataRobotAppFrameworkBaseSettings):
    """
    Finds variables in the priority order of: env
    variables (including Runtime Parameters), .env, file_secrets, then
    Pulumi output variables.
    """

    mem0_api_key: str | None = None
    agent_memory_space_id: str | None = None
    agent_memory_ttl_seconds: int | None = None
    agent_llm_model_name: str | None = None


def _get_default_memory_backend_config() -> Config:
    return Config()


def _get_default_mem0_api_key_for_memory_backend() -> str | None:
    config = _get_default_memory_backend_config()
    if config.agent_memory_space_id:
        return None
    return config.mem0_api_key


def _get_default_agent_memory_space_id() -> str | None:
    return _get_default_memory_backend_config().agent_memory_space_id


def _get_default_ttl_seconds() -> int | None:
    return Config().agent_memory_ttl_seconds


def _get_default_llm_model_name() -> str | None:
    return _get_default_memory_backend_config().agent_llm_model_name


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

    * If ``agent_memory_space_id`` is set, requests go to the DataRobot Memory
      Service's mem0-compatible endpoint at
      ``{datarobot_endpoint}/memory/{agent_memory_space_id}/`` authenticated with
      ``datarobot_api_token`` (or ``DATAROBOT_API_TOKEN``).
    * Otherwise, ``api_key`` (or ``MEM0_API_KEY``) is used against Mem0's
      hosted SaaS (``host`` defaults to ``https://api.mem0.ai``).
    """

    api_key: str | None = Field(
        default_factory=_get_default_mem0_api_key_for_memory_backend,
        description="Mem0 API key used when targeting Mem0's hosted SaaS.",
    )
    host: str | None = Field(
        default=None,
        description=(
            "Mem0 base URL for the SaaS backend. Ignored when ``agent_memory_space_id`` is set."
        ),
    )
    org_id: str | None = None
    project_id: str | None = None
    agent_memory_space_id: str | None = Field(
        default_factory=_get_default_agent_memory_space_id,
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
            "``agent_memory_space_id`` is set (e.g. ``https://app.datarobot.com/api/v2``). "
            "Defaults to the ``DATAROBOT_ENDPOINT`` env var."
        ),
    )
    datarobot_api_token: str | None = Field(
        default=None,
        description=(
            "DataRobot API token used when ``agent_memory_space_id`` is set. "
            "Defaults to the ``DATAROBOT_API_TOKEN`` env var."
        ),
    )
    llm_model_name: str | None = Field(
        default_factory=_get_default_llm_model_name,
        description=(
            "LLM model name to use for memory extraction in the DataRobot Memory Service. "
            "Required when ``agent_memory_space_id`` is set — the memory service no longer "
            "provides a built-in default. "
            "When set, the memory space is updated to this model on editor initialization. "
            "Defaults from the ``AGENT_LLM_MODEL_NAME`` env var."
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

    @model_validator(mode="after")
    def _require_llm_model_name_for_dr_backend(self) -> "DRMem0MemoryClientConfig":
        if self.agent_memory_space_id and not self.llm_model_name:
            raise ValueError(
                "llm_model_name is required when agent_memory_space_id is set. "
                "The DataRobot Memory Service no longer provides a built-in default. "
                "Set it via the llm_model_name config field or the "
                "AGENT_LLM_MODEL_NAME environment variable."
            )
        return self


class UnconfiguredMemoryEditor(MemoryEditor):  # type: ignore[misc]
    """No-op memory backend returned when ``dr_mem0_memory`` has no credentials."""

    async def add_items(self, items: list[MemoryItem], **kwargs: Any) -> None:
        return

    async def search(self, query: str, top_k: int = 5, **kwargs: Any) -> list[MemoryItem]:
        return []

    async def remove_items(self, **kwargs: Any) -> None:
        return


def is_memory_editor_configured(editor: MemoryEditor) -> bool:
    """Return ``False`` when ``dr_mem0_memory`` yielded an unconfigured no-op editor."""
    return not isinstance(editor, UnconfiguredMemoryEditor)


class DRMem0Editor(MemoryEditor):  # type: ignore[misc]
    """Adapt ``Mem0Client`` to NAT's ``MemoryEditor`` interface."""

    def __init__(
        self,
        client: Any,
        ttl_seconds: int | None = None,
        *,
        store_name: str = "mem0",
        store_id: str | None = None,
    ) -> None:
        self._client = client
        self._mem0 = client._memory
        self._ttl_seconds = ttl_seconds
        self._store_name = store_name
        self._store_id = store_id

    def _memory_span_attributes(
        self,
        *,
        user_id: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        attrs: dict[str, Any] = dict(extra or {})
        if user_id:
            attrs["gen_ai.memory.scope"] = "user"
            attrs["memory.user_id"] = user_id
        return attrs

    async def add_items(self, items: list[MemoryItem], **kwargs: Any) -> None:
        configured_user_id = kwargs.get("user_id")
        resolved_user_ids = {
            item.user_id or configured_user_id or self._mem0.user_id for item in items
        }
        user_id = next(iter(resolved_user_ids)) if len(resolved_user_ids) == 1 else None

        with trace_memory_operation(
            "update_memory",
            store_name=self._store_name,
            store_id=self._store_id,
            attributes=self._memory_span_attributes(
                user_id=user_id,
                extra={"memory.item_count": len(items)},
            ),
        ):
            await self._add_items_impl(items, **kwargs)

    async def _add_items_impl(self, items: list[MemoryItem], **kwargs: Any) -> None:
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
        user_id = kwargs.get("user_id") or self._mem0.user_id
        with trace_memory_operation(
            "search_memory",
            store_name=self._store_name,
            store_id=self._store_id,
            attributes=self._memory_span_attributes(
                user_id=user_id,
                extra={
                    "gen_ai.memory.query.text": truncate_memory_text(query),
                    "memory.top_k": top_k,
                },
            ),
        ) as span:
            memories = await self._search_impl(query, top_k=top_k, **kwargs)
            span.set_attribute("gen_ai.memory.search.result.count", len(memories))
            return memories

    async def _search_impl(self, query: str, top_k: int = 5, **kwargs: Any) -> list[MemoryItem]:
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
        has_memory_id = "memory_id" in kwargs
        user_id = kwargs.get("user_id")
        if not has_memory_id and not user_id:
            return

        memory_id = kwargs.get("memory_id") if has_memory_id else None
        with trace_memory_operation(
            "delete_memory",
            store_name=self._store_name,
            store_id=self._store_id,
            attributes=self._memory_span_attributes(
                user_id=user_id,
                extra={
                    **({"gen_ai.memory.record.id": memory_id} if memory_id else {}),
                    **({"memory.delete_all": True} if user_id and not has_memory_id else {}),
                },
            ),
        ):
            await self._remove_items_impl(**kwargs)

    async def _remove_items_impl(self, **kwargs: Any) -> None:
        if "memory_id" in kwargs:
            await self._mem0.delete(kwargs.pop("memory_id"))
            return
        user_id = kwargs.pop("user_id", None)
        if user_id:
            await self._mem0.delete_all(user_id=user_id)


async def _patch_memory_space_llm_name(config: DRMem0MemoryClientConfig) -> None:
    """PATCH the DataRobot Memory Space to set llm_model_name.

    The memory service no longer ships a default LLM name. When a workflow
    configures ``llm_model_name`` alongside ``agent_memory_space_id``, this
    function ensures the memory space reflects that name before any add/search
    calls are made, so the service can instantiate ``DataRobotMemoryConfig``
    successfully.
    """
    base = config.datarobot_endpoint or os.getenv("DATAROBOT_ENDPOINT")
    api_token = config.datarobot_api_token or os.getenv("DATAROBOT_API_TOKEN")
    if not base or not api_token:
        logger.warning(
            "Cannot update memory space llm_model_name: missing datarobot_endpoint or api_token"
        )
        return

    url = f"{base.rstrip('/')}/memory/{config.agent_memory_space_id}/"
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.patch(
            url,
            json={"llmModelName": config.llm_model_name},
            headers={"Authorization": f"Token {api_token}"},
        )
    if resp.status_code not in (200, 204):
        raise RuntimeError(
            f"Failed to update memory space llm_model_name: HTTP {resp.status_code} — {resp.text}"
        )


def _dr_mem0_endpoint(config: DRMem0MemoryClientConfig) -> str:
    """Build the DataRobot Memory Service mem0 endpoint for an agent memory space.

    Per PBMP-7431 ("API Layout"), the DR memory service exposes a
    mem0-compatible API at ``{DATAROBOT_ENDPOINT}/memory/{agent_memory_space_id}``.

    No trailing slash: mem0's ``AsyncMemoryClient._validate_api_key`` builds
    its ping URL as ``f"{host}/v1/ping/"`` via raw string concat (it does not
    use httpx's base-url joining for that call). A trailing slash on the
    host would produce a double slash there.
    """
    base = config.datarobot_endpoint or os.getenv("DATAROBOT_ENDPOINT")
    if not base:
        raise RuntimeError(
            "DataRobot endpoint is not set. Configure memory.datarobot_endpoint "
            "or DATAROBOT_ENDPOINT when using agent_memory_space_id."
        )
    return f"{base.rstrip('/')}/memory/{config.agent_memory_space_id}"


def _create_mem0_client(config: DRMem0MemoryClientConfig, api_key: str | None) -> Any:
    # Belt-and-suspenders: mem0 reads MEM0_TELEMETRY at import time.
    os.environ["MEM0_TELEMETRY"] = "False"
    try:
        from datarobot_genai.core.memory.mem0client import Mem0Client
    except ImportError as exc:
        raise RuntimeError(
            "The DataRobot Mem0 NAT memory provider requires the memory extra. "
            'Install it with `pip install "datarobot-genai[nat,memory]"`.'
        ) from exc

    host = _dr_mem0_endpoint(config) if config.agent_memory_space_id else config.host
    return Mem0Client(
        api_key=api_key,
        host=host,
        org_id=config.org_id,
        project_id=config.project_id,
    )


def _resolve_memory_backend(
    config: DRMem0MemoryClientConfig,
) -> tuple[str, str, str | None] | None:
    """Return ``(api_key, store_name, store_id)`` when configured, else ``None``."""
    if config.agent_memory_space_id:
        api_key = config.datarobot_api_token or os.getenv("DATAROBOT_API_TOKEN")
        if not api_key:
            return None
        return (api_key, "datarobot-memory", config.agent_memory_space_id)

    if config.api_key:
        return (config.api_key, "mem0", config.project_id or config.org_id)

    return None


@register_memory(config_type=DRMem0MemoryClientConfig)
async def dr_mem0_memory_client(
    config: DRMem0MemoryClientConfig, builder: Builder
) -> AsyncGenerator[MemoryEditor]:
    if config.agent_memory_space_id and config.api_key:
        # These point at different services with different tokens. Silently
        # picking one masks misconfiguration — e.g. a config copied from a
        # Mem0-SaaS deployment that left ``api_key`` populated, or a stray
        # ``MEM0_API_KEY`` in env hydrating ``api_key`` via its default
        # factory. Force the caller to disambiguate.
        raise RuntimeError(
            "agent_memory_space_id and api_key are mutually exclusive: they target "
            "different services (DataRobot Memory Service vs. Mem0 SaaS) with "
            "different auth tokens. Set exactly one. If MEM0_API_KEY is in env, "
            "either unset it or pass api_key=None explicitly when using agent_memory_space_id."
        )

    if config.agent_memory_space_id:
        await _patch_memory_space_llm_name(config)

    resolved = _resolve_memory_backend(config)
    if resolved is None:
        logger.info(
            "dr_mem0_memory: no memory backend configured "
            "(set agent_memory_space_id + DATAROBOT_API_TOKEN, or api_key / MEM0_API_KEY); "
            "memory operations are disabled"
        )
        yield UnconfiguredMemoryEditor()
        return

    api_key, store_name, store_id = resolved

    editor: MemoryEditor = DRMem0Editor(
        _create_mem0_client(config, api_key),
        ttl_seconds=config.default_ttl_seconds,
        store_name=store_name,
        store_id=store_id,
    )
    if isinstance(config, RetryMixin):
        editor = patch_with_retry(
            editor,
            retries=config.num_retries,
            retry_codes=config.retry_on_status_codes,
            retry_on_messages=config.retry_on_errors,
        )

    yield editor
