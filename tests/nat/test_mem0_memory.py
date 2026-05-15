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

from __future__ import annotations

from typing import Any

import pytest
from nat.builder.context import Context
from nat.memory.models import MemoryItem

from datarobot_genai.nat import datarobot_mem0_memory
from datarobot_genai.nat.datarobot_mem0_memory import Config
from datarobot_genai.nat.datarobot_mem0_memory import DRMem0Editor
from datarobot_genai.nat.datarobot_mem0_memory import DRMem0MemoryClientConfig

# Matches ``DataRobotMemoryClient.user_id`` (= ``sha256(api_key)``); the editor
# pins every add/search to this so add and search share scope.
FAKE_API_KEY_USER_ID = "fake-api-key-sha256"


class FakeMem0Api:
    def __init__(self, user_id: str = FAKE_API_KEY_USER_ID) -> None:
        self.user_id = user_id
        self.add_calls: list[dict[str, Any]] = []
        self.search_calls: list[dict[str, Any]] = []
        self.delete_calls: list[str] = []
        self.delete_all_calls: list[dict[str, Any]] = []
        self.search_result: dict[str, Any] = {"results": []}

    async def add(self, conversation: list[dict[str, str]], **kwargs: Any) -> None:
        self.add_calls.append({"conversation": conversation, "kwargs": kwargs})

    async def search(self, query: str, **kwargs: Any) -> dict[str, Any]:
        self.search_calls.append({"query": query, "kwargs": kwargs})
        return self.search_result

    async def delete(self, memory_id: str) -> None:
        self.delete_calls.append(memory_id)

    async def delete_all(self, **kwargs: Any) -> None:
        self.delete_all_calls.append(kwargs)


class FakeMem0Client:
    def __init__(self, mem0: FakeMem0Api | None = None) -> None:
        self._memory = mem0 or FakeMem0Api()


async def test_add_items_pins_user_id_to_mem0_client() -> None:
    # GIVEN a NAT MemoryItem whose user_id differs from the Mem0 client's.
    mem0 = FakeMem0Api()
    editor = DRMem0Editor(FakeMem0Client(mem0))
    item = MemoryItem(
        conversation=[{"role": "user", "content": "remember Python"}],
        user_id="user-123",
        tags=["preference"],
        metadata={"run_id": "run-456", "thread_id": "thread-789"},
    )

    # WHEN the editor stores it.
    await editor.add_items([item], agent_id="agent-abc")

    # THEN the Mem0 call uses ``DataRobotMemoryClient.user_id`` (api-key owner),
    # ignoring the wrapper-supplied ``MemoryItem.user_id``, while still forwarding
    # tags, metadata, and v1.1 payload shape.
    assert mem0.add_calls == [
        {
            "conversation": [{"role": "user", "content": "remember Python"}],
            "kwargs": {
                "user_id": FAKE_API_KEY_USER_ID,
                "run_id": "run-456",
                "tags": ["preference"],
                "metadata": {"thread_id": "thread-789"},
                "output_format": "v1.1",
                "agent_id": "agent-abc",
            },
        }
    ]


async def test_add_items_pins_user_id_even_with_conflicting_add_params() -> None:
    # GIVEN add_params and a MemoryItem that both set user_id.
    mem0 = FakeMem0Api()
    editor = DRMem0Editor(FakeMem0Client(mem0))
    item = MemoryItem(
        conversation=[{"role": "user", "content": "remember TypeScript"}],
        user_id="item-user",
        tags=["item-tag"],
        metadata={"run_id": "item-run", "thread_id": "item-thread"},
    )

    # WHEN the editor stores it with conflicting backend-specific kwargs.
    await editor.add_items(
        [item],
        user_id="configured-user",
        run_id="configured-run",
        tags=["configured-tag"],
        metadata={"thread_id": "configured-thread", "source": "workflow"},
        output_format="custom-format",
        async_mode=False,
    )

    # THEN the editor's pinned user_id wins over BOTH the kwarg and the item,
    # other NAT item fields still take precedence over configured add_params.
    assert mem0.add_calls == [
        {
            "conversation": [{"role": "user", "content": "remember TypeScript"}],
            "kwargs": {
                "user_id": FAKE_API_KEY_USER_ID,
                "run_id": "item-run",
                "tags": ["item-tag"],
                "metadata": {"thread_id": "item-thread", "source": "workflow"},
                "output_format": "custom-format",
                "async_mode": False,
            },
        }
    ]


async def test_search_builds_mem0_v2_filters_pinned_to_client_user_id() -> None:
    # GIVEN a Mem0 search result and NAT auto-memory user/run/app parameters.
    mem0 = FakeMem0Api()
    mem0.search_result = {
        "results": [
            {
                "input": [{"role": "user", "content": "I prefer Python"}],
                "memory": "User prefers Python.",
                "categories": ["preference"],
                "metadata": {"thread_id": "thread-789"},
            }
        ]
    }
    editor = DRMem0Editor(FakeMem0Client(mem0))

    # WHEN the editor searches memory.
    memories = await editor.search(
        "language",
        top_k=2,
        user_id="user-123",
        run_id="run-456",
        app_id="app-abc",
        metadata={"thread_id": "thread-789"},
        output_format="custom-format",
        threshold=0.75,
    )

    # THEN the search filters the v2 endpoint by the editor's pinned user_id
    # (api-key owner), ignoring the wrapper-supplied ``user_id`` kwarg, and the
    # returned NAT MemoryItem inherits that same pinned user_id.
    assert mem0.search_calls == [
        {
            "query": "language",
            "kwargs": {
                "filters": {
                    "AND": [
                        {"user_id": FAKE_API_KEY_USER_ID},
                        {"run_id": "run-456"},
                        {"app_id": "app-abc"},
                        {"metadata": {"thread_id": "thread-789"}},
                    ]
                },
                "top_k": 2,
                "output_format": "custom-format",
                "threshold": 0.75,
            },
        }
    ]
    assert memories == [
        MemoryItem(
            conversation=[{"role": "user", "content": "I prefer Python"}],
            user_id=FAKE_API_KEY_USER_ID,
            memory="User prefers Python.",
            tags=["preference"],
            metadata={"thread_id": "thread-789"},
        )
    ]


async def test_search_honors_explicit_filters() -> None:
    # GIVEN caller-provided Mem0 filters.
    mem0 = FakeMem0Api()
    editor = DRMem0Editor(FakeMem0Client(mem0))
    filters = {"AND": [{"user_id": "user-123"}, {"agent_id": "agent-abc"}]}

    # WHEN the editor searches with explicit filters.
    await editor.search("question", user_id="user-123", filters=filters)

    # THEN those filters are forwarded without rebuilding them.
    assert mem0.search_calls[0]["kwargs"]["filters"] == filters


def test_user_manager_shim_get_id_returns_none() -> None:
    # GIVEN the shim installed by ``datarobot_mem0_memory`` for NAT 1.6.
    shim = datarobot_mem0_memory._UserManagerShim()

    # THEN ``get_id`` returns None — the editor pins the real user_id, the shim
    # only exists to keep ``auto_memory_wrapper`` from raising AttributeError.
    assert shim.get_id() is None


def test_context_user_manager_property_is_patched() -> None:
    # GIVEN ``datarobot_mem0_memory`` is imported (this test module imports it).
    # THEN every ``Context`` exposes a ``user_manager`` attribute that is a shim.
    assert isinstance(Context.get().user_manager, datarobot_mem0_memory._UserManagerShim)


async def test_remove_items_deletes_by_memory_id_or_user_id() -> None:
    # GIVEN a Mem0-backed editor.
    mem0 = FakeMem0Api()
    editor = DRMem0Editor(FakeMem0Client(mem0))

    # WHEN deleting a single memory and then all memories for a user.
    await editor.remove_items(memory_id="memory-123")
    await editor.remove_items(user_id="user-123")

    # THEN the corresponding Mem0 delete APIs are used.
    assert mem0.delete_calls == ["memory-123"]
    assert mem0.delete_all_calls == [{"user_id": "user-123"}]


async def test_registered_memory_client_uses_config_api_key_and_retry(monkeypatch: Any) -> None:
    # GIVEN a configured NAT memory provider with an explicit Mem0 API key.
    created_clients: list[dict[str, Any]] = []
    patched_editors: list[dict[str, Any]] = []

    def fake_create_mem0_client(
        config: DRMem0MemoryClientConfig, api_key: str | None
    ) -> FakeMem0Client:
        created_clients.append({"config": config, "api_key": api_key})
        return FakeMem0Client()

    def fake_patch_with_retry(editor: Any, **kwargs: Any) -> Any:
        patched_editors.append({"editor": editor, "kwargs": kwargs})
        return editor

    monkeypatch.setattr(datarobot_mem0_memory, "_create_mem0_client", fake_create_mem0_client)
    monkeypatch.setattr(datarobot_mem0_memory, "patch_with_retry", fake_patch_with_retry)

    config = DRMem0MemoryClientConfig(
        api_key="secret-key",
        host="https://mem0.example.com",
        org_id="org-123",
        project_id="project-123",
    )

    # WHEN NAT builds the memory client.
    async with datarobot_mem0_memory.dr_mem0_memory_client(config, object()) as editor:
        # THEN it creates a DRMem0Editor with the configured key and retry wrapper.
        assert isinstance(editor, DRMem0Editor)

    assert created_clients == [{"config": config, "api_key": "secret-key"}]
    assert patched_editors == [
        {
            "editor": editor,
            "kwargs": {
                "retries": config.num_retries,
                "retry_codes": config.retry_on_status_codes,
                "retry_on_messages": config.retry_on_errors,
            },
        }
    ]


def test_memory_client_config_uses_settings_mem0_api_key(monkeypatch: Any) -> None:
    # GIVEN MEM0_API_KEY is available from DataRobot app framework settings.
    monkeypatch.setenv("MEM0_API_KEY", "settings-key")

    # WHEN the NAT memory config is created without an explicit api_key.
    config = DRMem0MemoryClientConfig()

    # THEN the api_key defaults from the settings config.
    assert Config().mem0_api_key == "settings-key"
    assert config.api_key == "settings-key"


async def test_registered_memory_client_requires_api_key(monkeypatch: Any) -> None:
    # GIVEN neither memory.api_key nor MEM0_API_KEY is configured.
    monkeypatch.delenv("MEM0_API_KEY", raising=False)

    # WHEN NAT builds the memory client, THEN it raises a clear configuration error.
    with pytest.raises(RuntimeError, match="MEM0_API_KEY"):
        async with datarobot_mem0_memory.dr_mem0_memory_client(
            DRMem0MemoryClientConfig(api_key=None), object()
        ):
            pass
