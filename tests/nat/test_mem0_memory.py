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

# Matches ``DataRobotMemoryClient.user_id`` (= ``sha256(api_key)``). The editor
# falls back to this when no per-session user_id is supplied (e.g. direct calls
# outside the auto-memory wrapper).
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


async def test_add_items_forwards_per_session_user_id_from_memory_item() -> None:
    # GIVEN a NAT MemoryItem whose user_id was set by ``auto_memory_wrapper``
    # from ``Context.user_id`` (= the per-session user id).
    mem0 = FakeMem0Api()
    editor = DRMem0Editor(FakeMem0Client(mem0))
    item = MemoryItem(
        conversation=[{"role": "user", "content": "remember Python"}],
        user_id="session-user-123",
        tags=["preference"],
        metadata={"run_id": "run-456", "thread_id": "thread-789"},
    )

    # WHEN the editor stores it.
    await editor.add_items([item], agent_id="agent-abc")

    # THEN the Mem0 call uses the MemoryItem's user_id so memories are scoped
    # per session, not globally per api key.
    assert mem0.add_calls == [
        {
            "conversation": [{"role": "user", "content": "remember Python"}],
            "kwargs": {
                "user_id": "session-user-123",
                "run_id": "run-456",
                "tags": ["preference"],
                "metadata": {"thread_id": "thread-789"},
                "output_format": "v1.1",
                "agent_id": "agent-abc",
            },
        }
    ]


async def test_add_items_falls_back_to_mem0_user_id_when_no_session_user_id() -> None:
    # GIVEN a MemoryItem with an empty user_id and no configured user_id kwarg
    # (direct call outside the wrapper, no session context).
    mem0 = FakeMem0Api()
    editor = DRMem0Editor(FakeMem0Client(mem0))
    item = MemoryItem(
        conversation=[{"role": "user", "content": "remember TypeScript"}],
        user_id="",
        tags=["preference"],
    )

    # WHEN the editor stores it.
    await editor.add_items([item])

    # THEN the api-key owner is used as a last-resort fallback so the call
    # still succeeds (Mem0 requires a user_id).
    assert mem0.add_calls[0]["kwargs"]["user_id"] == FAKE_API_KEY_USER_ID


async def test_add_items_item_user_id_beats_configured_user_id() -> None:
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

    # THEN the MemoryItem's per-session user_id wins over the configured kwarg
    # (the wrapper's identity is authoritative); other NAT item fields still
    # take precedence over configured add_params.
    assert mem0.add_calls == [
        {
            "conversation": [{"role": "user", "content": "remember TypeScript"}],
            "kwargs": {
                "user_id": "item-user",
                "run_id": "item-run",
                "tags": ["item-tag"],
                "metadata": {"thread_id": "item-thread", "source": "workflow"},
                "output_format": "custom-format",
                "async_mode": False,
            },
        }
    ]


async def test_search_builds_mem0_v2_filters_with_session_user_id() -> None:
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

    # WHEN the editor searches memory using the per-session user_id.
    memories = await editor.search(
        "language",
        top_k=2,
        user_id="session-user-123",
        run_id="run-456",
        app_id="app-abc",
        metadata={"thread_id": "thread-789"},
        output_format="custom-format",
        threshold=0.75,
    )

    # THEN the v2 endpoint filters by the per-session user_id, and the returned
    # NAT MemoryItem carries that same user_id so add and search share scope.
    assert mem0.search_calls == [
        {
            "query": "language",
            "kwargs": {
                "filters": {
                    "AND": [
                        {"user_id": "session-user-123"},
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
            user_id="session-user-123",
            memory="User prefers Python.",
            tags=["preference"],
            metadata={"thread_id": "thread-789"},
        )
    ]


async def test_search_falls_back_to_mem0_user_id_when_no_session_user_id() -> None:
    # GIVEN a search with no user_id (direct call outside the wrapper).
    mem0 = FakeMem0Api()
    editor = DRMem0Editor(FakeMem0Client(mem0))

    # WHEN the editor searches.
    await editor.search("language", top_k=2)

    # THEN the api-key owner is used as a last-resort fallback so the call
    # still succeeds (Mem0 v2 filters require user_id).
    assert mem0.search_calls[0]["kwargs"]["filters"] == {"AND": [{"user_id": FAKE_API_KEY_USER_ID}]}


async def test_search_honors_explicit_filters() -> None:
    # GIVEN caller-provided Mem0 filters.
    mem0 = FakeMem0Api()
    editor = DRMem0Editor(FakeMem0Client(mem0))
    filters = {"AND": [{"user_id": "user-123"}, {"agent_id": "agent-abc"}]}

    # WHEN the editor searches with explicit filters.
    await editor.search("question", user_id="user-123", filters=filters)

    # THEN those filters are forwarded without rebuilding them.
    assert mem0.search_calls[0]["kwargs"]["filters"] == filters


def test_user_manager_shim_get_id_returns_dr_memory_user_uuid(monkeypatch: Any) -> None:
    # GIVEN the shim installed by ``datarobot_mem0_memory`` for NAT 1.6 and a
    # request whose DR auth header decodes to a known UUIDv5.
    monkeypatch.setattr(datarobot_mem0_memory, "_memory_user_uuid", lambda: "uuid-from-dr-auth")
    shim = datarobot_mem0_memory._UserManagerShim(object())  # type: ignore[arg-type]

    # THEN ``get_id`` returns the canonical DR-user UUID so ``auto_memory_wrapper``
    # passes a stable per-user identity to the editor.
    assert shim.get_id() == "uuid-from-dr-auth"


def test_user_manager_shim_get_id_returns_none_when_dr_auth_header_missing(
    monkeypatch: Any,
) -> None:
    # GIVEN no DR auth header on the current request (e.g. local dev).
    monkeypatch.setattr(datarobot_mem0_memory, "_memory_user_uuid", lambda: None)
    shim = datarobot_mem0_memory._UserManagerShim(object())  # type: ignore[arg-type]

    # THEN ``get_id`` returns None so the wrapper falls through to its default
    # and the editor falls back to the api-key owner.
    assert shim.get_id() is None


def test_context_user_manager_property_is_patched() -> None:
    # GIVEN ``datarobot_mem0_memory`` is imported (this test module imports it).
    # THEN every ``Context`` exposes a ``user_manager`` attribute that is a shim.
    assert isinstance(Context.get().user_manager, datarobot_mem0_memory._UserManagerShim)


def test_memory_user_uuid_returns_uuid_for_decodable_dr_auth_header(
    monkeypatch: Any,
) -> None:
    # GIVEN a request whose headers carry a decodable DR auth context.
    class FakeUser:
        id = "datarobot-user-abc"

    class FakeAuthCtx:
        user = FakeUser()

    class FakeHandler:
        def get_context(self, headers: dict[str, str]) -> Any:
            assert headers == {"x-datarobot-authorization-context": "encoded-token"}
            return FakeAuthCtx()

    class FakeMetadata:
        headers = {"x-datarobot-authorization-context": "encoded-token"}

    class FakeContextInstance:
        metadata = FakeMetadata()

    monkeypatch.setattr(datarobot_mem0_memory, "AuthContextHeaderHandler", FakeHandler)
    monkeypatch.setattr(datarobot_mem0_memory.Context, "get", staticmethod(FakeContextInstance))

    # WHEN we resolve the memory user id.
    user_id = datarobot_mem0_memory._memory_user_uuid()

    # THEN it matches what NAT's ``UserInfo._from_session_cookie`` derives — a
    # UUIDv5 over the raw DR user id (the raw id never leaves the process).
    from nat.data_models.user_info import UserInfo

    assert user_id == UserInfo._from_session_cookie("datarobot-user-abc").get_user_id()


def test_memory_user_uuid_returns_none_when_no_metadata(monkeypatch: Any) -> None:
    # GIVEN a Context with no metadata at all.
    class FakeContextInstance:
        metadata = None

    monkeypatch.setattr(datarobot_mem0_memory.Context, "get", staticmethod(FakeContextInstance))

    # WHEN we resolve the memory user id.
    # THEN it returns None — caller will fall back to the api-key owner.
    assert datarobot_mem0_memory._memory_user_uuid() is None


def test_memory_user_uuid_returns_none_when_handler_returns_none(
    monkeypatch: Any,
) -> None:
    # GIVEN a request whose DR auth header is missing/undecodable (handler returns None).
    class FakeHandler:
        def get_context(self, headers: dict[str, str]) -> Any:
            return None

    class FakeMetadata:
        headers = {"other-header": "value"}

    class FakeContextInstance:
        metadata = FakeMetadata()

    monkeypatch.setattr(datarobot_mem0_memory, "AuthContextHeaderHandler", FakeHandler)
    monkeypatch.setattr(datarobot_mem0_memory.Context, "get", staticmethod(FakeContextInstance))

    # WHEN we resolve the memory user id.
    # THEN it returns None — caller will fall back to the api-key owner.
    assert datarobot_mem0_memory._memory_user_uuid() is None


def test_memory_user_uuid_returns_none_when_handler_raises(monkeypatch: Any) -> None:
    # GIVEN a handler that crashes (e.g. SESSION_SECRET_KEY unset in local dev).
    class FakeHandler:
        def get_context(self, headers: dict[str, str]) -> Any:
            raise RuntimeError("SESSION_SECRET_KEY not set")

    class FakeMetadata:
        headers = {"x-datarobot-authorization-context": "token"}

    class FakeContextInstance:
        metadata = FakeMetadata()

    monkeypatch.setattr(datarobot_mem0_memory, "AuthContextHeaderHandler", FakeHandler)
    monkeypatch.setattr(datarobot_mem0_memory.Context, "get", staticmethod(FakeContextInstance))

    # WHEN we resolve the memory user id.
    # THEN it swallows the error and returns None so memory writes don't crash
    # the request — caller falls back to the api-key owner.
    assert datarobot_mem0_memory._memory_user_uuid() is None


async def test_remove_items_deletes_by_memory_id_or_session_user_id() -> None:
    # GIVEN a Mem0-backed editor.
    mem0 = FakeMem0Api()
    editor = DRMem0Editor(FakeMem0Client(mem0))

    # WHEN deleting a single memory and then all memories for a session user.
    await editor.remove_items(memory_id="memory-123")
    await editor.remove_items(user_id="session-user-123")

    # THEN delete-by-id and delete_all both target the caller-supplied scope,
    # so a delete targeting a session user_id matches what add/search wrote.
    assert mem0.delete_calls == ["memory-123"]
    assert mem0.delete_all_calls == [{"user_id": "session-user-123"}]


async def test_remove_items_without_memory_id_or_user_id_is_a_noop() -> None:
    # GIVEN a Mem0-backed editor.
    mem0 = FakeMem0Api()
    editor = DRMem0Editor(FakeMem0Client(mem0))

    # WHEN remove_items is called with no targeting args.
    await editor.remove_items()

    # THEN no Mem0 delete API is invoked — we never wipe a scope implicitly.
    assert mem0.delete_calls == []
    assert mem0.delete_all_calls == []


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


def test_dr_mem0_endpoint_builds_path_prefixed_url(monkeypatch: Any) -> None:
    # GIVEN a memory_space_id and an explicit datarobot_endpoint.
    # WHEN the endpoint URL is built.
    # THEN it follows PBMP-7431's "API Layout": {endpoint}/memory/{id}, with no
    # trailing slash (mem0's _validate_api_key uses raw f"{host}/v1/ping/"
    # concat, so a trailing slash would produce a double slash). Any
    # caller-supplied trailing slash on the base is collapsed.
    config = DRMem0MemoryClientConfig(
        memory_space_id="space-123",
        datarobot_endpoint="https://app.datarobot.com/api/v2/",
    )
    assert (
        datarobot_mem0_memory._dr_mem0_endpoint(config)
        == "https://app.datarobot.com/api/v2/memory/space-123"
    )


def test_dr_mem0_endpoint_falls_back_to_datarobot_endpoint_env(monkeypatch: Any) -> None:
    # GIVEN no explicit datarobot_endpoint on the config but DATAROBOT_ENDPOINT in env.
    monkeypatch.setenv("DATAROBOT_ENDPOINT", "https://staging.datarobot.com/api/v2")
    config = DRMem0MemoryClientConfig(memory_space_id="space-xyz")

    # THEN the env var is used as the base.
    assert (
        datarobot_mem0_memory._dr_mem0_endpoint(config)
        == "https://staging.datarobot.com/api/v2/memory/space-xyz"
    )


def test_dr_mem0_endpoint_requires_a_base_url(monkeypatch: Any) -> None:
    # GIVEN no datarobot_endpoint configured and no env var.
    monkeypatch.delenv("DATAROBOT_ENDPOINT", raising=False)
    config = DRMem0MemoryClientConfig(memory_space_id="space-xyz")

    # THEN the builder refuses to fabricate a URL — better to fail loud than to
    # point at a wrong host.
    with pytest.raises(RuntimeError, match="DATAROBOT_ENDPOINT"):
        datarobot_mem0_memory._dr_mem0_endpoint(config)


def test_create_mem0_client_routes_to_dr_endpoint_when_memory_space_id_set(
    monkeypatch: Any,
) -> None:
    # GIVEN a memory_space_id and a DR endpoint.
    captured: dict[str, Any] = {}

    class FakeMem0Client:
        def __init__(
            self,
            api_key: str | None = None,
            host: str | None = None,
            org_id: str | None = None,
            project_id: str | None = None,
        ) -> None:
            captured.update(
                {"api_key": api_key, "host": host, "org_id": org_id, "project_id": project_id}
            )

    # Stub the import inside the function so we don't need the [memory] extra installed.
    import datarobot_genai.core.memory.mem0client as mem0client_module

    monkeypatch.setattr(mem0client_module, "Mem0Client", FakeMem0Client)

    config = DRMem0MemoryClientConfig(
        memory_space_id="space-42",
        datarobot_endpoint="https://app.datarobot.com/api/v2",
        org_id="org-1",
        project_id="proj-1",
    )

    # WHEN the Mem0Client is constructed.
    datarobot_mem0_memory._create_mem0_client(config, "dr-token")

    # THEN it points at the path-prefixed DR memory endpoint with the DR token
    # (not Mem0 SaaS), and forwards org/project as-is.
    assert captured == {
        "api_key": "dr-token",
        "host": "https://app.datarobot.com/api/v2/memory/space-42",
        "org_id": "org-1",
        "project_id": "proj-1",
    }


def test_create_mem0_client_uses_config_host_when_no_memory_space_id(
    monkeypatch: Any,
) -> None:
    # GIVEN no memory_space_id (Mem0 SaaS path) and an explicit host override.
    captured: dict[str, Any] = {}

    class FakeMem0Client:
        def __init__(
            self,
            api_key: str | None = None,
            host: str | None = None,
            org_id: str | None = None,
            project_id: str | None = None,
        ) -> None:
            captured.update({"api_key": api_key, "host": host})

    import datarobot_genai.core.memory.mem0client as mem0client_module

    monkeypatch.setattr(mem0client_module, "Mem0Client", FakeMem0Client)
    config = DRMem0MemoryClientConfig(api_key="mem0-key", host="https://mem0.example.com")

    # WHEN the Mem0Client is constructed.
    datarobot_mem0_memory._create_mem0_client(config, "mem0-key")

    # THEN the SaaS host wins (no DR prefix injection).
    assert captured == {"api_key": "mem0-key", "host": "https://mem0.example.com"}


async def test_registered_memory_client_uses_dr_token_when_memory_space_id_set(
    monkeypatch: Any,
) -> None:
    # GIVEN a memory_space_id config and a DATAROBOT_API_TOKEN in env.
    monkeypatch.setenv("DATAROBOT_API_TOKEN", "dr-secret-token")
    monkeypatch.delenv("MEM0_API_KEY", raising=False)

    created: list[dict[str, Any]] = []

    def fake_create_mem0_client(config: DRMem0MemoryClientConfig, api_key: str | None) -> Any:
        created.append({"config": config, "api_key": api_key})
        return FakeMem0Client()

    monkeypatch.setattr(datarobot_mem0_memory, "_create_mem0_client", fake_create_mem0_client)
    monkeypatch.setattr(datarobot_mem0_memory, "patch_with_retry", lambda ed, **_: ed)

    config = DRMem0MemoryClientConfig(
        api_key=None,
        memory_space_id="space-42",
        datarobot_endpoint="https://app.datarobot.com/api/v2",
    )

    # WHEN NAT builds the memory client.
    async with datarobot_mem0_memory.dr_mem0_memory_client(config, object()) as editor:
        assert isinstance(editor, DRMem0Editor)

    # THEN the DR token (not the mem0 api_key) is used as the auth credential
    # against the DR mem0 endpoint.
    assert created == [{"config": config, "api_key": "dr-secret-token"}]


async def test_registered_memory_client_prefers_explicit_dr_token_over_env(
    monkeypatch: Any,
) -> None:
    # GIVEN an explicit datarobot_api_token on the config and a different env var.
    monkeypatch.setenv("DATAROBOT_API_TOKEN", "env-token")
    created: list[dict[str, Any]] = []

    def fake_create_mem0_client(config: DRMem0MemoryClientConfig, api_key: str | None) -> Any:
        created.append({"api_key": api_key})
        return FakeMem0Client()

    monkeypatch.setattr(datarobot_mem0_memory, "_create_mem0_client", fake_create_mem0_client)
    monkeypatch.setattr(datarobot_mem0_memory, "patch_with_retry", lambda ed, **_: ed)

    config = DRMem0MemoryClientConfig(
        memory_space_id="space-42",
        datarobot_endpoint="https://app.datarobot.com/api/v2",
        datarobot_api_token="explicit-token",
    )

    # WHEN NAT builds the memory client.
    async with datarobot_mem0_memory.dr_mem0_memory_client(config, object()):
        pass

    # THEN the config field wins (env is only the fallback).
    assert created == [{"api_key": "explicit-token"}]


async def test_registered_memory_client_requires_dr_token_when_memory_space_id_set(
    monkeypatch: Any,
) -> None:
    # GIVEN a memory_space_id but neither datarobot_api_token nor DATAROBOT_API_TOKEN.
    monkeypatch.delenv("DATAROBOT_API_TOKEN", raising=False)

    # WHEN NAT builds the memory client, THEN it raises a DR-specific error so
    # users know to set the DR token, not the Mem0 api_key.
    with pytest.raises(RuntimeError, match="DATAROBOT_API_TOKEN"):
        async with datarobot_mem0_memory.dr_mem0_memory_client(
            DRMem0MemoryClientConfig(
                memory_space_id="space-42",
                datarobot_endpoint="https://app.datarobot.com/api/v2",
            ),
            object(),
        ):
            pass
