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

from datetime import UTC
from typing import Any

import pytest
from nat.builder.context import Context
from nat.memory.models import MemoryItem
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind
from opentelemetry.util._once import Once

from datarobot_genai.core.telemetry import memory
from datarobot_genai.dragent.plugins import datarobot_mem0_memory
from datarobot_genai.dragent.plugins.datarobot_mem0_memory import Config
from datarobot_genai.dragent.plugins.datarobot_mem0_memory import DRMem0Editor
from datarobot_genai.dragent.plugins.datarobot_mem0_memory import DRMem0MemoryClientConfig
from datarobot_genai.dragent.plugins.datarobot_mem0_memory import UnconfiguredMemoryEditor
from datarobot_genai.dragent.plugins.datarobot_mem0_memory import is_memory_editor_configured

# Matches ``DataRobotMemoryClient.user_id`` (= ``sha256(api_key)``). The editor
# falls back to this when no per-session user_id is supplied (e.g. direct calls
# outside the auto-memory wrapper).
FAKE_API_KEY_USER_ID = "fake-api-key-sha256"


@pytest.fixture
def memory_span_exporter(monkeypatch: pytest.MonkeyPatch) -> InMemorySpanExporter:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr("opentelemetry.trace._TRACER_PROVIDER", None)
    monkeypatch.setattr("opentelemetry.trace._TRACER_PROVIDER_SET_ONCE", Once())
    trace.set_tracer_provider(provider)
    monkeypatch.setattr(
        memory,
        "_tracer",
        trace.get_tracer("test.nat_mem0_memory"),
    )
    yield exporter
    exporter.clear()


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


async def test_add_items_injects_expiration_date_from_default_ttl() -> None:
    # GIVEN an editor configured with a positive TTL (e.g. AGENT_MEMORY_TTL_DAYS = 1).
    mem0 = FakeMem0Api()
    editor = DRMem0Editor(FakeMem0Client(mem0), ttl_days=1)
    item = MemoryItem(
        conversation=[{"role": "user", "content": "remember Python"}],
        user_id="session-user-123",
    )

    # WHEN the editor stores a memory.
    from datetime import datetime
    from datetime import timedelta

    before = datetime.now(UTC)
    await editor.add_items([item])
    after = datetime.now(UTC)

    # THEN the Mem0 call carries ``expiration_date = today + ttl`` as YYYY-MM-DD,
    # so the platform's expiration sweep will delete the memory after that date.
    valid_dates = {
        (before + timedelta(days=1)).strftime("%Y-%m-%d"),
        (after + timedelta(days=1)).strftime("%Y-%m-%d"),
    }
    assert mem0.add_calls[0]["kwargs"]["expiration_date"] in valid_dates


async def test_add_items_omits_expiration_date_when_no_default_ttl() -> None:
    # GIVEN an editor with no TTL configured (the default, backward-compatible path).
    mem0 = FakeMem0Api()
    editor = DRMem0Editor(FakeMem0Client(mem0))
    item = MemoryItem(
        conversation=[{"role": "user", "content": "remember Python"}],
        user_id="session-user-123",
    )

    # WHEN the editor stores a memory.
    await editor.add_items([item])

    # THEN no expiration_date is set — memories live forever, matching prior behavior.
    assert "expiration_date" not in mem0.add_calls[0]["kwargs"]


async def test_add_items_treats_ttl_zero_as_no_expiration() -> None:
    # GIVEN an editor explicitly configured with ttl=0 (the "no expiration" opt-out).
    mem0 = FakeMem0Api()
    editor = DRMem0Editor(FakeMem0Client(mem0), ttl_days=0)
    item = MemoryItem(
        conversation=[{"role": "user", "content": "remember Python"}],
        user_id="session-user-123",
    )

    # WHEN the editor stores a memory.
    await editor.add_items([item])

    # THEN ``expiration_date`` is unset — 0 means "no TTL", not "expire immediately".
    assert "expiration_date" not in mem0.add_calls[0]["kwargs"]


async def test_add_items_caller_supplied_expiration_date_beats_default_ttl() -> None:
    # GIVEN an editor with a configured TTL and a per-call expiration_date override.
    mem0 = FakeMem0Api()
    editor = DRMem0Editor(FakeMem0Client(mem0), ttl_days=1)
    item = MemoryItem(
        conversation=[{"role": "user", "content": "remember Python"}],
        user_id="session-user-123",
    )

    # WHEN the caller passes ``expiration_date`` explicitly (e.g. via
    # ``add_params`` in workflow.yaml) for a memory that needs a longer life.
    await editor.add_items([item], expiration_date="2099-12-31")

    # THEN the explicit override wins so callers can keep special memories
    # past the configured TTL.
    assert mem0.add_calls[0]["kwargs"]["expiration_date"] == "2099-12-31"


def test_ttl_to_expiration_date_returns_calendar_date() -> None:
    # GIVEN a TTL in days.
    # WHEN converted to mem0's expiration_date format.
    from datetime import datetime
    from datetime import timedelta

    from datarobot_genai.dragent.plugins.datarobot_mem0_memory import _ttl_to_expiration_date

    before = datetime.now(UTC)
    result = _ttl_to_expiration_date(7)
    after = datetime.now(UTC)

    # THEN the result is a YYYY-MM-DD string within +7 days of "now".
    # Accept either bound to avoid flakiness when the test straddles UTC midnight.
    valid_dates = {
        (before + timedelta(days=7)).strftime("%Y-%m-%d"),
        (after + timedelta(days=7)).strftime("%Y-%m-%d"),
    }
    assert result in valid_dates


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


def test_user_manager_shim_get_id_forwards_context_user_id() -> None:
    # GIVEN a Context whose ``user_id`` was set by ``DRAgentAGUISessionManager``
    # (which resolves the DR signed auth header via ``DRAgentUserManager``).
    class FakeContext:
        user_id = "uuid-from-dragent-session"

    shim = datarobot_mem0_memory._UserManagerShim(FakeContext())  # type: ignore[arg-type]

    # THEN ``get_id`` returns whatever the session manager wrote, so
    # ``auto_memory_wrapper`` keys memory on the resolved per-user identity.
    assert shim.get_id() == "uuid-from-dragent-session"


def test_user_manager_shim_get_id_returns_none_when_context_user_id_unset() -> None:
    # GIVEN a Context with no resolved user_id (e.g. ran outside a session).
    class FakeContext:
        user_id = None

    shim = datarobot_mem0_memory._UserManagerShim(FakeContext())  # type: ignore[arg-type]

    # THEN ``get_id`` returns None so the wrapper falls through to its default.
    assert shim.get_id() is None


def test_context_user_manager_property_is_patched() -> None:
    # GIVEN ``datarobot_mem0_memory`` is imported (this test module imports it).
    # THEN every ``Context`` exposes a ``user_manager`` attribute that is a shim.
    assert isinstance(Context.get().user_manager, datarobot_mem0_memory._UserManagerShim)


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


@pytest.mark.parametrize("memory_id", [None, ""])
async def test_remove_items_with_falsy_memory_id_still_deletes(memory_id: Any) -> None:
    mem0 = FakeMem0Api()
    editor = DRMem0Editor(FakeMem0Client(mem0))

    await editor.remove_items(memory_id=memory_id)

    assert mem0.delete_calls == [memory_id]
    assert mem0.delete_all_calls == []


async def test_registered_memory_client_forwards_default_ttl_to_editor(
    monkeypatch: Any,
) -> None:
    # GIVEN a config with a configured default_ttl_days and a stubbed mem0
    # client (the real one needs network access to validate the API key).
    monkeypatch.setattr(
        datarobot_mem0_memory, "_create_mem0_client", lambda *_a, **_k: FakeMem0Client()
    )
    monkeypatch.setattr(datarobot_mem0_memory, "patch_with_retry", lambda ed, **_: ed)

    config = DRMem0MemoryClientConfig(api_key="secret-key", default_ttl_days=7)

    # WHEN NAT builds the memory client.
    async with datarobot_mem0_memory.dr_mem0_memory_client(config, object()) as editor:
        # THEN the editor stores the TTL so subsequent add_items calls inject
        # expiration_date — without this hop the config value would be inert.
        assert isinstance(editor, DRMem0Editor)
        assert editor._ttl_days == 7


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


def test_memory_client_config_uses_settings_default_agent_memory_space_id(
    monkeypatch: Any,
) -> None:
    # GIVEN AGENT_MEMORY_SPACE_ID is set in the env by the recipe's agent runtime
    # parameter wiring.
    monkeypatch.setenv("AGENT_MEMORY_SPACE_ID", "space-from-runtime")
    monkeypatch.delenv("MEM0_API_KEY", raising=False)

    # WHEN the NAT memory config is created without an explicit agent_memory_space_id.
    config = DRMem0MemoryClientConfig(api_key=None)

    # THEN the agent_memory_space_id defaults from settings, so minimal workflow.yaml
    # memory blocks can still target the DataRobot Memory Service.
    assert Config().agent_memory_space_id == "space-from-runtime"
    assert config.agent_memory_space_id == "space-from-runtime"


def test_memory_client_config_explicit_agent_memory_space_id_beats_env(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("AGENT_MEMORY_SPACE_ID", "space-from-runtime")

    config = DRMem0MemoryClientConfig(api_key=None, agent_memory_space_id="space-explicit")

    assert config.agent_memory_space_id == "space-explicit"


def test_memory_client_config_uses_settings_default_ttl(monkeypatch: Any) -> None:
    # GIVEN AGENT_MEMORY_TTL_DAYS is set in the env (e.g. by infra/agent.py's
    # runtime parameter), exposing the recipe's configured retention to the
    # NAT memory editor.
    monkeypatch.setenv("AGENT_MEMORY_TTL_DAYS", "7")

    # WHEN the NAT memory config is created without an explicit default_ttl_days.
    config = DRMem0MemoryClientConfig(api_key="any")

    # THEN the field defaults from the AGENT_MEMORY_TTL_DAYS env var, so the
    # recipe's runtime parameter automatically reaches mem0 via expiration_date.
    assert config.default_ttl_days == 7


def test_memory_client_config_explicit_default_ttl_beats_env(monkeypatch: Any) -> None:
    # GIVEN AGENT_MEMORY_TTL_DAYS set in env but the workflow passing an
    # explicit override (e.g. a per-deployment retention different from the
    # global default).
    monkeypatch.setenv("AGENT_MEMORY_TTL_DAYS", "100")

    # WHEN the config is constructed with an explicit value.
    config = DRMem0MemoryClientConfig(api_key="any", default_ttl_days=42)

    # THEN the explicit value wins (env is only the default).
    assert config.default_ttl_days == 42


def test_memory_client_config_default_ttl_is_none_without_env(monkeypatch: Any) -> None:
    # GIVEN no AGENT_MEMORY_TTL_DAYS env var (e.g. a deployment that
    # opted out of TTL or never set the recipe's runtime parameter).
    monkeypatch.delenv("AGENT_MEMORY_TTL_DAYS", raising=False)

    # WHEN the config is constructed without an explicit override.
    config = DRMem0MemoryClientConfig(api_key="any")

    # THEN default_ttl_days is None — memories never expire, matching the
    # pre-TTL behavior so existing deployments don't silently change semantics.
    assert config.default_ttl_days is None


async def test_registered_memory_client_yields_unconfigured_editor_without_api_key(
    monkeypatch: Any,
) -> None:
    # GIVEN neither memory.api_key nor MEM0_API_KEY is configured.
    monkeypatch.delenv("MEM0_API_KEY", raising=False)
    monkeypatch.delenv("DATAROBOT_API_TOKEN", raising=False)

    # WHEN NAT builds the memory client, THEN it yields a no-op editor instead of failing.
    async with datarobot_mem0_memory.dr_mem0_memory_client(
        DRMem0MemoryClientConfig(api_key=None), object()
    ) as editor:
        assert isinstance(editor, UnconfiguredMemoryEditor)
        assert not is_memory_editor_configured(editor)
        await editor.add_items(
            [MemoryItem(conversation=[{"role": "user", "content": "hi"}], user_id="u1")]
        )
        assert await editor.search("hi", user_id="u1") == []


def test_dr_mem0_endpoint_builds_path_prefixed_url(monkeypatch: Any) -> None:
    # GIVEN an agent_memory_space_id and an explicit datarobot_endpoint.
    # WHEN the endpoint URL is built.
    # THEN it follows PBMP-7431's "API Layout": {endpoint}/memory/{id}, with no
    # trailing slash (mem0's _validate_api_key uses raw f"{host}/v1/ping/"
    # concat, so a trailing slash would produce a double slash). Any
    # caller-supplied trailing slash on the base is collapsed.
    config = DRMem0MemoryClientConfig(
        agent_memory_space_id="space-123",
        datarobot_endpoint="https://app.datarobot.com/api/v2/",
    )
    assert (
        datarobot_mem0_memory._dr_mem0_endpoint(config)
        == "https://app.datarobot.com/api/v2/memory/space-123"
    )


def test_dr_mem0_endpoint_falls_back_to_datarobot_endpoint_env(monkeypatch: Any) -> None:
    # GIVEN no explicit datarobot_endpoint on the config but DATAROBOT_ENDPOINT in env.
    monkeypatch.setenv("DATAROBOT_ENDPOINT", "https://staging.datarobot.com/api/v2")
    config = DRMem0MemoryClientConfig(agent_memory_space_id="space-xyz")

    # THEN the env var is used as the base.
    assert (
        datarobot_mem0_memory._dr_mem0_endpoint(config)
        == "https://staging.datarobot.com/api/v2/memory/space-xyz"
    )


def test_dr_mem0_endpoint_requires_a_base_url(monkeypatch: Any) -> None:
    # GIVEN no datarobot_endpoint configured and no env var.
    monkeypatch.delenv("DATAROBOT_ENDPOINT", raising=False)
    config = DRMem0MemoryClientConfig(agent_memory_space_id="space-xyz")

    # THEN the builder refuses to fabricate a URL — better to fail loud than to
    # point at a wrong host.
    with pytest.raises(RuntimeError, match="DATAROBOT_ENDPOINT"):
        datarobot_mem0_memory._dr_mem0_endpoint(config)


def test_create_mem0_client_routes_to_dr_endpoint_when_agent_memory_space_id_set(
    monkeypatch: Any,
) -> None:
    # GIVEN an agent_memory_space_id and a DR endpoint. This test exercises the real
    # ``_create_mem0_client`` body (host computation), which transitively
    # imports ``mem0``. Skip on minimal installs (the ``nat`` test module in
    # CI doesn't include the ``[memory]`` extra) — the factory-level tests
    # below already cover the routing decision by monkey-patching
    # ``_create_mem0_client`` directly.
    pytest.importorskip("mem0")

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

    import datarobot_genai.core.memory.mem0client as mem0client_module

    monkeypatch.setattr(mem0client_module, "Mem0Client", FakeMem0Client)

    config = DRMem0MemoryClientConfig(
        agent_memory_space_id="space-42",
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


def test_create_mem0_client_uses_config_host_when_no_agent_memory_space_id(
    monkeypatch: Any,
) -> None:
    # GIVEN no agent_memory_space_id (Mem0 SaaS path) and an explicit host override.
    # Same skip rationale as ``..._routes_to_dr_endpoint...``: this hits the
    # real ``_create_mem0_client`` body which imports ``mem0``.
    pytest.importorskip("mem0")

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


async def test_registered_memory_client_uses_dr_token_when_agent_memory_space_id_set(
    monkeypatch: Any,
) -> None:
    # GIVEN an agent_memory_space_id config and a DATAROBOT_API_TOKEN in env.
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
        agent_memory_space_id="space-42",
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
    # Clear MEM0_API_KEY so the api_key default_factory doesn't hydrate from
    # env and trip the mutually-exclusive guardrail (this is a DR-routing test,
    # we don't want the Mem0 SaaS field populated).
    monkeypatch.setenv("DATAROBOT_API_TOKEN", "env-token")
    monkeypatch.delenv("MEM0_API_KEY", raising=False)
    created: list[dict[str, Any]] = []

    def fake_create_mem0_client(config: DRMem0MemoryClientConfig, api_key: str | None) -> Any:
        created.append({"api_key": api_key})
        return FakeMem0Client()

    monkeypatch.setattr(datarobot_mem0_memory, "_create_mem0_client", fake_create_mem0_client)
    monkeypatch.setattr(datarobot_mem0_memory, "patch_with_retry", lambda ed, **_: ed)

    config = DRMem0MemoryClientConfig(
        agent_memory_space_id="space-42",
        datarobot_endpoint="https://app.datarobot.com/api/v2",
        datarobot_api_token="explicit-token",
    )

    # WHEN NAT builds the memory client.
    async with datarobot_mem0_memory.dr_mem0_memory_client(config, object()):
        pass

    # THEN the config field wins (env is only the fallback).
    assert created == [{"api_key": "explicit-token"}]


async def test_registered_memory_client_yields_unconfigured_editor_without_dr_token(
    monkeypatch: Any,
) -> None:
    # GIVEN an agent_memory_space_id but neither datarobot_api_token nor DATAROBOT_API_TOKEN.
    monkeypatch.delenv("DATAROBOT_API_TOKEN", raising=False)
    monkeypatch.delenv("MEM0_API_KEY", raising=False)

    async with datarobot_mem0_memory.dr_mem0_memory_client(
        DRMem0MemoryClientConfig(
            agent_memory_space_id="space-42",
            datarobot_endpoint="https://app.datarobot.com/api/v2",
        ),
        object(),
    ) as editor:
        assert isinstance(editor, UnconfiguredMemoryEditor)
        assert not is_memory_editor_configured(editor)


async def test_registered_memory_client_rejects_agent_memory_space_id_and_api_key_together(
    monkeypatch: Any,
) -> None:
    # GIVEN both agent_memory_space_id and api_key set (e.g. a config copied from a
    # Mem0-SaaS deployment that forgot to clear api_key after switching to DR,
    # or a stray MEM0_API_KEY in env hydrating api_key via its default factory).
    # The two fields point at different services with different auth tokens,
    # so silently picking one risks routing traffic to the wrong scope.

    # WHEN NAT builds the memory client, THEN it refuses to guess which one
    # the caller meant and raises a clear error that names both fields.
    with pytest.raises(RuntimeError, match="mutually exclusive"):
        async with datarobot_mem0_memory.dr_mem0_memory_client(
            DRMem0MemoryClientConfig(
                api_key="mem0-saas-key",
                agent_memory_space_id="space-42",
                datarobot_endpoint="https://app.datarobot.com/api/v2",
                datarobot_api_token="dr-token",
            ),
            object(),
        ):
            pass


async def test_registered_memory_client_rejects_agent_memory_space_id_with_env_mem0_key(
    monkeypatch: Any,
) -> None:
    # GIVEN MEM0_API_KEY contamination in env (e.g. another tool set it) and
    # an explicit agent_memory_space_id config. The default_factory will pick up
    # MEM0_API_KEY and populate api_key, creating an ambiguous config.
    monkeypatch.setenv("MEM0_API_KEY", "stray-env-key")
    monkeypatch.setenv("DATAROBOT_API_TOKEN", "dr-token")

    config = DRMem0MemoryClientConfig(agent_memory_space_id="space-42")
    # Confirm the env did hydrate api_key — this is the exact ambiguity the
    # guardrail exists to catch.
    assert config.api_key == "stray-env-key"

    # WHEN NAT builds the client, THEN the guardrail fires regardless of how
    # api_key got populated; the error message tells the user how to fix it
    # (pass api_key=None explicitly or unset MEM0_API_KEY).
    with pytest.raises(RuntimeError, match="api_key=None"):
        async with datarobot_mem0_memory.dr_mem0_memory_client(config, object()):
            pass


async def test_registered_memory_client_allows_agent_memory_space_id_with_explicit_null_api_key(
    monkeypatch: Any,
) -> None:
    # GIVEN MEM0_API_KEY in env but the caller explicitly passes api_key=None
    # to disambiguate — the documented escape hatch from the previous test.
    monkeypatch.setenv("MEM0_API_KEY", "stray-env-key")
    monkeypatch.setattr(
        datarobot_mem0_memory, "_create_mem0_client", lambda *_a, **_k: FakeMem0Client()
    )
    monkeypatch.setattr(datarobot_mem0_memory, "patch_with_retry", lambda ed, **_: ed)

    config = DRMem0MemoryClientConfig(
        api_key=None,  # explicit override beats the env default_factory
        agent_memory_space_id="space-42",
        datarobot_endpoint="https://app.datarobot.com/api/v2",
        datarobot_api_token="dr-token",
    )

    # WHEN NAT builds the client, THEN it routes to DR without complaint —
    # the explicit None signals the caller knows what they want.
    async with datarobot_mem0_memory.dr_mem0_memory_client(config, object()) as editor:
        assert isinstance(editor, DRMem0Editor)


async def test_registered_memory_client_sets_store_metadata_for_dr_route(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        datarobot_mem0_memory, "_create_mem0_client", lambda *_a, **_k: FakeMem0Client()
    )
    monkeypatch.setattr(datarobot_mem0_memory, "patch_with_retry", lambda ed, **_: ed)

    config = DRMem0MemoryClientConfig(
        agent_memory_space_id="space-42",
        datarobot_endpoint="https://app.datarobot.com/api/v2",
        datarobot_api_token="dr-token",
    )

    async with datarobot_mem0_memory.dr_mem0_memory_client(config, object()) as editor:
        assert editor._store_name == "datarobot-memory"
        assert editor._store_id == "space-42"


async def test_registered_memory_client_sets_store_metadata_for_mem0_saas(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        datarobot_mem0_memory, "_create_mem0_client", lambda *_a, **_k: FakeMem0Client()
    )
    monkeypatch.setattr(datarobot_mem0_memory, "patch_with_retry", lambda ed, **_: ed)

    config = DRMem0MemoryClientConfig(
        api_key="secret-key",
        org_id="org-123",
        project_id="project-456",
    )

    async with datarobot_mem0_memory.dr_mem0_memory_client(config, object()) as editor:
        assert editor._store_name == "mem0"
        assert editor._store_id == "project-456"


async def test_add_items_emits_update_memory_span(
    memory_span_exporter: InMemorySpanExporter,
) -> None:
    mem0 = FakeMem0Api()
    editor = DRMem0Editor(
        FakeMem0Client(mem0),
        store_name="datarobot-memory",
        store_id="space-42",
    )
    item = MemoryItem(
        conversation=[{"role": "user", "content": "remember Python"}],
        user_id="session-user-123",
    )

    await editor.add_items([item])

    span = memory_span_exporter.get_finished_spans()[0]
    assert span.name == "update_memory"
    assert span.kind == SpanKind.CLIENT
    assert span.attributes["gen_ai.operation.name"] == "update_memory"
    assert span.attributes["gen_ai.memory.store.name"] == "datarobot-memory"
    assert span.attributes["gen_ai.memory.store.id"] == "space-42"
    assert span.attributes["memory.item_count"] == 1
    assert span.attributes["memory.user_id"] == "session-user-123"


async def test_search_emits_search_memory_span(
    memory_span_exporter: InMemorySpanExporter,
) -> None:
    mem0 = FakeMem0Api()
    mem0.search_result = {
        "results": [
            {
                "input": [{"role": "user", "content": "I prefer Python"}],
                "memory": "User prefers Python.",
                "categories": ["preference"],
                "metadata": {},
            }
        ]
    }
    editor = DRMem0Editor(FakeMem0Client(mem0), store_name="mem0")

    await editor.search("language", top_k=2, user_id="session-user-123")

    span = memory_span_exporter.get_finished_spans()[0]
    assert span.name == "search_memory"
    assert span.attributes["gen_ai.operation.name"] == "search_memory"
    assert span.attributes["gen_ai.memory.query.text"] == "language"
    assert span.attributes["gen_ai.memory.search.result.count"] == 1
    assert span.attributes["memory.user_id"] == "session-user-123"


async def test_remove_items_emits_delete_memory_span(
    memory_span_exporter: InMemorySpanExporter,
) -> None:
    mem0 = FakeMem0Api()
    editor = DRMem0Editor(FakeMem0Client(mem0), store_name="mem0")

    await editor.remove_items(memory_id="memory-123")

    span = memory_span_exporter.get_finished_spans()[0]
    assert span.name == "delete_memory"
    assert span.attributes["gen_ai.operation.name"] == "delete_memory"
    assert span.attributes["gen_ai.memory.record.id"] == "memory-123"


async def test_remove_items_without_target_emits_no_span(
    memory_span_exporter: InMemorySpanExporter,
) -> None:
    mem0 = FakeMem0Api()
    editor = DRMem0Editor(FakeMem0Client(mem0), store_name="mem0")

    await editor.remove_items()

    assert not memory_span_exporter.get_finished_spans()


async def test_remove_items_with_falsy_memory_id_emits_delete_span_without_record_id(
    memory_span_exporter: InMemorySpanExporter,
) -> None:
    mem0 = FakeMem0Api()
    editor = DRMem0Editor(FakeMem0Client(mem0), store_name="mem0")

    await editor.remove_items(memory_id="")

    span = memory_span_exporter.get_finished_spans()[0]
    assert span.name == "delete_memory"
    assert "gen_ai.memory.record.id" not in span.attributes
