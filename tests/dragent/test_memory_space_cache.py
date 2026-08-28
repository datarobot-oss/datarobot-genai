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

import json
from typing import Any

import httpx
import pytest
import respx
from datarobot.application_utils.persistence import DRMemoryServiceClient
from datarobot.application_utils.persistence.markers import SYSTEM_PARTICIPANT

from datarobot_genai.dragent.memory_space_cache import MemorySpaceKVCache
from datarobot_genai.dragent.memory_space_cache import resolve_memory_space_id
from datarobot_genai.dragent.memory_space_cache import try_build_memory_service_client
from datarobot_genai.dragent.memory_space_cache import try_resolve_memory_space_id

BASE = "https://app.datarobot.com/api/v2"
MEMORY_BASE = f"{BASE}/memory"
SPACE_ID = "space-1"
SESSIONS_URL = f"{MEMORY_BASE}/{SPACE_ID}/sessions/"
SESSION_ID = "sess-1"
SESSION_URL = f"{SESSIONS_URL}{SESSION_ID}/"
EVENTS_URL = f"{SESSION_URL[:-1]}/events/"
LOGICAL_KEY = "dragent:agent_card:dep-1"


def _session_wire(
    *,
    session_id: str = SESSION_ID,
    dedup_key: str = LOGICAL_KEY,
    version: int = 1,
) -> dict[str, Any]:
    return {
        "id": session_id,
        "participants": [SYSTEM_PARTICIPANT],
        "description": "//dragent-cache/",
        "deduplicationKey": dedup_key,
        "metadata": {},
        "lifecycleStrategies": [],
        "version": version,
        "createdAt": "2026-06-30T00:00:00Z",
    }


def _event_wire(*, sequence_id: int = 0, content: str = "payload") -> dict[str, Any]:
    return {
        "sequenceId": sequence_id,
        "createdAt": "2026-06-30T00:00:01Z",
        "eventType": "status",
        "emitterType": "agent",
        "emitterId": None,
        "body": {"content": content},
    }


def _items(*wires: dict[str, Any]) -> dict[str, Any]:
    return {"items": list(wires), "total": len(wires)}


def _client() -> DRMemoryServiceClient:
    return DRMemoryServiceClient(
        endpoint=BASE,
        api_token="test-token",
        http_client=httpx.AsyncClient(),
    )


@pytest.fixture
def kv_cache() -> MemorySpaceKVCache:
    return MemorySpaceKVCache(memory_space_id=SPACE_ID, client=_client())


class TestResolveMemorySpaceId:
    def test_explicit_id(self) -> None:
        assert resolve_memory_space_id("space-explicit") == "space-explicit"

    def test_missing_id_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AGENT_CARD_REGISTRY_MEMORY_SPACE_ID", raising=False)
        with pytest.raises(ValueError, match="MemorySpace ID"):
            resolve_memory_space_id(None)

    def test_agent_memory_space_id_is_not_used(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AGENT_CARD_REGISTRY_MEMORY_SPACE_ID", raising=False)
        monkeypatch.setenv("AGENT_MEMORY_SPACE_ID", "mem0-space")
        assert try_resolve_memory_space_id(None) is None


class TestTryBuildMemoryServiceClient:
    def test_returns_none_without_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DATAROBOT_API_TOKEN", raising=False)
        assert try_build_memory_service_client() is None

    def test_returns_none_without_endpoint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DATAROBOT_API_TOKEN", "token")
        monkeypatch.delenv("DATAROBOT_ENDPOINT", raising=False)
        monkeypatch.delenv("DATAROBOT_PUBLIC_API_ENDPOINT", raising=False)
        assert try_build_memory_service_client() is None

    def test_uses_public_api_endpoint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DATAROBOT_API_TOKEN", "token")
        monkeypatch.setenv(
            "DATAROBOT_PUBLIC_API_ENDPOINT",
            "https://staging.datarobot.com/api/v2",
        )
        monkeypatch.setenv("DATAROBOT_ENDPOINT", "http://datarobot-nginx/api/v2")

        client = try_build_memory_service_client()

        assert client is not None
        assert client.base_url == "https://staging.datarobot.com/api/v2/memory"

    def test_explicit_args_take_precedence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DATAROBOT_API_TOKEN", raising=False)
        monkeypatch.delenv("DATAROBOT_ENDPOINT", raising=False)

        client = try_build_memory_service_client(
            endpoint="https://explicit.datarobot.com/api/v2",
            api_token="explicit-token",
        )

        assert client is not None
        assert client.base_url == "https://explicit.datarobot.com/api/v2/memory"


class TestMemorySpaceKVCache:
    @respx.mock
    async def test_get_value_returns_none_when_no_session(
        self, kv_cache: MemorySpaceKVCache
    ) -> None:
        respx.get(SESSIONS_URL).mock(return_value=httpx.Response(200, json=_items()))

        assert await kv_cache.get_value("dep-1") is None

    @respx.mock
    async def test_set_value_creates_session_and_event_when_missing(
        self, kv_cache: MemorySpaceKVCache
    ) -> None:
        captured: dict[str, Any] = {}

        def _capture_post(req: httpx.Request) -> httpx.Response:
            captured["session_body"] = json.loads(req.content)
            return httpx.Response(201, json=_session_wire())

        # Dedup lookup misses, so post() creates the session, then post()s the first event.
        respx.get(SESSIONS_URL).mock(return_value=httpx.Response(200, json=_items()))
        respx.post(SESSIONS_URL).mock(side_effect=_capture_post)
        respx.get(EVENTS_URL).mock(return_value=httpx.Response(200, json=_items()))

        def _capture_event(req: httpx.Request) -> httpx.Response:
            captured["event_body"] = json.loads(req.content)
            return httpx.Response(201, json=_event_wire(content="payload-v1"))

        respx.post(EVENTS_URL).mock(side_effect=_capture_event)

        await kv_cache.set_value("dep-1", "payload-v1")

        assert captured["session_body"]["deduplicationKey"] == LOGICAL_KEY
        assert captured["session_body"]["participants"] == [SYSTEM_PARTICIPANT]
        assert captured["event_body"]["body"]["content"] == "payload-v1"
        assert captured["event_body"]["emitter"]["type"] == "agent"

    @respx.mock
    async def test_set_value_patches_existing_event_in_place(
        self, kv_cache: MemorySpaceKVCache
    ) -> None:
        respx.get(SESSIONS_URL).mock(return_value=httpx.Response(200, json=_items(_session_wire())))
        respx.get(EVENTS_URL).mock(
            return_value=httpx.Response(200, json=_items(_event_wire(content="v1")))
        )

        patch_calls: list[dict[str, Any]] = []

        def _capture_patch(req: httpx.Request) -> httpx.Response:
            patch_calls.append(json.loads(req.content))
            return httpx.Response(200, json=_event_wire(content="v2"))

        patch_route = respx.patch(f"{EVENTS_URL}0/").mock(side_effect=_capture_patch)
        post_route = respx.post(EVENTS_URL).mock(
            return_value=httpx.Response(201, json=_event_wire())
        )

        await kv_cache.set_value("dep-1", "v2")

        assert patch_route.called
        assert not post_route.called
        assert patch_calls[0]["body"]["content"] == "v2"

    @respx.mock
    async def test_set_value_swallows_errors(self, kv_cache: MemorySpaceKVCache) -> None:
        respx.get(SESSIONS_URL).mock(return_value=httpx.Response(500, json={"detail": "boom"}))

        # Must not raise -- L2 failures degrade to an L1-only cache, never break the caller.
        await kv_cache.set_value("dep-1", "payload")

    @respx.mock
    async def test_get_value_uses_cached_session_id_and_skips_dedup_lookup(
        self, kv_cache: MemorySpaceKVCache
    ) -> None:
        kv_cache._session_ids[LOGICAL_KEY] = SESSION_ID  # noqa: SLF001
        list_route = respx.get(SESSIONS_URL).mock(return_value=httpx.Response(200, json=_items()))
        respx.get(SESSION_URL).mock(return_value=httpx.Response(200, json=_session_wire()))
        respx.get(EVENTS_URL).mock(
            return_value=httpx.Response(200, json=_items(_event_wire(content="cached")))
        )

        assert await kv_cache.get_value("dep-1") == "cached"
        assert not list_route.called

    @respx.mock
    async def test_get_value_falls_back_to_dedup_lookup_on_stale_session_id(
        self, kv_cache: MemorySpaceKVCache
    ) -> None:
        kv_cache._session_ids[LOGICAL_KEY] = "stale-session"  # noqa: SLF001
        respx.get(f"{SESSIONS_URL}stale-session/").mock(
            return_value=httpx.Response(404, json={"detail": "not found"})
        )
        respx.get(SESSIONS_URL).mock(return_value=httpx.Response(200, json=_items(_session_wire())))
        respx.get(EVENTS_URL).mock(
            return_value=httpx.Response(200, json=_items(_event_wire(content="recovered")))
        )

        assert await kv_cache.get_value("dep-1") == "recovered"
        assert kv_cache._session_ids[LOGICAL_KEY] == SESSION_ID  # noqa: SLF001

    @respx.mock
    async def test_delete_value_removes_session_and_invalidates_cache(
        self, kv_cache: MemorySpaceKVCache
    ) -> None:
        kv_cache._session_ids[LOGICAL_KEY] = SESSION_ID  # noqa: SLF001
        respx.get(SESSION_URL).mock(return_value=httpx.Response(200, json=_session_wire()))
        delete_route = respx.delete(SESSION_URL).mock(return_value=httpx.Response(204))

        await kv_cache.delete_value("dep-1")

        assert delete_route.called
        assert LOGICAL_KEY not in kv_cache._session_ids  # noqa: SLF001

    @respx.mock
    async def test_delete_value_no_op_when_missing(self, kv_cache: MemorySpaceKVCache) -> None:
        respx.get(SESSIONS_URL).mock(return_value=httpx.Response(200, json=_items()))
        delete_route = respx.delete(SESSION_URL)

        await kv_cache.delete_value("dep-1")

        assert not delete_route.called
