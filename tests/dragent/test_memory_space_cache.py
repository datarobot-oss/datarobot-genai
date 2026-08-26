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
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from datarobot_genai.dragent.memory_space_cache import CACHE_EVENT_TYPE
from datarobot_genai.dragent.memory_space_cache import DRAGENT_CACHE_PARTICIPANT_ID
from datarobot_genai.dragent.memory_space_cache import MemorySpaceKVCache
from datarobot_genai.dragent.memory_space_cache import resolve_memory_space_id
from datarobot_genai.dragent.memory_space_cache import try_resolve_memory_space_id


class _FakeEvent:
    def __init__(self, *, sequence_id: int, body: dict[str, Any] | None) -> None:
        self.sequence_id = sequence_id
        self.body = body


class _FakeSession:
    def __init__(self, session_id: str = "sess-1") -> None:
        self.id = session_id
        self.metadata: dict[str, Any] = {}
        self._events: list[_FakeEvent] = []
        self.post_event = MagicMock(side_effect=self._post_event)
        self.update_event = MagicMock(side_effect=self._update_event)
        self.delete = MagicMock()

    def events(self, **kwargs: Any) -> list[_FakeEvent]:
        if "last_n" in kwargs:
            return self._events[-kwargs["last_n"] :]
        return list(self._events)

    def _post_event(self, **kwargs: Any) -> _FakeEvent:
        event = _FakeEvent(sequence_id=len(self._events) + 1, body=kwargs.get("body"))
        self._events.append(event)
        return event

    def _update_event(self, sequence_id: int, **kwargs: Any) -> None:
        for event in self._events:
            if event.sequence_id == sequence_id:
                if "body" in kwargs:
                    event.body = kwargs["body"]
                return
        raise KeyError(sequence_id)


@pytest.fixture
def kv_cache() -> MemorySpaceKVCache:
    return MemorySpaceKVCache(memory_space_id="space-1")


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


class TestMemorySpaceKVCache:
    async def test_set_and_get_round_trip(self, kv_cache: MemorySpaceKVCache) -> None:
        session = _FakeSession()

        with (
            patch(
                "datarobot_genai.dragent.memory_space_cache._find_cache_session",
                return_value=None,
            ),
            patch(
                "datarobot_genai.dragent.memory_space_cache._create_cache_session",
                return_value=session,
            ),
        ):
            await kv_cache.set_value("dep-1", '{"version": 1}')

        with patch(
            "datarobot_genai.dragent.memory_space_cache._find_cache_session",
            return_value=session,
        ):
            assert await kv_cache.get_value("dep-1") == '{"version": 1}'

        session.post_event.assert_called_once()
        kwargs = session.post_event.call_args.kwargs
        assert kwargs["event_type"] == CACHE_EVENT_TYPE
        assert kwargs["body"]["payload"] == '{"version": 1}'

    async def test_update_existing_entry(self, kv_cache: MemorySpaceKVCache) -> None:
        session = _FakeSession()
        session.post_event(body={"v": 1, "payload": "v1"})

        with patch(
            "datarobot_genai.dragent.memory_space_cache._find_cache_session",
            return_value=session,
        ):
            await kv_cache.set_value("dep-1", "v2")

        session.update_event.assert_called_once_with(1, body={"v": 1, "payload": "v2"})
        assert session.post_event.call_count == 1

    async def test_create_uses_cache_participant(self, kv_cache: MemorySpaceKVCache) -> None:
        session = _FakeSession()

        with (
            patch(
                "datarobot_genai.dragent.memory_space_cache._find_cache_session",
                return_value=None,
            ),
            patch(
                "datarobot_genai.dragent.memory_space_cache.Session.create",
                return_value=session,
            ) as create_mock,
        ):
            await kv_cache.set_value("dep-1", "payload")

        create_mock.assert_called_once()
        assert create_mock.call_args.args[1] == [DRAGENT_CACHE_PARTICIPANT_ID]

    async def test_delete_removes_session(self, kv_cache: MemorySpaceKVCache) -> None:
        session = _FakeSession()

        with patch(
            "datarobot_genai.dragent.memory_space_cache._find_cache_session",
            return_value=session,
        ):
            await kv_cache.delete_value("dep-1")

        session.delete.assert_called_once()
