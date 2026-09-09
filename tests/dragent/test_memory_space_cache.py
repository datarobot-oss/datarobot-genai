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
import requests

import datarobot_genai.dragent.memory_space_cache as memory_space_cache_module
from datarobot_genai.dragent.memory_space_cache import _STALE_CONNECTION_RETRIES
from datarobot_genai.dragent.memory_space_cache import CACHE_EVENT_TYPE
from datarobot_genai.dragent.memory_space_cache import DRAGENT_CACHE_PARTICIPANT_ID
from datarobot_genai.dragent.memory_space_cache import MemorySpaceKVCache
from datarobot_genai.dragent.memory_space_cache import _find_cache_session
from datarobot_genai.dragent.memory_space_cache import configure_datarobot_memory_client
from datarobot_genai.dragent.memory_space_cache import is_enclave_l2_workload
from datarobot_genai.dragent.memory_space_cache import registry_cache_deduplication_key
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


_ENCLAVE_HOST_ENV = "DR_WORKLOAD_EXTERNAL_URL_HOST"
_ENCLAVE_PREFIX_ENV = "DR_WORKLOAD_EXTERNAL_URL_PREFIX"


def _set_enclave_gateway_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_ENCLAVE_HOST_ENV, "enclave-x.datarobot.com")
    monkeypatch.setenv(_ENCLAVE_PREFIX_ENV, "/workloads/abc123")


@pytest.fixture(autouse=True)
def _reset_provisioned_registry_cache_space_state() -> None:
    memory_space_cache_module._ProvisionedRegistryCacheSpaceState.space_id = None
    yield
    memory_space_cache_module._ProvisionedRegistryCacheSpaceState.space_id = None


@pytest.fixture
def kv_cache() -> MemorySpaceKVCache:
    return MemorySpaceKVCache(memory_space_id="space-1")


class TestEnclaveL2Workload:
    def test_is_false_without_enclave_gateway(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WORKLOAD_ID", "wl-abc123")
        monkeypatch.delenv(_ENCLAVE_HOST_ENV, raising=False)
        monkeypatch.delenv(_ENCLAVE_PREFIX_ENV, raising=False)
        assert is_enclave_l2_workload() is False

    def test_is_false_on_enclave_gateway_without_workload_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("WORKLOAD_ID", raising=False)
        _set_enclave_gateway_env(monkeypatch)
        assert is_enclave_l2_workload() is False

    def test_deduplication_key(self) -> None:
        assert registry_cache_deduplication_key("wl-abc123") == (
            "dragent:agent-card-registry:workload:wl-abc123"
        )


class TestResolveMemorySpaceId:
    def test_returns_none_when_not_on_enclave(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("WORKLOAD_ID", raising=False)
        monkeypatch.delenv(_ENCLAVE_HOST_ENV, raising=False)
        monkeypatch.delenv(_ENCLAVE_PREFIX_ENV, raising=False)
        assert try_resolve_memory_space_id() is None

    def test_skips_on_enclave_gateway_without_workload_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("WORKLOAD_ID", raising=False)
        _set_enclave_gateway_env(monkeypatch)
        assert try_resolve_memory_space_id() is None

    def test_creates_space_on_enclave_workload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WORKLOAD_ID", "wl-abc123")
        _set_enclave_gateway_env(monkeypatch)
        space = MagicMock(id="space-new")
        create_mock = MagicMock(return_value=space)

        with (
            patch(
                "datarobot_genai.dragent.memory_space_cache.try_configure_datarobot_memory_client",
                return_value=True,
            ),
            patch(
                "datarobot_genai.dragent.memory_space_cache.MemorySpace.create",
                create_mock,
            ),
        ):
            assert try_resolve_memory_space_id() == "space-new"

        create_mock.assert_called_once_with(
            description="Agent card registry L2 cache",
            deduplication_key="dragent:agent-card-registry:workload:wl-abc123",
        )

    def test_adopts_existing_space_on_dedup_collision(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from datarobot.errors import MemorySpaceDeduplicationError

        monkeypatch.setenv("WORKLOAD_ID", "wl-xyz")
        _set_enclave_gateway_env(monkeypatch)
        existing = MagicMock(id="space-existing")
        create_mock = MagicMock(
            side_effect=MemorySpaceDeduplicationError(
                "conflict",
                409,
                json={"existingMemorySpaceId": "space-existing"},
            )
        )
        get_mock = MagicMock(return_value=existing)

        with (
            patch(
                "datarobot_genai.dragent.memory_space_cache.try_configure_datarobot_memory_client",
                return_value=True,
            ),
            patch(
                "datarobot_genai.dragent.memory_space_cache.MemorySpace.create",
                create_mock,
            ),
            patch(
                "datarobot_genai.dragent.memory_space_cache.MemorySpace.get",
                get_mock,
            ),
        ):
            assert try_resolve_memory_space_id() == "space-existing"

        get_mock.assert_called_once_with("space-existing")

    def test_returns_cached_space_id_without_reprovisioning(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        memory_space_cache_module._ProvisionedRegistryCacheSpaceState.space_id = "space-cached"
        create_mock = MagicMock()

        with patch(
            "datarobot_genai.dragent.memory_space_cache.MemorySpace.create",
            create_mock,
        ):
            assert try_resolve_memory_space_id() == "space-cached"

        create_mock.assert_not_called()


class TestConfigureDatarobotMemoryClient:
    def test_configures_enclave_gateway_without_dr_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DATAROBOT_API_TOKEN", "token")
        _set_enclave_gateway_env(monkeypatch)
        mock_client = MagicMock()
        mock_config = MagicMock()

        with (
            patch("datarobot.config.create_drconfig", return_value=mock_config) as create_mock,
            patch(
                "datarobot.rest.RESTClientObject.from_config",
                return_value=mock_client,
            ) as from_config_mock,
            patch("datarobot.client.set_client") as set_client_mock,
            patch("datarobot.Client") as client_ctor_mock,
        ):
            configure_datarobot_memory_client()

        create_mock.assert_called_once_with(
            token="token",
            endpoint="https://enclave-x.datarobot.com/api/v2",
        )
        from_config_mock.assert_called_once_with(mock_config)
        set_client_mock.assert_called_once_with(mock_client)
        client_ctor_mock.assert_not_called()

    def test_raises_without_enclave_gateway(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DATAROBOT_API_TOKEN", "token")
        monkeypatch.delenv(_ENCLAVE_HOST_ENV, raising=False)
        monkeypatch.delenv(_ENCLAVE_PREFIX_ENV, raising=False)

        with pytest.raises(ValueError, match="enclave API gateway"):
            configure_datarobot_memory_client()


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
            patch(
                "datarobot_genai.dragent.memory_space_cache.Session.get",
                return_value=session,
            ),
        ):
            await kv_cache.set_value("dep-1", '{"version": 1}')
            assert await kv_cache.get_value("dep-1") == '{"version": 1}'

        session.post_event.assert_called_once()
        kwargs = session.post_event.call_args.kwargs
        assert kwargs["event_type"] == CACHE_EVENT_TYPE
        assert kwargs["body"]["content"] == '{"version": 1}'

    async def test_get_reuses_cached_session_id_without_list(
        self, kv_cache: MemorySpaceKVCache
    ) -> None:
        session = _FakeSession()
        find_mock = MagicMock(return_value=None)
        get_mock = MagicMock(return_value=session)
        session.post_event(body={"content": "cached"})

        with (
            patch(
                "datarobot_genai.dragent.memory_space_cache._find_cache_session",
                find_mock,
            ),
            patch(
                "datarobot_genai.dragent.memory_space_cache._create_cache_session",
                return_value=session,
            ),
            patch(
                "datarobot_genai.dragent.memory_space_cache.Session.get",
                get_mock,
            ),
        ):
            await kv_cache.set_value("dep-1", "cached")
            find_mock.reset_mock()
            get_mock.reset_mock()
            assert await kv_cache.get_value("dep-1") == "cached"

        find_mock.assert_not_called()
        get_mock.assert_called_once_with("space-1", "sess-1")

    async def test_update_existing_entry(self, kv_cache: MemorySpaceKVCache) -> None:
        session = _FakeSession()
        session.post_event(body={"content": "v1"})

        with (
            patch(
                "datarobot_genai.dragent.memory_space_cache._find_cache_session",
                return_value=session,
            ),
            patch(
                "datarobot_genai.dragent.memory_space_cache.Session.get",
                return_value=session,
            ),
        ):
            await kv_cache.set_value("dep-1", "v2")

        session.update_event.assert_called_once_with(1, body={"content": "v2"})
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

        with (
            patch(
                "datarobot_genai.dragent.memory_space_cache._find_cache_session",
                return_value=session,
            ),
            patch(
                "datarobot_genai.dragent.memory_space_cache.Session.get",
                return_value=session,
            ),
        ):
            await kv_cache.delete_value("dep-1")

        session.delete.assert_called_once()


class TestStaleConnectionRetry:
    """Regression tests for the stale pooled-connection retry (RemoteDisconnected)."""

    async def test_find_cache_session_retries_transient_connection_error(self) -> None:
        session = _FakeSession()
        list_mock = MagicMock(
            side_effect=[
                requests.exceptions.ConnectionError("stale connection"),
                [session],
            ]
        )

        with patch("datarobot_genai.dragent.memory_space_cache.Session.list", list_mock):
            result = _find_cache_session("space-1", "dragent:agent_card:dep-1")

        assert result is session
        assert list_mock.call_count == 2

    async def test_find_cache_session_raises_after_exhausting_retries(self) -> None:
        list_mock = MagicMock(side_effect=requests.exceptions.ConnectionError("stale connection"))

        with (
            patch("datarobot_genai.dragent.memory_space_cache.Session.list", list_mock),
            pytest.raises(requests.exceptions.ConnectionError),
        ):
            _find_cache_session("space-1", "dragent:agent_card:dep-1")

        assert list_mock.call_count == _STALE_CONNECTION_RETRIES + 1

    async def test_get_value_survives_a_single_transient_connection_error(
        self, kv_cache: MemorySpaceKVCache
    ) -> None:
        session = _FakeSession()
        session.post_event(body={"content": "cached"})
        list_mock = MagicMock(
            side_effect=[
                requests.exceptions.ConnectionError("stale connection"),
                [session],
            ]
        )

        with patch("datarobot_genai.dragent.memory_space_cache.Session.list", list_mock):
            assert await kv_cache.get_value("dep-1") == "cached"

        assert list_mock.call_count == 2

    async def test_get_value_falls_back_to_none_once_retries_are_exhausted(
        self, kv_cache: MemorySpaceKVCache
    ) -> None:
        list_mock = MagicMock(side_effect=requests.exceptions.ConnectionError("stale connection"))

        with patch("datarobot_genai.dragent.memory_space_cache.Session.list", list_mock):
            # Still degrades to a cache miss rather than raising -- get_value's own
            # try/except is the last line of defense once the retry is exhausted.
            assert await kv_cache.get_value("dep-1") is None

        assert list_mock.call_count == _STALE_CONNECTION_RETRIES + 1
