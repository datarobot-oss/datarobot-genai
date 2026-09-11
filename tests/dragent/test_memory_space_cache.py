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

import warnings
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from datarobot.application_utils.persistence.exceptions import DRMemoryNotFoundError
from datarobot.application_utils.persistence.exceptions import DRMemoryUnavailableError

import datarobot_genai.dragent.memory_space_cache as memory_space_cache_module
from datarobot_genai.dragent.memory_space_cache import _TRANSPORT_RETRIES
from datarobot_genai.dragent.memory_space_cache import DRAGENT_CACHE_PARTICIPANT_ID
from datarobot_genai.dragent.memory_space_cache import MemorySpaceKVCache
from datarobot_genai.dragent.memory_space_cache import _cache_deduplication_key
from datarobot_genai.dragent.memory_space_cache import configure_datarobot_memory_client
from datarobot_genai.dragent.memory_space_cache import is_enclave_l2_workload
from datarobot_genai.dragent.memory_space_cache import registry_cache_deduplication_key
from datarobot_genai.dragent.memory_space_cache import try_resolve_memory_space_id
from datarobot_genai.dragent.memory_space_cache import try_resolve_memory_space_id_async


class _FakeEvent:
    def __init__(self, *, content: str) -> None:
        self.content = content
        self.patch = AsyncMock()


class _FakeSession:
    def __init__(self, session_id: str = "sess-1") -> None:
        self.id = session_id
        self.delete = AsyncMock()


_ENCLAVE_HOST_ENV = "DR_WORKLOAD_EXTERNAL_URL_HOST"
_ENCLAVE_PREFIX_ENV = "DR_WORKLOAD_EXTERNAL_URL_PREFIX"


def _set_enclave_gateway_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_ENCLAVE_HOST_ENV, "enclave-x.datarobot.com")
    monkeypatch.setenv(_ENCLAVE_PREFIX_ENV, "/workloads/abc123")


@pytest.fixture(autouse=True)
def _reset_provisioned_registry_cache_space_state() -> None:
    memory_space_cache_module._ProvisionedRegistryCacheSpaceState.space_id = None
    memory_space_cache_module._MemoryClientState.client = None
    yield
    memory_space_cache_module._ProvisionedRegistryCacheSpaceState.space_id = None
    memory_space_cache_module._MemoryClientState.client = None


@pytest.fixture
def kv_cache() -> MemorySpaceKVCache:
    client = MagicMock()
    return MemorySpaceKVCache(memory_space_id="space-1", client=client)


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
        post_mock = AsyncMock(return_value=space)

        with (
            patch(
                "datarobot_genai.dragent.memory_space_cache.try_configure_datarobot_memory_client",
                return_value=True,
            ),
            patch(
                "datarobot_genai.dragent.memory_space_cache.DRMemorySpace.post",
                post_mock,
            ),
            patch(
                "datarobot_genai.dragent.memory_space_cache._require_memory_client",
                return_value=MagicMock(),
            ),
        ):
            assert try_resolve_memory_space_id() == "space-new"

        post_mock.assert_awaited_once_with(
            post_mock.await_args.args[0],
            description="Agent card registry L2 cache",
            deduplication_key="dragent:agent-card-registry:workload:wl-abc123",
        )

    def test_adopts_existing_space_on_dedup_collision(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WORKLOAD_ID", "wl-xyz")
        _set_enclave_gateway_env(monkeypatch)
        space = MagicMock(id="space-existing")
        post_mock = AsyncMock(return_value=space)

        with (
            patch(
                "datarobot_genai.dragent.memory_space_cache.try_configure_datarobot_memory_client",
                return_value=True,
            ),
            patch(
                "datarobot_genai.dragent.memory_space_cache.DRMemorySpace.post",
                post_mock,
            ),
            patch(
                "datarobot_genai.dragent.memory_space_cache._require_memory_client",
                return_value=MagicMock(),
            ),
        ):
            assert try_resolve_memory_space_id() == "space-existing"

    def test_returns_cached_space_id_without_reprovisioning(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        memory_space_cache_module._ProvisionedRegistryCacheSpaceState.space_id = "space-cached"
        post_mock = AsyncMock()

        with patch(
            "datarobot_genai.dragent.memory_space_cache.DRMemorySpace.post",
            post_mock,
        ):
            assert try_resolve_memory_space_id() == "space-cached"

        post_mock.assert_not_called()

    async def test_sync_from_running_loop_returns_already_provisioned_space_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GIVEN bootstrap already stored a space id WHEN resolved from a running loop
        THEN the cached id is returned without asyncio.run.
        """
        memory_space_cache_module._ProvisionedRegistryCacheSpaceState.space_id = "space-cached"
        resolve_mock = AsyncMock()

        with patch(
            "datarobot_genai.dragent.memory_space_cache._try_resolve_memory_space_id",
            resolve_mock,
        ):
            assert try_resolve_memory_space_id() == "space-cached"

        resolve_mock.assert_not_called()

    async def test_async_creates_space_from_running_loop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GIVEN an enclave workload WHEN resolved from a running loop
        THEN the space is provisioned.
        """
        monkeypatch.setenv("WORKLOAD_ID", "wl-abc123")
        _set_enclave_gateway_env(monkeypatch)
        space = MagicMock(id="space-new")
        post_mock = AsyncMock(return_value=space)

        with (
            patch(
                "datarobot_genai.dragent.memory_space_cache.try_configure_datarobot_memory_client",
                return_value=True,
            ),
            patch(
                "datarobot_genai.dragent.memory_space_cache.DRMemorySpace.post",
                post_mock,
            ),
            patch(
                "datarobot_genai.dragent.memory_space_cache._require_memory_client",
                return_value=MagicMock(),
            ),
        ):
            assert await try_resolve_memory_space_id_async() == "space-new"

        post_mock.assert_awaited_once()

    async def test_sync_from_running_loop_does_not_leave_unawaited_coroutine(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GIVEN a running loop WHEN the sync helper is used
        THEN no provision coroutine is created.
        """
        monkeypatch.setenv("WORKLOAD_ID", "wl-abc123")
        _set_enclave_gateway_env(monkeypatch)
        resolve_mock = AsyncMock(return_value="space-new")

        with (
            patch(
                "datarobot_genai.dragent.memory_space_cache._try_resolve_memory_space_id",
                resolve_mock,
            ),
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter("always")
            assert try_resolve_memory_space_id() is None

        resolve_mock.assert_not_called()
        assert not any("never awaited" in str(w.message) for w in caught)

    async def test_async_retry_succeeds_after_sync_helper_misses_on_running_loop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GIVEN a missed sync bootstrap from a running loop
        WHEN async resolve retries THEN L2 is provisioned.
        """
        monkeypatch.setenv("WORKLOAD_ID", "wl-abc123")
        _set_enclave_gateway_env(monkeypatch)
        space = MagicMock(id="space-new")
        post_mock = AsyncMock(return_value=space)

        with (
            patch(
                "datarobot_genai.dragent.memory_space_cache.try_configure_datarobot_memory_client",
                return_value=True,
            ),
            patch(
                "datarobot_genai.dragent.memory_space_cache.DRMemorySpace.post",
                post_mock,
            ),
            patch(
                "datarobot_genai.dragent.memory_space_cache._require_memory_client",
                return_value=MagicMock(),
            ),
        ):
            assert try_resolve_memory_space_id() is None
            assert await try_resolve_memory_space_id_async() == "space-new"

        post_mock.assert_awaited_once()


class TestConfigureDatarobotMemoryClient:
    def test_configures_enclave_gateway_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DATAROBOT_API_TOKEN", "token")
        _set_enclave_gateway_env(monkeypatch)
        client_ctor_mock = MagicMock()

        with patch(
            "datarobot_genai.dragent.memory_space_cache.DRMemoryServiceClient",
            client_ctor_mock,
        ):
            configure_datarobot_memory_client()

        client_ctor_mock.assert_called_once_with(
            endpoint="https://enclave-x.datarobot.com/api/v2",
            api_token="token",
        )
        assert memory_space_cache_module._MemoryClientState.client is client_ctor_mock.return_value

    def test_raises_without_enclave_gateway(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DATAROBOT_API_TOKEN", "token")
        monkeypatch.delenv(_ENCLAVE_HOST_ENV, raising=False)
        monkeypatch.delenv(_ENCLAVE_PREFIX_ENV, raising=False)

        with pytest.raises(ValueError, match="enclave API gateway"):
            configure_datarobot_memory_client()


class TestMemorySpaceKVCache:
    async def test_set_and_get_round_trip(self, kv_cache: MemorySpaceKVCache) -> None:
        session = _FakeSession()
        event = _FakeEvent(content='{"version": 1}')
        space = MagicMock()
        logical_key = "dragent:agent_card:dep-1"
        dedup_key = _cache_deduplication_key(logical_key)

        with (
            patch(
                "datarobot_genai.dragent.memory_space_cache.DRMemorySpace.get",
                AsyncMock(return_value=space),
            ),
            patch(
                "datarobot_genai.dragent.memory_space_cache.AgentCardCacheSession.get",
                AsyncMock(side_effect=DRMemoryNotFoundError("missing", status_code=404)),
            ),
            patch(
                "datarobot_genai.dragent.memory_space_cache.AgentCardCacheSession.post",
                AsyncMock(return_value=session),
            ) as create_mock,
            patch(
                "datarobot_genai.dragent.memory_space_cache.AgentCardCacheEvent.last",
                AsyncMock(side_effect=[[], [event]]),
            ),
        ):
            await kv_cache.set_value("dep-1", '{"version": 1}')
            assert await kv_cache.get_value("dep-1") == '{"version": 1}'

        create_mock.assert_awaited_once_with(
            space,
            cache_kind="agent_card",
            dedup_key=dedup_key,
            logical_key=logical_key,
            participants=[DRAGENT_CACHE_PARTICIPANT_ID],
        )

    async def test_get_reuses_cached_session_without_second_lookup(
        self, kv_cache: MemorySpaceKVCache
    ) -> None:
        session = _FakeSession()
        event = _FakeEvent(content="cached")
        space = MagicMock()
        get_session_mock = AsyncMock(return_value=session)

        with (
            patch(
                "datarobot_genai.dragent.memory_space_cache.DRMemorySpace.get",
                AsyncMock(return_value=space),
            ),
            patch(
                "datarobot_genai.dragent.memory_space_cache.AgentCardCacheSession.get",
                get_session_mock,
            ),
            patch(
                "datarobot_genai.dragent.memory_space_cache.AgentCardCacheEvent.last",
                AsyncMock(return_value=[event]),
            ),
        ):
            await kv_cache.set_value("dep-1", "cached")
            get_session_mock.reset_mock()
            assert await kv_cache.get_value("dep-1") == "cached"

        get_session_mock.assert_not_awaited()

    async def test_update_existing_entry(self, kv_cache: MemorySpaceKVCache) -> None:
        session = _FakeSession()
        event = _FakeEvent(content="v1")
        space = MagicMock()

        with (
            patch(
                "datarobot_genai.dragent.memory_space_cache.DRMemorySpace.get",
                AsyncMock(return_value=space),
            ),
            patch(
                "datarobot_genai.dragent.memory_space_cache.AgentCardCacheSession.get",
                AsyncMock(return_value=session),
            ),
            patch(
                "datarobot_genai.dragent.memory_space_cache.AgentCardCacheEvent.last",
                AsyncMock(return_value=[event]),
            ),
        ):
            await kv_cache.set_value("dep-1", "v2")

        event.patch.assert_awaited_once_with(content="v2")

    async def test_create_uses_cache_participant(self, kv_cache: MemorySpaceKVCache) -> None:
        session = _FakeSession()
        space = MagicMock()

        with (
            patch(
                "datarobot_genai.dragent.memory_space_cache.DRMemorySpace.get",
                AsyncMock(return_value=space),
            ),
            patch(
                "datarobot_genai.dragent.memory_space_cache.AgentCardCacheSession.get",
                AsyncMock(side_effect=DRMemoryNotFoundError("missing", status_code=404)),
            ),
            patch(
                "datarobot_genai.dragent.memory_space_cache.AgentCardCacheSession.post",
                AsyncMock(return_value=session),
            ) as create_mock,
            patch(
                "datarobot_genai.dragent.memory_space_cache.AgentCardCacheEvent.last",
                AsyncMock(return_value=[]),
            ),
            patch(
                "datarobot_genai.dragent.memory_space_cache.AgentCardCacheEvent.post",
                AsyncMock(),
            ),
        ):
            await kv_cache.set_value("dep-1", "payload")

        assert create_mock.await_args.kwargs["participants"] == [DRAGENT_CACHE_PARTICIPANT_ID]

    async def test_delete_removes_session(self, kv_cache: MemorySpaceKVCache) -> None:
        session = _FakeSession()
        space = MagicMock()

        with (
            patch(
                "datarobot_genai.dragent.memory_space_cache.DRMemorySpace.get",
                AsyncMock(return_value=space),
            ),
            patch(
                "datarobot_genai.dragent.memory_space_cache.AgentCardCacheSession.get",
                AsyncMock(return_value=session),
            ),
        ):
            await kv_cache.delete_value("dep-1")

        session.delete.assert_awaited_once()


class TestTransportRetry:
    async def test_resolve_session_retries_transient_transport_error(
        self, kv_cache: MemorySpaceKVCache
    ) -> None:
        session = _FakeSession()
        space = MagicMock()
        get_mock = AsyncMock(
            side_effect=[
                DRMemoryUnavailableError("stale connection"),
                session,
            ]
        )

        with (
            patch(
                "datarobot_genai.dragent.memory_space_cache.DRMemorySpace.get",
                AsyncMock(return_value=space),
            ),
            patch(
                "datarobot_genai.dragent.memory_space_cache.AgentCardCacheSession.get",
                get_mock,
            ),
            patch(
                "datarobot_genai.dragent.memory_space_cache.AgentCardCacheEvent.last",
                AsyncMock(return_value=[_FakeEvent(content="cached")]),
            ),
        ):
            assert await kv_cache.get_value("dep-1") == "cached"

        assert get_mock.await_count == 2

    async def test_get_value_falls_back_to_none_once_retries_are_exhausted(
        self, kv_cache: MemorySpaceKVCache
    ) -> None:
        space = MagicMock()
        get_mock = AsyncMock(side_effect=DRMemoryUnavailableError("stale connection"))

        with (
            patch(
                "datarobot_genai.dragent.memory_space_cache.DRMemorySpace.get",
                AsyncMock(return_value=space),
            ),
            patch(
                "datarobot_genai.dragent.memory_space_cache.AgentCardCacheSession.get",
                get_mock,
            ),
        ):
            assert await kv_cache.get_value("dep-1") is None

        assert get_mock.await_count == _TRANSPORT_RETRIES + 1
