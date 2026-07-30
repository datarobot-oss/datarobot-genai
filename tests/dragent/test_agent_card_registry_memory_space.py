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

from datetime import UTC
from datetime import datetime
from datetime import timedelta
from unittest.mock import patch

import pytest
from a2a.types import AgentCard

from datarobot_genai.dragent.agent_card_registry import AgentCardRegistryConfig
from datarobot_genai.dragent.agent_card_registry_backends import AgentCardCacheRecord
from datarobot_genai.dragent.agent_card_registry_backends import LayeredAgentCardCacheBackend
from datarobot_genai.dragent.agent_card_registry_backends import MemorySpaceAgentCardCacheBackend
from datarobot_genai.dragent.agent_card_registry_backends import create_agent_card_cache_backend
from datarobot_genai.dragent.memory_space_cache import MemorySpaceKVCache


class _FakeMemoryBackend:
    def __init__(self) -> None:
        self._entries: dict[str, dict] = {}
        self._next_id = 1

    async def get_all(self, **kwargs):
        filters = kwargs.get("filters", {})
        metadata_filter = None
        for condition in filters.get("AND", []):
            if "metadata" in condition:
                metadata_filter = condition["metadata"]
                break
        results = []
        for entry in self._entries.values():
            meta = entry.get("metadata") or {}
            if metadata_filter and all(meta.get(k) == v for k, v in metadata_filter.items()):
                results.append(entry)
        return {"results": results}

    async def add(self, messages, **kwargs):
        metadata = kwargs.get("metadata") or {}
        memory_id = f"mem-{self._next_id}"
        self._next_id += 1
        payload = messages[0]["content"] if messages else ""
        self._entries[memory_id] = {
            "id": memory_id,
            "memory": payload,
            "metadata": metadata,
            "user_id": kwargs.get("user_id"),
        }
        return {"results": [{"id": memory_id}]}

    async def update(self, memory_id, text=None, **kwargs):
        entry = self._entries[memory_id]
        if text is not None:
            entry["memory"] = text
        if kwargs.get("metadata"):
            entry["metadata"] = kwargs["metadata"]
        return {"id": memory_id}


class _FakeMem0Client:
    def __init__(self) -> None:
        self._memory = _FakeMemoryBackend()


_SAMPLE_AGENT_CARD = AgentCard.model_validate(
    {
        "name": "MemorySpace Agent",
        "description": "Cached in MemorySpace",
        "url": "https://agent.example.com/a2a/",
        "version": "1.0.0",
        "skills": [],
        "defaultInputModes": ["text"],
        "defaultOutputModes": ["text"],
        "capabilities": {"streaming": False},
    }
)


@pytest.fixture
def memory_space_backend():
    kv = MemorySpaceKVCache(_FakeMem0Client(), key_prefix="dragent:")
    return MemorySpaceAgentCardCacheBackend(kv_cache=kv)


class TestMemorySpaceAgentCardCacheBackend:
    async def test_store_and_get_fresh_by_deployment_id(self, memory_space_backend):
        await memory_space_backend.store(
            {"dep-1": _SAMPLE_AGENT_CARD},
            key_types={"dep-1": "dep"},
        )
        record = await memory_space_backend.get_fresh("dep-1", cache_ttl=3600)
        assert record is not None
        assert record.card.name == "MemorySpace Agent"

    async def test_get_stale_after_soft_ttl(self, memory_space_backend):
        await memory_space_backend.store(
            {"dep-1": _SAMPLE_AGENT_CARD},
            key_types={"dep-1": "dep"},
        )

        payload = await memory_space_backend._kv.get_value("dep:dep-1", kind="agent_card")
        record = AgentCardCacheRecord.model_validate_json(payload)
        record.fetched_at = datetime.now(UTC) - timedelta(seconds=120)
        await memory_space_backend._kv.set_value(
            "dep:dep-1",
            record.model_dump_json(),
            kind="agent_card",
        )

        assert await memory_space_backend.get_fresh("dep-1", cache_ttl=60) is None
        stale = await memory_space_backend.get_stale("dep-1", max_staleness_seconds=3600)
        assert stale is not None
        assert stale.card.name == "MemorySpace Agent"


class TestCreateAgentCardCacheBackendMemorySpace:
    def test_memory_space_backend_requires_space_id(self):
        config = AgentCardRegistryConfig(agent_card_registry_backend="memory_space")
        env = {
            "DATAROBOT_ENDPOINT": "https://app.datarobot.com/api/v2",
            "DATAROBOT_API_TOKEN": "token",
        }
        with patch.dict("os.environ", env, clear=True):
            with pytest.raises(ValueError, match="MemorySpace ID"):
                create_agent_card_cache_backend(config)

    def test_memory_space_backend_creates_layered_backend(self):
        config = AgentCardRegistryConfig(
            agent_card_registry_backend="memory_space",
            agent_card_registry_memory_space_id="space-123",
        )
        env = {
            "DATAROBOT_ENDPOINT": "https://app.datarobot.com/api/v2",
            "DATAROBOT_API_TOKEN": "token",
        }
        with patch.dict("os.environ", env, clear=False):
            with patch(
                "datarobot_genai.dragent.agent_card_registry_backends.create_memory_space_client"
            ) as mock_create:
                mock_create.return_value = _FakeMem0Client()
                backend = create_agent_card_cache_backend(config)
        assert isinstance(backend, LayeredAgentCardCacheBackend)
        mock_create.assert_called_once_with(memory_space_id="space-123")
