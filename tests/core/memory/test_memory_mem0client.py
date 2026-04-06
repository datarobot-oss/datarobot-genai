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

import importlib
import sys
import types
from typing import Any

import pytest


class _FakeAsyncClient:
    def __init__(
        self, base_url: Any = None, headers: dict[str, str] | None = None, timeout: Any = None
    ) -> None:
        self.base_url = base_url
        self.headers = dict(headers or {})
        self.timeout = timeout

    def close(self) -> None:
        return None


class _FakeAsyncProject:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _FakeAsyncMemoryClient:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.last_search_kwargs: dict[str, Any] | None = None
        self.last_add_args: tuple[Any, ...] | None = None
        self.last_add_kwargs: dict[str, Any] | None = None

    def _validate_api_key(self) -> str:
        return "user@example.com"

    async def search(self, **kwargs: Any) -> str:
        self.last_search_kwargs = kwargs
        return "mocked-result"

    async def add(self, *args: Any, **kwargs: Any) -> None:
        self.last_add_args = args
        self.last_add_kwargs = kwargs


def _load_mem0client_module(monkeypatch: Any) -> Any:
    for module_name in (
        "datarobot_genai.core.memory.mem0client",
        "datarobot_genai.core.memory.datarobot_memory_client",
    ):
        sys.modules.pop(module_name, None)

    fake_mem0 = types.ModuleType("mem0")
    fake_mem0.AsyncMemoryClient = _FakeAsyncMemoryClient

    fake_client = types.ModuleType("mem0.client")
    fake_client_project = types.ModuleType("mem0.client.project")
    fake_client_project.AsyncProject = _FakeAsyncProject
    fake_client.project = fake_client_project

    fake_memory = types.ModuleType("mem0.memory")
    fake_memory_telemetry = types.ModuleType("mem0.memory.telemetry")
    fake_memory_telemetry.capture_client_event = lambda *a, **k: None
    fake_memory.telemetry = fake_memory_telemetry

    fake_mem0.client = fake_client
    fake_mem0.memory = fake_memory

    for name, module in (
        ("mem0", fake_mem0),
        ("mem0.client", fake_client),
        ("mem0.client.project", fake_client_project),
        ("mem0.memory", fake_memory),
        ("mem0.memory.telemetry", fake_memory_telemetry),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    fake_httpx = types.ModuleType("httpx")
    fake_httpx.AsyncClient = _FakeAsyncClient
    fake_httpx.URL = lambda value: value
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    module = importlib.import_module("datarobot_genai.core.memory.mem0client")
    return importlib.reload(module)


@pytest.mark.asyncio
async def test_retrieve_builds_filter_with_ids(monkeypatch: Any) -> None:
    mem0client = _load_mem0client_module(monkeypatch)
    client = mem0client.Mem0Client(api_key="test-key")

    result = await client.retrieve(
        prompt="hello",
        run_id="r-1",
        agent_id="a-1",
        app_id="app-1",
    )

    assert result == "mocked-result"
    assert client._memory.last_search_kwargs == {
        "query": "hello",
        "filters": {
            "AND": [
                {"user_id": client._memory.user_id},
                {"run_id": "r-1"},
                {"agent_id": "a-1"},
                {"app_id": "app-1"},
            ]
        },
    }


@pytest.mark.asyncio
async def test_retrieve_with_optional_attributes_adds_them_to_filter(monkeypatch: Any) -> None:
    mem0client = _load_mem0client_module(monkeypatch)
    client = mem0client.Mem0Client(api_key="test-key")

    result = await client.retrieve(
        prompt="hello",
        attributes={"project_id": "p-1", "session_id": "s-1"},
        run_id="r-1",
        agent_id="a-1",
        app_id="app-1",
    )

    assert result == "mocked-result"
    assert client._memory.last_search_kwargs == {
        "query": "hello",
        "filters": {
            "AND": [
                {"user_id": client._memory.user_id},
                {"run_id": "r-1"},
                {"agent_id": "a-1"},
                {"app_id": "app-1"},
                {"metadata": {"project_id": "p-1", "session_id": "s-1"}},
            ]
        },
    }


@pytest.mark.asyncio
async def test_store_adds_user_message(monkeypatch: Any) -> None:
    mem0client = _load_mem0client_module(monkeypatch)
    client = mem0client.Mem0Client(api_key="test-key")

    await client.store(
        user_message="hello",
        run_id="r-2",
        agent_id="a-2",
        app_id="app-2",
    )

    assert client._memory.last_add_args == ([{"role": "user", "content": "hello"}],)
    assert client._memory.last_add_kwargs == {
        "version": "v1",
        "output_format": "v1.1",
        "user_id": client._memory.user_id,
        "run_id": "r-2",
        "agent_id": "a-2",
        "app_id": "app-2",
    }


@pytest.mark.asyncio
async def test_store_merges_optional_attributes(monkeypatch: Any) -> None:
    mem0client = _load_mem0client_module(monkeypatch)
    client = mem0client.Mem0Client(api_key="test-key")

    await client.store(
        user_message="hello",
        run_id="r-2",
        agent_id="a-2",
        app_id="app-2",
        attributes={"project_id": "p-1"},
    )

    assert client._memory.last_add_kwargs == {
        "version": "v1",
        "output_format": "v1.1",
        "metadata": {"project_id": "p-1"},
        "user_id": client._memory.user_id,
        "run_id": "r-2",
        "agent_id": "a-2",
        "app_id": "app-2",
    }


@pytest.mark.asyncio
async def test_retrieve_formats_search_results(monkeypatch: Any) -> None:
    mem0client = _load_mem0client_module(monkeypatch)
    client = mem0client.Mem0Client(api_key="test-key")
    client._memory.last_search_kwargs = None

    async def fake_search(**kwargs: Any) -> dict[str, Any]:
        client._memory.last_search_kwargs = kwargs
        return {"results": [{"memory": "first"}, {"text": "second"}, {"content": "third"}]}

    client._memory.search = fake_search  # type: ignore[method-assign]

    result = await client.retrieve(prompt="hello")

    assert result == "first\nsecond\nthird"
