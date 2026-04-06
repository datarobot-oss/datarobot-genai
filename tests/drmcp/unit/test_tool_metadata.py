# Copyright 2026 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import inspect

import pytest

from datarobot_genai.drtools.core import tool_metadata

tool_metadata_module = importlib.import_module("datarobot_genai.drtools.core.tool_metadata")


@pytest.fixture(autouse=True)
def clear_tool_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tool_metadata_module, "_TOOL_REGISTRY", [])


@pytest.mark.asyncio
async def test_tool_metadata_preserves_async_detection() -> None:
    @tool_metadata(tags={"test"})
    async def async_tool(x: int) -> int:
        return x + 1

    assert inspect.iscoroutinefunction(async_tool)
    assert await async_tool(1) == 2
    assert async_tool._tool_metadata["tags"] == {"test"}  # type: ignore[attr-defined]


def test_tool_metadata_preserves_sync_detection() -> None:
    @tool_metadata(tags={"test"})
    def sync_tool(x: int) -> int:
        return x + 1

    assert not inspect.iscoroutinefunction(sync_tool)
    assert sync_tool(1) == 2
    assert sync_tool._tool_metadata["tags"] == {"test"}  # type: ignore[attr-defined]


def test_tool_registry_keeps_registered_function_and_metadata() -> None:
    @tool_metadata(tags={"registry"}, enabled=True)
    def registry_tool() -> str:
        return "ok"

    registered_tools = tool_metadata_module.get_registered_tools()
    assert len(registered_tools) == 1
    func, metadata = registered_tools[0]
    assert func.__name__ == "registry_tool"
    assert metadata == {"tags": {"registry"}, "enabled": True}
