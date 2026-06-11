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

"""Unit tests for the resource_metadata decorator and registry."""

from importlib import import_module

import pytest

from datarobot_genai.drtools.core import get_registered_resources
from datarobot_genai.drtools.core import resource_metadata

# The package re-exports the decorator under the same name as its module, so
# fetch the module itself to patch the registry it holds.
_registry_module = import_module("datarobot_genai.drtools.core.resource_metadata")


@pytest.fixture(autouse=True)
def _fresh_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give each test an empty registry so registrations don't leak between runs."""
    monkeypatch.setattr(_registry_module, "_RESOURCE_REGISTRY", [])


def _metadata_for(name: str) -> dict:
    """Return the metadata recorded for the registered function named ``name``."""
    matches = [meta for func, meta in get_registered_resources() if func.__name__ == name]
    assert len(matches) == 1, f"expected exactly one registration for {name}, got {len(matches)}"
    return matches[0]


async def test_async_resource_registered_and_callable() -> None:
    @resource_metadata(
        uri="panels://{source}/{panel_id}",
        name="get_panel",
        mime_type="application/json",
        tags={"panels", "read"},
    )
    async def get_panel(source: str, panel_id: str) -> dict:
        return {"source": source, "panel_id": panel_id}

    # Metadata is attached to the returned wrapper for direct access.
    assert get_panel._resource_metadata["uri"] == "panels://{source}/{panel_id}"
    # The wrapped function still behaves like the original.
    assert await get_panel("main", "abc12") == {"source": "main", "panel_id": "abc12"}
    # And it is discoverable via the registry with its full metadata.
    meta = _metadata_for("get_panel")
    assert meta["mime_type"] == "application/json"
    assert meta["name"] == "get_panel"
    assert meta["tags"] == {"panels", "read"}


def test_sync_resource_registered_and_callable() -> None:
    @resource_metadata(uri="panels://main/list", name="list_panels")
    def list_panels() -> list[str]:
        return ["a", "b"]

    assert list_panels() == ["a", "b"]
    assert list_panels._resource_metadata["uri"] == "panels://main/list"
    assert _metadata_for("list_panels")["uri"] == "panels://main/list"


def test_registry_returns_a_copy() -> None:
    before = get_registered_resources()
    before.append(("garbage", {}))  # type: ignore[arg-type]
    after = get_registered_resources()
    assert ("garbage", {}) not in after, "get_registered_resources() must return a defensive copy"
