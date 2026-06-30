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

"""Unit tests for the panel MCP resource handlers (drmcpbase)."""

import inspect
import json
from collections.abc import Callable
from typing import Any

import pytest

from datarobot_genai.drmcpbase.panels import register_panel_resources
from datarobot_genai.drmcpbase.panels import resources as res_mod
from datarobot_genai.drmcputils.panels.models import Dataset
from datarobot_genai.drmcputils.panels.models import Json
from datarobot_genai.drmcputils.panels.models import Text
from datarobot_genai.drmcputils.panels.store import PanelStore

from .conftest import FakeBlobStore


@pytest.fixture
def panel_resources(monkeypatch: pytest.MonkeyPatch) -> PanelStore:
    store = PanelStore(FakeBlobStore())
    monkeypatch.setattr(res_mod, "_require_mcp_sandbox", lambda: None)
    monkeypatch.setattr(res_mod, "_store", lambda: store)
    return store


async def test_panels_list_resource(panel_resources: PanelStore) -> None:
    await panel_resources.create(Text(title="a", text="hi"), source="staging")
    out = json.loads(await res_mod.panels_list_resource("staging"))
    assert out["source"] == "staging"
    assert out["count"] == 1
    assert out["panels"][0]["title"] == "a"


async def test_panel_metadata_resource(panel_resources: PanelStore) -> None:
    panel = await panel_resources.create(Json(title="j", data={"x": 1}), source="main")
    assert panel.id is not None
    out = json.loads(await res_mod.panel_metadata_resource("main", panel.id))
    assert out["id"] == panel.id
    assert out["type"] == "json"


async def test_panel_content_resource_text(panel_resources: PanelStore) -> None:
    panel = await panel_resources.create(Text(title="t", text="body"), source="main")
    assert panel.id is not None
    out = json.loads(await res_mod.panel_content_resource("main", panel.id))
    assert out == {"type": "text", "text": "body"}


async def test_panel_content_resource_json(panel_resources: PanelStore) -> None:
    panel = await panel_resources.create(Json(title="j", data={"k": "v"}), source="main")
    assert panel.id is not None
    out = json.loads(await res_mod.panel_content_resource("main", panel.id))
    assert out == {"type": "json", "data": {"k": "v"}}


async def test_panel_content_resource_dataset_is_reference(panel_resources: PanelStore) -> None:
    panel = await panel_resources.create(
        Dataset(title="d"), source="main", payload=b"PARQUET", payload_name="d.parquet"
    )
    assert panel.id is not None
    out = json.loads(await res_mod.panel_content_resource("main", panel.id))
    assert out["type"] == "dataset"
    assert out["payload_files_id"] == panel.payload_files_id


def test_register_panel_resources_registers_expected_uris() -> None:
    """register_panel_resources wires every handler onto the passed instance.

    The handlers must be registered *unwrapped* (real coroutine functions) so
    FastMCP awaits them rather than returning an un-awaited coroutine.
    """
    calls: list[tuple[dict[str, Any], Callable]] = []

    class FakeMCP:
        def resource(self, **kwargs: Any) -> Callable[[Callable], Callable]:
            def decorator(func: Callable) -> Callable:
                calls.append((kwargs, func))
                return func

            return decorator

    register_panel_resources(FakeMCP())

    uris = {kwargs["uri"] for kwargs, _func in calls}
    assert uris == {
        "panels://{source}",
        "panels://{source}/{panel_id}",
        "panels://{source}/{panel_id}/content",
    }
    for _kwargs, func in calls:
        assert inspect.iscoroutinefunction(func)
