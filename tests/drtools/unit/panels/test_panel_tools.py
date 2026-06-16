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

"""Unit tests for the panel CRUD tools and the MCP_SANDBOX guard."""

from contextlib import contextmanager

import pytest

from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.panels import access as access_mod
from datarobot_genai.drmcputils.panels.store import PanelStore
from datarobot_genai.drtools.panels import tools as tools_mod

from .conftest import FakeBlobStore


@pytest.fixture
def panel_tools(monkeypatch: pytest.MonkeyPatch) -> PanelStore:
    """Patch the tools to skip the entitlement guard and use an in-memory store."""
    store = PanelStore(FakeBlobStore())
    monkeypatch.setattr(tools_mod, "_require_mcp_sandbox", lambda: None)
    monkeypatch.setattr(tools_mod, "_get_store", lambda: store)
    return store


async def test_create_text_panel_then_list_get_delete(panel_tools: PanelStore) -> None:
    created = await tools_mod.create_text_panel(title="T", text="hi", source="staging")
    assert created["type"] == "text"
    panel_id = created["id"]
    assert panel_id

    listed = await tools_mod.list_panels(source="staging")
    assert listed["count"] == 1
    assert listed["source"] == "staging"
    assert listed["panels"][0]["id"] == panel_id

    got = await tools_mod.get_panel(panel_id)
    assert got["title"] == "T"
    assert got["text"] == "hi"

    deleted = await tools_mod.delete_panel(panel_id)
    assert deleted == {"deleted": True, "panel_id": panel_id}
    assert (await tools_mod.list_panels(source="staging"))["count"] == 0


async def test_create_json_panel(panel_tools: PanelStore) -> None:
    created = await tools_mod.create_json_panel(title="J", data={"a": 1}, source="main")
    assert created["type"] == "json"
    assert created["data"] == {"a": 1}


async def test_get_panel_requires_id(panel_tools: PanelStore) -> None:
    with pytest.raises(ToolError):
        await tools_mod.get_panel("")


async def test_tool_propagates_guard_denial(monkeypatch: pytest.MonkeyPatch) -> None:
    def _deny() -> None:
        raise ToolError("denied", kind=tools_mod.ToolErrorKind.AUTHENTICATION)

    monkeypatch.setattr(tools_mod, "_require_mcp_sandbox", _deny)
    monkeypatch.setattr(tools_mod, "_get_store", lambda: PanelStore(FakeBlobStore()))
    with pytest.raises(ToolError):
        await tools_mod.list_panels()


def test_require_mcp_sandbox_denies_when_entitlement_off(monkeypatch: pytest.MonkeyPatch) -> None:
    @contextmanager
    def _fake_client(**_kwargs: object):
        yield object()

    monkeypatch.setattr(access_mod, "request_user_dr_client", _fake_client)
    monkeypatch.setattr(
        access_mod.FeatureFlag,
        "is_enabled",
        staticmethod(lambda _name, *, client, **_kw: False),
    )
    with pytest.raises(ToolError):
        access_mod._require_mcp_sandbox()


def test_require_mcp_sandbox_allows_when_entitlement_on(monkeypatch: pytest.MonkeyPatch) -> None:
    @contextmanager
    def _fake_client(**_kwargs: object):
        yield object()

    monkeypatch.setattr(access_mod, "request_user_dr_client", _fake_client)
    monkeypatch.setattr(
        access_mod.FeatureFlag,
        "is_enabled",
        staticmethod(lambda _name, *, client, **_kw: True),
    )
    # Should not raise.
    access_mod._require_mcp_sandbox()


def test_get_store_returns_panelstore() -> None:
    assert isinstance(access_mod._get_store(), PanelStore)
