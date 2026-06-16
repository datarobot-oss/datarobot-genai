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

"""Unit tests for the panel review tools (inspect_panel, view_json_panel)."""

import pytest

from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.panels.models import Dataset
from datarobot_genai.drmcputils.panels.models import Json
from datarobot_genai.drmcputils.panels.models import Text
from datarobot_genai.drmcputils.panels.store import PanelStore
from datarobot_genai.drtools.panels import review as review_mod
from datarobot_genai.drtools.panels.truncate import truncate_for_llm

from .conftest import FakeBlobStore


@pytest.fixture
def review_env(monkeypatch: pytest.MonkeyPatch) -> PanelStore:
    store = PanelStore(FakeBlobStore())
    monkeypatch.setattr(review_mod, "_require_mcp_sandbox", lambda: None)
    monkeypatch.setattr(review_mod, "_get_store", lambda: store)
    return store


async def test_inspect_panel_walks_lineage(review_env: PanelStore) -> None:
    root = await review_env.create(
        Dataset(title="src", execution_context={"kind": "connector_query", "sql": "SELECT 1"}),
        source="staging",
    )
    child = await review_env.create(
        Dataset(
            title="filtered",
            parents=[root.id],
            execution_context={"kind": "sandbox_transform", "code": "x = 1"},
        ),
        source="staging",
    )

    out = await review_mod.inspect_panel(child.id)
    assert out["panel_id"] == child.id
    assert out["node_count"] == 2
    assert out["graph"][child.id]["parents"] == [root.id]
    assert out["graph"][root.id]["execution_context"]["kind"] == "connector_query"


async def test_inspect_panel_handles_missing_ancestor(review_env: PanelStore) -> None:
    child = await review_env.create(
        Dataset(title="orphan-child", parents=["gone123"]), source="staging"
    )
    out = await review_mod.inspect_panel(child.id)
    assert out["graph"]["gone123"] == {"id": "gone123", "error": "panel unavailable"}


async def test_view_json_panel_truncates(review_env: PanelStore) -> None:
    panel = await review_env.create(
        Json(title="J", data={"items": list(range(100))}), source="staging"
    )
    out = await review_mod.view_json_panel(panel.id)
    assert out["title"] == "J"
    assert len(out["data"]["items"]) == 6  # 5 items + truncation marker
    assert "95 more items" in out["data"]["items"][-1]


async def test_view_json_panel_rejects_non_json(review_env: PanelStore) -> None:
    text = await review_env.create(Text(title="T", text="hi"), source="staging")
    with pytest.raises(ToolError):
        await review_mod.view_json_panel(text.id)


def test_truncate_for_llm_depth_and_strings() -> None:
    deep = {"a": {"b": {"c": {"d": {"e": {"f": {"g": 1}}}}}}}
    out = truncate_for_llm(deep)
    assert out["a"]["b"]["c"]["d"]["e"]["f"] == "{...} (1 keys)"
    long_string = "x" * 300
    assert truncate_for_llm(long_string).endswith("(100 more chars)")


async def test_inspect_panel_tolerates_toolerror_ancestor(
    review_env: PanelStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    from datarobot_genai.drmcputils.exceptions import ToolErrorKind

    child = await review_env.create(
        Dataset(title="child", parents=["stale-parent"]), source="staging"
    )
    original_get = review_env.get

    async def _get(pid: str):
        if pid == "stale-parent":
            raise ToolError("not found", kind=ToolErrorKind.NOT_FOUND)
        return await original_get(pid)

    monkeypatch.setattr(review_env, "get", _get)
    out = await review_mod.inspect_panel(child.id)
    assert out["graph"]["stale-parent"] == {"id": "stale-parent", "error": "panel unavailable"}
