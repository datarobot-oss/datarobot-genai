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

"""Unit tests for sandbox-backed panel transform/filter tools (execute_code mocked)."""

import io
from typing import Any

import polars as pl
import pytest

from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.panels.models import Dataset
from datarobot_genai.drmcputils.panels.models import Text
from datarobot_genai.drmcputils.panels.store import PanelStore
from datarobot_genai.drtools.panels import transform as tf_mod
from datarobot_genai.drtools.panels.datasource import _rows_to_parquet

from .conftest import FakeBlobStore

_ROWS = [{"region": "EMEA", "rev": 5}, {"region": "AMER", "rev": 20}]


@pytest.fixture
def transform_env(monkeypatch: pytest.MonkeyPatch) -> PanelStore:
    """Patch the guard, inject an in-memory store, and stub the sandbox execute_code."""
    store = PanelStore(FakeBlobStore())
    monkeypatch.setattr(tf_mod, "_require_mcp_sandbox", lambda: None)
    monkeypatch.setattr(tf_mod, "_get_store", lambda: store)

    async def _fake_execute_code(
        code: str, *, inputs: dict[str, Any], **_kw: Any
    ) -> dict[str, Any]:
        # Simulate the sandbox: keep rows with rev > 10 (independent of the code text).
        kept = [row for row in inputs["rows"] if row["rev"] > 10]
        return {"return_value": kept, "stdout": "", "stderr": "", "exit_code": 0}

    monkeypatch.setattr(tf_mod, "_execute_code", _fake_execute_code)
    return store


async def _make_source(store: PanelStore) -> str:
    src = await store.create(
        Dataset(title="src"),
        source="staging",
        payload=_rows_to_parquet(_ROWS),
        payload_name="src.parquet",
    )
    assert src.id is not None
    return src.id


async def test_transform_panel_creates_child_with_lineage(transform_env: PanelStore) -> None:
    store = transform_env
    src_id = await _make_source(store)

    child = await tf_mod.transform_panel(
        panel_id=src_id,
        code="_return = df.to_dicts()",
        title="High revenue",
        source="staging",
    )

    assert child["type"] == "dataset"
    assert child["parents"] == [src_id]
    assert child["row_count"] == 1
    assert child["execution_context"]["kind"] == "sandbox_transform"
    assert child["execution_context"]["source_panel"] == src_id
    # Derived payload reads back as the sandbox's output rows.
    blobs: FakeBlobStore = store._blobs  # type: ignore[assignment]
    frame = pl.read_parquet(io.BytesIO(blobs.blobs[child["payload_files_id"]][0]))
    assert frame.to_dicts() == [{"region": "AMER", "rev": 20}]


async def test_filter_panel_creates_child(transform_env: PanelStore) -> None:
    store = transform_env
    src_id = await _make_source(store)

    child = await tf_mod.filter_panel(
        panel_id=src_id,
        where="pl.col('rev') > 10",
        title="Filtered",
        source="staging",
    )
    assert child["row_count"] == 1
    assert child["parents"] == [src_id]


async def test_transform_requires_dataset_payload(transform_env: PanelStore) -> None:
    store = transform_env
    text_panel = await store.create(Text(title="note", text="hi"), source="staging")
    assert text_panel.id is not None
    with pytest.raises(ToolError):
        await tf_mod.transform_panel(panel_id=text_panel.id, code="_return = []", title="x")


async def test_filter_panel_requires_where(transform_env: PanelStore) -> None:
    with pytest.raises(ToolError):
        await tf_mod.filter_panel(panel_id="p1", where="", title="x")


async def test_tools_fail_closed_when_sandbox_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tf_mod, "_require_mcp_sandbox", lambda: None)
    monkeypatch.setattr(tf_mod, "_execute_code", None)
    with pytest.raises(ToolError):
        await tf_mod.transform_panel(panel_id="p1", code="_return = []", title="x")


async def test_transform_empty_result_keeps_source_columns(
    transform_env: PanelStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = transform_env
    src_id = await _make_source(store)

    async def _empty_execute_code(code: str, *, inputs: dict[str, Any], **_kw: Any):
        return {"return_value": [], "stdout": "", "stderr": "", "exit_code": 0}

    monkeypatch.setattr(tf_mod, "_execute_code", _empty_execute_code)
    out = await tf_mod.transform_panel(
        panel_id=src_id, code="_return = []", title="Empty", source="staging"
    )
    assert out["row_count"] == 0
    assert out["columns"] == ["region", "rev"]
    payload = await store.get_payload(out["id"])
    frame = pl.read_parquet(io.BytesIO(payload))
    assert frame.columns == ["region", "rev"]
    assert frame.height == 0


async def test_transform_rejects_non_dataset_panel(transform_env: PanelStore) -> None:
    store = transform_env
    text_panel = await store.create(Text(title="T", text="hi"), source="staging")
    assert text_panel.id is not None
    with pytest.raises(ToolError) as exc_info:
        await tf_mod.transform_panel(
            panel_id=text_panel.id, code="_return = []", title="X", source="staging"
        )
    assert "only Dataset panels" in str(exc_info.value)


def test_sandbox_import_is_wired() -> None:
    # Guards against the defensive import silently failing (wrong module path):
    # with the sandbox backend present in this repo, _execute_code must be bound.
    assert tf_mod._execute_code is not None


async def test_transform_rejects_non_dict_rows(
    transform_env: PanelStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = transform_env
    src_id = await _make_source(store)

    async def _bad_execute_code(code: str, *, inputs: dict[str, Any], **_kw: Any):
        return {"return_value": [1, 2, 3], "stdout": "", "stderr": "", "exit_code": 0}

    monkeypatch.setattr(tf_mod, "_execute_code", _bad_execute_code)
    with pytest.raises(ToolError):
        await tf_mod.transform_panel(
            panel_id=src_id, code="_return = [1,2,3]", title="X", source="staging"
        )
