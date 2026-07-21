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

"""Unit tests for connector-sourced Dataset panels."""

import io
from typing import Any

import polars as pl
import pytest

from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.panels.store import PanelStore
from datarobot_genai.drtools.panels import datasource as ds_mod

from .conftest import FakeBlobStore


def test_rows_to_parquet_roundtrips() -> None:
    rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    parquet = ds_mod._rows_to_parquet(rows)
    frame = pl.read_parquet(io.BytesIO(parquet))
    assert frame.columns == ["a", "b"]
    assert frame.to_dicts() == rows


@pytest.fixture
def patched_datasource(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[FakeBlobStore, list[dict[str, Any]]]:
    """Skip the entitlement guard, inject an in-memory store, and stub the connector query."""
    blobs = FakeBlobStore()
    rows = [{"region": "EMEA", "rev": 10}, {"region": "AMER", "rev": 20}]
    monkeypatch.setattr(ds_mod, "_require_mcp_sandbox", lambda: None)
    monkeypatch.setattr(ds_mod, "_get_store", lambda: PanelStore(blobs))

    async def _fake_query(*, datastore_id: str, sql: str, limit: int = 100) -> dict[str, Any]:
        return {"rows": rows, "columns": ["region", "rev"], "row_count": len(rows)}

    monkeypatch.setattr(ds_mod, "catalog_query_datastore", _fake_query)
    return blobs, rows


async def test_create_dataset_panel_from_connector(
    patched_datasource: tuple[FakeBlobStore, list[dict[str, Any]]],
) -> None:
    blobs, rows = patched_datasource
    created = await ds_mod.create_dataset_panel_from_connector(
        datastore_id="ds-1",
        sql="SELECT region, rev FROM sales WHERE rev > 5",
        title="Sales",
        source="staging",
    )

    assert created["type"] == "dataset"
    assert created["row_count"] == 2
    assert created["columns"] == ["region", "rev"]
    assert created["execution_context"] == {
        "kind": "connector_query",
        "datastore_id": "ds-1",
        "sql": "SELECT region, rev FROM sales WHERE rev > 5",
    }
    # The Parquet payload was stored and is readable back as the queried rows.
    assert created["payload_files_id"] is not None
    frame = pl.read_parquet(io.BytesIO(blobs.container[created["payload_path"]]))
    assert frame.to_dicts() == rows


async def test_create_dataset_panel_requires_sql(
    patched_datasource: tuple[FakeBlobStore, list[dict[str, Any]]],
) -> None:
    with pytest.raises(ToolError):
        await ds_mod.create_dataset_panel_from_connector(datastore_id="ds-1", sql="", title="X")


def test_rows_to_parquet_empty_keeps_columns() -> None:
    parquet = ds_mod._rows_to_parquet([], ["region", "rev"])
    frame = pl.read_parquet(io.BytesIO(parquet))
    assert frame.columns == ["region", "rev"]
    assert frame.height == 0


async def test_create_dataset_panel_empty_query_keeps_column_schema(
    patched_datasource: tuple[FakeBlobStore, list[dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blobs, _rows = patched_datasource

    async def _empty_query(*, datastore_id: str, sql: str, limit: int = 100) -> dict[str, Any]:
        return {"rows": [], "columns": ["region", "rev"], "row_count": 0}

    monkeypatch.setattr(ds_mod, "catalog_query_datastore", _empty_query)
    created = await ds_mod.create_dataset_panel_from_connector(
        datastore_id="ds-1", sql="SELECT 1", title="Empty", source="staging"
    )
    assert created["columns"] == ["region", "rev"]
    payload = await blobs.get(created["payload_path"])
    frame = pl.read_parquet(io.BytesIO(payload))
    assert frame.columns == ["region", "rev"]
    assert frame.height == 0


async def test_preview_dataset_panel(
    patched_datasource: tuple[FakeBlobStore, list[dict[str, Any]]],
) -> None:
    _blobs, rows = patched_datasource
    created = await ds_mod.create_dataset_panel_from_connector(
        datastore_id="ds-1", sql="SELECT 1", title="Sales", source="staging"
    )
    preview = await ds_mod.preview_dataset_panel(created["id"], sample_size=1)
    assert preview["columns"] == ["region", "rev"]
    assert preview["row_count"] == len(rows)
    assert preview["sample"] == rows[:1]
    assert preview["dtypes"]["rev"] == "Int64"
    assert preview["execution_context"]["kind"] == "connector_query"


async def test_preview_dataset_panel_rejects_non_dataset(
    patched_datasource: tuple[FakeBlobStore, list[dict[str, Any]]],
) -> None:
    from datarobot_genai.drmcputils.panels.models import Text

    store = ds_mod._get_store()
    text_panel = await store.create(Text(title="T", text="hi"), source="staging")
    with pytest.raises(ToolError):
        await ds_mod.preview_dataset_panel(text_panel.id)


def test_rows_to_parquet_normalizes_sparse_rows_to_columns() -> None:
    rows = [{"a": 1}, {"b": "y"}]  # connector JSON often omits null keys
    parquet = ds_mod._rows_to_parquet(rows, ["a", "b"])
    frame = pl.read_parquet(io.BytesIO(parquet))
    assert frame.columns == ["a", "b"]
    assert frame.to_dicts() == [{"a": 1, "b": None}, {"a": None, "b": "y"}]
