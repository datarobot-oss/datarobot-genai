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

"""Unit tests for the panel-composition tools in ``drtools/panels/compose.py``.

Ported from wren-mcp's ``test_facades_compose.py`` (MODEL-24090). Two flavors:

* The panel-store / SDK boundary is mocked — an in-memory ``FakeBlobStore``
  backs the ``PanelStore``, the ``ENABLE_MCP_SANDBOX`` guard is neutralized, and
  the DataRobot SDK module (``dr``) plus the request-scoped client are patched
  so nothing touches the network.
* The deterministic cores — ``apply_what_if``'s adjustment math and
  ``query_datasets_to_panel``'s polars ``SQLContext`` execution — run against
  *real* polars frames with no mocks, exercising the exact expressions the
  tools apply.
"""

import base64
import contextlib
import datetime
import io
from typing import Any
from unittest.mock import MagicMock

import polars as pl
import pytest

from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.panels.models import Dataset
from datarobot_genai.drmcputils.panels.models import Json
from datarobot_genai.drmcputils.panels.store import PanelStore
from datarobot_genai.drtools.panels import compose

from .conftest import FakeBlobStore


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> PanelStore:
    """Inject an in-memory store and skip the entitlement guard."""
    blobs = FakeBlobStore()
    panel_store = PanelStore(blobs)
    monkeypatch.setattr(compose, "_require_mcp_sandbox", lambda: None)
    monkeypatch.setattr(compose, "_get_store", lambda: panel_store)
    return panel_store


@pytest.fixture
def fake_sdk(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch the SDK module, the request-scoped client, and the guard to be inert."""
    monkeypatch.setattr(compose, "_require_mcp_sandbox", lambda: None)
    dr_mock = MagicMock(name="dr")
    monkeypatch.setattr(compose, "dr", dr_mock)

    client = MagicMock(name="ThreadSafeDataRobotClient")
    client.return_value.request_user_client.return_value = contextlib.nullcontext()
    monkeypatch.setattr(compose, "ThreadSafeDataRobotClient", client)
    return dr_mock


async def _make_dataset_panel(
    store: PanelStore, frame: pl.DataFrame, *, source: str = "staging"
) -> str:
    buf = io.BytesIO()
    frame.write_parquet(buf)
    panel = Dataset(
        title="src",
        row_count=frame.height,
        columns=frame.columns,
    )
    created = await store.create(
        panel,
        source=source,
        payload=buf.getvalue(),
        payload_name="src.parquet",
        content_type=compose._PARQUET_CONTENT_TYPE,
    )
    return created.id  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# apply_what_if adjustment math — real polars frames, no mocks.
# --------------------------------------------------------------------------- #


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [
                datetime.date(2025, 1, 1),
                datetime.date(2025, 2, 1),
                datetime.date(2025, 3, 1),
            ],
            "series": ["a", "a", "b"],
            "price": [100.0, 200.0, 300.0],
        }
    )


def test_apply_mul() -> None:
    # GIVEN a frame of prices — WHEN a mul adjustment is applied
    out = compose._apply_adjustments(
        _frame(),
        [{"column": "price", "op": "mul", "value": 0.5}],
        date_col="date",
        series_col="series",
    )
    # THEN every price is halved
    assert out["price"].to_list() == [50.0, 100.0, 150.0]


def test_apply_add() -> None:
    # GIVEN a frame of prices — WHEN an add adjustment is applied
    out = compose._apply_adjustments(
        _frame(),
        [{"column": "price", "op": "add", "value": 10.0}],
        date_col="date",
        series_col="series",
    )
    # THEN every price is shifted by the addend
    assert out["price"].to_list() == [110.0, 210.0, 310.0]


def test_apply_set() -> None:
    # GIVEN a frame of prices — WHEN a set adjustment is applied
    out = compose._apply_adjustments(
        _frame(),
        [{"column": "price", "op": "set", "value": 1.0}],
        date_col="date",
        series_col="series",
    )
    # THEN every price is replaced by the constant
    assert out["price"].to_list() == [1.0, 1.0, 1.0]


def test_apply_date_window() -> None:
    # GIVEN an adjustment bounded by an inclusive date window
    out = compose._apply_adjustments(
        _frame(),
        [
            {
                "column": "price",
                "op": "set",
                "value": 0.0,
                "from_date": "2025-01-15",
                "to_date": "2025-02-15",
            }
        ],
        date_col="date",
        series_col="series",
    )
    # THEN only the 2025-02-01 row falls in the window
    assert out["price"].to_list() == [100.0, 0.0, 300.0]


def test_apply_date_window_datetime_column_includes_boundary_times() -> None:
    # GIVEN a pl.Datetime date column with a row at 10:00 on the boundary date
    frame = pl.DataFrame(
        {
            "date": [
                datetime.datetime(2025, 1, 1, 10, 0),
                datetime.datetime(2025, 1, 2, 0, 0),
            ],
            "series": ["a", "a"],
            "price": [100.0, 200.0],
        }
    )
    # WHEN the window's inclusive upper bound is that boundary date
    out = compose._apply_adjustments(
        frame,
        [
            {
                "column": "price",
                "op": "set",
                "value": 0.0,
                "from_date": "2024-12-01",
                "to_date": "2025-01-01",
            }
        ],
        date_col="date",
        series_col="series",
    )
    # THEN comparisons run on the date portion, so the 10:00 row still matches
    assert out["price"].to_list() == [0.0, 200.0]


def test_apply_series_filter() -> None:
    # GIVEN an adjustment scoped to series "b" — WHEN applied
    out = compose._apply_adjustments(
        _frame(),
        [{"column": "price", "op": "mul", "value": 2.0, "series": ["b"]}],
        date_col="date",
        series_col="series",
    )
    # THEN only the series-"b" row is adjusted
    assert out["price"].to_list() == [100.0, 200.0, 600.0]


def test_apply_sequential() -> None:
    # GIVEN two adjustments — WHEN applied in order
    out = compose._apply_adjustments(
        _frame(),
        [
            {"column": "price", "op": "add", "value": 100.0},
            {"column": "price", "op": "mul", "value": 2.0},
        ],
        date_col="date",
        series_col="series",
    )
    # THEN the second operates on the first's output ((p + 100) * 2)
    assert out["price"].to_list() == [400.0, 600.0, 800.0]


def test_apply_unknown_column_raises() -> None:
    # GIVEN an adjustment naming a column not in the frame — THEN it is rejected
    with pytest.raises(ToolError):
        compose._apply_adjustments(
            _frame(),
            [{"column": "nope", "op": "mul", "value": 1.0}],
            date_col="date",
            series_col="series",
        )


def test_apply_bad_op_raises() -> None:
    # GIVEN an unsupported op — THEN it is rejected
    with pytest.raises(ToolError):
        compose._apply_adjustments(
            _frame(),
            [{"column": "price", "op": "divide", "value": 1.0}],
            date_col="date",
            series_col="series",
        )


def test_apply_series_on_single_series_raises() -> None:
    # GIVEN a series filter against a single-series deployment — THEN it is rejected
    with pytest.raises(ToolError):
        compose._apply_adjustments(
            _frame(),
            [{"column": "price", "op": "mul", "value": 2.0, "series": ["a"]}],
            date_col="date",
            series_col=None,
        )


# --------------------------------------------------------------------------- #
# create_dataset_panel_from_catalog (wren: get_datarobot_dataset_as_panel)
# --------------------------------------------------------------------------- #


async def test_create_dataset_panel_from_catalog(store: PanelStore, fake_sdk: MagicMock) -> None:
    import pandas as pd

    # GIVEN an AI Catalog dataset materializable as a dataframe
    dataset_obj = MagicMock()
    dataset_obj.get_as_dataframe.return_value = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    fake_sdk.Dataset.get.return_value = dataset_obj

    # WHEN it is converted into a Dataset panel
    result = await compose.create_dataset_panel_from_catalog(
        dataset_id="ds-1", title="My Data", source="staging"
    )

    # THEN the panel metadata and the stored Parquet payload match the dataset
    assert result["type"] == "dataset"
    assert result["row_count"] == 3
    assert result["columns"] == ["a", "b"]
    assert result["execution_context"] == {
        "kind": "catalog_dataset",
        "dataset_id": "ds-1",
    }
    frame = pl.read_parquet(io.BytesIO(await store.get_payload(result["id"])))
    assert frame.to_dicts() == [
        {"a": 1, "b": "x"},
        {"a": 2, "b": "y"},
        {"a": 3, "b": "z"},
    ]


async def test_create_dataset_panel_from_catalog_limit(
    store: PanelStore, fake_sdk: MagicMock
) -> None:
    import pandas as pd

    # GIVEN a 5-row catalog dataset
    dataset_obj = MagicMock()
    dataset_obj.get_as_dataframe.return_value = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
    fake_sdk.Dataset.get.return_value = dataset_obj

    # WHEN converted with limit=2 — THEN only the head is materialized
    result = await compose.create_dataset_panel_from_catalog(
        dataset_id="ds-1", title="Capped", limit=2
    )
    assert result["row_count"] == 2


# --------------------------------------------------------------------------- #
# upload_dataset_panel_to_catalog (wren: upload_panel_dataset_to_datarobot)
# --------------------------------------------------------------------------- #


async def test_upload_panel(store: PanelStore, monkeypatch: pytest.MonkeyPatch) -> None:
    # GIVEN a Dataset panel with a Parquet payload
    panel_id = await _make_dataset_panel(store, pl.DataFrame({"a": [1, 2], "b": ["x", "y"]}))

    captured: dict[str, Any] = {}

    async def _fake_upload(*, file_content_base64: str, dataset_filename: str) -> dict[str, Any]:
        captured["b64"] = file_content_base64
        captured["filename"] = dataset_filename
        return {"dataset_id": "new-1", "dataset_name": dataset_filename}

    monkeypatch.setattr(compose, "catalog_upload_dataset", _fake_upload)

    # WHEN it is uploaded to the AI Catalog with an explicit name
    result = await compose.upload_dataset_panel_to_catalog(panel_id=panel_id, name="sales")

    # THEN the payload went up as base64 CSV under that name
    assert result == {"dataset_id": "new-1", "dataset_name": "sales.csv"}
    assert captured["filename"] == "sales.csv"
    csv = base64.b64decode(captured["b64"]).decode("utf-8")
    assert csv == "a,b\n1,x\n2,y\n"


async def test_upload_panel_default_name(
    store: PanelStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    # GIVEN a Dataset panel titled "src" (see _make_dataset_panel)
    panel_id = await _make_dataset_panel(store, pl.DataFrame({"a": [1]}))
    captured: dict[str, Any] = {}

    async def _fake_upload(*, file_content_base64: str, dataset_filename: str) -> dict[str, Any]:
        captured["filename"] = dataset_filename
        return {"dataset_id": "x"}

    monkeypatch.setattr(compose, "catalog_upload_dataset", _fake_upload)

    # WHEN uploaded without a name — THEN the panel title becomes the filename
    await compose.upload_dataset_panel_to_catalog(panel_id=panel_id)
    assert captured["filename"] == "src.csv"


async def test_upload_panel_non_dataset_raises(store: PanelStore) -> None:
    # GIVEN a non-Dataset panel — THEN uploading it is rejected
    created = await store.create(Json(title="j", data={"k": 1}), source="staging")
    with pytest.raises(ToolError):
        await compose.upload_dataset_panel_to_catalog(panel_id=created.id)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# query_datasets_to_panel (wren: query_datarobot_dataset) — real SQLContext.
# --------------------------------------------------------------------------- #


@pytest.fixture
def patch_materialize(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace SDK dataset materialization with in-memory frames t0, t1."""
    import pandas as pd

    frames = {
        "d0": pd.DataFrame({"id": [1, 2, 3], "v": [10, 20, 30]}),
        "d1": pd.DataFrame({"id": [1, 2, 3], "w": [100, 200, 300]}),
    }

    async def _fake_materialize(dataset_ids: list[str]) -> dict[str, pl.DataFrame]:
        return {f"t{i}": pl.from_pandas(frames[ds_id]) for i, ds_id in enumerate(dataset_ids)}

    monkeypatch.setattr(compose, "_materialize_frames", _fake_materialize)


async def test_query_dataset_binds_tables(store: PanelStore, patch_materialize: None) -> None:
    # GIVEN two catalog datasets bound as t0, t1 — WHEN a join query runs
    result = await compose.query_datasets_to_panel(
        query="SELECT t0.id, t0.v, t1.w FROM t0 JOIN t1 ON t0.id = t1.id WHERE t0.v > 10",
        title="Joined",
        description="joined data",
        dataset_ids=["d0", "d1"],
    )
    # THEN the joined/filtered result is stored as a Dataset panel
    assert result["type"] == "dataset"
    assert result["row_count"] == 2
    frame = pl.read_parquet(io.BytesIO(await store.get_payload(result["id"])))
    assert frame.sort("id").to_dicts() == [
        {"id": 2, "v": 20, "w": 200},
        {"id": 3, "v": 30, "w": 300},
    ]


async def test_query_dataset_always_writes_staging(
    store: PanelStore, patch_materialize: None
) -> None:
    # GIVEN persist=True (wren compat: it only controlled DR-side Wrangle
    # artifact preservation, a no-op here) — WHEN the query runs
    result = await compose.query_datasets_to_panel(
        query="SELECT * FROM t0",
        title="All",
        description="all",
        dataset_ids=["d0"],
        persist=True,
    )
    # THEN the result panel still lands in staging, never main
    staging = await store.list(source="staging")
    assert any(p.id == result["id"] for p in staging)
    main = await store.list(source="main")
    assert not any(p.id == result["id"] for p in main)


async def test_query_dataset_bad_sql_raises(store: PanelStore, patch_materialize: None) -> None:
    # GIVEN SQL the polars engine cannot execute — THEN the error suggests
    # transform_panel as the fallback
    with pytest.raises(ToolError) as exc:
        await compose.query_datasets_to_panel(
            query="SELECT bogusfn(v) FROM t0",
            title="Bad",
            description="bad",
            dataset_ids=["d0"],
        )
    assert "transform_panel" in str(exc.value)


# --------------------------------------------------------------------------- #
# get_prediction_history
# --------------------------------------------------------------------------- #


async def test_prediction_history(store: PanelStore, monkeypatch: pytest.MonkeyPatch) -> None:
    # GIVEN a deployment with stored prediction results
    async def _fake_history(**kwargs: Any) -> dict[str, Any]:
        return {
            "deployment_id": "dep-1",
            "row_count": 2,
            "rows": [
                {"timestamp": "2025-01-01", "prediction": 1.0},
                {"timestamp": "2025-01-02", "prediction": 2.0},
            ],
            "has_more": False,
        }

    monkeypatch.setattr(compose, "deployment_get_prediction_history", _fake_history)

    # WHEN the history is fetched
    result = await compose.get_prediction_history(deployment_id="dep-1")

    # THEN the rows are stored as a Dataset panel and paging metadata surfaces
    assert result["panel"]["type"] == "dataset"
    assert result["panel"]["row_count"] == 2
    assert result["row_count"] == 2
    assert result["has_more"] is False
    frame = pl.read_parquet(io.BytesIO(await store.get_payload(result["panel"]["id"])))
    assert frame.columns == ["timestamp", "prediction"]


async def test_prediction_history_empty(store: PanelStore, monkeypatch: pytest.MonkeyPatch) -> None:
    # GIVEN a deployment with no stored predictions
    async def _fake_history(**kwargs: Any) -> dict[str, Any]:
        return {"deployment_id": "dep-1", "row_count": 0, "rows": [], "has_more": False}

    monkeypatch.setattr(compose, "deployment_get_prediction_history", _fake_history)
    # WHEN fetched — THEN an empty panel is still created
    result = await compose.get_prediction_history(deployment_id="dep-1")
    assert result["panel"]["row_count"] == 0


# --------------------------------------------------------------------------- #
# get_autopilot_status
# --------------------------------------------------------------------------- #


def _fake_project(*, autopilot_done: bool, queued: int, running: int, models: int) -> MagicMock:
    from datarobot.enums import QUEUE_STATUS

    project = MagicMock()
    project.get_status.return_value = {
        "autopilot_done": autopilot_done,
        "stage": "modeling",
        "stage_description": "Building models",
    }
    jobs = [MagicMock(status=QUEUE_STATUS.QUEUE) for _ in range(queued)]
    jobs += [MagicMock(status=QUEUE_STATUS.INPROGRESS) for _ in range(running)]
    project.get_model_jobs.return_value = jobs
    project.get_models.return_value = [MagicMock() for _ in range(models)]
    project.use_case_id = "uc-1"
    return project


async def test_autopilot_status_complete(fake_sdk: MagicMock) -> None:
    # GIVEN a project whose AutoPilot has finished
    fake_sdk.Project.get.return_value = _fake_project(
        autopilot_done=True, queued=0, running=0, models=5
    )
    # WHEN status is fetched — THEN the report starts with the completion line
    result = await compose.get_autopilot_status(project_id="p-1")
    assert result["report"].startswith("AutoPilot status: complete")
    assert result["autopilot_done"] is True
    assert result["models_on_leaderboard"] == 5


async def test_autopilot_status_in_progress(fake_sdk: MagicMock) -> None:
    # GIVEN a project mid-AutoPilot with queued and running jobs
    fake_sdk.Project.get.return_value = _fake_project(
        autopilot_done=False, queued=3, running=2, models=1
    )
    # WHEN status is fetched — THEN the report shows in-progress counts
    result = await compose.get_autopilot_status(project_id="p-1")
    assert result["report"].startswith("AutoPilot status: in progress")
    assert result["models_queued"] == 3
    assert result["models_running"] == 2


# --------------------------------------------------------------------------- #
# apply_what_if — full tool with mocked deployment introspection.
# --------------------------------------------------------------------------- #


def _patch_partition(fake_sdk: MagicMock, *, date_col: str | None, series_col: str | None) -> None:
    deployment = MagicMock()
    deployment.model = {"project_id": "proj-1"}
    fake_sdk.Deployment.get.return_value = deployment

    project = MagicMock()
    if date_col is None:
        project.list_datetime_partition_spec.return_value = None
    else:
        spec = {
            "datetime_partition_column": date_col,
            "multiseries_id_columns": [series_col] if series_col else [],
        }
        project.list_datetime_partition_spec.return_value = spec
    fake_sdk.Project.get.return_value = project


async def test_apply_what_if_writes_child_panel(store: PanelStore, fake_sdk: MagicMock) -> None:
    # GIVEN a datetime-partitioned deployment and a scoring panel
    _patch_partition(fake_sdk, date_col="date", series_col="series")
    panel_id = await _make_dataset_panel(store, _frame())

    # WHEN a series-scoped mul adjustment is applied
    result = await compose.apply_what_if(
        panel_id=panel_id,
        deployment_id="dep-1",
        adjustments=[{"column": "price", "op": "mul", "value": 2.0, "series": ["a"]}],
    )

    # THEN a lineage-linked child panel holds the adjusted rows
    assert result["type"] == "dataset"
    assert result["parents"] == [panel_id]
    assert result["execution_context"]["kind"] == "what_if"
    frame = pl.read_parquet(io.BytesIO(await store.get_payload(result["id"])))
    assert frame["price"].to_list() == [200.0, 400.0, 300.0]


async def test_apply_what_if_non_datetime_raises(store: PanelStore, fake_sdk: MagicMock) -> None:
    # GIVEN a deployment that is not datetime-partitioned
    _patch_partition(fake_sdk, date_col=None, series_col=None)
    panel_id = await _make_dataset_panel(store, _frame())
    # WHEN applying what-if — THEN it is rejected before any adjustment runs
    with pytest.raises(ToolError):
        await compose.apply_what_if(
            panel_id=panel_id,
            deployment_id="dep-1",
            adjustments=[{"column": "price", "op": "mul", "value": 2.0}],
        )


# --------------------------------------------------------------------------- #
# predict_with_deployment — mocked predict_score_inline_realtime delegate.
# --------------------------------------------------------------------------- #


def _patch_predict(monkeypatch: pytest.MonkeyPatch, predictions_csv: str) -> dict[str, Any]:
    """Mock the inline realtime scoring delegate; record its kwargs."""
    captured: dict[str, Any] = {}

    async def _fake_predict(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"type": "inline", "data": predictions_csv, "show_explanations": False}

    monkeypatch.setattr(compose, "predict_score_inline_realtime", _fake_predict)
    return captured


async def test_predict_serializes_panel_to_csv_and_stores_child(
    store: PanelStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    # GIVEN a scoring Dataset panel and a scoring delegate returning inline CSV
    panel_id = await _make_dataset_panel(store, pl.DataFrame({"a": [1, 2], "b": ["x", "y"]}))
    captured = _patch_predict(monkeypatch, "a,prediction\n1,0.9\n2,0.1\n")

    # WHEN predictions are made with the deployment
    result = await compose.predict_with_deployment(panel_id=panel_id, deployment_id="dep-1")

    # THEN the scoring frame went over the wire as CSV (header + rows)
    assert captured["dataset"] == "a,b\n1,x\n2,y\n"
    assert captured["deployment_id"] == "dep-1"
    assert captured["max_explanations"] == 0
    assert captured["forecast_point"] is None

    # THEN a child Dataset panel landed in staging, linked to the scoring panel
    assert result["type"] == "dataset"
    assert result["parents"] == [panel_id]
    assert result["title"] == "Predictions (src)"
    assert result["execution_context"] == {
        "kind": "prediction",
        "deployment_id": "dep-1",
    }
    assert result["row_count"] == 2
    assert result["columns"] == ["a", "prediction"]
    staging = await store.list(source="staging")
    assert any(p.id == result["id"] for p in staging)

    frame = pl.read_parquet(io.BytesIO(await store.get_payload(result["id"])))
    assert frame["prediction"].to_list() == [0.9, 0.1]


async def test_predict_passes_ts_args_and_explanations(
    store: PanelStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    # GIVEN a time-series deployment call with explanations requested
    panel_id = await _make_dataset_panel(store, pl.DataFrame({"a": [1]}))
    captured = _patch_predict(monkeypatch, "a,prediction\n1,0.5\n")

    # WHEN predicting with add_explanations and a forecast point
    result = await compose.predict_with_deployment(
        panel_id=panel_id,
        deployment_id="dep-ts",
        add_explanations=True,
        forecast_point="2025-06-01",
    )

    # THEN wren's mapping add_explanations=True -> max_explanations=10 holds
    assert captured["max_explanations"] == 10
    assert captured["forecast_point"] == "2025-06-01"
    assert result["title"] == "Predictions (src 2025-06-01) with explanations"
    assert result["description"] == "Predictions with explanations"


async def test_predict_non_dataset_panel_raises(
    store: PanelStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    # GIVEN a non-Dataset panel — THEN scoring it is rejected
    created = await store.create(Json(title="j", data={"k": 1}), source="staging")
    _patch_predict(monkeypatch, "a\n1\n")
    with pytest.raises(ToolError):
        await compose.predict_with_deployment(panel_id=created.id, deployment_id="dep-1")  # type: ignore[arg-type]


async def test_predict_empty_delegate_response_raises(
    store: PanelStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    # GIVEN a delegate response with no inline data
    panel_id = await _make_dataset_panel(store, pl.DataFrame({"a": [1]}))

    async def _fake_predict(**kwargs: Any) -> dict[str, Any]:
        return {"type": "inline", "data": None}

    monkeypatch.setattr(compose, "predict_score_inline_realtime", _fake_predict)
    # WHEN predicting — THEN the empty upstream response is surfaced as an error
    with pytest.raises(ToolError):
        await compose.predict_with_deployment(panel_id=panel_id, deployment_id="dep-1")


async def test_predict_missing_args_raise(store: PanelStore) -> None:
    # GIVEN blank ids — THEN validation rejects the call before any work
    with pytest.raises(ToolError):
        await compose.predict_with_deployment(panel_id="", deployment_id="dep-1")
    with pytest.raises(ToolError):
        await compose.predict_with_deployment(panel_id="p-1", deployment_id="")
