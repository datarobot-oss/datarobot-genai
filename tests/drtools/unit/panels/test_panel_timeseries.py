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

"""Unit tests for the time-series scoring dataset panel builder.

Ported from wren-mcp's ``test_facades_heavy.py`` (TS half). The
deployment-introspection SDK boundary is mocked, but the scoring-frame math
(``_build_scoring_frame`` and ``_coerce_column_to_date``) runs on *real* polars
frames; the ``_build_scoring_frame`` cases are adapted from wren's
``test_dr_helpers.TestGetScoringDataset`` fixtures.
"""

import contextlib
import datetime
import io
from datetime import datetime as dtime
from typing import Any
from unittest.mock import MagicMock

import polars as pl
import pytest

from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.panels.models import Dataset
from datarobot_genai.drmcputils.panels.models import Json
from datarobot_genai.drmcputils.panels.store import PanelStore
from datarobot_genai.drtools.panels import timeseries as ts_mod

from .conftest import FakeBlobStore


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> PanelStore:
    """Inject an in-memory store and skip the entitlement guard."""
    panel_store = PanelStore(FakeBlobStore())
    monkeypatch.setattr(ts_mod, "_require_mcp_sandbox", lambda: None)
    monkeypatch.setattr(ts_mod, "_get_store", lambda: panel_store)
    return panel_store


async def _make_dataset_panel(
    store: PanelStore,
    frame: pl.DataFrame,
    *,
    title: str = "src",
    source: str = "staging",
) -> str:
    buf = io.BytesIO()
    frame.write_parquet(buf)
    panel = Dataset(title=title, row_count=frame.height, columns=frame.columns)
    created = await store.create(
        panel,
        source=source,
        payload=buf.getvalue(),
        payload_name="src.parquet",
        content_type=ts_mod._PARQUET_CONTENT_TYPE,
    )
    assert created.id is not None
    return created.id


def _reqs(**overrides: Any) -> ts_mod.ScoringDataRequirements:
    base: dict[str, Any] = dict(
        target="target",
        multi_series_id_column="series_id",
        date_time_column="timestamp",
        effective_feature_derivation_window_start=-7,
        forecast_distance=3,
        windows_basis_unit="DAY",
        known_in_advance_features=[],
        date_format="%Y-%m-%dT%H:%M:%S%z",
    )
    base.update(overrides)
    return ts_mod.ScoringDataRequirements(**base)


# --------------------------------------------------------------------------- #
# _build_scoring_frame -- real polars frames, adapted from wren's fixtures.
# --------------------------------------------------------------------------- #


def test_build_scoring_frame_handles_rfc3339_strings() -> None:
    # GIVEN a daily deployment spec and a dataset with RFC 3339 timestamp strings
    reqs = _reqs(effective_feature_derivation_window_start=-7, forecast_distance=3)
    dataset = pl.DataFrame(
        {
            "timestamp": [
                "2024-06-01T00:00:00Z",
                "2024-06-02T00:30:00Z",
                "2024-06-03T00:00:00Z",
                "2024-06-04T00:00:00Z",
            ],
            "series_id": ["A"] * 4,
            "target": [1.0, 2.0, 3.0, 4.0],
        }
    )

    # WHEN the scoring frame is built for a 2024-06-04 forecast point
    result = ts_mod._build_scoring_frame(dataset, reqs, dtime(2024, 6, 4))

    # THEN the strings are parsed to dates and the future window is appended
    assert result["timestamp"].dtype == pl.Date
    assert result["timestamp"].to_list() == [
        datetime.date(2024, 6, 1),
        datetime.date(2024, 6, 2),
        datetime.date(2024, 6, 3),
        datetime.date(2024, 6, 4),
        datetime.date(2024, 6, 5),
        datetime.date(2024, 6, 6),
        datetime.date(2024, 6, 7),
    ]


def test_build_scoring_frame_hourly_uses_datetime() -> None:
    # GIVEN an HOUR-basis deployment spec and hourly timestamp strings
    reqs = _reqs(
        windows_basis_unit="HOUR",
        effective_feature_derivation_window_start=-2,
        forecast_distance=1,
        date_format="%Y-%m-%d %H:%M:%S",
    )
    dataset = pl.DataFrame(
        {
            "timestamp": [
                "2024-06-01 00:00:00",
                "2024-06-01 01:00:00",
                "2024-06-01 02:00:00",
                "2024-06-01 03:00:00",
            ],
            "series_id": ["A"] * 4,
            "target": [1.0, 2.0, 3.0, 4.0],
        }
    )

    # WHEN the scoring frame is built for a mid-day forecast point
    result = ts_mod._build_scoring_frame(dataset, reqs, dtime(2024, 6, 1, 2, 0, 0))

    # THEN the datetime column stays a Datetime (not truncated to Date)
    assert result["timestamp"].dtype == pl.Datetime
    assert result["timestamp"].to_list() == [
        dtime(2024, 6, 1, 0, 0),
        dtime(2024, 6, 1, 1, 0),
        dtime(2024, 6, 1, 2, 0),
        dtime(2024, 6, 1, 3, 0),
    ]


def test_build_scoring_frame_fills_future_window() -> None:
    # GIVEN a dataset whose history ends at the forecast point
    reqs = _reqs(
        effective_feature_derivation_window_start=-2,
        forecast_distance=3,
        date_format="%Y-%m-%d",
    )
    dataset = pl.DataFrame(
        {
            "timestamp": ["2024-06-01", "2024-06-02", "2024-06-03", "2024-06-04"],
            "series_id": ["A"] * 4,
            "target": [1.0, 2.0, 3.0, 4.0],
        }
    )

    # WHEN the scoring frame is built
    result = ts_mod._build_scoring_frame(dataset, reqs, dtime(2024, 6, 4))

    # THEN rows are generated through forecast_point + forecast_distance
    assert result["timestamp"].dtype == pl.Date
    assert result["timestamp"].to_list() == [
        datetime.date(2024, 6, 2),
        datetime.date(2024, 6, 3),
        datetime.date(2024, 6, 4),
        datetime.date(2024, 6, 5),
        datetime.date(2024, 6, 6),
        datetime.date(2024, 6, 7),
    ]
    # THEN historic rows keep the target; future rows are null (diagonal concat)
    assert result["target"].to_list() == [2.0, 3.0, None, None, None, None]


def test_build_scoring_frame_single_series() -> None:
    # GIVEN a single-series deployment spec (no multiseries id column)
    reqs = _reqs(
        multi_series_id_column=None,
        effective_feature_derivation_window_start=-2,
        forecast_distance=2,
        date_format="%Y-%m-%d",
    )
    dataset = pl.DataFrame(
        {
            "timestamp": ["2024-06-01", "2024-06-02", "2024-06-03", "2024-06-04"],
            "target": [1.0, 2.0, 3.0, 4.0],
        }
    )

    # WHEN the scoring frame is built
    result = ts_mod._build_scoring_frame(dataset, reqs, dtime(2024, 6, 4))

    # THEN no series column is invented and the window math still applies
    assert "series_id" not in result.columns
    assert result["timestamp"].to_list() == [
        datetime.date(2024, 6, 2),
        datetime.date(2024, 6, 3),
        datetime.date(2024, 6, 4),
        datetime.date(2024, 6, 5),
        datetime.date(2024, 6, 6),
    ]


def test_build_scoring_frame_missing_series_column_raises() -> None:
    # GIVEN a multi-series spec but a dataset without the series column
    reqs = _reqs(
        effective_feature_derivation_window_start=-2,
        forecast_distance=1,
        date_format="%Y-%m-%d",
    )
    dataset = pl.DataFrame({"timestamp": ["2024-06-01", "2024-06-02"], "target": [1.0, 2.0]})

    # WHEN/THEN building the frame raises an actionable validation error
    with pytest.raises(ToolError, match="series_id"):
        ts_mod._build_scoring_frame(dataset, reqs, dtime(2024, 6, 2))


def test_build_scoring_frame_known_in_advance_future_kept() -> None:
    # GIVEN a known-in-advance feature with values supplied past the forecast point
    reqs = _reqs(
        effective_feature_derivation_window_start=-2,
        forecast_distance=2,
        date_format="%Y-%m-%d",
        known_in_advance_features=["promo"],
    )
    dataset = pl.DataFrame(
        {
            "timestamp": [
                "2024-06-01",
                "2024-06-02",
                "2024-06-03",
                "2024-06-04",
                "2024-06-05",
            ],
            "series_id": ["A"] * 5,
            "target": [1.0, 2.0, 3.0, None, None],
            "promo": [0, 0, 1, 1, 0],
        }
    )

    # WHEN the scoring frame is built at 2024-06-03
    result = ts_mod._build_scoring_frame(dataset, reqs, dtime(2024, 6, 3)).sort("timestamp")

    # THEN future rows (>= forecast point) carry the known-in-advance promo values
    future = result.filter(pl.col("timestamp") >= datetime.date(2024, 6, 3))
    assert future["promo"].to_list() == [1, 1, 0]


def test_build_scoring_frame_unsupported_basis_unit_raises() -> None:
    # GIVEN a deployment spec with a basis unit the builder does not support
    reqs = _reqs(windows_basis_unit="YEAR")
    dataset = pl.DataFrame({"timestamp": ["2024-06-01"], "series_id": ["A"], "target": [1.0]})

    # WHEN/THEN building the frame raises a validation error naming the unit
    with pytest.raises(ToolError, match="basis unit"):
        ts_mod._build_scoring_frame(dataset, reqs, dtime(2024, 6, 1))


# --------------------------------------------------------------------------- #
# get_time_series_scoring_dataset_panel -- full facade, introspection mocked.
# --------------------------------------------------------------------------- #


async def test_ts_scoring_panel_writes_child(
    store: PanelStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    # GIVEN a source Dataset panel and mocked deployment scoring requirements
    reqs = _reqs(
        multi_series_id_column=None,
        effective_feature_derivation_window_start=-2,
        forecast_distance=2,
        date_format="%Y-%m-%d",
    )

    async def _fake_reqs(deployment_id: str) -> ts_mod.ScoringDataRequirements:
        return reqs

    monkeypatch.setattr(ts_mod, "_get_scoring_data_requirements", _fake_reqs)

    panel_id = await _make_dataset_panel(
        store,
        pl.DataFrame(
            {
                "timestamp": ["2024-06-01", "2024-06-02", "2024-06-03", "2024-06-04"],
                "target": [1.0, 2.0, 3.0, 4.0],
            }
        ),
        title="Sales",
    )

    # WHEN the tool builds the scoring dataset panel
    result = await ts_mod.get_time_series_scoring_dataset_panel(
        panel_id=panel_id, deployment_id="dep-1", forecast_point="2024-06-04"
    )

    # THEN a derived Dataset panel is written with lineage + scoring context
    assert result["type"] == "dataset"
    assert result["parents"] == [panel_id]
    # THEN the default source is the session-scoped staging area (kept from wren
    # so BPA facade delegation doesn't silently change where TS panels land)
    assert result["payload_path"].startswith("staging/")
    assert result["execution_context"]["kind"] == "ts_scoring"
    assert result["execution_context"]["deployment_id"] == "dep-1"
    assert result["title"].startswith("Scoring dataset (Sales")
    # THEN the stored parquet payload holds the built scoring frame
    payload = await store.get_payload(result["id"])
    assert payload is not None
    frame = pl.read_parquet(io.BytesIO(payload))
    assert frame["timestamp"].to_list() == [
        datetime.date(2024, 6, 2),
        datetime.date(2024, 6, 3),
        datetime.date(2024, 6, 4),
        datetime.date(2024, 6, 5),
        datetime.date(2024, 6, 6),
    ]


async def test_ts_scoring_panel_infers_forecast_point(
    store: PanelStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    # GIVEN a source panel and no explicit forecast_point
    reqs = _reqs(
        multi_series_id_column=None,
        effective_feature_derivation_window_start=-2,
        forecast_distance=1,
        date_format="%Y-%m-%d",
    )

    async def _fake_reqs(deployment_id: str) -> ts_mod.ScoringDataRequirements:
        return reqs

    monkeypatch.setattr(ts_mod, "_get_scoring_data_requirements", _fake_reqs)

    panel_id = await _make_dataset_panel(
        store,
        pl.DataFrame(
            {
                "timestamp": ["2024-06-01", "2024-06-02", "2024-06-03"],
                "target": [1.0, 2.0, 3.0],
            }
        ),
    )

    # WHEN the tool runs without a forecast_point
    result = await ts_mod.get_time_series_scoring_dataset_panel(
        panel_id=panel_id, deployment_id="dep-1"
    )

    # THEN the latest timestamp (2024-06-03) is the inferred forecast point
    assert result["execution_context"]["forecast_point"].startswith("2024-06-03")


def test_parse_forecast_point_converts_offset_to_utc() -> None:
    # GIVEN forecast-point strings in various ISO 8601 shapes
    # WHEN parsed
    # THEN a non-UTC offset is converted to UTC (matching the UTC-normalized
    # datetime column), not just stripped of its tz label
    assert ts_mod._parse_forecast_point("2025-01-15T12:00:00-05:00") == dtime(2025, 1, 15, 17)
    assert ts_mod._parse_forecast_point("2024-06-04T00:00:00Z") == dtime(2024, 6, 4)
    assert ts_mod._parse_forecast_point("2024-06-04") == dtime(2024, 6, 4)
    # THEN garbage input raises a validation error
    with pytest.raises(ToolError):
        ts_mod._parse_forecast_point("not-a-date")


async def test_ts_scoring_panel_non_dataset_raises(store: PanelStore) -> None:
    # GIVEN a non-Dataset (Json) source panel
    created = await store.create(Json(title="j", data={"k": 1}), source="staging")
    assert created.id is not None

    # WHEN/THEN the tool rejects it with a validation error
    with pytest.raises(ToolError):
        await ts_mod.get_time_series_scoring_dataset_panel(
            panel_id=created.id, deployment_id="dep-1"
        )


# --------------------------------------------------------------------------- #
# _get_scoring_data_requirements -- SDK introspection boundary mocked.
# --------------------------------------------------------------------------- #


async def test_get_scoring_data_requirements_introspection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # GIVEN a mocked SDK exposing a TS deployment's project/model/partitioning
    dr_mock = MagicMock(name="dr")
    monkeypatch.setattr(ts_mod, "dr", dr_mock)
    client = MagicMock(name="ThreadSafeDataRobotClient")
    client.return_value.request_user_client.return_value = contextlib.nullcontext()
    monkeypatch.setattr(ts_mod, "ThreadSafeDataRobotClient", client)

    deployment = MagicMock()
    deployment.model = {"project_id": "proj-1", "id": "model-1"}
    dr_mock.Deployment.get.return_value = deployment

    project = MagicMock()
    project.target = "sales (actual)"
    project.list_datetime_partition_spec.return_value = {
        "datetime_partition_column": "date",
        "multiseries_id_columns": ["store_id"],
        "feature_settings": None,
    }
    date_feature = MagicMock()
    date_feature.name = "date"
    date_feature.date_format = "%Y-%m-%d"
    project.get_features.return_value = [date_feature]
    dr_mock.Project.get.return_value = project

    model = MagicMock()
    model.effective_feature_derivation_window_start = -14
    model.forecast_window_end = 7
    model.windows_basis_unit = "DAY"
    dr_mock.DatetimeModel.get.return_value = model

    # WHEN scoring requirements are derived for the deployment
    reqs = await ts_mod._get_scoring_data_requirements("dep-1")

    # THEN every field is read from the SDK spec ('(actual)' suffixes stripped)
    assert reqs.target == "sales"
    assert reqs.multi_series_id_column == "store_id"
    assert reqs.date_time_column == "date"
    assert reqs.effective_feature_derivation_window_start == -14
    assert reqs.forecast_distance == 7
    assert reqs.windows_basis_unit == "DAY"
    assert reqs.date_format == "%Y-%m-%d"
    assert reqs.known_in_advance_features == []


async def test_get_scoring_data_requirements_no_partitioning_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # GIVEN a mocked SDK whose project has no datetime partitioning spec
    dr_mock = MagicMock(name="dr")
    monkeypatch.setattr(ts_mod, "dr", dr_mock)
    client = MagicMock(name="ThreadSafeDataRobotClient")
    client.return_value.request_user_client.return_value = contextlib.nullcontext()
    monkeypatch.setattr(ts_mod, "ThreadSafeDataRobotClient", client)

    deployment = MagicMock()
    deployment.model = {"project_id": "proj-1", "id": "model-1"}
    dr_mock.Deployment.get.return_value = deployment
    project = MagicMock()
    project.list_datetime_partition_spec.return_value = None
    dr_mock.Project.get.return_value = project
    dr_mock.DatetimeModel.get.return_value = MagicMock()

    # WHEN/THEN requirement derivation rejects the non-TS deployment
    with pytest.raises(ToolError, match="datetime-partitioned"):
        await ts_mod._get_scoring_data_requirements("dep-1")
