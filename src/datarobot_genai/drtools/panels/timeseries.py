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

"""Time-series scoring dataset panel builder.

Port of wren-mcp's ``get_time_series_scoring_dataset_panel`` facade (itself a
port of wren's ``bi_forecasting`` + ``dr_helpers.get_scoring_dataset``
machinery). It introspects a deployment's datetime-partitioning / model
settings via the DataRobot SDK (scoped through
:class:`ThreadSafeDataRobotClient`) to derive the scoring-data requirements,
then builds a forward-looking scoring frame — recent history for the feature
derivation window plus the future forecast window, with known-in-advance
features carried into the future rows — in polars and stores it as a derived
Dataset panel. The frame-building math (and the ``_coerce_column_to_date``
date parsing) is ported column-for-column from wren.
"""

from __future__ import annotations

import asyncio
import io
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC
from datetime import date
from datetime import datetime
from datetime import timedelta
from typing import Annotated
from typing import Any

import datarobot as dr
import numpy as np
import pandas as pd
import polars as pl
from dateutil.relativedelta import relativedelta
from pandas.core.tools.datetimes import (  # type: ignore[attr-defined]
    _guess_datetime_format_for_array,
)

from datarobot_genai.drmcputils.clients.datarobot import ThreadSafeDataRobotClient
from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.exceptions import ToolErrorKind
from datarobot_genai.drmcputils.panels.access import _get_store
from datarobot_genai.drmcputils.panels.access import _require_mcp_sandbox
from datarobot_genai.drmcputils.panels.models import Dataset
from datarobot_genai.drmcputils.panels.store import DEFAULT_SOURCE
from datarobot_genai.drtools.core import tool_metadata

logger = logging.getLogger(__name__)

_PARQUET_CONTENT_TYPE = "application/vnd.apache.parquet"


def _frame_to_parquet(frame: pl.DataFrame) -> bytes:
    """Serialize a polars frame to Parquet bytes (panel Dataset payload format)."""
    buffer = io.BytesIO()
    frame.write_parquet(buffer)
    return buffer.getvalue()


@dataclass
class ScoringDataRequirements:
    """Deployment-derived requirements for building a TS scoring frame.

    Ported from wren's ``dr_helpers.ScoringDataRequirements``.
    """

    target: str
    multi_series_id_column: str | None
    date_time_column: str
    effective_feature_derivation_window_start: int
    forecast_distance: int
    windows_basis_unit: str
    known_in_advance_features: list[str]
    date_format: str | None = None

    def get_required_columns(self) -> list[str]:
        columns = [self.target, self.date_time_column, *self.known_in_advance_features]
        if self.multi_series_id_column is not None:
            columns.insert(1, self.multi_series_id_column)
        return columns


def _coerce_column_to_date(
    frame: pl.DataFrame,
    column: str,
    *,
    format_hints: Sequence[str | None] = (),
    as_datetime: bool = False,
) -> pl.DataFrame:
    """Convert a column to ``pl.Date``/``pl.Datetime``, trying format hints then pandas.

    Ported verbatim from wren's ``dr_helpers._coerce_column_to_date``. Existing
    temporal columns are normalized to the requested type.
    """
    if column not in frame.columns:
        raise ValueError(f"Column '{column}' not found in frame.")

    column_expr = pl.col(column)
    series = frame[column]

    target_dtype = pl.Datetime if as_datetime else pl.Date
    if series.dtype == target_dtype:
        return frame
    if series.dtype.is_temporal():
        if as_datetime:
            return frame.with_columns(column_expr.cast(pl.Datetime))
        return frame.with_columns(column_expr.dt.date())

    pandas_series = series.to_pandas()
    candidates: list[str | None] = [fmt for fmt in format_hints if fmt]

    sample_guess = _guess_datetime_format_for_array(np.asarray(pandas_series[:10], dtype=object))
    if sample_guess and sample_guess not in candidates:
        candidates.append(sample_guess)

    candidates.append(None)

    for fmt in candidates:
        parsed = pd.to_datetime(pandas_series, format=fmt, errors="coerce", utc=True)
        non_null_mask = ~pd.isna(pandas_series)
        parsed_non_null = parsed[non_null_mask]
        if parsed_non_null.isna().any():
            continue
        parsed_series = pl.from_pandas(pd.DataFrame({column: parsed}))[column]
        if as_datetime:
            parsed_series = parsed_series.cast(pl.Datetime)
        else:
            parsed_series = parsed_series.dt.date()
        return frame.with_columns(parsed_series.alias(column))

    raise ValueError(
        f"Unable to coerce column '{column}' to date using provided formats: {candidates}"
    )


def _build_future_grid(
    dataset: pl.DataFrame,
    *,
    date_col: str,
    series_col: str | None,
    date_values: list[date | datetime],
    extra_cols: list[str],
) -> pl.DataFrame:
    """Cartesian grid of (series x future dates). Ported from wren's helper."""
    if series_col is None:
        if not date_values:
            columns = [date_col] + extra_cols
            return dataset.head(0).select(columns)
        return pl.DataFrame({date_col: date_values})

    series_values = dataset[series_col].unique().to_list()
    if not series_values or not date_values:
        columns = [date_col, series_col] + extra_cols
        return dataset.head(0).select(columns)

    series_df = pl.DataFrame({series_col: series_values})
    dates_df = pl.DataFrame({date_col: date_values})
    return series_df.join(dates_df, how="cross")


def _validate_scoring_columns(columns: Sequence[str], reqs: ScoringDataRequirements) -> None:
    """Raise a clear error if the scoring data misses required columns.

    Ported from wren's ``dr_helpers._validate_scoring_columns``.
    """
    missing: list[str] = []
    if reqs.date_time_column not in columns:
        missing.append(reqs.date_time_column)
    if reqs.multi_series_id_column is not None and reqs.multi_series_id_column not in columns:
        missing.append(reqs.multi_series_id_column)
    if not missing:
        return

    hint = (
        " This deployment is multi-series; each scoring row must identify the series "
        f"it belongs to via the '{reqs.multi_series_id_column}' column. If you are "
        "scoring data for a single series, add the column with a constant value."
        if reqs.multi_series_id_column is not None and reqs.multi_series_id_column in missing
        else ""
    )
    raise ToolError(
        f"Scoring data is missing required column(s): {missing}. "
        f"The deployment expects the datetime partition column "
        f"'{reqs.date_time_column}'"
        + (
            f" and the series id column '{reqs.multi_series_id_column}'"
            if reqs.multi_series_id_column is not None
            else ""
        )
        + "."
        + hint,
        kind=ToolErrorKind.VALIDATION,
    )


def _build_scoring_frame(
    dataset: pl.DataFrame,
    reqs: ScoringDataRequirements,
    forecast_point: datetime,
) -> pl.DataFrame:
    """Build a forward-looking scoring frame (recent history + future window).

    Pure polars port of wren's ``dr_helpers.get_scoring_dataset`` body (minus the
    SDK ``get_scoring_data_requirements`` fetch, which the caller supplies). The
    output columns and row semantics match wren:

    * Historic rows: all source columns, for timestamps in the feature
      derivation window ``[forecast_point + fdw_start, forecast_point)``.
    * Future rows: the datetime column, the (optional) series column, and any
      known-in-advance features, for timestamps in
      ``[forecast_point, forecast_point + forecast_distance]``. Missing future
      known-in-advance values are left null via the left join / diagonal concat.
    """
    _validate_scoring_columns(dataset.columns, reqs)
    use_datetime = reqs.windows_basis_unit in {"HOUR"}
    forecast_point_value: date | datetime
    if use_datetime:
        forecast_point_value = forecast_point
    else:
        forecast_point_value = forecast_point.date()

    if reqs.multi_series_id_column is not None:
        if dataset[reqs.multi_series_id_column].dtype.is_integer():
            dataset = dataset.with_columns(
                pl.col(reqs.multi_series_id_column).cast(pl.Float64).cast(pl.String)
            )
        else:
            dataset = dataset.with_columns(pl.col(reqs.multi_series_id_column).cast(pl.String))

    dataset = _coerce_column_to_date(
        dataset,
        reqs.date_time_column,
        format_hints=(reqs.date_format,),
        as_datetime=use_datetime,
    )

    start_offset: timedelta | relativedelta
    if reqs.windows_basis_unit == "DAY":
        start_offset = timedelta(days=reqs.effective_feature_derivation_window_start)
        future_end = forecast_point_value + timedelta(days=reqs.forecast_distance)
        date_values = [
            forecast_point_value + timedelta(days=offset)
            for offset in range(reqs.forecast_distance + 1)
        ]
    elif reqs.windows_basis_unit == "WEEK":
        start_offset = timedelta(weeks=reqs.effective_feature_derivation_window_start)
        future_end = forecast_point_value + timedelta(weeks=reqs.forecast_distance)
        date_values = [
            forecast_point_value + timedelta(weeks=offset)
            for offset in range(reqs.forecast_distance + 1)
        ]
    elif reqs.windows_basis_unit == "HOUR":
        start_offset = timedelta(hours=reqs.effective_feature_derivation_window_start)
        future_end = forecast_point_value + timedelta(hours=reqs.forecast_distance)
        date_values = [
            forecast_point_value + timedelta(hours=offset)
            for offset in range(reqs.forecast_distance + 1)
        ]
    elif reqs.windows_basis_unit == "MONTH":
        start_offset = relativedelta(months=reqs.effective_feature_derivation_window_start)
        future_end = forecast_point_value + relativedelta(months=reqs.forecast_distance)
        date_values = [
            forecast_point_value + relativedelta(months=offset)
            for offset in range(reqs.forecast_distance + 1)
        ]
    else:
        raise ToolError(
            f"Unsupported windows basis unit: {reqs.windows_basis_unit}",
            kind=ToolErrorKind.VALIDATION,
        )

    historic_part = dataset.filter(
        (pl.col(reqs.date_time_column) >= forecast_point_value + start_offset)
        & (pl.col(reqs.date_time_column) < forecast_point_value)
    )

    future_columns: list[str] = [reqs.date_time_column]
    if reqs.multi_series_id_column is not None:
        future_columns.append(reqs.multi_series_id_column)
    future_columns.extend(reqs.known_in_advance_features)

    future_known = dataset.filter(
        (pl.col(reqs.date_time_column) >= forecast_point_value)
        & (pl.col(reqs.date_time_column) <= future_end)
    ).select(future_columns)

    extra_cols_offset = 2 if reqs.multi_series_id_column is not None else 1
    future_grid = _build_future_grid(
        dataset,
        date_col=reqs.date_time_column,
        series_col=reqs.multi_series_id_column,
        date_values=date_values,
        extra_cols=future_columns[extra_cols_offset:],
    )
    join_keys = [reqs.date_time_column]
    if reqs.multi_series_id_column is not None:
        join_keys.append(reqs.multi_series_id_column)
    future_part = future_grid.join(
        future_known,
        on=join_keys,
        how="left",
    ).select(future_columns)

    sort_keys = (
        [reqs.multi_series_id_column, reqs.date_time_column]
        if reqs.multi_series_id_column is not None
        else [reqs.date_time_column]
    )
    return pl.concat([historic_part, future_part], how="diagonal").sort(sort_keys)


# --- Deployment introspection (SDK) -> ScoringDataRequirements -------------- #


def _strip_actual_suffix(name: str) -> str:
    """Drop a trailing ' (actual)' marker and surrounding whitespace."""
    if name.endswith(" (actual)"):
        name = name[: -len(" (actual)")]
    return name.rstrip()


def _get_target_column(project: Any) -> str:
    return _strip_actual_suffix(str(project.target))


def _get_date_time_column(spec: Any) -> str:
    return _strip_actual_suffix(str(spec["datetime_partition_column"]))


def _get_multi_series_id_column(spec: Any) -> str | None:
    columns = spec["multiseries_id_columns"]
    if not columns:
        return None
    return _strip_actual_suffix(str(columns[0]))


def _get_date_format(features: list[Any], date_time_column: str) -> str | None:
    for feature in features:
        if feature.name == date_time_column:
            return None if feature.date_format is None else str(feature.date_format)
    return None


def _get_feature_lookup(project: Any) -> dict[str, str]:
    """Map modeling feature names back to their raw source feature names.

    Ported from wren's ``dr_helpers.get_feature_lookup``.
    """
    informative = project.get_featurelist_by_name("Informative Features")
    if informative is None or informative.features is None:
        raise ToolError(
            "Informative Features list is unavailable for this project.",
            kind=ToolErrorKind.UPSTREAM,
        )
    raw = project.get_featurelist_by_name("Raw Features")
    if raw is None or raw.features is None:
        raise ToolError(
            "Raw Features list is unavailable for this project.",
            kind=ToolErrorKind.UPSTREAM,
        )
    all_modeling_features = informative.features
    all_raw_features = raw.features
    feature_lookup: dict[str, str] = {}
    for f in all_modeling_features:
        try:
            raw_f = next(raw_f for raw_f in all_raw_features if f.startswith(raw_f))
            feature_lookup[f] = raw_f
        except StopIteration:
            continue
    for raw_f in all_raw_features:
        if raw_f not in feature_lookup.values():
            feature_lookup[raw_f] = raw_f
    return feature_lookup


def _get_known_in_advance_features(spec: Any, project: Any) -> list[str]:
    """Resolve known-in-advance feature names. Ported from wren."""
    feature_settings = spec["feature_settings"]
    if not feature_settings:
        return []
    feature_lookup = _get_feature_lookup(project)
    known = [
        feature_lookup[fs.feature_name]
        for fs in feature_settings
        if fs.known_in_advance and fs.feature_name in feature_lookup
    ]
    return list(set(known))


async def _get_scoring_data_requirements(deployment_id: str) -> ScoringDataRequirements:
    """Introspect a deployment's project/model/partitioning to derive scoring reqs.

    Port of wren's ``dr_helpers.get_scoring_data_requirements``: reads the
    datetime-partitioning spec, target, date format, model derivation-window /
    forecast settings, and known-in-advance features via the DataRobot SDK
    (scoped through :class:`ThreadSafeDataRobotClient`).
    """

    def _introspect() -> ScoringDataRequirements:
        deployment = dr.Deployment.get(deployment_id)
        if deployment.model is None:
            raise ToolError(
                f"Deployment {deployment_id} has no associated model.",
                kind=ToolErrorKind.VALIDATION,
            )
        project_id = str(deployment.model["project_id"])
        model_id = str(deployment.model["id"])
        project = dr.Project.get(project_id=project_id)
        model = dr.DatetimeModel.get(project=project_id, model_id=model_id)  # type: ignore[no-untyped-call]

        spec = project.list_datetime_partition_spec()
        if spec is None:
            raise ToolError(
                f"Deployment {deployment_id} is not datetime-partitioned; "
                "get_time_series_scoring_dataset_panel requires a time-series deployment.",
                kind=ToolErrorKind.VALIDATION,
            )

        date_time_column = _get_date_time_column(spec)
        try:
            features = project.get_features()
            date_format = _get_date_format(features, date_time_column)
        except Exception as exc:  # noqa: BLE001 - date_format is best-effort
            logger.debug("Unable to retrieve date format for '%s': %s", date_time_column, exc)
            date_format = None

        return ScoringDataRequirements(
            target=_get_target_column(project),
            multi_series_id_column=_get_multi_series_id_column(spec),
            date_time_column=date_time_column,
            effective_feature_derivation_window_start=int(
                model.effective_feature_derivation_window_start
            ),
            forecast_distance=int(model.forecast_window_end),
            windows_basis_unit=model.windows_basis_unit,
            known_in_advance_features=_get_known_in_advance_features(spec, project),
            date_format=date_format,
        )

    with ThreadSafeDataRobotClient().request_user_client():
        return await asyncio.to_thread(_introspect)


def _parse_forecast_point(value: str) -> datetime:
    """Parse an ISO 8601 forecast-point string into a naive UTC datetime.

    A tz-aware value is converted to UTC before the tz label is dropped, so it
    lines up with the datetime column, which ``_coerce_column_to_date``
    normalizes to UTC wall-clock (``pd.to_datetime(..., utc=True)``) before
    stripping the tz.
    """
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ToolError(
            f"Could not parse forecast_point '{value}'; use ISO 8601 "
            "(e.g. '2025-01-15' or '2025-01-15T00:00:00').",
            kind=ToolErrorKind.VALIDATION,
        ) from exc
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(UTC).replace(tzinfo=None)
    return parsed


def _infer_forecast_point(frame: pl.DataFrame, reqs: ScoringDataRequirements) -> datetime:
    """Derive a forecast point from the latest timestamp in the datetime column.

    Mirrors wren's ``bi_forecasting._infer_forecast_point`` / the DataRobot
    backend default when ``forecastPoint`` is omitted.
    """
    date_col = reqs.date_time_column
    if date_col not in frame.columns:
        raise ToolError(
            "forecast_point was not provided and cannot be inferred: scoring data is "
            f"missing the datetime partition column '{date_col}'.",
            kind=ToolErrorKind.VALIDATION,
        )
    parsed = _coerce_column_to_date(
        frame.select(date_col),
        date_col,
        format_hints=(reqs.date_format,),
        as_datetime=True,
    )
    max_value = parsed[date_col].drop_nulls().max()
    if max_value is None:
        raise ToolError(
            "forecast_point was not provided and cannot be inferred: datetime column "
            f"'{date_col}' contains no parseable values.",
            kind=ToolErrorKind.VALIDATION,
        )
    if isinstance(max_value, datetime):
        return max_value
    if isinstance(max_value, date):
        return datetime(max_value.year, max_value.month, max_value.day)
    return pd.Timestamp(max_value).to_pydatetime()  # type: ignore[arg-type]


@tool_metadata(
    tags={"panels", "write", "dataset", "timeseries", "daria"},
    description=(
        "[Panels—TS scoring dataset] Prepare a Dataset panel for time-series scoring against "
        "a deployment: introspects the deployment's datetime partitioning + forecast settings "
        "and builds a scoring frame (recent history + future forecast window), saved as a "
        "derived Dataset panel. Use this before scoring a time-series deployment."
    ),
    display_name="Panels — Create time-series scoring dataset",
    description_ui=(
        "Builds a correctly-shaped time-series scoring dataset panel for a deployment, "
        "combining recent history with the future forecast window."
    ),
)
async def get_time_series_scoring_dataset_panel(
    panel_id: Annotated[str, "Source Dataset panel to prepare for scoring."],
    deployment_id: Annotated[str, "The DataRobot deployment ID to score against."],
    forecast_point: Annotated[
        str | None,
        "Optional ISO 8601 forecast point. When omitted, the latest timestamp in the "
        "dataset's datetime partition column is used, matching the DataRobot backend default.",
    ] = None,
    source: Annotated[str, "Target source ('main' or 'staging')."] = DEFAULT_SOURCE,
) -> dict[str, Any]:
    """Build a time-series scoring Dataset panel for a deployment.

    Applies the deployment-specific transformations (feature derivation window +
    forecast window) needed to score a time-series deployment, producing a frame
    with the recent history and the future rows to forecast.
    """
    _require_mcp_sandbox()
    if not panel_id:
        raise ToolError("panel_id must be provided", kind=ToolErrorKind.VALIDATION)
    if not deployment_id:
        raise ToolError("deployment_id must be provided", kind=ToolErrorKind.VALIDATION)

    store = _get_store()
    source_panel = await store.get(panel_id)
    if not isinstance(source_panel, Dataset):
        raise ToolError(
            f"Panel {panel_id} is a {source_panel.type.value} panel; "
            "get_time_series_scoring_dataset_panel only supports Dataset panels.",
            kind=ToolErrorKind.VALIDATION,
        )
    raw = await store.get_payload(source_panel)
    if raw is None:
        raise ToolError(
            f"Dataset panel {panel_id} has no stored payload to prepare.",
            kind=ToolErrorKind.VALIDATION,
        )
    frame = pl.read_parquet(io.BytesIO(raw))

    reqs = await _get_scoring_data_requirements(deployment_id)

    if forecast_point is None:
        resolved_point = _infer_forecast_point(frame, reqs)
    else:
        resolved_point = _parse_forecast_point(forecast_point)

    scoring_frame = _build_scoring_frame(frame, reqs, resolved_point)

    title = f"Scoring dataset ({source_panel.title} {resolved_point})"
    panel = Dataset(
        title=title,
        description="Scoring dataset",
        parents=[panel_id],
        row_count=scoring_frame.height,
        columns=scoring_frame.columns,
        execution_context={
            "kind": "ts_scoring",
            "deployment_id": deployment_id,
            "forecast_point": resolved_point.isoformat(),
        },
    )
    created = await store.create(
        panel,
        source=source,
        payload=_frame_to_parquet(scoring_frame),
        payload_name="scoring-dataset.parquet",
        content_type=_PARQUET_CONTENT_TYPE,
    )
    return created.model_dump(mode="json")
