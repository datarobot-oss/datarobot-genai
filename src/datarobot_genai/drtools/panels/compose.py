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

"""Panel-composition tools (ported from wren-mcp, MODEL-24090).

These tools compose several primitives — the DataRobot SDK (scoped through
:class:`ThreadSafeDataRobotClient`), the panel store, and the predictive
delegates — to reproduce wren-mcp behaviors that have no single delegate
equivalent: materializing AI Catalog datasets as panels, round-tripping panels
back to the Catalog, SQL over datasets, prediction history/scoring into
lineage-linked child panels, AutoPilot progress reports, and deterministic
what-if adjustments.

Every panel write goes through :func:`_get_store` after the
``_require_mcp_sandbox()`` entitlement gate, and Dataset payloads are Parquet
(matching the in-tree ``datasource.py`` convention). Composition results land
in ``staging`` by default (wren's contract: staged first, promoted via
``move_panel``).

wren → genai name mapping:

* ``get_datarobot_dataset_as_panel``  → ``create_dataset_panel_from_catalog``
* ``upload_panel_dataset_to_datarobot`` → ``upload_dataset_panel_to_catalog``
* ``query_datarobot_dataset``         → ``query_datasets_to_panel``
* ``get_prediction_history``, ``get_autopilot_status``,
  ``predict_with_deployment``, ``apply_what_if`` keep their names.
"""

from __future__ import annotations

import asyncio
import base64
import datetime
import io
import logging
from typing import Annotated
from typing import Any

import datarobot as dr
import polars as pl

from datarobot_genai.drmcputils.clients.datarobot import ThreadSafeDataRobotClient
from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.exceptions import ToolErrorKind
from datarobot_genai.drmcputils.panels.access import _get_store
from datarobot_genai.drmcputils.panels.access import _require_mcp_sandbox
from datarobot_genai.drmcputils.panels.models import Dataset
from datarobot_genai.drtools.core import tool_metadata
from datarobot_genai.drtools.predictive.data import catalog_upload_dataset
from datarobot_genai.drtools.predictive.deployment import deployment_get_prediction_history
from datarobot_genai.drtools.predictive.predict_realtime import predict_score_inline_realtime

logger = logging.getLogger(__name__)

_PARQUET_CONTENT_TYPE = "application/vnd.apache.parquet"
_VALID_OPS = ("mul", "add", "set")
_STAGING_SOURCE = "staging"


def _frame_to_parquet(frame: pl.DataFrame) -> bytes:
    """Serialize a polars frame to Parquet bytes (panel Dataset payload format)."""
    buffer = io.BytesIO()
    frame.write_parquet(buffer)
    return buffer.getvalue()


def _instance_url(endpoint: str) -> str:
    """Strip the ``/api/v2`` suffix off an API endpoint to get the UI base URL."""
    return endpoint.rstrip("/").removesuffix("/api/v2")


@tool_metadata(
    tags={"panels", "write", "catalog", "dataset", "daria"},
    description=(
        "[Panels—from catalog] Convert a DataRobot AI Catalog dataset into a Dataset panel "
        "(Parquet payload stored via the Files API) for use with the panel tools. Records "
        "dataset_id for lineage/refresh."
    ),
    display_name="Panels — Create dataset from catalog",
    description_ui=(
        "Converts a DataRobot AI Catalog dataset into a dataset panel for use with the panel tools."
    ),
)
async def create_dataset_panel_from_catalog(
    *,
    dataset_id: Annotated[str, "The AI Catalog dataset ID to convert."],
    title: Annotated[str, "Human-readable panel title."],
    description: Annotated[str | None, "Optional short description."] = None,
    source: Annotated[str, "Target source ('main' or 'staging')."] = _STAGING_SOURCE,
    limit: Annotated[
        int | None,
        "Optional cap on rows to materialize; omit to load the whole dataset.",
    ] = None,
) -> dict[str, Any]:
    _require_mcp_sandbox()
    if not dataset_id:
        raise ToolError("dataset_id must be provided", kind=ToolErrorKind.VALIDATION)
    if not title:
        raise ToolError("title must be provided", kind=ToolErrorKind.VALIDATION)

    with ThreadSafeDataRobotClient().request_user_client():
        dataset = await asyncio.to_thread(dr.Dataset.get, dataset_id)
        pandas_df = await asyncio.to_thread(dataset.get_as_dataframe)

    frame = pl.from_pandas(pandas_df)
    if limit is not None and frame.height > limit:
        frame = frame.head(limit)

    panel = Dataset(
        title=title,
        description=description,
        row_count=frame.height,
        columns=frame.columns,
        execution_context={"kind": "catalog_dataset", "dataset_id": dataset_id},
    )
    created = await _get_store().create(
        panel,
        source=source,
        payload=_frame_to_parquet(frame),
        payload_name=f"{title}.parquet",
        content_type=_PARQUET_CONTENT_TYPE,
    )
    return created.model_dump(mode="json")


@tool_metadata(
    tags={"panels", "read", "catalog", "dataset", "daria"},
    description=(
        "[Panels—to catalog] Upload a Dataset panel's tabular data to the DataRobot AI "
        "Catalog as a new dataset (CSV-backed). Name defaults to the panel title."
    ),
    display_name="Panels — Upload dataset to catalog",
    description_ui=(
        "Uploads a dataset panel's tabular data to the DataRobot AI Catalog as a new dataset."
    ),
)
async def upload_dataset_panel_to_catalog(
    *,
    panel_id: Annotated[str, "The Dataset panel to upload."],
    name: Annotated[
        str | None,
        "Optional catalog dataset name (without extension); defaults to the panel title.",
    ] = None,
) -> dict[str, Any]:
    _require_mcp_sandbox()
    if not panel_id:
        raise ToolError("panel_id must be provided", kind=ToolErrorKind.VALIDATION)

    store = _get_store()
    panel = await store.get(panel_id)
    if not isinstance(panel, Dataset):
        raise ToolError(
            f"Panel {panel_id} is a {panel.type.value} panel; "
            "upload_dataset_panel_to_catalog only supports Dataset panels.",
            kind=ToolErrorKind.VALIDATION,
        )
    payload = await store.get_payload(panel)
    if payload is None:
        raise ToolError(
            f"Dataset panel {panel_id} has no stored payload to upload.",
            kind=ToolErrorKind.VALIDATION,
        )

    frame = pl.read_parquet(io.BytesIO(payload))
    csv_bytes = frame.write_csv().encode("utf-8")
    file_content_base64 = base64.b64encode(csv_bytes).decode("ascii")
    filename = f"{name or panel.title}.csv"
    return await catalog_upload_dataset(
        file_content_base64=file_content_base64,
        dataset_filename=filename,
    )


async def _materialize_frames(dataset_ids: list[str]) -> dict[str, pl.DataFrame]:
    """Fetch each AI Catalog dataset as a polars frame, bound to t0, t1, ..."""
    frames: dict[str, pl.DataFrame] = {}
    with ThreadSafeDataRobotClient().request_user_client():
        for index, dataset_id in enumerate(dataset_ids):
            dataset = await asyncio.to_thread(dr.Dataset.get, dataset_id)
            pandas_df = await asyncio.to_thread(dataset.get_as_dataframe)
            frames[f"t{index}"] = pl.from_pandas(pandas_df)
    return frames


@tool_metadata(
    tags={"panels", "write", "catalog", "dataset", "sql", "daria"},
    description=(
        "[Panels—query datasets] Query one or more DataRobot AI Catalog datasets with SQL "
        "and store the result as a Dataset panel. Datasets are bound positionally as t0, "
        "t1, ... in the query (polars SQL engine); the result always lands in 'staging'."
    ),
    display_name="Panels — Query datasets",
    description_ui=(
        "Queries one or more DataRobot AI Catalog datasets with SQL and stores the result "
        "as a dataset panel."
    ),
)
async def query_datasets_to_panel(
    *,
    query: Annotated[str, "SQL to execute. Tables are named t0, t1, ... by dataset position."],
    title: Annotated[str, "Title for the resulting Dataset panel."],
    description: Annotated[str, "Description for the resulting Dataset panel."],
    dataset_ids: Annotated[list[str], "AI Catalog dataset IDs to expose as t0, t1, ..."],
    use_case_id: Annotated[
        str | None,
        "Accepted for wren-mcp compatibility; not used by the local SQL engine.",
    ] = None,
    persist: Annotated[
        bool,
        "Retained for wren-mcp signature compatibility; a no-op here (in wren it "
        "controlled DataRobot-side Wrangle artifact preservation). The result panel "
        "always goes to 'staging', as in wren.",
    ] = False,
) -> dict[str, Any]:
    _require_mcp_sandbox()
    if not query:
        raise ToolError("query must be provided", kind=ToolErrorKind.VALIDATION)
    if not dataset_ids:
        raise ToolError("dataset_ids must be provided", kind=ToolErrorKind.VALIDATION)

    frames = await _materialize_frames(dataset_ids)

    sql_context = pl.SQLContext(frames=frames)
    try:
        result_frame = sql_context.execute(query, eager=True)
    except Exception as exc:  # noqa: BLE001 - any SQL failure -> actionable guidance
        raise ToolError(
            f"Could not execute the query with the polars SQL engine: {exc}. "
            "Rephrase the SQL (tables are named t0, t1, ...), or use "
            "transform_panel to run arbitrary Python over a panel instead.",
            kind=ToolErrorKind.VALIDATION,
        ) from exc

    panel = Dataset(
        title=title,
        description=description,
        row_count=result_frame.height,
        columns=result_frame.columns,
        execution_context={
            "kind": "sql_query",
            "src": query,
            "dataset_ids": dataset_ids,
        },
    )
    created = await _get_store().create(
        panel,
        source=_STAGING_SOURCE,
        payload=_frame_to_parquet(result_frame),
        payload_name=f"{title}.parquet",
        content_type=_PARQUET_CONTENT_TYPE,
    )
    return created.model_dump(mode="json")


@tool_metadata(
    tags={"panels", "write", "deployment", "dataset", "daria"},
    description=(
        "[Panels—prediction history] Retrieve a deployment's stored prediction history "
        "and store it as a Dataset panel (in 'staging'). Supports limit/offset paging and "
        "optional ISO 8601 time bounds."
    ),
    display_name="Panels — Get prediction history",
    description_ui=(
        "Retrieves a deployment's stored prediction history and stores it as a dataset panel."
    ),
)
async def get_prediction_history(
    *,
    deployment_id: Annotated[str, "The MLOps deployment ID."],
    limit: Annotated[int, "Max prediction rows to fetch."] = 100,
    offset: Annotated[int, "Rows to skip (pagination)."] = 0,
    start_time: Annotated[str | None, "Optional ISO 8601 lower bound on prediction time."] = None,
    end_time: Annotated[str | None, "Optional ISO 8601 upper bound on prediction time."] = None,
) -> dict[str, Any]:
    _require_mcp_sandbox()
    if not deployment_id:
        raise ToolError("deployment_id must be provided", kind=ToolErrorKind.VALIDATION)

    history = await deployment_get_prediction_history(
        deployment_id=deployment_id,
        limit=limit,
        offset=offset,
        start_time=start_time,
        end_time=end_time,
    )
    rows: list[dict[str, Any]] = history.get("rows") or []
    columns = list(rows[0].keys()) if rows else []
    frame = pl.DataFrame(rows) if rows else pl.DataFrame()

    panel = Dataset(
        title=f"Prediction history ({deployment_id})",
        description="Prediction history retrieved from the DataRobot deployment.",
        row_count=frame.height,
        columns=frame.columns if rows else columns,
        execution_context={
            "kind": "prediction_history",
            "deployment_id": deployment_id,
        },
    )
    created = await _get_store().create(
        panel,
        source=_STAGING_SOURCE,
        payload=_frame_to_parquet(frame),
        payload_name=f"prediction-history-{deployment_id}.parquet",
        content_type=_PARQUET_CONTENT_TYPE,
    )
    return {
        "panel": created.model_dump(mode="json"),
        "deployment_id": deployment_id,
        "row_count": history.get("row_count", frame.height),
        "has_more": history.get("has_more", False),
    }


@tool_metadata(
    tags={"panels", "read", "modeling", "daria"},
    description=(
        "[Panels—autopilot status] Check the progress of an AutoPilot run: whether it is "
        "complete, queued/running model counts, leaderboard size, and the leaderboard URL. "
        "Poll until the report starts with 'AutoPilot status: complete'."
    ),
    display_name="Panels — Get AutoPilot status",
    description_ui=(
        "Checks the progress of an AutoPilot run, including queued/running model counts "
        "and the leaderboard URL."
    ),
)
async def get_autopilot_status(
    *,
    project_id: Annotated[str, "DataRobot project ID of the AutoPilot run."],
    use_case_id: Annotated[
        str | None,
        "Optional use case ID, used only to build the leaderboard URL when the project "
        "record does not carry one.",
    ] = None,
) -> dict[str, Any]:
    from datarobot.enums import QUEUE_STATUS

    _require_mcp_sandbox()
    if not project_id:
        raise ToolError("project_id must be provided", kind=ToolErrorKind.VALIDATION)

    client = ThreadSafeDataRobotClient()
    with client.request_user_client():
        project = await asyncio.to_thread(dr.Project.get, project_id)
        status, jobs, models = await asyncio.gather(
            asyncio.to_thread(project.get_status),
            asyncio.to_thread(project.get_model_jobs),
            asyncio.to_thread(project.get_models),
        )

    queued = sum(
        1
        for job in jobs
        if (getattr(job, "status", "") or "").lower() == QUEUE_STATUS.QUEUE.lower()
    )
    running = sum(
        1
        for job in jobs
        if (getattr(job, "status", "") or "").lower() == QUEUE_STATUS.INPROGRESS.lower()
    )
    leaderboard_count = len(models)

    autopilot_done = bool(status.get("autopilot_done"))
    stage = status.get("stage") or ""
    stage_description = status.get("stage_description") or ""

    if autopilot_done:
        first_line = "AutoPilot status: complete"
    else:
        if stage and stage_description:
            stage_part = f" (stage: {stage} - {stage_description})"
        elif stage:
            stage_part = f" (stage: {stage})"
        else:
            stage_part = ""
        first_line = f"AutoPilot status: in progress{stage_part}"

    lines = [
        first_line,
        f"Models queued: {queued}",
        f"Models running: {running}",
        f"Models on leaderboard: {leaderboard_count}",
    ]

    resolved_use_case = getattr(project, "use_case_id", None) or use_case_id
    leaderboard_url: str | None = None
    if resolved_use_case:
        leaderboard_url = (
            f"{_instance_url(client.endpoint)}/usecases/{resolved_use_case}"
            f"/experiment/{project_id}/leaderboard"
        )
        lines.append(f"Leaderboard: {leaderboard_url}")

    return {
        "report": "\n".join(lines),
        "autopilot_done": autopilot_done,
        "stage": stage,
        "models_queued": queued,
        "models_running": running,
        "models_on_leaderboard": leaderboard_count,
        "leaderboard_url": leaderboard_url,
    }


@tool_metadata(
    tags={"panels", "write", "deployment", "predictions", "dataset", "daria"},
    description=(
        "[Panels—predict] Make predictions using a DataRobot deployment on a scoring "
        "Dataset panel and store the results as a lineage-linked child Dataset panel "
        "(in 'staging'). add_explanations includes prediction explanations (slow — only "
        "if needed). For time series deployments, forecast_point is optional: when "
        "omitted the backend defaults to the latest timestamp in the scoring data."
    ),
    display_name="Panels — Predict with deployment",
    description_ui=(
        "Makes predictions using a DataRobot deployment on a scoring dataset panel and "
        "stores the results as a child dataset panel."
    ),
)
async def predict_with_deployment(
    *,
    panel_id: Annotated[str, "Dataset panel containing scoring data."],
    deployment_id: Annotated[str, "The DataRobot deployment ID to use for predictions."],
    add_explanations: Annotated[bool, "Whether to include prediction explanations (slow)."] = False,
    forecast_point: Annotated[
        str | None,
        "ISO 8601 forecast point for the predictions. Optional even for time series "
        "deployments; when omitted, the latest timestamp in the scoring data's datetime "
        "partition column is used (the DataRobot backend's default).",
    ] = None,
) -> dict[str, Any]:
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
            "predict_with_deployment only supports Dataset panels.",
            kind=ToolErrorKind.VALIDATION,
        )
    payload = await store.get_payload(source_panel)
    if payload is None:
        raise ToolError(
            f"Dataset panel {panel_id} has no stored payload to score.",
            kind=ToolErrorKind.VALIDATION,
        )
    scoring_csv = pl.read_parquet(io.BytesIO(payload)).write_csv()

    # Wren mapped add_explanations=True to max_explanations=10 in its predict
    # helpers; the delegate handles TS/non-TS branching internally.
    result = await predict_score_inline_realtime(
        deployment_id=deployment_id,
        dataset=scoring_csv,
        forecast_point=forecast_point,
        max_explanations=10 if add_explanations else 0,
    )
    predictions_csv = result.get("data")
    if not predictions_csv:
        raise ToolError(
            f"Deployment {deployment_id} returned no inline prediction data "
            f"(response type: {result.get('type')!r}).",
            kind=ToolErrorKind.UPSTREAM,
        )
    predictions = pl.read_csv(io.StringIO(predictions_csv))

    suffix = " with explanations" if add_explanations else ""
    title_point = f" {forecast_point}" if forecast_point else ""
    panel = Dataset(
        title=f"Predictions ({source_panel.title}{title_point}){suffix}",
        description=f"Predictions{suffix}",
        parents=[panel_id],
        row_count=predictions.height,
        columns=predictions.columns,
        execution_context={
            "kind": "prediction",
            "deployment_id": deployment_id,
        },
    )
    created = await store.create(
        panel,
        source=_STAGING_SOURCE,
        payload=_frame_to_parquet(predictions),
        payload_name="predictions.parquet",
        content_type=_PARQUET_CONTENT_TYPE,
    )
    return created.model_dump(mode="json")


def _coerce_date(value: Any) -> datetime.date | None:
    """Coerce an adjustment date bound (str/date/datetime) to its date portion.

    Comparisons are done on the date portion only (wren's contract), so a
    datetime bound is truncated to its date.
    """
    if value is None:
        return None
    if isinstance(value, datetime.datetime):
        return value.date()
    if isinstance(value, datetime.date):
        return value
    if isinstance(value, str):
        try:
            return datetime.date.fromisoformat(value)
        except ValueError:
            try:
                return datetime.datetime.fromisoformat(value).date()
            except ValueError as exc:
                raise ToolError(
                    f"Could not parse date '{value}'; use ISO 8601 (YYYY-MM-DD).",
                    kind=ToolErrorKind.VALIDATION,
                ) from exc
    raise ToolError(f"Unsupported date value: {value!r}", kind=ToolErrorKind.VALIDATION)


def _apply_adjustments(
    frame: pl.DataFrame,
    adjustments: list[dict[str, Any]],
    *,
    date_col: str,
    series_col: str | None,
) -> pl.DataFrame:
    """Apply what-if adjustments to a polars frame sequentially (deterministic).

    Each adjustment: ``{column, op(mul|add|set), value, from_date?, to_date?,
    series?}``. Date bounds are inclusive and compared on the date portion
    only: a ``pl.Datetime`` date column is truncated via ``.dt.date()`` so
    rows with a time component on a boundary date are still included.
    Operations are applied in order, mirroring wren's ``apply_what_if``.
    """
    # Compare on the date portion: Datetime columns are truncated to Date so a
    # row at 10:00 on the boundary date still matches an inclusive bound.
    if isinstance(frame.schema.get(date_col), pl.Datetime):
        date_expr = pl.col(date_col).dt.date()
    else:
        date_expr = pl.col(date_col)

    existing_columns = set(frame.columns)
    for adjustment in adjustments:
        column_name = adjustment.get("column")
        if not isinstance(column_name, str) or column_name not in existing_columns:
            raise ToolError(
                f"Unknown or missing column in adjustment: {column_name}",
                kind=ToolErrorKind.VALIDATION,
            )

        op = (adjustment.get("op") or "mul").lower()
        if op not in _VALID_OPS:
            raise ToolError(
                f"Unsupported op '{op}'. Use one of 'mul', 'add', 'set'.",
                kind=ToolErrorKind.VALIDATION,
            )

        value = adjustment.get("value")
        if value is None:
            raise ToolError("Adjustment missing 'value'.", kind=ToolErrorKind.VALIDATION)

        series_filter = adjustment.get("series")
        from_date = _coerce_date(adjustment.get("from_date"))
        to_date = _coerce_date(adjustment.get("to_date"))

        cond = pl.lit(True)
        if series_filter is not None:
            if series_col is None:
                raise ToolError(
                    "This deployment is single-series; 'series' filter is not applicable.",
                    kind=ToolErrorKind.VALIDATION,
                )
            if not isinstance(series_filter, (list, tuple)):
                raise ToolError(
                    "'series' must be a list of series identifiers.",
                    kind=ToolErrorKind.VALIDATION,
                )
            cond = cond & pl.col(series_col).is_in(list(series_filter))
        if from_date is not None and to_date is not None:
            cond = cond & date_expr.is_between(from_date, to_date, closed="both")
        elif from_date is not None:
            cond = cond & (date_expr >= from_date)
        elif to_date is not None:
            cond = cond & (date_expr <= to_date)

        if op == "mul":
            updated = pl.col(column_name) * pl.lit(value)
        elif op == "add":
            updated = pl.col(column_name) + pl.lit(value)
        else:  # set
            updated = pl.lit(value)

        frame = frame.with_columns(
            pl.when(cond).then(updated).otherwise(pl.col(column_name)).alias(column_name)
        )
    return frame


async def _datetime_partition_columns(deployment_id: str) -> tuple[str, str | None]:
    """Resolve the (date_col, series_col) of a deployment's datetime partitioning.

    Raises ToolError if the deployment's project is not datetime-partitioned
    (i.e. a standard-CV/tabular deployment), before any adjustment runs.
    """
    with ThreadSafeDataRobotClient().request_user_client():
        deployment = await asyncio.to_thread(dr.Deployment.get, deployment_id)
        if deployment.model is None:
            raise ToolError(
                f"Deployment {deployment_id} has no associated model.",
                kind=ToolErrorKind.VALIDATION,
            )
        project_id = str(deployment.model["project_id"])
        project = await asyncio.to_thread(dr.Project.get, project_id)
        spec = await asyncio.to_thread(project.list_datetime_partition_spec)

    if spec is None:
        raise ToolError(
            f"Deployment {deployment_id} is not datetime-partitioned; apply_what_if "
            "requires a time-series or OTV deployment.",
            kind=ToolErrorKind.VALIDATION,
        )

    # DatetimePartitioningSpecification is a trafaret record: subscriptable at
    # runtime, but not typed as such (matches wren's access pattern).
    date_col = str(spec["datetime_partition_column"]).removesuffix(" (actual)").rstrip()  # type: ignore[index]
    series_columns = spec["multiseries_id_columns"]  # type: ignore[index]
    series_col: str | None = None
    if series_columns:
        series_col = str(series_columns[0]).removesuffix(" (actual)").rstrip()
    return date_col, series_col


@tool_metadata(
    tags={"panels", "write", "deployment", "dataset", "daria"},
    description=(
        "[Panels—what-if] Apply deterministic what-if adjustments (mul/add/set, optionally "
        "scoped by inclusive date window and series) to a scoring Dataset panel and store "
        "the result as a lineage-linked child panel. Requires a datetime-partitioned "
        "(time-series or OTV) deployment; operations apply sequentially. If this tool "
        "raises, surface the error rather than silently swapping to transform_panel."
    ),
    display_name="Panels — Apply what-if",
    description_ui=(
        "Applies deterministic what-if adjustments to a scoring dataset panel and stores "
        "the result as a child panel."
    ),
)
async def apply_what_if(
    *,
    panel_id: Annotated[str, "The scoring Dataset panel to adjust."],
    deployment_id: Annotated[
        str, "The DataRobot deployment ID (validated for datetime partitioning)."
    ],
    adjustments: Annotated[
        list[dict[str, Any]],
        "Adjustment dicts with keys: column, op (mul|add|set; default mul), value, "
        "optional from_date/to_date (inclusive, ISO 8601), optional series (list of "
        "series ids). Applied sequentially.",
    ],
    title: Annotated[str | None, "Optional title for the result panel."] = None,
    source: Annotated[str, "Target source ('main' or 'staging')."] = _STAGING_SOURCE,
) -> dict[str, Any]:
    _require_mcp_sandbox()
    if not panel_id:
        raise ToolError("panel_id must be provided", kind=ToolErrorKind.VALIDATION)
    if not deployment_id:
        raise ToolError("deployment_id must be provided", kind=ToolErrorKind.VALIDATION)

    store = _get_store()
    panel = await store.get(panel_id)
    if not isinstance(panel, Dataset):
        raise ToolError(
            f"Panel {panel_id} is a {panel.type.value} panel; apply_what_if only "
            "supports Dataset panels.",
            kind=ToolErrorKind.VALIDATION,
        )
    payload = await store.get_payload(panel)
    if payload is None:
        raise ToolError(
            f"Dataset panel {panel_id} has no stored payload to adjust.",
            kind=ToolErrorKind.VALIDATION,
        )

    date_col, series_col = await _datetime_partition_columns(deployment_id)

    frame = pl.read_parquet(io.BytesIO(payload))
    adjusted = _apply_adjustments(frame, adjustments, date_col=date_col, series_col=series_col)

    result_panel = Dataset(
        title=title or "What-if adjusted dataset",
        description="What-if adjusted dataset",
        parents=[panel_id],
        row_count=adjusted.height,
        columns=adjusted.columns,
        execution_context={
            "kind": "what_if",
            "deployment_id": deployment_id,
            "adjustments": adjustments,
        },
    )
    created = await store.create(
        result_panel,
        source=source,
        payload=_frame_to_parquet(adjusted),
        payload_name="what-if-adjusted.parquet",
        content_type=_PARQUET_CONTENT_TYPE,
    )
    return created.model_dump(mode="json")
