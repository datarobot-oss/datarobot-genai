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

"""cuOpt solution -> panels persistence (wren's ``_persist_panels`` over PanelStore).

Writes the extracted solution tables as Dataset panels (Parquet payloads), the
summary as an inline Text panel, and the raw solver output as an inline Json
panel -- all through the shared per-request :func:`_get_store` (conversation
scoping is automatic) into the ``staging`` source for review-before-promote.
"""

from __future__ import annotations

import io
import json
from typing import Any

import pandas as pd
import polars as pl

from datarobot_genai.drmcputils.panels.access import _get_store
from datarobot_genai.drmcputils.panels.models import Dataset
from datarobot_genai.drmcputils.panels.models import Json
from datarobot_genai.drmcputils.panels.models import Text

_PARQUET_CONTENT_TYPE = "application/vnd.apache.parquet"
_STAGING_SOURCE = "staging"


def _frame_to_parquet(frame: pl.DataFrame) -> bytes:
    """Serialize a polars frame to Parquet bytes (panel Dataset payload format)."""
    buffer = io.BytesIO()
    frame.write_parquet(buffer)
    return buffer.getvalue()


def _pandas_to_polars(df: pd.DataFrame) -> pl.DataFrame:
    """Convert a pandas frame to polars, stringifying columns polars can't infer.

    cuOpt normalization can leave object columns holding nested lists/dicts;
    those are JSON-encoded so the frame always serializes to Parquet cleanly.
    """
    try:
        return pl.from_pandas(df)
    except Exception:  # noqa: BLE001 - any conversion failure falls back to JSON-encoding
        safe = df.copy()
        for col in safe.columns:
            if safe[col].dtype == object:
                safe[col] = safe[col].apply(
                    lambda v: (
                        v if v is None or isinstance(v, (int, float, str, bool)) else json.dumps(v)
                    )
                )
        return pl.from_pandas(safe)


async def persist_cuopt_panels(
    *,
    tables: list[dict[str, Any]],
    summary_text: str,
    result_dict: dict[str, Any],
    problem_type: str,
    parent_id: str | None,
) -> dict[str, Any]:
    """Persist solution tables (Dataset), summary (Text), and raw solution (Json).

    Returns
    -------
        A dict carrying the summary text and the created panel manifests,
        ordered summary -> solution -> dataset tables (matching wren's
        BIResponse payload ordering). All panels link to ``parent_id`` (the
        input Json panel) when one is provided, preserving lineage.
    """
    store = _get_store()
    parents = [parent_id] if parent_id else []

    dataset_panels: list[dict[str, Any]] = []
    for table in tables:
        df: pd.DataFrame = table["dataframe"]
        frame = _pandas_to_polars(df)
        panel = Dataset(
            title=table["title"],
            description=table["description"],
            parents=list(parents),
            row_count=frame.height,
            columns=frame.columns,
            execution_context={
                "kind": "cuopt_solution",
                "problem_type": problem_type,
            },
        )
        created = await store.create(
            panel,
            source=_STAGING_SOURCE,
            payload=_frame_to_parquet(frame),
            payload_name=f"{table['title']}.parquet",
            content_type=_PARQUET_CONTENT_TYPE,
        )
        dataset_panels.append(created.model_dump(mode="json"))

    summary_panel: dict[str, Any] | None = None
    if summary_text:
        created_summary = await store.create(
            Text(
                title="cuOpt Optimization Summary",
                text=summary_text,
                description="Summary of cuOpt optimization results.",
                parents=list(parents),
                execution_context={
                    "kind": "cuopt_summary",
                    "problem_type": problem_type,
                },
            ),
            source=_STAGING_SOURCE,
        )
        summary_panel = created_summary.model_dump(mode="json")

    solution_panel: dict[str, Any] | None = None
    solution_data = result_dict.get("solution") or result_dict.get("solver_response")
    if solution_data:
        created_solution = await store.create(
            Json(
                title=f"cuOpt Solution - {problem_type.upper()}",
                data=solution_data,
                description="Raw solution data from cuOpt solver",
                parents=list(parents),
                execution_context={
                    "kind": "cuopt_solution_raw",
                    "problem_type": problem_type,
                },
            ),
            source=_STAGING_SOURCE,
        )
        solution_panel = created_solution.model_dump(mode="json")

    panels: list[dict[str, Any]] = []
    if summary_panel:
        panels.append(summary_panel)
    if solution_panel:
        panels.append(solution_panel)
    panels.extend(dataset_panels)

    return {
        "status": result_dict.get("status", "success"),
        "problem_type": problem_type,
        "comments": summary_text or "",
        "panels": panels,
    }
