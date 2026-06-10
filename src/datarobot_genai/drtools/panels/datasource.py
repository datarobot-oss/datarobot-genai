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

"""Connector-sourced panels.

Lets a Dataset panel contain (and filter) data from *any* datasource supported
by the DataRobot connectors framework. Source-side filtering is pushed down via
SQL (``catalog_query_datastore``); the result is materialized to Parquet and
stored as the panel's payload through the Files API. The originating
``datastore_id`` + ``sql`` are recorded in ``execution_context`` for lineage and
later refresh.
"""

from __future__ import annotations

import io
import logging
from typing import Annotated
from typing import Any

import polars as pl

from datarobot_genai.drtools.core import tool_metadata
from datarobot_genai.drtools.core.exceptions import ToolError
from datarobot_genai.drtools.core.exceptions import ToolErrorKind
from datarobot_genai.drtools.pagination import PAGINATION_MAX
from datarobot_genai.drtools.panels.models import Dataset
from datarobot_genai.drtools.panels.store import DEFAULT_SOURCE
from datarobot_genai.drtools.panels.tools import _get_store
from datarobot_genai.drtools.panels.tools import _require_mcp_sandbox
from datarobot_genai.drtools.predictive.data import catalog_query_datastore

logger = logging.getLogger(__name__)

_PARQUET_CONTENT_TYPE = "application/vnd.apache.parquet"


def _rows_to_parquet(rows: list[dict[str, Any]], columns: list[str] | None = None) -> bytes:
    """Serialize query rows (list of column→value dicts) to Parquet bytes.

    ``columns`` keeps the Parquet schema in agreement with the panel's
    ``columns`` metadata: rows are normalized to exactly those keys (missing
    keys become null), and an empty result still carries the column schema.
    """
    if columns:
        frame = (
            pl.DataFrame([{c: row.get(c) for c in columns} for row in rows])
            if rows
            else pl.DataFrame({c: [] for c in columns})
        )
    else:
        frame = pl.DataFrame(rows)
    buffer = io.BytesIO()
    frame.write_parquet(buffer)
    return buffer.getvalue()


@tool_metadata(
    tags={"panels", "write", "connector", "dataset", "daria"},
    description=(
        "[Panels—from connector] Run SQL against a DataRobot connector/datastore and "
        "materialize the result as a Dataset panel (Parquet payload stored via the Files API). "
        "Push WHERE/LIMIT into the SQL to filter at the source. Records datastore_id + sql for "
        "lineage/refresh. datastore_id comes from list_datastores."
    ),
)
async def create_dataset_panel_from_connector(
    *,
    datastore_id: Annotated[str, "Connector/datastore id (from list_datastores)."],
    sql: Annotated[str, "SQL to run; push WHERE/LIMIT here to filter at the source."],
    title: Annotated[str, "Human-readable panel title."],
    description: Annotated[str | None, "Optional short description."] = None,
    source: Annotated[str, "Target source ('main' or 'staging')."] = DEFAULT_SOURCE,
    limit: Annotated[int, "Max rows to fetch (default 100)."] = PAGINATION_MAX,
) -> dict[str, Any]:
    _require_mcp_sandbox()
    if not datastore_id:
        raise ToolError("datastore_id must be provided", kind=ToolErrorKind.VALIDATION)
    if not sql:
        raise ToolError("sql must be provided", kind=ToolErrorKind.VALIDATION)
    if not title:
        raise ToolError("title must be provided", kind=ToolErrorKind.VALIDATION)

    result = await catalog_query_datastore(datastore_id=datastore_id, sql=sql, limit=limit)
    rows: list[dict[str, Any]] = result.get("rows") or []
    columns = result.get("columns") or (
        list(rows[0].keys()) if rows and isinstance(rows[0], dict) else []
    )
    row_count = result.get("row_count", len(rows))

    panel = Dataset(
        title=title,
        description=description,
        row_count=row_count,
        columns=columns,
        execution_context={"kind": "connector_query", "datastore_id": datastore_id, "sql": sql},
    )
    created = await _get_store().create(
        panel,
        source=source,
        payload=_rows_to_parquet(rows, columns),
        payload_name=f"{title}.parquet",
        content_type=_PARQUET_CONTENT_TYPE,
    )
    return created.model_dump(mode="json")


@tool_metadata(
    tags={"panels", "read", "dataset", "preview", "daria"},
    description=(
        "[Panels—preview dataset] Preview a Dataset panel's tabular payload: columns, "
        "dtypes, row count, and the first sample_size rows. Read-only; tabular panels "
        "only (for Json panels use get_panel). Ported from wren-mcp's preview_data_source."
    ),
)
async def preview_dataset_panel(
    panel_id: Annotated[str, "Id of the Dataset panel to preview."],
    *,
    sample_size: Annotated[int, "Rows to include in the sample (1-100, default 10)."] = 10,
) -> dict[str, Any]:
    _require_mcp_sandbox()
    if not panel_id:
        raise ToolError("panel_id must be provided", kind=ToolErrorKind.VALIDATION)
    sample_size = min(max(sample_size, 1), PAGINATION_MAX)

    store = _get_store()
    panel = await store.get(panel_id)
    if not isinstance(panel, Dataset):
        raise ToolError(
            f"Panel {panel_id} is a {panel.type.value} panel; preview_dataset_panel only "
            "supports Dataset panels (use get_panel for other types).",
            kind=ToolErrorKind.VALIDATION,
        )
    payload = await store.get_payload(panel)
    if payload is None:
        raise ToolError(
            f"Dataset panel {panel_id} has no stored payload to preview.",
            kind=ToolErrorKind.VALIDATION,
        )

    frame = pl.read_parquet(io.BytesIO(payload))
    return {
        "panel_id": panel_id,
        "title": panel.title,
        "columns": frame.columns,
        "dtypes": {name: str(dtype) for name, dtype in zip(frame.columns, frame.dtypes)},
        "row_count": frame.height,
        "sample": frame.head(sample_size).to_dicts(),
        "execution_context": panel.execution_context,
    }
