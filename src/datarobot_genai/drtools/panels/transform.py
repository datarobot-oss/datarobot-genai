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

"""Sandbox-backed panel filter/transform tools.

These run user-supplied filtering/transform code over a Dataset panel inside the
isolated DataRobot workload sandbox (``execute_code``), then materialize the
result as a derived child panel (with lineage back to the source). Use this for
derived in-memory transforms/aggregations; push *source-side* filtering into the
connector SQL instead (see :mod:`datarobot_genai.drtools.panels.datasource`).
"""

from __future__ import annotations

import io
import logging
from typing import Annotated
from typing import Any

import polars as pl

from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.exceptions import ToolErrorKind
from datarobot_genai.drmcputils.panels.access import _get_store
from datarobot_genai.drmcputils.panels.access import _require_mcp_sandbox
from datarobot_genai.drmcputils.panels.models import Dataset
from datarobot_genai.drmcputils.panels.store import DEFAULT_SOURCE
from datarobot_genai.drtools.core import tool_metadata
from datarobot_genai.drtools.panels.datasource import _rows_to_parquet
from datarobot_genai.drtools.sandbox import execute_code as _execute_code

logger = logging.getLogger(__name__)

_PARQUET_CONTENT_TYPE = "application/vnd.apache.parquet"

# The source dataset is bound as a polars DataFrame `df`; user code assigns the
# result rows to `_return`.
_TRANSFORM_PREAMBLE = "import polars as pl\ndf = pl.DataFrame(inputs['rows'])"


async def _run_transform(
    *,
    panel_id: str,
    code: str,
    title: str,
    description: str | None,
    source: str,
) -> dict[str, Any]:
    _require_mcp_sandbox()
    if not panel_id:
        raise ToolError("panel_id must be provided", kind=ToolErrorKind.VALIDATION)
    if not title:
        raise ToolError("title must be provided", kind=ToolErrorKind.VALIDATION)

    store = _get_store()
    source_panel = await store.get(panel_id)
    if not isinstance(source_panel, Dataset):
        raise ToolError(
            f"Panel {panel_id} is a {source_panel.type.value} panel; only Dataset panels "
            "can be transformed.",
            kind=ToolErrorKind.VALIDATION,
        )
    raw = await store.get_payload(source_panel)
    if raw is None:
        raise ToolError(
            "Source panel has no dataset payload to transform.",
            kind=ToolErrorKind.VALIDATION,
        )
    source_frame = pl.read_parquet(io.BytesIO(raw))
    rows = source_frame.to_dicts()

    result = await _execute_code(f"{_TRANSFORM_PREAMBLE}\n{code}", inputs={"rows": rows})
    out_rows = result.get("return_value")
    if not isinstance(out_rows, list) or not all(isinstance(row, dict) for row in out_rows):
        raise ToolError(
            "Transform code must assign a list of row dicts to `_return`.",
            kind=ToolErrorKind.VALIDATION,
        )

    if out_rows:
        columns = list(out_rows[0].keys())
    else:
        # Zero rows: keep the source schema so downstream consumers still see
        # the column names (e.g. a filter that matched nothing).
        columns = source_frame.columns
    child = Dataset(
        title=title,
        description=description,
        row_count=len(out_rows),
        columns=columns,
        parents=[panel_id],
        execution_context={"kind": "sandbox_transform", "source_panel": panel_id, "code": code},
    )
    created = await store.create(
        child,
        source=source,
        payload=_rows_to_parquet(out_rows, columns),
        payload_name=f"{title}.parquet",
        content_type=_PARQUET_CONTENT_TYPE,
    )
    return created.model_dump(mode="json")


@tool_metadata(
    tags={"panels", "write", "transform", "sandbox", "daria"},
    description=(
        "[Panels—transform] Run Python over a Dataset panel in the sandbox and save the result "
        "as a derived child panel (lineage preserved). The source dataset is bound as a polars "
        "DataFrame `df`; assign the resulting rows to `_return` (a list of dicts), e.g. "
        "`_return = df.group_by('region').agg(pl.col('rev').sum()).to_dicts()`."
    ),
)
async def transform_panel(
    *,
    panel_id: Annotated[str, "Source Dataset panel id."],
    code: Annotated[str, "Python operating on `df`; assign result rows to `_return`."],
    title: Annotated[str, "Title for the derived panel."],
    description: Annotated[str | None, "Optional description."] = None,
    source: Annotated[str, "Target source ('main' or 'staging')."] = DEFAULT_SOURCE,
) -> dict[str, Any]:
    return await _run_transform(
        panel_id=panel_id, code=code, title=title, description=description, source=source
    )


@tool_metadata(
    tags={"panels", "write", "filter", "sandbox", "daria"},
    description=(
        "[Panels—filter] Filter a Dataset panel by a polars boolean expression in the sandbox, "
        "saving the filtered rows as a derived child panel. `where` is a polars expression over "
        "`df`, e.g. \"pl.col('rev') > 1000\"."
    ),
)
async def filter_panel(
    *,
    panel_id: Annotated[str, "Source Dataset panel id."],
    where: Annotated[str, "polars boolean expression, e.g. \"pl.col('rev') > 1000\"."],
    title: Annotated[str, "Title for the filtered panel."],
    description: Annotated[str | None, "Optional description."] = None,
    source: Annotated[str, "Target source ('main' or 'staging')."] = DEFAULT_SOURCE,
) -> dict[str, Any]:
    if not where:
        raise ToolError("where expression must be provided", kind=ToolErrorKind.VALIDATION)
    return await _run_transform(
        panel_id=panel_id,
        code=f"_return = df.filter({where}).to_dicts()",
        title=title,
        description=description,
        source=source,
    )
