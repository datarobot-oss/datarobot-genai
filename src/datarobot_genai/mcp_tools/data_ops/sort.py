"""Generic dataset sort tool."""
from __future__ import annotations

import logging
from typing import Any

from datarobot_genai.mcp_tools._registry import register_tool

logger = logging.getLogger(__name__)


async def sort_data(
    dataset_id: str,
    sort_by: list[dict],
    limit: int = 100,
    columns: list[str] | None = None,
) -> dict[str, Any]:
    """Sort a DataRobot dataset by one or more columns. Returns sorted rows.

    sort_by format:
      [{"column": "date", "direction": "desc"}, {"column": "revenue", "direction": "asc"}]

    direction must be 'asc' or 'desc' (default 'asc').
    columns: optional list of columns to return.

    This is a generic operation — works on any dataset accessible via DataRobot API.
    """
    import datarobot as dr

    dataset = dr.Dataset.get(dataset_id)
    df = dataset.get_as_dataframe()

    sort_cols = []
    ascending = []
    for s in sort_by:
        col = s["column"]
        direction = s.get("direction", "asc").lower()
        if col not in df.columns:
            raise ValueError(f"Sort column '{col}' not found. Available: {list(df.columns)}")
        if direction not in ("asc", "desc"):
            raise ValueError(f"direction must be 'asc' or 'desc', got '{direction}'")
        sort_cols.append(col)
        ascending.append(direction == "asc")

    df = df.sort_values(by=sort_cols, ascending=ascending)

    if columns:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"Columns not found: {missing}")
        df = df[columns]

    df = df.head(limit)

    return {
        "dataset_id": dataset_id,
        "sort_by": sort_by,
        "row_count": len(df),
        "columns": list(df.columns),
        "rows": df.to_dict(orient="records"),
    }


register_tool(
    "sort_data",
    sort_data,
    "Sort a DataRobot dataset by one or more columns and return sorted rows.",
    "data_ops",
)
