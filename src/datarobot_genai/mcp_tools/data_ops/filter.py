"""Generic dataset filter tool.

Allows filtering any DataRobot dataset by column conditions.
This is a generic operation — the panels client library uses it
for panel filtering, but any MCP client can use it too.
"""
from __future__ import annotations

import logging
from typing import Any

from datarobot_genai.mcp_tools._registry import register_tool

logger = logging.getLogger(__name__)

_SUPPORTED_OPS = {"==", "!=", ">", ">=", "<", "<=", "in", "not in", "contains", "startswith"}


def _apply_filter(df: Any, f: dict) -> Any:
    """Apply a single filter condition to a pandas DataFrame."""
    col = f["column"]
    op = f["op"]
    val = f["value"]

    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found. Available: {list(df.columns)}")

    series = df[col]
    if op == "==":
        return df[series == val]
    elif op == "!=":
        return df[series != val]
    elif op == ">":
        return df[series > val]
    elif op == ">=":
        return df[series >= val]
    elif op == "<":
        return df[series < val]
    elif op == "<=":
        return df[series <= val]
    elif op == "in":
        return df[series.isin(val)]
    elif op == "not in":
        return df[~series.isin(val)]
    elif op == "contains":
        return df[series.astype(str).str.contains(str(val), na=False)]
    elif op == "startswith":
        return df[series.astype(str).str.startswith(str(val), na=False)]
    else:
        raise ValueError(f"Unsupported operator '{op}'. Supported: {_SUPPORTED_OPS}")


async def filter_data(
    dataset_id: str,
    filters: list[dict],
    limit: int = 100,
    columns: list[str] | None = None,
) -> dict[str, Any]:
    """Filter a DataRobot dataset by column conditions. Returns matching rows.

    filters format:
      [{"column": "age", "op": ">", "value": 30}, ...]

    Supported operators: ==, !=, >, >=, <, <=, in, not in, contains, startswith

    columns: optional list of columns to return (all columns returned if None).

    This is a generic operation — works on any dataset accessible via DataRobot API.
    """
    import datarobot as dr

    dataset = dr.Dataset.get(dataset_id)
    df = dataset.get_as_dataframe()

    original_count = len(df)
    for f in filters:
        df = _apply_filter(df, f)

    if columns:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"Columns not found: {missing}")
        df = df[columns]

    df = df.head(limit)

    return {
        "dataset_id": dataset_id,
        "filters_applied": len(filters),
        "original_row_count": original_count,
        "filtered_row_count": len(df),
        "columns": list(df.columns),
        "rows": df.to_dict(orient="records"),
    }


register_tool(
    "filter_data",
    filter_data,
    "Filter a DataRobot dataset by column conditions and return matching rows.",
    "data_ops",
)
