"""Generic dataset aggregation tool.

Allows grouping and aggregating any DataRobot dataset.
"""
from __future__ import annotations

import logging
from typing import Any

from datarobot_genai.mcp_tools._registry import register_tool

logger = logging.getLogger(__name__)

_SUPPORTED_FUNCS = {"sum", "mean", "min", "max", "count", "std", "median", "first", "last"}


async def aggregate_data(
    dataset_id: str,
    group_by: list[str],
    aggregations: list[dict],
    limit: int = 100,
) -> dict[str, Any]:
    """Aggregate a DataRobot dataset by groups. Returns grouped results.

    aggregations format:
      [{"column": "revenue", "func": "sum"}, {"column": "quantity", "func": "mean"}]

    Supported functions: sum, mean, min, max, count, std, median, first, last

    This is a generic operation — works on any dataset accessible via DataRobot API.
    """
    import datarobot as dr

    dataset = dr.Dataset.get(dataset_id)
    df = dataset.get_as_dataframe()

    missing_group_cols = [c for c in group_by if c not in df.columns]
    if missing_group_cols:
        raise ValueError(f"Group-by columns not found: {missing_group_cols}")

    agg_dict: dict[str, Any] = {}
    for agg in aggregations:
        col = agg["column"]
        func = agg["func"]
        if col not in df.columns:
            raise ValueError(f"Aggregation column '{col}' not found.")
        if func not in _SUPPORTED_FUNCS:
            raise ValueError(f"Unsupported aggregation function '{func}'. Supported: {_SUPPORTED_FUNCS}")
        if col in agg_dict:
            # Multiple aggs on same column: use list form
            existing = agg_dict[col]
            agg_dict[col] = existing if isinstance(existing, list) else [existing]
            agg_dict[col].append(func)
        else:
            agg_dict[col] = func

    grouped = df.groupby(group_by).agg(agg_dict).reset_index()

    # Flatten multi-level column names if they exist
    if hasattr(grouped.columns, "levels"):
        grouped.columns = ["_".join(filter(None, col)) for col in grouped.columns]

    grouped = grouped.head(limit)

    return {
        "dataset_id": dataset_id,
        "group_by": group_by,
        "aggregations": aggregations,
        "row_count": len(grouped),
        "columns": list(grouped.columns),
        "rows": grouped.to_dict(orient="records"),
    }


register_tool(
    "aggregate_data",
    aggregate_data,
    "Aggregate a DataRobot dataset by groups using sum, mean, count, etc.",
    "data_ops",
)
