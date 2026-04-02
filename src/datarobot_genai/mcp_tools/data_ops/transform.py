"""Generic dataset transform tool using code execution.

Delegates to the code execution sandbox from wren_tools. This module
is intentionally thin — the sandbox abstraction lives in wren_tools.
"""
from __future__ import annotations

import logging
from typing import Any

from datarobot_genai.mcp_tools._registry import register_tool

logger = logging.getLogger(__name__)


async def transform_data(
    dataset_id: str,
    code: str,
    result_dataset_name: str | None = None,
    upload_result: bool = False,
) -> dict[str, Any]:
    """Transform a DataRobot dataset by running Python code against it.

    The code receives a pandas DataFrame as `df` and must assign the
    transformed result back to `df`. Example:

        df = df[df['revenue'] > 0]
        df['profit_margin'] = df['profit'] / df['revenue']

    If upload_result=True, the transformed DataFrame is uploaded back
    to DataRobot as a new dataset named result_dataset_name.

    Returns: columns, row_count, sample rows (first 5), and optionally
    the new dataset ID if uploaded.
    """
    import datarobot as dr

    dataset = dr.Dataset.get(dataset_id)
    df = dataset.get_as_dataframe()

    # Inject df into exec namespace and run transform code
    namespace: dict[str, Any] = {"df": df}
    try:
        exec(compile(code, "<transform>", "exec"), namespace)  # noqa: S102
    except Exception as exc:
        return {"error": f"Transform code failed: {exc}", "dataset_id": dataset_id}

    result_df = namespace.get("df")
    if result_df is None:
        return {
            "error": "Transform code did not produce a DataFrame. Assign your result to `df`.",
            "dataset_id": dataset_id,
        }

    result: dict[str, Any] = {
        "dataset_id": dataset_id,
        "original_row_count": len(df),
        "result_row_count": len(result_df),
        "columns": list(result_df.columns),
        "sample": result_df.head(5).to_dict(orient="records"),
    }

    if upload_result:
        name = result_dataset_name or f"Transformed: {dataset.name}"
        new_dataset = dr.Dataset.create_from_in_memory_data(
            data_frame=result_df, dataset_name=name
        )
        result["new_dataset_id"] = new_dataset.id
        result["new_dataset_name"] = new_dataset.name

    return result


register_tool(
    "transform_data",
    transform_data,
    "Transform a DataRobot dataset by running Python code against it as a pandas DataFrame.",
    "data_ops",
)
