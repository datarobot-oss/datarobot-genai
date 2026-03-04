"""DataRobot dataset and datastore tools.

Ported from wren-mcp bi_data.py. Panel-specific types (DatasetRef, staging)
are replaced with plain dict returns.
"""
from __future__ import annotations

import logging

from datarobot_genai.mcp_tools._registry import register_tool

logger = logging.getLogger(__name__)


async def list_datarobot_datasets(
    search: str | None = None,
    limit: int = 100,
    use_case_id: str | None = None,
) -> list[dict]:
    """List DataRobot AI Catalog datasets with optional search filter.

    Returns a list of dicts with id, name, and uri fields.
    """
    import datarobot as dr

    client = dr.Client()
    params: dict = {"limit": limit}
    if search:
        params["search"] = search
    if use_case_id:
        params["useCaseId"] = use_case_id
    response = client.get("catalogItems/", params=params)
    items = response.json().get("data", [])
    return [
        {"id": item["id"], "name": item.get("name", ""), "uri": f"dataset://{item['id']}"}
        for item in items
    ]


async def get_datarobot_dataset(
    dataset_id: str,
    include_sample: bool = True,
    sample_rows: int = 10,
) -> dict:
    """Get DataRobot dataset metadata and optional sample rows.

    Returns id, name, created_at, row_count, columns, and sample data.
    """
    import datarobot as dr

    dataset = dr.Dataset.get(dataset_id)
    result: dict = {
        "id": dataset.id,
        "name": dataset.name,
        "created_at": str(dataset.created_at),
        "row_count": getattr(dataset, "row_count", None),
    }
    if include_sample:
        try:
            df = dataset.get_as_dataframe()
            result["columns"] = list(df.columns)
            result["sample"] = df.head(sample_rows).to_dict(orient="records")
        except Exception as exc:
            result["sample_error"] = str(exc)
    return result


async def upload_dataset_to_datarobot(
    name: str,
    rows: list[dict],
    use_case_id: str | None = None,
) -> dict:
    """Upload in-memory data to DataRobot AI Catalog as a new dataset.

    Returns the new dataset's id and name.
    """
    import datarobot as dr
    import pandas as pd

    df = pd.DataFrame(rows)
    dataset = dr.Dataset.create_from_in_memory_data(data_frame=df, dataset_name=name)
    result: dict = {"id": dataset.id, "name": dataset.name}
    if use_case_id:
        try:
            use_case = dr.UseCase.get(use_case_id)
            use_case.add_dataset(dataset.id)
            result["use_case_id"] = use_case_id
        except Exception as exc:
            result["use_case_add_error"] = str(exc)
    return result


async def list_datastores(
    show_all: bool = False,
) -> list[dict]:
    """List available DataRobot data connections (datastores).

    Returns a list of dicts with id, canonical_name, and params.
    """
    import datarobot as dr

    datastores = dr.DataStore.list()
    return [
        {
            "id": ds.id,
            "canonical_name": getattr(ds, "canonical_name", ""),
            "creator_id": getattr(ds, "creator_id", ""),
            "params": getattr(ds, "params", {}),
        }
        for ds in datastores
    ]


async def browse_datastore(
    datastore_id: str,
    path: str | None = None,
    offset: int = 0,
    limit: int = 100,
    search: str | None = None,
) -> str:
    """Browse a DataRobot data connection at a path.

    Navigate datastores to list catalogs, schemas, and tables.
    Returns formatted text listing of items at the given path.
    """
    import datarobot as dr

    client = dr.Client()
    params: dict = {"offset": offset, "limit": limit}
    if path:
        params["path"] = path
    if search:
        params["search"] = search
    response = client.get(f"externalDataDrivers/{datastore_id}/tables/", params=params)
    data = response.json()
    items = data.get("data", data) if isinstance(data, dict) else data
    lines = [f"Datastore {datastore_id} at path={path or '/'}:"]
    for item in items:
        lines.append(f"  {item}")
    return "\n".join(lines)


async def query_datastore(
    datastore_id: str,
    sql: str,
    limit: int = 1000,
) -> dict:
    """Execute a SQL query against a DataRobot datastore connection.

    Returns rows, columns, and row_count. Use limit to cap large results.
    """
    import datarobot as dr

    client = dr.Client()
    payload = {"query": sql, "limit": limit}
    response = client.post(f"externalDataDrivers/{datastore_id}/execute/", json=payload)
    data = response.json()
    return {
        "rows": data.get("data", []),
        "row_count": len(data.get("data", [])),
        "columns": data.get("columns", []),
    }


register_tool(
    "list_datarobot_datasets",
    list_datarobot_datasets,
    "List DataRobot AI Catalog datasets with optional search filter.",
    "wren_tools",
)
register_tool(
    "get_datarobot_dataset",
    get_datarobot_dataset,
    "Get DataRobot dataset metadata and optional sample rows.",
    "wren_tools",
)
register_tool(
    "upload_dataset_to_datarobot",
    upload_dataset_to_datarobot,
    "Upload in-memory data to DataRobot AI Catalog as a new dataset.",
    "wren_tools",
)
register_tool(
    "list_datastores",
    list_datastores,
    "List available DataRobot data connections (datastores).",
    "wren_tools",
)
register_tool(
    "browse_datastore",
    browse_datastore,
    "Browse a DataRobot data connection at a path.",
    "wren_tools",
)
register_tool(
    "query_datastore",
    query_datastore,
    "Execute a SQL query against a DataRobot datastore connection.",
    "wren_tools",
)
